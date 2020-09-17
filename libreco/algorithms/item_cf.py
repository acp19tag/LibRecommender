import random
from operator import itemgetter
from itertools import islice, takewhile
from collections import defaultdict
import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm
from .base import Base
from ..utils.similarities import cosine_sim, pearson_sim, jaccard_sim
from ..utils.misc import time_block, colorize
from ..evaluate.evaluate import EvalMixin


class ItemCF(Base, EvalMixin):
    def __init__(
            self,
            task,
            data_info,
            sim_type="cosine",
            k=20,
            lower_upper_bound=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        EvalMixin.__init__(self, task)

        self.task = task
        self.k = k
        self.default_prediction = data_info.global_mean if (
                task == "rating") else 0.0
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.sim_type = sim_type
        self.user_consumed = data_info.user_consumed
        # sparse matrix, user as row and item as column
        self.user_interaction = None
        # sparse matrix, item as row and user as column
        self.item_interaction = None
        # sparse similarity matrix
        self.sim_matrix = None
        self.topk_sim = None
        self.print_count = 0
        self._caution_sim_type()

    def fit(self, train_data, block_size=None, num_threads=1, min_common=1,
            mode="invert", verbose=1, eval_data=None, metrics=None,
            store_top_k=True):
        self.show_start_time()
        self.user_interaction = train_data.sparse_interaction
        self.item_interaction = self.user_interaction.T.tocsr()

        with time_block("sim_matrix", verbose=1):
            if self.sim_type == "cosine":
                sim_func = cosine_sim
            elif self.sim_type == "pearson":
                sim_func = pearson_sim
            elif self.sim_type == "jaccard":
                sim_func = jaccard_sim
            else:
                raise ValueError(
                    "sim_type must be one of ('cosine', 'pearson', 'jaccard')"
                )

            self.sim_matrix = sim_func(
                self.item_interaction, self.user_interaction, self.n_items,
                self.n_users, block_size, num_threads, min_common, mode
            )

        assert self.sim_matrix.has_sorted_indices
        if issparse(self.sim_matrix):
            n_elements = self.sim_matrix.getnnz()
            sparsity_ratio = 100*n_elements / (self.n_users*self.n_users)
            print(f"sim_matrix, shape: {self.sim_matrix.shape}, "
                  f"num_elements: {n_elements}, "
                  f"sparsity: {sparsity_ratio:5.4f} %")
        if store_top_k:
            self.compute_top_k()

        if verbose > 1:
            self.print_metrics(eval_data=eval_data, metrics=metrics)
            print("=" * 30)

    def predict(self, user, item):
        user = (
            np.asarray([user])
            if isinstance(user, int)
            else np.asarray(user)
        )
        item = (
            np.asarray([item])
            if isinstance(item, int)
            else np.asarray(item)
        )
        unknown_num, unknown_index, user, item = self._check_unknown(
            user, item)

        preds = []
        sim_matrix = self.sim_matrix
        interaction = self.user_interaction
        for u, i in zip(user, item):
            item_slice = slice(sim_matrix.indptr[i], sim_matrix.indptr[i+1])
            sim_items = sim_matrix.indices[item_slice]
            sim_values = sim_matrix.data[item_slice]

            user_slice = slice(interaction.indptr[u], interaction.indptr[u+1])
            user_interacted_i = interaction.indices[user_slice]
            user_interacted_values = interaction.data[user_slice]
            common_items, indices_in_i, indices_in_u = np.intersect1d(
                sim_items, user_interacted_i,
                assume_unique=True, return_indices=True)

            common_sims = sim_values[indices_in_i]
            common_labels = user_interacted_values[indices_in_u]
            if common_items.size == 0 or np.all(common_sims <= 0.):
                self.print_count += 1
                no_str = (f"No common interaction or similar neighbor "
                          f"for user {u} and item {i}, "
                          f"proceed with default prediction")
                if self.print_count < 7:
                    print(f"{colorize(no_str, 'red')}")
                preds.append(self.default_prediction)
            else:
                k_neighbor_labels, k_neighbor_sims = zip(
                    *islice(
                        takewhile(
                            lambda x: x[1] > 0,
                            sorted(zip(common_labels, common_sims),
                                   key=itemgetter(1), reverse=True),
                        ),
                        self.k,
                    )
                )

                if self.task == "rating":
                    sims_distribution = (
                            k_neighbor_sims / np.sum(k_neighbor_sims)
                    )
                    weighted_pred = np.average(
                        k_neighbor_labels, weights=sims_distribution
                    )
                    preds.append(
                        np.clip(weighted_pred, self.lower_bound,
                                self.upper_bound)
                    )
                elif self.task == "ranking":
                    preds.append(np.mean(k_neighbor_sims))

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, random_rec=False):
        user = self._check_unknown_user(user)
        if not user:
            return   # popular ?

        u_consumed = set(self.user_consumed[user])
        user_slice = slice(self.user_interaction.indptr[user],
                           self.user_interaction.indptr[user + 1])
        user_interacted_i = self.user_interaction.indices[user_slice]
        user_interacted_labels = self.user_interaction.data[user_slice]

        result = defaultdict(lambda: 0.0)
        for i, i_label in zip(user_interacted_i, user_interacted_labels):
            if self.topk_sim is not None:
                item_sim_topk = self.topk_sim[i]
            else:
                item_slice = slice(self.sim_matrix.indptr[i],
                                   self.sim_matrix.indptr[i+1])
                sim_items = self.sim_matrix.indices[item_slice]
                sim_values = self.sim_matrix.data[item_slice]
                item_sim_topk = sorted(
                    zip(sim_items, sim_values),
                    key=itemgetter(1),
                    reverse=True
                )[:self.k]

            for j, sim in item_sim_topk:
                if j in u_consumed:
                    continue
                result[j] += sim * i_label

        if len(result) == 0:
            self.print_count += 1
            no_str = (f"no suitable recommendation for user {user}, "
                      f"return default recommendation")
            if self.print_count < 7:
                print(f"{colorize(no_str, 'red')}")
            return -1

        rank_items = [(k, v) for k, v in result.items()]
        rank_items.sort(key=lambda x: -x[1])
        if random_rec:
            if len(rank_items) < n_rec:
                item_candidates = rank_items
            else:
                item_candidates = random.sample(rank_items, k=n_rec)
            return item_candidates
        else:
            return rank_items[:n_rec]

    def _caution_sim_type(self):
        if self.task == "ranking" and self.sim_type == "pearson":
            caution_str = (f"Warning: {self.sim_type} is not suitable "
                           f"for implicit data")
            print(f"{colorize(caution_str, 'red')}")
        if self.task == "rating" and self.sim_type == "jaccard":
            caution_str = (f"Warning: {self.sim_type} is not suitable "
                           f"for explicit data")
            print(f"{colorize(caution_str, 'red')}")

    def compute_top_k(self):
        top_k = dict()
        for i in tqdm(range(self.n_items), desc="top_k"):
            item_slice = slice(self.sim_matrix.indptr[i],
                               self.sim_matrix.indptr[i+1])
            sim_items = self.sim_matrix.indices[item_slice].tolist()
            sim_values = self.sim_matrix.data[item_slice].tolist()
            top_k[i] = sorted(
                zip(sim_items, sim_values),
                key=itemgetter(1),
                reverse=True
            )[:self.k]
        self.topk_sim = top_k

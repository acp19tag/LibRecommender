import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse_svd
from ..utils.similarities import *
from ..utils.intersect import get_intersect
from ..utils.baseline_estimates import baseline_als, baseline_sgd
try:
    import tensorflow as tf
    tf.enable_eager_execution()
    tfe = tf.contrib.eager
except ModuleNotFoundError:
    print("you need tensorflow-eager for tf-version of this model")


class superSVD:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_training=True, k=50, min_support=1,
                 sim_option="pearson", seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_training = batch_training
        self.seed = seed
        self.k = k
        self.min_support = min_support
        if sim_option == "cosine":
            self.sim_option = cosine_sim
        elif sim_option == "msd":
            self.sim_option = msd_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)


    def fit(self, dataset):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        self.train_user_indices = dataset.train_user_indices
        self.train_item_indices = dataset.train_item_indices
        self.train_ratings = dataset.train_ratings
        self.test_user_indices = dataset.test_user_indices
        self.test_item_indices = dataset.test_item_indices
        self.test_ratings = dataset.test_ratings
        self.bbu, self.bbi = baseline_als(dataset)

        self.bu = np.zeros((self.n_users,))
        self.bi = np.zeros((self.n_items,))
        self.pu = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_users, self.n_factors))
        self.qi = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_items, self.n_factors))
        self.yj = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_items, self.n_factors))
        self.w = np.random.normal(loc=0.0, scale=0.1,
                                  size=(self.n_items, self.n_items))
        self.c = np.random.normal(loc=0.0, scale=0.1,
                                  size=(self.n_items, self.n_items))
        time_sim = time.time()
        self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
                                                       self.min_support, self.k)
        print("sim intersect time: {:.4f}".format(time.time() - time_sim))
        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(self.train_user_indices,
                                   self.train_item_indices,
                                   self.train_ratings):
                    u_items = list(self.train_user[u].keys())
                    nu_sqrt = np.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[i], self.pu[u] + nui)
                    intersect_items, index_u = self.intersect_user_item_train[(u, i)]

                    if len(intersect_items) == 0:
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])

                    else:
                        u_ratings = np.array(list(self.train_user[u].values()))[index_u]
                        base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                        user_sqrt = np.sqrt(len(intersect_items))
                        ru = np.sum((u_ratings - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                        nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot + ru + nu)

                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])
                        self.w[i][intersect_items] += \
                            self.lr * (err * (u_ratings - base_neighbor) / user_sqrt -
                                                                 self.reg * self.w[i][intersect_items])
                        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
                                                                 self.reg * self.c[i][intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))

        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                random_users = np.random.permutation(list(self.train_user.keys()))
                for u in random_users:
                    u_items = list(self.train_user[u].keys())
                    u_ratings = np.array(list(self.train_user[u].values()))
                    nu_sqrt = np.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[u_items], self.pu[u] + nui)

                    ru = []
                    nu = []
                    for single_item in u_items:
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            ru.append(0)
                            nu.append(0)
                        else:
                            u_ratings_intersect = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = np.sqrt(len(intersect_items))
                            ru_single = np.sum((u_ratings_intersect - base_neighbor) *
                                               self.w[single_item][intersect_items]) / user_sqrt
                            nu_single = np.sum(self.c[single_item][intersect_items]) / user_sqrt
                            ru.append(ru_single)
                            nu.append(nu_single)
                    ru = np.array(ru)
                    nu = np.array(nu)

                    err = u_ratings - (self.global_mean + self.bu[u] + self.bi[u_items] + dot + ru + nu)
                    err = err.reshape(len(u_items), 1)
                    self.bu[u] += self.lr * (np.sum(err) - self.reg * self.bu[u])
                    self.bi[u_items] += self.lr * (err.flatten() - self.reg * self.bi[u_items])
                    self.qi[u_items] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[u_items])
                    self.pu[u] += self.lr * (np.sum(err * self.qi[u_items], axis=0) - self.reg * self.pu[u])
                    self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt - self.reg * self.yj[u_items])

                    for single_item, error in zip(u_items, err):
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            continue
                        else:
                            u_ratings = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = np.sqrt(len(intersect_items))
                            self.w[single_item, intersect_items] += self.lr * (
                                    error.flatten() * (u_ratings - base_neighbor) / user_sqrt -
                                    self.reg * self.w[single_item, intersect_items])
                            self.c[single_item, intersect_items] += self.lr * (error.flatten() / user_sqrt -
                                                                    self.reg * self.c[single_item, intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))

    def predict(self, u, i):
        try:
            u_items = list(self.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nui, self.qi[i])

            try:
                intersect_items, index_u = self.intersect_user_item_train[(u, i)]
            except KeyError:
                intersect_items, index_u = [], -1

            if len(intersect_items) == 0:
                pass
            else:
                u_ratings = np.array(list(self.train_user[u].values()))[index_u]
                base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_ratings - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred

    def rmse(self, dataset, mode="train"):
        if mode == "train":
            user_indices = dataset.train_user_indices
            item_indices = dataset.train_item_indices
            ratings = dataset.train_ratings
        elif mode == "test":
            user_indices = dataset.test_user_indices
            item_indices = dataset.test_item_indices
            ratings = dataset.test_ratings

        pred = []
        for u, i in zip(user_indices, item_indices):
            p = self.predict(u, i)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score



class superSVD_tf:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_training=True, k=50, min_support=1,
                 sim_option="pearson", seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_training = batch_training
        self.seed = seed
        self.k = k
        self.min_support = min_support
        if sim_option == "cosine":
            self.sim_option = cosine_sim
        elif sim_option == "msd":
            self.sim_option = msd_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)

    def fit(self, dataset):
        start_time = time.time()
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_ratings = dataset.train_ratings
        test_ratings = dataset.test_ratings
        global_mean = dataset.global_mean

        bu = tf.Variable(tf.zeros([dataset.n_users]))
        bi = tf.Variable(tf.zeros([dataset.n_items]))
        pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        w = tf.Variable(tf.random_normal([dataset.n_items, dataset.n_items], 0.0, 0.01))
        c = tf.Variable(tf.random_normal([dataset.n_items, dataset.n_items], 0.0, 0.01))
        bbu, bbi = baseline_als(dataset)

    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
                                                       self.min_support, self.k)

        def compute_loss():
            pred_whole = []
            for u, i in zip(dataset.train_user_indices, dataset.train_item_indices):
                u_items = np.array(list(dataset.train_user[u].keys()))
                nui = tf.reduce_sum(tf.gather(yj, u_items), axis=0) / \
                           tf.sqrt(tf.cast(tf.size(u_items), tf.float32))

                dot = tf.reduce_sum(tf.multiply(tf.gather(pu, u) + nui, tf.gather(qi, i)))
                pred = global_mean + tf.gather(bu, u) + tf.gather(bi, i) + dot

                try:
                    intersect_items, index_u = self.intersect_user_item_train[(u, i)]
                except KeyError:
                    intersect_items, index_u = [], -1

                if len(intersect_items) == 0:
                    pass
                else:
                    u_ratings = np.array(list(dataset.train_user[u].values()))[index_u]
                    base_neighbor = global_mean + bbu[u] + bbi[intersect_items]
                    user_sqrt = tf.sqrt(tf.cast(tf.size(intersect_items), tf.float32))
                    ru = tf.cast(tf.reduce_sum(
                            (u_ratings - base_neighbor) *
                                tf.gather(tf.gather(w, i), intersect_items)), tf.float32) / user_sqrt
                    nu = tf.cast(tf.reduce_sum(tf.gather(tf.gather(c, i), intersect_items)), tf.float32) / user_sqrt
                    pred += ru + nu
                pred_whole.append(pred)

            pred_whole = tf.convert_to_tensor(np.array(pred_whole))
            score = tf.reduce_sum(tf.pow(pred_whole - dataset.train_ratings, 2)) + \
                        self.reg * (tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi) + tf.nn.l2_loss(pu) +
                           tf.nn.l2_loss(qi) + tf.nn.l2_loss(yj) + tf.nn.l2_loss(w) + tf.nn.l2_loss(c))
            return score

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            for u, i in zip(train_user_indices, train_item_indices):
                with tf.GradientTape() as tape:
                    variables = [bu, bi, pu, qi, yj, w, c]
                    loss = compute_loss()
                    grads = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(grads, variables))

            train_loss = compute_loss().numpy()
            print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
            print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

        self.pu = pu.numpy()
        self.qi = qi.numpy()
        self.yj = yj.numpy()
        self.bu = bu.numpy()
        self.bi = bi.numpy()
        self.w = w.numpy()
        self.c = c.numpy()
        self.bbu = bbu
        self.bbi = bbi
        self.global_mean = global_mean
        self.dataset = dataset

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nui, self.qi[i])

            try:
                intersect_items, index_u = self.intersect_user_item_train[(u, i)]
            except KeyError:
                intersect_items, index_u = [], -1

            if len(intersect_items) == 0:
                pass
            else:
                u_ratings = np.array(list(self.dataset.train_user[u].values()))[index_u]
                base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_ratings - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred













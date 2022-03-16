import numpy as np
import pandas as pd
from libreco.data import split_by_ratio_chrono, DatasetFeat
from libreco.algorithms import YouTubeRanking

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### PARAMETERS

embed_size = 300
n_epochs = 20
lr = 1e-4
batch_size = 512
hidden_units = "128,64,32"

### FUNCTIONS

def split_train_test_by_windowid(df, test_size = 0.2):
    first_time = True
    for windowid in df['WindowID'].unique():
        if first_time:
            test_df = df.groupby('WindowID').get_group(windowid).sample(frac = test_size)
            first_time = False
        else:
            test_df.append(df.groupby('WindowID').get_group(windowid).sample(frac = test_size))
    train_df = df.drop(test_df.index)
    return train_df, test_df

def generate_recommendation_output(user_list, job_list, output_dir):
    output_dict = {}
    for i, user in enumerate(user_list):
      if user not in output_dict:
        output_dict[user] = [job_list[i]]
      else:
        output_dict[user].append(job_list[i])

    with open(output_dir, 'w') as outfile:
        outfile.write("UserId, JobIds\n")
        for user, value in output_dict.items():
            outfile.write(f'{str(user)},' + " ".join([str(x) for x in value]) + "\n")

def split_train_test(df):
    return df.loc[df['Split'] == 'Train'], df.loc[df['Split'] == 'Test']

### READ DATA

data = pd.read_csv(
    "../../data/kaggle-job-recommendation/apps_with_item_user_data.tsv",
    sep = "\t"
)

### SAMPLE DATA - SUPPRESS WHEN LIVE

# data = data.sample(10000)

### REFORMAT DATA

# column names need to play ball with library
data.rename(columns = {
    "UserID": "user",
    "JobID": "item",
    "Label": "label"
}, inplace=True)

# replace nan on int columns with 0
data['WorkHistoryCount'] = data['WorkHistoryCount'].fillna(0)
data['TotalYearsExperience'] = data['TotalYearsExperience'].fillna(0)
data['ManagedHowMany'] = data['ManagedHowMany'].fillna(0)

# define types
data = data.astype({
    "user":                 int,
    "item":                 int,
    "label":                int,
    "Split":                str,
    "WindowID":             int,
    "Title":                str,
    "Popularity":           int,
    "DegreeType":           str,
    "Major":                str,
    "WorkHistoryCount":     int,
    "TotalYearsExperience": int,
    "ManagedHowMany":       int
})

# split train and test sets by WindowID
# train_df, test_df = split_train_test_by_windowid(data)
train_df, test_df = split_train_test(data)

print(f"Shape of Train Set: {train_df.shape}")
print(f"Shape of Test Set: {test_df.shape}")

# define columns
sparse_col = ["Title", "DegreeType", "Major"]
dense_col = ["Popularity", "WorkHistoryCount", "TotalYearsExperience", "ManagedHowMany"]
user_col = ["DegreeType", "Major", "WorkHistoryCount", "TotalYearsExperience", "ManagedHowMany"]
item_col = ["Title", "Popularity"]

# put data into format accepted by library
train_data, data_info = DatasetFeat.build_trainset(
        train_df, user_col, item_col, sparse_col, dense_col
    )
test_data = DatasetFeat.build_testset(test_df)

# sample negative items for each record
train_data.build_negative_samples(data_info)
test_data.build_negative_samples(data_info)

### CREATE AND TRAIN MODEL

ytb_ranking = YouTubeRanking(task="ranking", data_info=data_info,
                                 embed_size=embed_size, 
                                 n_epochs=n_epochs, 
                                 lr=lr,
                                 batch_size=batch_size, 
                                 use_bn=True,
                                 hidden_units=hidden_units)

ytb_ranking.fit(train_data, verbose=2, shuffle=True,
                    eval_data=test_data,
                    metrics=["loss", "roc_auc", "precision",
                             "recall", "map", "ndcg"])

### MAKE PREDICTIONS

output_user_id_list = []
output_job_id_list = []

for window_id in test_df['WindowID'].unique():
    for user in list(set(test_df['user'][test_df['WindowID'] == window_id])):
        if user in list(set(train_df['user'])):
            for prediction in ytb_ranking.recommend_user(
                user = user, 
                n_rec = 150, 
                cold_start= "popular"
                ):

                output_user_id_list.append(user)
                output_job_id_list.append(prediction)

    # users = list(set(test_df['user'][test_df['WindowID'] == window_id]))
    # predictions = [ytb_ranking.recommend_user(
    #     user = user, 
    #     n_rec = 150, 
    #     cold_start= "popular"
    #     ) for user in users]
    # output_user_id_list.extend(users)
    # output_job_id_list.extend(predictions)

generate_recommendation_output(output_user_id_list, output_job_id_list, "model_predictions/yt_recommender.csv")


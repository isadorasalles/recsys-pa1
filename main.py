import sys
import pandas as pd
import numpy as np

ratings_path = sys.argv[1]
targets_path = sys.argv[2]

df_ratings = pd.read_csv(ratings_path)
df_ratings[['UserId', 'ItemId']] = pd.DataFrame(df_ratings["UserId:ItemId"].str.split(':',1).tolist(),
                                                columns = ['UserId','ItemId'])

df_targets = pd.read_csv(targets_path)
df_targets[['UserId', 'ItemId']] = pd.DataFrame(df_targets["UserId:ItemId"].str.split(':',1).tolist(),
                                                columns = ['UserId','ItemId'])

list_user = df_ratings["UserId"].tolist()
list_user_t = df_targets["UserId"].tolist()
list_items = df_ratings["ItemId"].tolist()
list_items_t = df_targets["ItemId"].tolist()

unique_users = list(set(list_user))
unique_items = list(set(list_items))

for i in list_user_t:
    if i not in list_user:
        print("puuuts")
 
# matrix = np.zeros((len(unique_users), len(unique_items)))

# for i, user in enumerate(unique_users):
#     # print(user)
#     for j, item in enumerate(unique_items):
#         if user+":"+item in df_ratings["UserId:ItemId"]: 
#             print("ola")
#             print(df_ratings[df_ratings["UserId:ItemId"] == user+":"+item])

#df = pd.DataFrame(columns=unique_items, index=unique_users)


# print(df.head())
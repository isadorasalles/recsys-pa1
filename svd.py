import numpy as np
import pandas as pd

class svd():

    def __init__(self, ratings_path, targets_path):
        # self.ratings = pd.read_csv(ratings_path)
        # self.ratings[['UserId', 'ItemId']] = pd.DataFrame(self.ratings["UserId:ItemId"].str.split(':',1).tolist(),
        #                                                 columns = ['UserId','ItemId'])

        # self.targets = pd.read_csv(targets_path)
        # self.targets[['UserId', 'ItemId']] = pd.DataFrame(self.targets["UserId:ItemId"].str.split(':',1).tolist(),
        #                                                 columns = ['UserId','ItemId'])

        # list_user = self.ratings["UserId"].tolist()
        # #list_user.extend(self.targets["UserId"].tolist())
        # list_items = self.ratings["ItemId"].tolist()
        # #list_items.extend(self.targets["ItemId"].tolist())

        # self.unique_users = list(set(list_user))
        # self.unique_items = list(set(list_items))
        # self.users_index = list(range(0, len(self.unique_users)-1))
        # self.items_index = list(range(0, len(self.unique_items)-1))
        self.read_ratings(ratings_path)
        self.stochastic_gradient()
        self.submission(targets_path)

        ## pegar o index antes, já deixar uma tupla com index_user, index_item, prediction

    def read_ratings(self, ratings_path):
        df = pd.read_csv(ratings_path)
        self.user_index = {}
        self.item_index = {}
        u_ind = 0
        i_ind = 0
        self.ratings = []
        for _, row in df.iterrows():
            userid = row["UserId:ItemId"].split(":")[0]
            itemid = row["UserId:ItemId"].split(":")[1]
            if userid not in self.user_index.keys():
                self.user_index[userid] = u_ind
                u_ind+=1
            if itemid not in self.item_index.keys():
                self.item_index[itemid] = i_ind
                i_ind+=1
            self.ratings.append((self.user_index[userid], self.item_index[itemid], float(row["Prediction"])))
        self.average_rating = df["Prediction"].mean()
        # print("User ratings saved!")

    def predict(self, u, i, n_features):
        pred = 0.0
        for f in range(n_features):
            pred += self.P[u][f] * self.Q[f][i]
        return pred

    # melhor setagem= steps: 20, n_features:5, reg=0.01 
    # pensar em mudar a inicialização
    def stochastic_gradient(self, steps=20, alpha=0.005, n_features=5, reg=0.02):
        self.P = np.ones((len(self.user_index), n_features))
        self.Q = np.ones((len(self.item_index), n_features))
        self.Q = self.Q.T

        sum_ = 0
        for s in range(steps):
            # print(s)
            rmse = 0
            for u, i, rui in self.ratings:
                eui = pd.to_numeric(rui) -  self.predict(u, i, n_features)
                rmse += eui*eui
                
                for f in range(n_features):
                    self.P[u][f] = self.P[u][f] + alpha* (2*eui*self.Q[f][i] - reg*self.P[u][f])
                    self.Q[f][i] = self.Q[f][i] + alpha* (2*eui*self.P[u][f] - reg*self.Q[f][i])
                sum_+=1
                
            rmse/=sum_
            # print(rmse)
        
    
    def submission(self, targets_path):
        df = pd.read_csv(targets_path)
        print("UserId:ItemId,Prediction")
        for _, row in df.iterrows():
            userid = row["UserId:ItemId"].split(":")[0]
            itemid = row["UserId:ItemId"].split(":")[1]
            if userid in self.user_index.keys() and itemid in self.item_index.keys():
                # print(self.P[self.user_index[userid]])
                prediction = self.predict(self.user_index[userid], self.item_index[itemid], 5)
                if prediction > 10:
                    prediction = 10
                elif prediction < 0:
                    prediction = 0
            else:
                prediction = self.average_rating  # arrumar isso aqui
            print("{}:{},{}".format(userid, itemid, prediction))

if __name__ == "__main__":
    svd("ratings.csv", "targets.csv")
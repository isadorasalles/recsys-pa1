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
        print("ACABOU!")

    def d0(self, v1,v2):
        """                                                                                                     
        d0 is Nominal approach:                                                                                 
        multiply/add in a loop                                                                                  
        """
        out = 0
        for k in range(len(v1)):
            out += v1[k] * v2[k]
        return out

    def predict(self, u, i, n_features):
        pred = 0.0
        for f in range(n_features):
            pred += self.P[u][f] * self.Q[f][i]
        return pred

    def stochastic_gradient(self, steps=5, alpha=0.005, n_features=3):

        # print(self.unique_items)
        # print(self.unique_users)
        
        self.P = np.random.rand(len(self.user_index), n_features)
        self.Q = np.random.rand(len(self.item_index), n_features)
        self.Q = self.Q.T

        sum_ =0
        for s in range(steps):
            print(s)
            for u, i, rui in self.ratings:
                # u = self.unique_users.index(user)
                # i = self.unique_items.index(item) 
                # print(rui)
                eui = pd.to_numeric(rui) -  self.predict(u, i, n_features)
                # print(eui)
                for f in range(n_features):
                    self.P[u][f] = self.P[u][f] + 2*alpha*eui*self.Q[f][i]
                    self.Q[f][i] = self.Q[f][i] + 2*alpha*eui*self.P[u][f]
                sum_+=1
                print(sum_)
        
        print(self.P)
        print(self.Q.T)
        print("RESULTADO")
        #print (np.dot(self.P, self.Q))

        # return P, Q.T

if __name__ == "__main__":
    factorization("ratings.csv", "targets.csv")
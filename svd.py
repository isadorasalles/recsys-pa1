import numpy as np
import pandas as pd

class svd():

    def __init__(self, epochs = 20, lr = 0.005, n_factors=5, reg=0.1):
        ## best configuration
        self.epochs = epochs
        self.lr = lr
        self.n_factors = n_factors
        self.reg = reg

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

    def predict(self, u, i):
        pred = 0.0
        for f in range(self.n_factors):
            pred += self.P[u][f] * self.Q[f][i]
        return pred

    # melhor setagem= steps: 20, n_features:5, reg=0.1, alpha
    def stochastic_gradient(self):
        self.P = np.ones((len(self.user_index), self.n_factors))
        self.Q = np.ones((len(self.item_index), self.n_factors))
        self.Q = self.Q.T

        sum_ = 0
        for s in range(self.epochs):
            rmse = 0
            for u, i, rui in self.ratings:
                eui = rui -  self.predict(u, i)
                rmse += eui*eui
                
                for f in range(self.n_factors):
                    self.P[u][f] = self.P[u][f] + self.lr * (2*eui*self.Q[f][i] - self.reg*self.P[u][f])
                    self.Q[f][i] = self.Q[f][i] + self.lr * (2*eui*self.P[u][f] - self.reg*self.Q[f][i])
                sum_+=1
                
            rmse/=sum_
    
    def submission(self, targets_path):
        df = pd.read_csv(targets_path)
        print("UserId:ItemId,Prediction")
        for _, row in df.iterrows():
            userid = row["UserId:ItemId"].split(":")[0]
            itemid = row["UserId:ItemId"].split(":")[1]
            if userid in self.user_index.keys() and itemid in self.item_index.keys():
                prediction = self.predict(self.user_index[userid], self.item_index[itemid])
                if prediction > 10:
                    prediction = 10
                elif prediction < 0:
                    prediction = 0
            else:
                prediction = self.average_rating 
            print("{}:{},{}".format(userid, itemid, prediction))

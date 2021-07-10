import numpy as np
import pandas as pd

class svd():

    def __init__(self, epochs = 20, lr = 0.005, n_factors=5, reg=0.1):
        self.epochs = epochs
        self.lr = lr
        self.n_factors = n_factors
        self.reg = reg

    def read_ratings(self, ratings_path):
        df = pd.read_csv(ratings_path)
        self.user_index = {}
        self.item_index = {}
        self.user_avg = {}
        self.item_avg = {}
        u_ind = 0
        i_ind = 0
        self.ratings = []
        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]
            if userid not in self.user_index.keys():
                self.user_index[userid] = u_ind
                self.user_avg[userid] = []
                u_ind+=1
            self.user_avg[userid].append(row.Prediction)
            if itemid not in self.item_index.keys():
                self.item_index[itemid] = i_ind
                self.item_avg[itemid] = []
                i_ind+=1
            self.item_avg[itemid].append(row.Prediction)
            self.ratings.append((self.user_index[userid], self.item_index[itemid], float(row.Prediction)))
        self.average_rating = df["Prediction"].mean()

    def predict(self, u, i):
        pred = 0.0
        for f in range(self.n_factors):
            pred += self.P[u][f] * self.Q[f][i]
        return pred

    def stochastic_gradient(self):
        self.P = np.ones((len(self.user_index), self.n_factors))
        self.Q = np.ones((self.n_factors, (len(self.item_index)))
        print(type(self.P))
        # self.Q = self.Q.T
        # self.lr = 0.01
        # sum_ = 0
        for s in range(self.epochs):
            rmse = 0
            for u, i, rui in self.ratings:
                eui = rui -  self.predict(u, i)
                # rmse += eui*eui
                
                for f in range(self.n_factors):
                    self.P[u][f] = self.P[u][f] + self.lr * (2*eui*self.Q[f][i] - self.reg*self.P[u][f])
                    self.Q[f][i] = self.Q[f][i] + self.lr * (2*eui*self.P[u][f] - self.reg*self.Q[f][i])
                sum_+=1
            # self.lr -= 0.00025
            # rmse/=sum_
            # print("Epoch {}, rmse: {}".format(s, rmse))


    def submission(self, targets_path):
        df = pd.read_csv(targets_path)
        print("UserId:ItemId,Prediction")
        for row in df.itertuples():
            userid = row[1].split(":")[0]
            itemid = row[1].split(":")[1]
            if userid in self.user_index.keys() and itemid in self.item_index.keys():
                prediction = self.predict(self.user_index[userid], self.item_index[itemid])
                if prediction > 10:
                    prediction = 10
                elif prediction < 0:
                    prediction = 0
            elif userid in self.user_index.keys():
                prediction = np.mean(self.user_avg[userid])
            elif itemid in self.item_index.keys():
                prediction = np.mean(self.item_avg[itemid])
            else:
                prediction = self.average_rating 
            print("{}:{},{}".format(userid, itemid, prediction))

# Create logistic regression class for multi-class classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils import get_data,get_regular_data, split_data,sigmoid, init_weight_and_bias, softmax,softmax_cost, error_rate, classification_rate, y2indicator, standardize_data, save_model

class LogisticModel(object):
    def __init__(self):
        self.learning_rate = 10e-7
        self.reg = 0.02
        self.epochs = 200000

    def fit(self,X,Y, Xvalid, Yvalid, show_fig = False):
        Tvalid = y2indicator(Yvalid)

        N,D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W = np.random.randn(D,K) / np.sqrt(D)
        self.b = np.zeros(K)

        costs = []
        best_validation_error = 1

        for i in range(self.epochs):
            pY = self.forward(X)
            self.W -= self.learning_rate*(X.T.dot(pY - T) + self.reg*self.W)
            self.b -= self.learning_rate*((pY - T).sum(axis = 0) + self.reg*self.b)

            if i%100 == 0:
                pYvalid = self.forward(Xvalid)
                c = softmax_cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis = 1))
                print("i:", i, "cost:", c, "error_rate:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.title('Cost')
            plt.show()
    
    def forward(self,X):
        return softmax(X.dot(self.W) + self.b)
    
    def predict(self,X):
        pY = self.forward(X)
        return np.argmax(pY, axis = 1)
    
    def score(self,X,Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)
    
    def get_weights(self):
        return self.W
    
    def get_bias(self):
        return self.b

def main():
    path = 'Data/diabetes.csv'
    # XTrain, YTrain, XTest, YTest = get_data('Data/diabetes.csv')
    data = get_regular_data(path)
    XTrain, YTrain, XTest, YTest = split_data(data)
    XTrain, XTest = standardize_data(XTrain, XTest)
    model = LogisticModel()
    print("Train shape:", XTrain.shape, "Test shape:", XTest.shape)
    model.fit(XTrain, YTrain, XTest, YTest, show_fig = True)
    print("Train score:", model.score(XTrain, YTrain))
    print("Test score:", model.score(XTest, YTest))
    print("Final W:", model.get_weights())
    print("Final b:", model.get_bias())
    
if __name__ == '__main__':
    main()


        

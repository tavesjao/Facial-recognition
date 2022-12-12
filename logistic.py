# Create logistic regression class for binary classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils import getBinaryData, sigmoid, init_weight_and_bias, sigmoid_cost, error_rate, classification_rate

class LogisticModel(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-7, reg=0, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        #split into validation set and training set
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        #train set
        X, Y = X[:-1000], Y[:-1000]

        #dimensionality of the data
        N, D = X.shape
        
        #initialize weights and bias
        self.w = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        #collect costs for plotting
        costs = []
        best_validation_error = 1
        
        #gradient descent
        for i in range(epochs):
            pY = self.forward(X)

            #gradient descent step                    #added regularization
            self.w -= learning_rate*(X.T.dot(pY - Y) + reg*self.w)
            self.b -= learning_rate*((pY - Y).sum() + reg*self.b)
            if i%100 == 0:
                #calculate cost and error rate
                pYvalid = self.forward(X_valid)
                cost = sigmoid_cost(Y_valid, pYvalid)
                costs.append(cost)
                error = error_rate(Y_valid, np.round(pYvalid))
                print("i:", i, "cost:", cost, "error_rate:", error)
                if error < best_validation_error:
                    best_validation_error = error
        print("best_validation_error:", best_validation_error)
        
        if show_fig:
            plt.plot(costs)
            plt.title('Cost')
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        prediction = self.predict(X)
        return classification_rate(Y, prediction)

def main():
    X, Y = getBinaryData()
    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    #balance the data through oversampling
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    print(model.score(X, Y))

if __name__ == '__main__':
    main()

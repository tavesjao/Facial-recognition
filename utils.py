import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# load uncompressed fer2013.csv
def getData(balance_ones=True, Ntest=1000):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('Data/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    # shuffle and split
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

    if balance_ones:
        # balance the 1 class
        X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        Xtrain = np.vstack([X0, X1])
        Ytrain = np.concatenate((Y0, [1]*len(X1)))

    return Xtrain, Ytrain, Xvalid, Yvalid

def getBinaryData(filename:str='Data/fer2013.csv'):
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)

def get_data(balanced_data=True, filename:str='Data/fer2013.csv'):
    Y = []
    X = []

    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X,Y =  np.array(X) / 255.0, np.array(Y)

    X, y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-1000], Y[:-1000]
    Xvalid, Yvalid = X[-1000:], Y[-1000:]

    if balanced_data:
        X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        Xtrain = np.vstack([X0, X1])
        Ytrain = np.concatenate((Y0, [1]*len(X1)))
    
    return Xtrain, Ytrain, Xvalid, Yvalid


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def softmax_cost(T, Y):
    return -(T*np.log(Y)).sum()

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def classification_rate(Y, P):
    return np.mean(Y == P)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
    
def sign(x:list):
    for element in x:
        if element>0:
            element=1
        else:
            element=0
    return x

def get_regular_data(path):
    import pandas as pd
    data = pd.read_csv(path)
    return data

def split_data(data):
    #data = get_regular_data
    split_pctg = int(0.6*len(data))
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X = X.to_numpy()
    y = y.to_numpy()
    Xtrain, Xvalid = X[:split_pctg], X[split_pctg:]
    Ytrain, Yvalid = y[:split_pctg], y[split_pctg:]
    return Xtrain, Ytrain, Xvalid, Yvalid

#create function to standardize data
def standardize_data(Xtrain, Xvalid):
    mean = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    Xtrain = (Xtrain-mean)/std
    Xvalid = (Xvalid-mean)/std
    return Xtrain, Xvalid
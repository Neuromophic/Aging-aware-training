import numpy as np
import matplotlib.pyplot as plt

def OneHot(y, K):
    M = y.shape[1]
    Y = np.zeros([K, M])
    for i in range(M):
        Y[int(y[0, i]), i] = 1
    return Y

def GenerateData(N=80, var=0.3, plot=False):
    x1 = np.random.randn(1, int(N/4)) * var + 1
    y1 = np.random.randn(1, int(N/4)) * var + 1
    l1 = np.random.randn(1, int(N/4)) * 0 + 0
    x2 = np.random.randn(1, int(N/4)) * var + 1
    y2 = np.random.randn(1, int(N/4)) * var - 1
    l2 = np.random.randn(1, int(N/4)) * 0 + 1
    x3 = np.random.randn(1, int(N/4)) * var - 1
    y3 = np.random.randn(1, int(N/4)) * var + 1
    l3 = np.random.randn(1, int(N/4)) * 0 + 2
    x4 = np.random.randn(1, int(N/4)) * var - 1
    y4 = np.random.randn(1, int(N/4)) * var - 1
    l4 = np.random.randn(1, int(N/4)) * 0 + 3
    X1 = np.hstack((x1, x2, x3, x4))
    X2 = np.hstack((y1, y2, y3, y4))
    X = np.vstack((X1, X2))
    y = np.hstack((l1, l2, l3, l4))
    if plot:
        plt.figure()
        plt.scatter(X[0, :], X[1, :], c=y)
        plt.show()
    return X.astype(float), y

def SplitData(X, Y, training_rate):
    N, M = X.shape
    K = Y.shape[0]
    M_train = int(M * training_rate)
    DATA = np.vstack((X, Y))
    index = np.arange(0, M, 1)
    np.random.shuffle(index)
    DATA = DATA[:, index]
    X = DATA[:-K, :].reshape(N, -1)
    Y = DATA[-K:, :].reshape(K, -1)
    X_train = X[:, :M_train]
    Y_train = Y[:, :M_train]
    X_valid = X[:, M_train:]
    Y_valid = Y[:, M_train:]
    return X_train, Y_train, X_valid, Y_valid

def Normalization(X):
    return (X - np.mean(X, 1).reshape(-1, 1)) / np.std(X, 1).reshape(-1, 1)
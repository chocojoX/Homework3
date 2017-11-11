import numpy as np
import pandas as pd
import random
from Interior_point_and_barrier import *
from SVM_Problem import *
from sklearn import preprocessing


def  import_data():
    data = pd.read_csv('data/iris.txt')
    X = np.array(data)[:, :4]
    Y = np.array(data)[:, 4]

    idx = np.where(Y=='Iris-versicolor')[0].tolist() + np.where(Y=='Iris-virginica')[0].tolist()
    X = np.concatenate((X[idx, :], np.ones((len(idx), 1))), axis = 1)
    Y = Y[idx]

    idx = np.where(Y=='Iris-versicolor')[0].tolist()
    Y[idx] = 1
    idx =  np.where(Y=='Iris-virginica')[0].tolist()
    Y[idx] = -1
    X =preprocessing.scale(X)

    return X, Y


def split_data(X, Y, p=0.8):
    n = X.shape[0]
    n_train = int(p*n)
    idx = [i for i in range(n)]
    random.shuffle(idx)

    X_train = X[idx[:n_train]]
    Y_train = Y[idx[:n_train]]

    X_test = X[idx[n_train:]]
    Y_test = Y[idx[n_train:]]

    return X_train, Y_train, X_test, Y_test


def solve_SVM(X_train, Y_train, tau, mu=2, tol=0.01):
    n, d = X_train.shape
    Q, p, A, b = transform_svm_primal(tau, X_train, Y_train)
    x_0 = np.zeros(n+d)
    x_0[d:] = 2

    w_z, hist = barr_method(Q, p, A, b, x_0, mu, tol)
    w = w_z[:d]
    z = w_z[d:]
    print(w, len(hist))
    return w, z


def test_SVM(X_test, Y_test, w):
    pred = np.dot(X_test, w)
    print(pred)
    foo = pred*Y_test
    perf = 0
    for i in range(X_test.shape[0]):
        if foo[i]<0:
            perf+=1.
    return pred, perf/X_test.shape[0]

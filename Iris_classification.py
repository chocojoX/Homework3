import numpy as np
import pandas as pd
import random
from Interior_point_and_barrier import *
from SVM_Problem import *
from sklearn import preprocessing
import matplotlib.pyplot as plt


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

    w_z, hist, duality_gaps = barr_method(Q, p, A, b, x_0, mu, tol)
    w = w_z[:d]
    z = w_z[d:]
    return w, z


def solve_dual(X_train, Y_train, tau, mu=2, tol=0.01):
    n, d = X_train.shape
    Q, p, A, b = transform_svm_dual(tau, X_train, Y_train)
    x_0 = 1/(2*tau*n)*np.ones(n)

    lbd, hist, duality_gaps = barr_method(Q, p, A, b, x_0, mu, tol)
    return lbd


def test_SVM(X_test, Y_test, w):
    pred = np.dot(X_test, w)
    foo = pred*Y_test
    perf = 0
    for i in range(X_test.shape[0]):
        if foo[i]<0:
            perf+=1.
    return pred, perf/X_test.shape[0]


def cross_validate(X, Y):
    K = 3  # Number of folds of cross validation
    perf_test = []
    perf_train = []
    for tau in [0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        perf_0 = 0
        perf_1 = 0
        for k in range(K):
            X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
            w, z = solve_SVM(X_train, Y_train, tau=tau, mu=5, tol=0.1)
            pred, perf = test_SVM(X_train, Y_train, w)
            perf_0 += perf
            pred, perf = test_SVM(X_test, Y_test, w)
            perf_1 += perf
        perf_train.append(100*perf_0/K)
        perf_test.append(100*perf_1/K)

    plt.plot( [0.0001, 0.001, 0.001, 0.1, 1, 10], perf_test)
    plt.xscale('log')
    plt.title("Evolution of performance as a function of tau")
    plt.xlabel("tau")
    plt.ylabel("Accuracy (%)")
    plt.show()


def duality_gap(X, Y):
    # primal duality gap
    tau = 1
    tol = 0.001



    for mu in [2, 15, 50, 100]:
        X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
        n, d = X_train.shape
        Q, p, A, b = transform_svm_primal(tau, X_train, Y_train)
        x_0 = np.zeros(n+d)
        x_0[d:] = 2

        w_z, hist, duality_gaps = barr_method(Q, p, A, b, x_0, mu, tol)

        plt.plot(duality_gaps, label='mu = %i' %mu)

    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Duality gap as a function of the number of newton steps \n for the primal problem with damped Newton")
    plt.show()


    for mu in [2, 15, 50, 100]:
        X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
        n, d = X_train.shape
        Q, p, A, b = transform_svm_dual(tau, X_train, Y_train)
        x_0 = 1/(2*tau*n)*np.ones(n)

        lbd, hist, duality_gaps = barr_method(Q, p, A, b, x_0, mu, tol)

        plt.plot(duality_gaps, label='mu = %i' %mu)

    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Duality gap as a function of the number of newton steps \n for the dual problem with damped Newton")
    plt.show()


    for mu in [2, 15, 50, 100]:
        X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
        n, d = X_train.shape
        Q, p, A, b = transform_svm_primal(tau, X_train, Y_train)
        x_0 = np.zeros(n+d)
        x_0[d:] = 2

        w_z, hist, duality_gaps = barr_method_LS(Q, p, A, b, x_0, mu, tol)

        plt.plot(duality_gaps, label='mu = %i' %mu)

    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Duality gap as a function of the number of newton steps \n for the primal problem with damped Newton with Newton backtracking line search")
    plt.show()

    for mu in [2, 15, 50, 100]:
        X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
        n, d = X_train.shape
        Q, p, A, b = transform_svm_dual(tau, X_train, Y_train)
        x_0 = 1/(2*tau*n)*np.ones(n)

        lbd, hist, duality_gaps = barr_method_LS(Q, p, A, b, x_0, mu, tol)

        plt.plot(duality_gaps, label='mu = %i' %mu)

    plt.legend(loc='best')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Duality gap as a function of the number of newton steps \n for the dual problem with Newton backtracking line search")
    plt.show()

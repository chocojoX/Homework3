import numpy as np


def transform_svm_primal(tau, X, y):
    n = X.shape[0]; d=X.shape[1]
    In = np.eye(n)
    A = np.zeros((d+n, 2*n))
    A[d:, :n] = -In
    A[d:, n:] = -In
    for i in range(n):
        A[i, :d] = y[i] * X[i, :]

    b = np.zeros(2*n)
    b[:n] = -1

    Q = np.zeros((d+n, d+n))
    Q[:d, :d] = np.eye(d)

    p = np.zeros(d+n)
    p[d:] = 1/(tau*n)
    return Q, p, A, b


def transform_svm_dual(tau, X, y):
    n = X.shape[0]; d = X.shape[1]
    In = np.eye(n)

    A = np.zeros((2*n, n))
    A[:n, :] = In
    A[n:, :] = -In
    b = np.zeros(2*n)
    b[:n] = 1/(tau*n)

    Q =np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.dot(y[i]*X[i, :], y[j]*X[j, :])
            Q[i, j] = val
            Q[j, i] = val

    p = -np.ones(n)
    return Q, p, A, b

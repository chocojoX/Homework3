import numpy as np


def  phi(x, t, Q, p, A, b):
    quad = 0.5 * np.dot((np.dot(np.transpose(x), Q)), x)
    lin = np.dot(p, x)
    constraint = np.dot(A, x) - b
    barrier = -np.sum(np.log(constraint))

    return t*(quad+lin)+barrier


def grad(x, t, Q, p, A, b):
    quad = np.dot(Q, x)
    lin = p
    barrier = - np.dot(  A,   1/(np.dot(A, x) -b))

    return t*(quad+lin) + barrier


def hess(x, t, Q, p, A, b):
    quad = Q
    barrier = np.dot(  np.dot(A, np.transpose(A)) ,  1/(np.dot(A, x)-b)**2)

    return t*quad + barrier


if __name__=='__main__':
    Q=10; p=1; A=2; b=-1; # Declare some parameters
    t=0.001; # Set the barrier parameter
    x = 2
    print(phi(x, t, Q, p, A, b))
    print(grad(x, t, Q, p, A, b))
    print(hess(x, t, Q, p, A, b))
    pass

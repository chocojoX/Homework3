from Interior_point_and_barrier import *
from SVM_Problem import *
from Iris_classification import *
import numpy as np


if __name__=='__main__':
    Q=10; p=1; A=2; b=-1; # Declare some parameters
    t=0.001; # Set the barrier parameter
    x = -1
    print(phi(x, t, Q, p, A, b))
    print(grad(x, t, Q, p, A, b))
    print(hess(x, t, Q, p, A, b))

    f = lambda x: phi(x,t,Q,p,A,b);
    g = lambda x: grad(x,t,Q,p,A,b);
    h = lambda x: hess(x,t,Q,p,A,b);
    tol = 0.01

    # Testing one newton step
    x_new, gap = dampedNewtonStep(x, f, g, h)
    print( "x=",x_new, "gap=%.3f" %gap, "Phi_t(x)=%.2f" %f(x_new), "Phi_10000=", phi(x_new, 10000, Q, p, A, b)/10000)

    # Testing damped Newton
    x_star, hist = dampedNewton(x,f,g,h,tol)
    print(x_star, f(x_star), phi(x_star, 10000, Q, p, A, b)/10000)

    # Testing Newton backtracking line-search
    x_star, hist = newtonLS(x,f,g,h,tol)
    print(x_star, f(x_star), phi(x_star, 10000, Q, p, A, b)/10000)

    # Testing the barrier method
    mu = 2
    x_star, hist = barr_method(Q, p, A, b, x, mu, tol)
    print(x_star, t, f(x_star), phi(x_star, 1000, Q, p, A, b)/1000)

    # Testing the SVM on the Iris data
    X, Y = import_data()
    X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.8)
    w, z = solve_SVM(X_train, Y_train, tau=0.02, mu=5, tol=0.01)
    pred, perf = test_SVM(X_train, Y_train, w)
    print("Performance of classifier on training data : %.1f %% accuracy" %(100*perf))
    pred, perf = test_SVM(X_test, Y_test, w)
    print("Performance of classifier on testing data : %.1f %% accuracy" %(100*perf))

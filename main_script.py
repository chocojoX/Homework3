from Interior_point_and_barrier import *


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

    x_new, gap = dampedNewtonStep(x, f, g, h)
    print(x_new, gap, f(x_new))
    tol = 0.01
    x0 = -1
    x_star, hist = dampedNewton(x0,f,g,h,tol)
    print(x_star, f(x_star))

    x_star, hist = newtonLS(x0,f,g,h,tol)
    print(x_star, f(x_star))

    pass

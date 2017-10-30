import numpy as np


def  phi(x, t, Q, p, A, b):
    quad = 0.5 * np.dot((np.dot(np.transpose(x), Q)), x)
    lin = np.dot(p, x)
    constraint = b - np.dot(A, x)
    barrier = - np.sum(np.log(constraint))

    return t*(quad+lin)+barrier


def grad(x, t, Q, p, A, b):
    quad = np.dot(Q, x)
    lin = p
    barrier = np.dot(  A,   1/(b- np.dot(A, x))  )

    return t * (quad+lin) + barrier


def hess(x, t, Q, p, A, b):
    quad = Q
    barrier = np.dot(  np.dot(A, np.transpose(A)) ,  1/(b - np.dot(A, x))**2)

    return t*quad + barrier


def dampedNewtonStep(x, f, g, h):
    grad_x = g(x)
    hess_x = h(x)
    phi_x = f(x)
    if len(hess_x.shape) == 0:
        inv_hess = 1/hess_x
    else:
        inv_hess = np.linalg.inv(hess_x)

    lambda_x = np.dot(   np.dot(  np.transpose(grad_x),  inv_hess)   ,   grad_x)
    x_new = x - 1/(1+lambda_x) * np.dot( inv_hess,  grad_x)
    estimated_gap = lambda_x**2 / 2
    return x_new, estimated_gap


def dampedNewton(x0,f,g,h,tol):
    assert tol<= (3-np.sqrt(5))/2, "tol should be lower than (3-V5)/2"
    x_hist = []
    estimated_gap = tol + 1
    x_new = x0
    while estimated_gap>tol:
        x_new, estimated_gap = dampedNewtonStep(x_new, f, g, h)
        print( "gap : %.3f" %estimated_gap)
        x_hist.append(x_new)
    return x_new, x_hist


def backtrackingNewtonStep(x, f, g, h):
    grad_x = g(x)
    hess_x = h(x)
    phi_x = f(x)
    if len(hess_x.shape) == 0:
        inv_hess = 1/hess_x
    else:
        inv_hess = np.linalg.inv(hess_x)

    lambda_x = np.dot(   np.dot(  np.transpose(grad_x),  inv_hess)   ,   grad_x)
    estimated_gap = lambda_x**2 / 2

    direction = np.dot( inv_hess,  grad_x)
    t = 1
    beta = 0.7
    phi_new = phi_x+1
    while phi_new>phi_x:
        x_new = x - t * np.dot( inv_hess,  grad_x)
        phi_new = f(x_new)
        t = beta*t
    return x_new, estimated_gap


def  newtonLS(x0,f,g,h,tol) :
    x_hist = []
    estimated_gap = tol + 1
    x_new = x0
    while estimated_gap>tol:
        x_new, estimated_gap = backtrackingNewtonStep(x_new, f, g, h)
        print( "gap : %.3f" %estimated_gap)
        x_hist.append(x_new)
    return x_new, x_hist




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

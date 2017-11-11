import numpy as np


def phi(x, t, Q, p, A, b):
    quad = 0.5 * np.dot((np.dot(np.transpose(x), Q)), x)
    lin = np.dot(p, x)
    constraint = b - np.dot(A, x)
    try:
        barrier = - np.sum(np.log(constraint))
    except:
        import pdb; pdb.set_trace()

    return t*(quad+lin)+barrier


def grad(x, t, Q, p, A, b):
    quad = np.dot(Q, x)
    lin = p
    barrier = np.dot(  np.transpose(A),   1/(b- np.dot(A, x))  )

    return t * (quad+lin) + barrier


def hess(x, t, Q, p, A, b):
    quad = Q
    try:
        barrier = np.dot(  np.dot(np.transpose(A), np.diag(1/(b - np.dot(A, x))**2) ) , A)
    except:
        barrier = np.dot(  np.dot(np.transpose(A), (1/(b - np.dot(A, x))**2 )) , A)

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
        # print( "gap : %.3f" %estimated_gap)
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
        # print( "gap : %.3f" %estimated_gap)
        x_hist.append(x_new)
    return x_new, x_hist


def barr_method(Q, p, A, b, x_0, mu, tol):
    try:
        m = A.shape[0]
    except:
        m=1
    t = 0.001
    x_start = x_0
    hist = [x_start]

    while(m/t > tol):
        f = lambda x: phi(x,t,Q,p,A,b)
        g = lambda x: grad(x,t,Q,p,A,b)
        h = lambda x: hess(x,t,Q,p,A,b)

        x_start, hist_ = dampedNewton(x_start, f, g, h, tol)
        hist.append(x_start)
        t = mu * t

    return x_start, hist

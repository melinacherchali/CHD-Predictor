import numpy as np
import matplotlib.pyplot as plt

def msd_error(y,ty):
    n=ty.shape[0]
    e=y-ty
    return 1/n*(e).dot(e.T)

def mean_squared_error_gd(y,tx,initial_w,max_iters,gama):
    """
    Compute the linear regression using gradient descent
    y: the target values
    tx: the input data
    initial_w: the initial weight
    max_iters: the maximum number of iterations
    gama: the learning rate
    """
    n=tx.shape[0]
    while max_iters>0:
        y_=tx.dot(initial_w)
        error=y-y_
        if error.all()==0:
            return y_,initial_w
        grad=-(1/n)*(tx.T).dot(error)
        initial_w=initial_w-gama*grad
        max_iters-=1
    return (initial_w,msd_error(y_,y))


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Compute the linear regression using stochastic gradient descent
    y: the target values
    tx: the input data
    initial_w: the initial weight
    max_iters: the maximum number of iterations
    gamma: the learning rate """
    n=tx.shape[0]
    while max_iters>0:
        i=np.random.randint(0,n)
        y_=tx[i].dot(initial_w)
        error=y[i]-y_
        grad=-(tx[i].T).dot(error)
        initial_w=initial_w-gamma*grad
        max_iters-=1
    return (initial_w,msd_error(y,tx.dot(initial_w)))
        
def test_function(func):
    y = np.array([5,10,17])
    tx = np.array([[1, 2], [1, 3], [1, 4]])
    initial_w = np.array([0, 0])
    max_iters = 1000
    gamma = 0.1
    a=func(y, tx, initial_w, max_iters, gamma)
    print(a[0],"MSE : ",a[1])
    
def least_squares(y, tx):
    """
    Calculate the least squares solution.
    y: the target values
    tx: the input data
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return (w,msd_error(y,tx.dot(w)))

def ridge_regression(y, tx, lambda_):
    """
    Calculate the ridge regression solution.
    y: the target values
    tx: the input data
    lambda_: the regularization parameter
    """
    n=tx.shape[0]
    d=tx.shape[1]
    a=tx.T.dot(tx)+2*n*lambda_*np.eye(d)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    return (w,msd_error(y,tx.dot(w)))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Compute the logistic regression using gradient descent
    y: the target values
    tx: the input data
    initial_w: the initial weight
    max_iters: the maximum number of iterations
    gamma: the learning rate
    """
    n=tx.shape[0]
    while max_iters>0:
        y_=1/(1+np.exp(-tx.dot(initial_w)))
        error=y-y_
        if error.all()==0:
            return y_,initial_w
        grad=(1/n)*(tx.T).dot(error)
        initial_w=initial_w+gamma*grad
        max_iters-=1
    return (initial_w,msd_error(y_,y))

def reg_logistic_regression(y,tx,lambda_, initial_w, max_iters, gamma):
    """
    Compute the regularized logistic regression using gradient descent
    y: the target values
    tx: the input data
    lambda_: the regularization parameter
    initial_w: the initial weight
    max_iters: the maximum number of iterations
    gamma: the learning rate
    """
    n=tx.shape[0]
    while max_iters>0:
        y_=1/(1+np.exp(-tx.dot(initial_w)))
        error=y-y_
        if error.all()==0:
            return y_,initial_w
        grad=(1/n)*(tx.T).dot(error)+2*lambda_*initial_w
        initial_w=initial_w+gamma*grad
        max_iters-=1
    return (initial_w,msd_error(y_,y))
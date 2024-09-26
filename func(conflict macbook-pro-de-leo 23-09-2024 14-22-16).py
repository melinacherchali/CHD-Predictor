import numpy as np

def msd_error(y,tx):
    e=y-tx
    return 1/tx.shape[0]*(e.T).dot(e)

def mean_squared_error_gd(y,tx,initial_w,max_iters,gama):
    """Compute the linear regression using gradient descent"
    y: the target values"
    tx: the input data"
    initial_w: the initial weight"
    max_iters: the maximum number of iterations"
    gama: the learning rate"
    n=tx.shape[0]
    while max_iters>0:
        y_=initial_w.dot(tx)
        error=y-y_
        if error.all()==0:
            return y_,initial_w
        grad=-(1/n)*(tx.T).dot(error)
        initial_w=initial_w-gama*grad
        max_iters-=1
    return y_,initial_w
        
        
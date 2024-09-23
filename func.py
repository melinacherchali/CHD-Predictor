import numpy as np

def msd_error(y,tx):
    return ((y-tx)**2).mean()

def mean_squared_error_gd(y,tx,initial_w,max_iters,gama):
    while max_iters>0:
        y_=initial_w.dot(y)
        error=msd_error(y_,tx)
        if error==0:
            return y_,initial_w
        grad=np.gradient(error)
        initial_w=initial_w-gama*grad
        max_iters-=1
    return y_,initial_w
        
        
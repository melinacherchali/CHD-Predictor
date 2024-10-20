import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp

path="/Users/leocusumano/myCloud/EPFL/MA1/ML/Proj1/DATASETS/f_sets/"

x_train=np.load(path+"f_x_train_.npy")
x_test=np.load(path+"f_x_test_.npy")
y_train=np.load(path+"f_y_train_.npy")

x_tr, y_tr, x_te, y_te = pp.split_data(x_train, y_train, 0.8)  

x_t2,t2=pp.clean_data(x_tr,x_te,1,0.3,0)
print(x_t2.shape)



x_pca=pp.PCA(x_t2,70)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def msd_sigm(y,tx,initial_w,max_iters,gama):
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
        y_=sigmoid(tx.dot(initial_w))
        error=y-y_
        if error.all()==0:
            return (initial_w,imp.msd_error(y_,y))
        grad=-(1/2*n)*(tx.T).dot(error)
        initial_w=initial_w-gama*grad
        max_iters-=1
    return (initial_w,imp.msd_error(y_,y))

y_tr=np.where(y_tr==-1,0,1)

print("Least squares",msd_sigm(y_tr,x_pca,np.zeros(x_pca.shape[1]),1000,0.01))



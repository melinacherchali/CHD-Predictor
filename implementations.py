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

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    
    N = np.shape(y)[0]
    e = y - np.dot(tx,w)
    grad = -1/N * np.dot(tx.T,e)
    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * (gradient)
        # store w and loss
        ws.append(w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        # print(
        #     "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    return losses, ws


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    batches = batch_iter(y, tx, batch_size=batch_size, num_batches=1, shuffle=True)
    gradient = np.mean([compute_gradient(y_batch, tx_batch, w) for y_batch, tx_batch in batches], axis=0)
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        # print(
        #     "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )
    return losses, ws


import numpy as np
import matplotlib.pyplot as plt
import helpers 

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















#########


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




def grid_search(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.

    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """
    
    losses = np.zeros((len(grid_w0), len(grid_w1)))
    
    for i in range(len(grid_w0)):
        for j in range(len(grid_w1)):
            w = np.array([grid_w0[i], grid_w1[j]])
            losses[j, i] = compute_loss(y, tx, w)
            
    return losses


def generate_w(num_intervals, w_size):
    """Generate a grid of values for w0 and w1."""
    w = np.zeros((w_size, w_size))
    for i in range(w_size):
        for j in range(w_size):
            w[i, j] = np.array([i, j])
    return w
        
        
        
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = np.shape(y)[0]
    e = y - np.dot(tx, w)
    return 1/(2*N) * (e.T.dot(e))



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

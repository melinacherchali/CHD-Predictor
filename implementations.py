import numpy as np
import matplotlib.pyplot as plt
from functions import *


### Required functions ###


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: weights that minimize the loss function
        loss: the loss value (scalar) for the last iteration of GD
    """

    ws, losses = gradient_descent(y, tx, initial_w, max_iters, gamma)
    return ws[-1], losses[-1]  # return the last weight and loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Compute the mean squared error using stochastic gradient descent.

    Args:
        y (numpy.ndarray): The target values.
        tx (numpy.ndarray): The input data.
        initial_w (numpy.ndarray): The initial weights.
        max_iters (int): The maximum number of iterations.
        gamma (float): The learning rate.

    Returns:
        tuple containing the final weight vector and the final loss value.
            w: weights that minimize the loss function
            loss: the loss value (scalar) for the last iteration of GD
    """

    ws, losses = stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma)
    return ws[-1], losses[-1]  # return the last weight and loss


def least_squares(y, tx):
    """
    Calculate the least squares solution.

    Args :
        y (numpy.ndarray): the target values
        tx (numpy.ndarray): the input data

    Returns :
        tuple containing the final weight vector and the final loss value.
            w: weights that minimize the loss function
            loss: the loss value

    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return (w, compute_loss(y, tx, w))


def ridge_regression(y, tx, lambda_):
    """
    Calculate the ridge regression solution.

    Args:
        y: the target values
        tx: the input data
        lambda_: the regularization parameter

    Returns :
        tuple containing the final weight vector and the final loss value.
            w: weights that minimize the loss function
            loss: the loss value
    """
    n = tx.shape[0]
    d = tx.shape[1]
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return (w, compute_loss(y, tx, w))


def logistic_regression(y, tx, initial_w, max_iters, gamma, losses_return=False):
    """
    Perform logistic regression using gradient descent.

    Args:
        y (numpy.ndarray): The target variable. Binary (0,1)
        tx (numpy.ndarray): The input features.
        initial_w (numpy.ndarray): The initial weight vector.
        max_iters (int): The maximum number of iterations.
        gamma (float): The learning rate.

    Returns:
        tuple containing the final weight vector and the final loss value.
        w: final weight vector
        loss: final loss value
    """

    w = initial_w
    loss = logistic_loss(y, tx, w)
    losses = [loss]
    for i in range(max_iters):
        gradient = logistic_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = logistic_loss(y, tx, w)
        losses.append(loss)
    if losses_return:
        return w, loss, losses
    return (w, loss)


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, return_losses=False
):
    """
    Regularized logistic regression using gradient descent.
    Args:
        y (numpy.ndarray): The target variable.
        tx (numpy.ndarray): The input features.
        lambda_ (float): The regularization parameter.
        initial_w (numpy.ndarray): The initial weight vector.
        max_iters (int): The maximum number of iterations.
        gamma (float): The learning rate.
    Returns:
        tuple containing the final weight vector and the final loss value.
        w: The optimized weight vector.
        loss: The final loss value.
    """
    N = tx.shape[0]
    w = initial_w
    loss = logistic_loss(y, tx, w)
    losses = [loss]
    for i in range(max_iters):
        gradient = penalized_logistic_regression(
            y, tx, w, lambda_
        )  # penality term included
        w = w - gamma * gradient
        loss = logistic_loss(y, tx, w)  # penality term not cinluded
        losses.append(loss)
    if return_losses:
        return w, loss, losses
    return w, loss

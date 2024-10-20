import numpy as np
import implementations as imp
import helpers as hlp
import preprocessing as pre
import os

cwd = os.getcwd()
path = cwd + "/dataset_to_release"
abs_path = path

x_train_, x_test_, y_train_, train_ids_, test_ids_ = hlp.load_csv_data(path)

x = x_train_.copy()
x_submit = x_test_.copy()
y = y_train_.copy()

x_train, y_train, x_test, y_test = pre.split_data(x, y, 0.8)

correlation_thr = 0.8
nan_thr = 0.8
std_thr = 0.1

x_train_cleaned, x_test_cleaned = pre.clean_data(
    x, x_submit, correlation_thr, nan_thr, std_thr
)


def sigmoid(t):
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid


def calculate_gradient(y, tx, w):
    y_pred = sigmoid(tx @ w)
    gradient = np.dot(tx.T, y_pred - y) / y.shape[0]
    return gradient


def calculate_hessian(y, tx, w):
    N = y.shape[0]
    y_pred = sigmoid(np.dot(tx, w))
    S = np.diagflat(y_pred * (1 - y_pred))
    hessian = 1 / N * tx.T @ S @ tx
    return hessian


def learning_by_gradient_descent(y, tx, w, gamma):
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return w


def calculate_hessian_opti(y, tx, w):
    N = y.shape[0]
    y_pred = sigmoid(np.dot(tx, w))
    S = np.diagflat(y_pred * (1 - y_pred))
    hessian = 1 / N * tx.T @ S @ tx
    return hessian


# w_descent = learning_by_gradient_descent(y, x_train_cleaned, w, gamma)
def penalized_logistic_regression(y, tx, w, lambda_):
    N = y.shape[0]
    # loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    # loss = loss + (lambda_ / (2 * N)) * np.sum(np.square(w))
    gradient = gradient + lambda_ * N * w
    return gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w


def calculate_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    y_pred = sigmoid(np.dot(tx, w))
    loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss


# Generate a logistic regression using gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
    return w


w = np.zeros(x_train_cleaned.shape[1])
print("code is running")
# Gradient Descent
# w_gradient = learning_by_gradient_descent(y, x_train_cleaned, w, gamma)

# Newton Method
# w_newton = learning_by_newton_method(y, x_train_cleaned, w)

# Penalized Gradient Descent
# _, w_penalized = learning_by_penalized_gradient(y, x_train_cleaned, w, gamma, 0.1)

# Regularized Logistic Regression
# Hyperparameters tuning max_iters, gamma, lambda_
# Define the parameter grid
param_grid = {
    "max_iters": [500, 1000, 1500],
    "gamma": [0.001, 0.01, 0.1],
    "lambda_": [0.01, 0.1, 1],
}


# Define a function to perform the grid search
def grid_search_logistic_regression(y, tx, param_grid):
    best_params = None
    best_score = float("inf")

    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
            for lambda_ in param_grid["lambda_"]:
                w = np.zeros(tx.shape[1])
                w = reg_logistic_regression(y, tx, lambda_, w, max_iters, gamma)
                loss = calculate_loss(y, tx, w)

                if loss < best_score:
                    best_score = loss
                    best_params = {
                        "max_iters": max_iters,
                        "gamma": gamma,
                        "lambda_": lambda_,
                    }

    return best_params


# Perform the grid search
best_params = grid_search_logistic_regression(y, x_train_cleaned, param_grid)
print("Best hyperparameters found: ", best_params)

w_penalized = reg_logistic_regression(y, x_train_cleaned, 0.1, w, 1000, 0.01)
y_pred = sigmoid(np.dot(x_test_cleaned, w_penalized))
y_pred_ = np.copy(y_pred)

y_pred_threshold = np.where(y_pred_ > np.mean(y_pred_), 1, -1)
print(y_pred_threshold)

# max_iters = 1000
# gamma = 0.01
hlp.create_csv_submission(test_ids_, y_pred_threshold, "y_pred_ridge.csv")

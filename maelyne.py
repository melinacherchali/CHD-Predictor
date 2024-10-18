import numpy as np 
import implementations as imp
import helpers as hlp
import preprocessing as pre


path = '/Users/maelynenguyen/Desktop/dataset_to_release'
abs_path = path

x_train_, x_test_, y_train_, train_ids_, test_ids_ = hlp.load_csv_data(path)

x = x_train_.copy()
x_submit = x_test_.copy()
y = y_train_.copy()

x_train, y_train, x_test, y_test = pre.split_data(x, y, 0.8)  

correlation_thr = 0.8
nan_thr = 0.8
std_thr = 0.1

x_train_cleaned, x_test_cleaned = pre.clean_data(x, x_submit, correlation_thr, nan_thr, std_thr)


def sigmoid(t):
    sigmoid = 1 / (1 + np.exp(-t))
    return sigmoid



def calculate_gradient(y, tx, w):
    y_pred = sigmoid(tx@w)
    gradient = np.dot(tx.T, y_pred - y) / y.shape[0]
    return gradient

def calculate_hessian(y, tx, w):
    N = y.shape[0]
    y_pred = sigmoid(np.dot(tx, w))
    S = np.diagflat(y_pred * (1 - y_pred))
    hessian = 1/N * tx.T @ S @ tx
    return hessian


def learning_by_gradient_descent(y, tx, w, gamma):
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return w


def calculate_hessian_opti(y, tx, w):
    N = y.shape[0]
    y_pred = sigmoid(np.dot(tx, w))
    S = np.diagflat(y_pred * (1 - y_pred))
    hessian = 1/N * tx.T @ S @ tx
    return hessian



#w_descent = learning_by_gradient_descent(y, x_train_cleaned, w, gamma)
def penalized_logistic_regression(y, tx, w, lambda_):
    N = y.shape[0]
    #loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    #loss = loss + (lambda_ / (2 * N)) * np.sum(np.square(w))
    gradient = gradient + lambda_ / N * w
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

#Generate a logistic regression using gradient descent 
def logistic_regression(y,tx,initial_w,max_iters, gamma):
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return w 
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for i in range(max_iters):
        gradient = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
    return w


# Assuming you have already defined y, x_train_cleaned, and w
max_iters = 1000
gamma = 0.01

w = np.zeros(x_train_cleaned.shape[1])
print("code is running")
#Gradient Descent
#w_gradient = learning_by_gradient_descent(y, x_train_cleaned, w, gamma)

#Newton Method 
#w_newton = learning_by_newton_method(y, x_train_cleaned, w)

#Penalized Gradient Descent
#_, w_penalized = learning_by_penalized_gradient(y, x_train_cleaned, w, gamma, 0.1)

#Regularized Logistic Regression
w_penalized = reg_logistic_regression(y_train, x_train_cleaned, 0.1, w, 1000, 0.01)
y_pred = sigmoid(np.dot(x_test_cleaned, w_penalized))

y_pred_ = np.round(y_pred).copy()

print("marche")

y_pred_[y_pred_ == 0] = -1

int_y_pred = y_pred_.astype(int)
print(y_pred_)
hlp.create_csv_submission(test_ids_, int_y_pred,"y_pred_Logistic_Descent.csv")
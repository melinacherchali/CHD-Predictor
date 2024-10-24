import numpy as np
import implementations as imp
import helpers as hlp
import preprocessing as pre
import os

cwd = "/Users/maelynenguyen/Desktop"
path = cwd + "/dataset_to_release"
abs_path = path

x_train_, x_test_, y_train_, train_ids_, test_ids_ = hlp.load_csv_data(path)

x = x_train_.copy()
x_submit = x_test_.copy()
y = y_train_.copy()


correlation_thr = 0.8
nan_thr = 0.8
std_thr = 0.1

x_train_cleaned, x_test_cleaned = pre.clean_data(x, x_submit, correlation_thr, nan_thr, std_thr)



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


import implementations as imp 

def grid_search_logistic_regression(y_train,x_train_cleaned,param_grid,w_initial):

    best_params = None
    best_score = float("inf")
    best_w = w_initial

    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
                w, loss = imp.logistic_regression(y_train, x_train_cleaned, w_initial, max_iters, gamma)
                print("Loss:", loss)
                if loss < best_score:
                    best_score = loss
                    best_w = w
                    best_params = {
                        "max_iters": max_iters,
                        "gamma": gamma,
                    }
                    
        return best_w, best_params

    # Perform the grid search
#best_params = grid_search_logistic_regression(y, x_train_cleaned, param_grid)
#print("Best hyperparameters found: ", best_params)

#w_penalized = reg_logistic_regression(y, x_train_cleaned, 0.1, w, 1000, 0.01)
#y_pred = sigmoid(np.dot(x_test_cleaned, w_penalized))
#y_pred_ = np.copy(y_pred)

#y_pred_threshold = np.where(y_pred_ > np.mean(y_pred_), 1, -1)
#print(y_pred_threshold)

# max_iters = 1000
# gamma = 0.01
#hlp.create_csv_submission(test_ids_, y_pred_threshold, "y_pred_ridge.csv")





def cross_validate(x, y, model, k=5, seed=1):
    """
    Perform k-fold cross-validation on the given data and model.
        
    parameters:
    x: the input data
    y: the target values
    model: a function that takes train_x, train_y, test_x as input and returns predictions for test_x
    k: the number of folds
    seed: random seed for reproducibility
        
    returns: list of accuracy scores for each fold
    """
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    fold_size = x.shape[0] // k
    #fold_size_test = x_submit.shape[0] // k
    scores = []

    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        print(test_indices)
        x_train, y_train = x[train_indices], y[train_indices]
        x_test,y_test = x[test_indices], y[test_indices]
    
        y_pred = model(x_train, y_train, x_test)
        accuracy = np.mean(y_pred == y_test)
        scores.append(accuracy)

    return scores

def logistic (x_train,y_train,x_test) :
    #best_w_log = grid_search_logistic_regression(y_train,x_train,param_grid,np.zeros(x_train.shape[1]))
    best_w_log, _ = imp.logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 1000, 0.01)
    y_sub = imp.sigmoid(x_test @ best_w_log)
    y_sub = np.where(y_sub > .76, 1, -1)
    hlp.create_csv_submission(test_ids_, y_sub, "y_pred_logistic.csv")
    return y_sub 

#scores = cross_validate(x_train_cleaned, x_test_cleaned, y, logistic, k=5, seed=1)

#print(scores)
def k_fold_split(x, y, k):
    """Utility function to split data into k folds."""
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    fold_size = len(y) // k
    folds = []
    
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        folds.append((train_indices, test_indices))
    
    return folds

def grid_search_logistic_regression(y_train, x_train_cleaned, param_grid, w_initial, k=5):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    
    # Generate folds for cross-validation
    folds = k_fold_split(x_train_cleaned, y_train, k)

    # Iterate over each combination of parameters
    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
            total_loss = 0
            
            # Perform k-fold cross-validation
            for train_indices, test_indices in folds:
                x_train_fold = x_train_cleaned[train_indices]
                y_train_fold = y_train[train_indices]
                x_test_fold = x_train_cleaned[test_indices]
                y_test_fold = y_train[test_indices]
                
    
                w, loss = imp.logistic_regression(y_train_fold, x_train_fold, w_initial, max_iters, gamma)
                total_loss += loss

            avg_loss = total_loss / k
            print(f"Max Iters: {max_iters}, Gamma: {gamma}, Avg Loss: {avg_loss}")

            if avg_loss < best_score:
                best_score = avg_loss
                best_w = w
                best_params = {
                    "max_iters": max_iters,
                    "gamma": gamma,
                }

    return best_w, best_params

#Cross validation with a grid search 
#best_w, best_params = grid_search_logistic_regression(y, x_train_cleaned, param_grid, np.zeros(x_train_cleaned.shape[1]))
#print("Best hyperparameters found: ", best_params)
#y_pred = imp.sigmoid(np.dot(x_test_cleaned, best_w))
#y_pred = np.where(y_pred > .75 , 1, -1)
#hlp.create_csv_submission(test_ids_, y_pred, "y_pred_.csv")



def adam_optimizer(y, x, w, max_iters, gamma, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = np.zeros_like(w), np.zeros_like(w)
    for iter in range(max_iters):
        # Compute predictions and gradients
        
        predictions = imp.sigmoid(np.dot(x, w))
        gradient = np.dot(x.T, predictions - y)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** (iter + 1))
        v_hat = v / (1 - beta2 ** (iter + 1))

        # Update weights
        w -= gamma * m_hat / (np.sqrt(v_hat) + epsilon)
    return w

from sklearn.metrics import f1_score

def adam_optimizer_better(y, x, w, max_iters, gamma, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=None, tolerance=1e-5):
    m, v = np.zeros_like(w), np.zeros_like(w)
    beta1_power, beta2_power = 1, 1  # For bias correction
    initial_gamma = gamma  # Store initial learning rate for decay
    decay_rate = 0.01  # Learning rate decay factor
    
    for iter in range(max_iters):
        if batch_size:
            indices = np.random.choice(len(y), batch_size, replace=False)
            x_batch, y_batch = x[indices], y[indices]
        else:
            x_batch, y_batch = x, y
        
        predictions = imp.sigmoid(np.dot(x_batch, w))
        gradient = np.dot(x_batch.T, predictions - y_batch)
        
        if np.linalg.norm(gradient) < tolerance:  # Early stopping
            print(f"Stopping early at iteration {iter}")
            break
        
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        beta1_power *= beta1
        beta2_power *= beta2
        m_hat = m / (1 - beta1_power)
        v_hat = v / (1 - beta2_power)
        
        # Update weights
        gamma = initial_gamma * (1 / (1 + decay_rate * iter))  # Decaying learning rate
        w -= gamma * m_hat / (np.sqrt(np.maximum(v_hat, epsilon)) + epsilon)
        
        if iter % 100 == 0:  # Logging every 100 iterations
            loss = np.mean((y_batch - predictions) ** 2)
            print(f"Iteration {iter}, Loss: {loss}")

    return w


def best_threshold(y_pred,y):
    # Search for the best threshold
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred_thres = np.where(y_pred > threshold, 1, -1)
        f1 = f1_score(y, y_pred_thres)
        f1_scores.append(f1)

# Find the threshold with the highest F1 score
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold



w_initial_adam = np.zeros(x_train_cleaned.shape[1])
#w_adam = adam_optimizer(y, x_train_cleaned, w_initial_adam, 1000, 0.01,best_beta1,best_beta2)

w_adam = adam_optimizer_better(y, x_train_cleaned, w_initial_adam, 1000, 0.01, batch_size=100, tolerance=1e-5)
y_train_adam = imp.sigmoid(np.dot(x_train_cleaned,w_adam))
optimal_threshold = best_threshold(y_train_adam,y)

y_pred_adam = imp.sigmoid(np.dot(x_test_cleaned, w_adam))

#print("Optimal Threshold based on F1 score:", optimal_threshold)
#y_sub = np.where(y_pred_adam > optimal_threshold , 1, -1)
print("Optimal Threshold based on F1 score:", optimal_threshold)
y_sub = np.where(y_pred_adam > optimal_threshold , 1, -1)
hlp.create_csv_submission(test_ids_, y_sub, "y_pred_adam.csv")

#support vector machine 

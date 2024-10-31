import numpy as np
import implementations as imp

### Helper functions ###


def test_function(func):
    y = np.array([5, 10, 17])
    tx = np.array([[1, 2], [1, 3], [1, 4]])
    initial_w = np.array([0, 0])
    max_iters = 1000
    gamma = 0.1
    a = func(y, tx, initial_w, max_iters, gamma)
    print(a[0], "MSE : ", a[1])


def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = np.shape(y)[0]
    e = y - np.dot(tx, w)
    return 1 / (2 * N) * (e.T.dot(e))


### Gradient Descent ###


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
    e = y - np.dot(tx, w)
    grad = -1 / N * np.dot(tx.T, e)
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
    w = initial_w
    loss = compute_loss(y, tx, w)
    losses = [loss]
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

    return ws, losses


### Stochastic Gradient Descent ###


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=1):
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
    return ws, losses


def compute_stoch_gradient(y, tx, w, batch_size=1):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    batches = batch_iter(y, tx, batch_size=batch_size, num_batches=1, shuffle=True)
    gradient = np.mean(
        [compute_gradient(y_batch, tx_batch, w) for y_batch, tx_batch in batches],
        axis=0,
    )
    return gradient


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


### Logistic Regression ###


def sigmoid(t):
    sigmoid = 1 / (1 + np.exp(-t))
    sigmoid = np.where(sigmoid == 0, 1e-10, sigmoid)
    sigmoid = np.where(sigmoid == 1, 1 - 1e-10, sigmoid)
    return sigmoid


def logistic_gradient(y, tx, w):
    y_pred = sigmoid(tx @ w)
    gradient = np.dot(tx.T, y_pred - y) / y.shape[0]
    return gradient


def penalized_logistic_regression(y, tx, w, lambda_):
    N = y.shape[0]
    gradient = logistic_gradient(y, tx, w) + lambda_ * 2 * w  # penalized gradient
    return gradient


def logistic_loss(y, tx, w):
    """
    y : binary label (0,1)
    tx : features
    w : weights
    """
    N = tx.shape[0]
    loss = (
        -1
        / N
        * np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w)))
    )
    return loss


### Evaluation ###

def accuracy_score_(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Validate input
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")

    # Map -1 to 0 for binary classification
    y_true = np.where(y_true == -1, 0, 1)
    y_pred = np.where(y_pred == -1, 0, 1)

    # Correct predictions
    correct_predictions = np.sum(y_true == y_pred)

    # Accuracy: Correct predictions / Total predictions
    accuracy = correct_predictions / len(y_true)

    return accuracy


def f1_score_(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Validate input
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")

    # Map -1 to 0 for binary classification
    y_true = np.where(y_true == -1, 0, 1)
    y_pred = np.where(y_pred == -1, 0, 1)

    # True Positives (TP): Both predicted and actual are positive (1)
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False Positives (FP): Predicted positive (1) but actual negative (0)
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # False Negatives (FN): Predicted negative (0) but actual positive (1)
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def k_fold_split(x, y, k):
    """Utility function to split data into k folds."""
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    fold_size = len(y) // k
    folds = []

    for i in range(k):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.concatenate(
            (indices[: i * fold_size], indices[(i + 1) * fold_size :])
        )
        folds.append((train_indices, test_indices))

    return folds


def grid_search_gd(y_train, x_train, param_grid, w_initial, k=5):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    losses = []
    
    # Generate folds for cross-validation
    folds = k_fold_split(x_train, y_train, k)
    
    # Iterate over each combination of parameters
    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
            total_loss = 0
            # Perform k-fold cross-validation
            for train_indices, test_indices in folds:
                x_train_fold = x_train[train_indices]
                y_train_fold = y_train[train_indices]
                x_test_fold = x_train[test_indices]
                y_test_fold = y_train[test_indices]
                
                # Train the model with training data
                w, _ = imp.mean_squared_error_gd(y_train_fold, x_train_fold, w_initial, max_iters, gamma)
                # Test the model with testing data
                loss = imp.compute_loss(y_test_fold, x_test_fold, w)
                
                total_loss += loss
            
            avg_loss = total_loss / k
            losses.append((gamma, avg_loss))
            print(f"Max Iters: {max_iters}, Gamma: {gamma}, Avg Loss: {avg_loss}")
            
            if avg_loss < best_score:
                best_score = avg_loss
                best_w = w
                best_params = {
                    "max_iters": max_iters,
                    "gamma": gamma,
                }
                
    return best_w, best_params, losses            
    

def grid_search_sgd(y_train, x_train, param_grid, w_initial, k=5):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    losses = []
    
    # Generate folds for cross-validation
    folds = k_fold_split(x_train, y_train, k)
    
    # Iterate over each combination of parameters
    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
            for batch_size in param_grid["batch_size"]:
                total_loss = 0
                # Perform k-fold cross-validation
                for train_indices, test_indices in folds:
                    x_train_fold = x_train[train_indices]
                    y_train_fold = y_train[train_indices]
                    x_test_fold = x_train[test_indices]
                    y_test_fold = y_train[test_indices]
                    
                    # Train the model with training data
                    ws, _ = imp.stochastic_gradient_descent(y_train_fold, x_train_fold, w_initial, max_iters, gamma, batch_size)
                    w = ws[-1]
                    # Test the model with testing data
                    loss = imp.compute_loss(y_test_fold, x_test_fold, w)
                    
                    total_loss += loss
                
                avg_loss = total_loss / k
                losses.append((gamma, avg_loss))
                print(f"Max Iters: {max_iters}, Gamma: {gamma}, Avg Loss: {avg_loss}")
                
                if avg_loss < best_score:
                    best_score = avg_loss
                    best_w = w
                    best_params = {
                        "max_iters": max_iters,
                        "gamma": gamma,
                        "batch_size": batch_size,
                    }
                
    return best_w, best_params, losses

def grid_search_ridge(y_train, x_train, param_grid, w_initial, k=5):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    losses = []
    
    # Generate folds for cross-validation
    folds = k_fold_split(x_train, y_train, k)
    
    # Iterate over each combination of parameters
    for lambda_ in param_grid["lambdas"]:
        total_loss = 0
        # Perform k-fold cross-validation
        for train_indices, test_indices in folds:
            x_train_fold = x_train[train_indices]
            y_train_fold = y_train[train_indices]
            x_test_fold = x_train[test_indices]
            y_test_fold = y_train[test_indices]
            
            # Train the model with training data
            w, _ = imp.ridge_regression(y_train_fold, x_train_fold, lambda_)
            # Test the model with testing data
            loss = imp.compute_loss(y_test_fold, x_test_fold, w)
            
            total_loss += loss
        
        avg_loss = total_loss / k
        losses.append((lambda_, avg_loss))
        print(f"Lambda: {lambda_}, Avg Loss: {avg_loss}")
        
        if avg_loss < best_score:
            best_score = avg_loss
            best_w = w
            best_params = {
                "lambdas": lambda_,
            }
    
    return best_w, best_params, losses
        
        


def grid_search_logistic_regression(
    y_train, x_train_cleaned, param_grid, w_initial, k=5
):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    losses = []

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

                # Train the model with training data
                w, _ = imp.logistic_regression(
                    y_train_fold, x_train_fold, w_initial, max_iters, gamma
                )
                # Test the model with testing data
                loss = imp.logistic_loss(y_test_fold, x_test_fold, w)

                total_loss += loss

            avg_loss = total_loss / k
            losses.append((max_iters, gamma, avg_loss))
            print(f"Max Iters: {max_iters}, Gamma: {gamma}, Avg Loss: {avg_loss}")

            if avg_loss < best_score:
                best_score = avg_loss
                best_w = w
                best_params = {
                    "max_iters": max_iters,
                    "gamma": gamma,
                }

    return best_w, best_params, losses


def grid_search_reg_logistic_regression(
    y_train, x_train_cleaned, param_grid, w_initial, k=5
):
    best_params = None
    best_score = float("inf")
    best_w = w_initial
    losses = []

    # Generate folds for cross-validation
    folds = k_fold_split(x_train_cleaned, y_train, k)

    # Iterate over each combination of parameters
    for max_iters in param_grid["max_iters"]:
        for gamma in param_grid["gamma"]:
            for lambda_ in param_grid["lambda_"]:
                total_loss = 0
                # Perform k-fold cross-validation
                for train_indices, test_indices in folds:
                    x_train_fold = x_train_cleaned[train_indices]
                    y_train_fold = y_train[train_indices]
                    x_test_fold = x_train_cleaned[test_indices]
                    y_test_fold = y_train[test_indices]

                    # Train the model with training data
                    w, _ = imp.reg_logistic_regression(
                        y_train_fold, x_train_fold, lambda_, w_initial, max_iters, gamma
                    )
                    # Test the model with testing data
                    loss = imp.logistic_loss(y_test_fold, x_test_fold, w)

                    total_loss += loss

                avg_loss = total_loss / k
                losses.append((max_iters, gamma, lambda_, avg_loss))
                print(
                    f"Max Iters: {max_iters}, Gamma: {gamma}, Lambda: {lambda_}, Avg Loss: {avg_loss}"
                )

                if avg_loss < best_score:
                    best_score = avg_loss
                    best_w = w
                    best_params = {
                        "max_iters": max_iters,
                        "gamma": gamma,
                        "lambda_": lambda_,
                    }

    return best_w, best_params, losses


def best_threshold(y_pred, y):
    """y : binary (-1,1)
    y_pred : binary (0,1)
    """
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred_thres = np.where(y_pred > threshold, 1, -1)
        f1 = f1_score_(y, y_pred_thres)
        f1_scores.append(f1)

    # Find the threshold with the highest F1 score
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold


### Optimizer  ###


def adam_optimizer(y, x, w, max_iters, gamma, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m, v = np.zeros_like(w), np.zeros_like(w)
    for iter in range(max_iters):
        # Compute predictions and gradients
        predictions = imp.sigmoid(np.dot(x, w))
        gradient = np.dot(x.T, predictions - y)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient**2)

        # Bias correction
        m_hat = m / (1 - beta1 ** (iter + 1))
        v_hat = v / (1 - beta2 ** (iter + 1))

        # Update weights
        w -= gamma * m_hat / (np.sqrt(v_hat) + epsilon)
    return w

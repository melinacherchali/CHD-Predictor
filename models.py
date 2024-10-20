import numpy as np
import implementations as imp


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    A = np.linalg.inv(tx.T @ tx)
    w = A @ tx.T @ y
    N = tx.shape[0]
    mse = 1 / (2 * N) * np.sum((y - tx @ w) ** 2)

    return w, mse


def test_model(y_test, x_test, w):
    """Test the model using the test data.
    Args:
        y_test: numpy array of shape (N,), N is the number of samples.
        x_test: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape(D,), D is the number of features.

    Returns:
        mse: scalar.

    >>> test_model(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([ 0.21212121, -0.12121212]))
    8.666684749742561e-33
    """
    N = x_test.shape[0]
    mse = 1 / (2 * N) * np.sum((y_test - x_test @ w) ** 2)
    return mse


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Logistic regression cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5  # Avoid log(0)
    cost = (
        -1
        / m
        * (np.dot(y, np.log(h + epsilon)) + np.dot((1 - y), np.log(1 - h + epsilon)))
    )
    return cost


# Gradient descent for logistic regression
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# Predict function for logistic regression
def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]

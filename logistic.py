import numpy as np 
import implementations as imp
import helpers as hlp
import preprocessing as pre
import models as mdl
path = '/Users/maelynenguyen/Desktop/ML/dataset_to_release'
abs_path = path
#os.getcwd() + 
x_train_, x_test_, y_train_, train_ids_, test_ids_ = hlp.load_csv_data(abs_path)

x = x_train_.copy()
x_submit = x_test_.copy()
y = y_train_.copy()

x_train, y_train, x_test, y_test = pre.split_data(x, y, 0.8)  

correlation_thr = 0.8
nan_thr = 0.8
std_thr = 0.1

x_train_cleaned, x_test_cleaned = pre.clean_data(x_train, x_test, correlation_thr, nan_thr, std_thr)

#Logistic Regression 
theta = np.zeros(x_train_cleaned.shape[1])
learning_rate = 0.01
num_iterations = 100

# Train logistic regression model
theta, cost_history = mdl.gradient_descent(x_train_cleaned, y_train, theta, learning_rate, num_iterations)

y_pred = mdl.sigmoid(np.dot(x_train_cleaned, theta))
y_pred = np.round(y_pred)
y_pred_copy = np.copy(y_pred).astype(int)

y_pred_copy[y_pred == 0] = -1

print(y_pred_copy)
hlp.create_csv_submission(test_ids_, y_pred,"y_pred_logistic_regression.csv")



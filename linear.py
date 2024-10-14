#Standardize the data
import numpy as np 
import implementations as imp
import helpers as hlp
import preprocessing as pre

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

#Model 1 : Gradient Descent 
w, loss = imp.mean_squared_error_gd(y_train, x_train_cleaned, np.zeros(x_train_cleaned.shape[1]), 200, 0.1)
y_pred = np.dot(x_train_cleaned, w)
y_pred_GD = pre.threshold(y_pred)
hlp.create_csv_submission(test_ids_, y_pred_GD,"y_pred_Gradient_Descent.csv")

#Model 2 : Stochastic Gradient Descent
w_, loss_ = imp.mean_squared_error_sgd(y_train, x_train_cleaned, np.zeros(x_train_cleaned.shape[1]), 100, 0.1)
y_pred_ = np.dot(x_train_cleaned, w_)
y_pred_SGD = pre.threshold(y_pred_)
hlp.create_csv_submission(test_ids_, y_pred_SGD,"y_pred_sgd.csv")

#Model 3 : Least Squares
w_least, loss_least = imp.least_squares(y_train, x_train_cleaned)
y_pred_least = np.dot(x_train_cleaned, w_least)
y_pred_thresholded_least = pre.threshold(y_pred_least)

hlp.create_csv_submission(test_ids_, y_pred_thresholded_least,"y_pred_least.csv")

#Compute the accuracy of the models
#accuracy_SGD = np.mean(y_test == y_pred_SGD)
#print(f"Stochastic Gradient Descent Accuracy: {accuracy_SGD * 100:.2f}%")

#accuracy_GD = np.mean(y_test == y_pred_GD)
#print(f"Gradient Descent Accuracy: {accuracy_GD * 100:.2f}%")

#accuracy_least = np.mean(y_test == y_pred_least)
#print(f"Least Squares Accuracy: {accuracy_least * 100:.2f}%")




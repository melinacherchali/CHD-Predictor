
import numpy as np
import helpers as hp



# data loading



abs_path="/Users/maelynenguyen/Desktop/dataset_to_release"
x_train_, x_test_, y_train_, train_ids_, test_ids_ =hp.load_csv_data(abs_path)

# # save the data in a file to load it faster

path_to_save= "/Users/maelynenguyen/Desktop/"

np.save(path_to_save+"f_x_train_",x_train_)
np.save(path_to_save+"f_x_test_",x_test_)
np.save(path_to_save+"f_y_train_",y_train_)
np.save(path_to_save+"f_train_ids_",train_ids_)
np.save(path_to_save+"f_test_ids_",test_ids_)




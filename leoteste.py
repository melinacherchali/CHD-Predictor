import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp


path="/Users/leocusumano/myCloud/EPFL/MA1/ML/Proj1/DATASETS/f_sets/"

x_train=np.load(path+"f_x_train_.npy")
x_test=np.load(path+"f_x_test_.npy")
y_train=np.load(path+"f_y_train_.npy")
test_ids=np.load(path+"f_test_ids_.npy")

nan_col_ratio=(np.isnan(x_train).sum(axis=0)/x_train.shape[0])
nan_row_ratio=(np.isnan(x_train).sum(axis=1)/x_train.shape[1])

a=plt.subplots(2)
a[1][0].hist(nan_col_ratio,bins=100)
a[1][0].set_title("Column NaN ratio")
a[1][1].set_title("Row NaN ratio")
a[1][1].hist(nan_row_ratio,bins=100)
plt.show()

x,y=np.meshgrid(nan_col_ratio,nan_row_ratio)
# to find a score for each row and column that take into account the NaN ratio, we can multiply the two matrices and take the square root
dual_ratios=np.sqrt(x*y)
plt.matshow(dual_ratios[::300,:].T)
plt.colorbar(label="SQRT(rowNaNratio*colNaNratio)")

plt.show()

def nan_to_drop(data,dual_nan_ratio_thr):
    i_drop,j_drop=np.where(data>dual_nan_ratio_thr)
    return np.unique(i_drop),np.unique(j_drop)

def drop_col_row_nan(data_x,data_y,nan_trd=0.7):
    nan_col_ratio=(np.isnan(x_train).sum(axis=0)/x_train.shape[0])
    nan_row_ratio=(np.isnan(x_train).sum(axis=1)/x_train.shape[1])
    x,y=np.meshgrid(nan_col_ratio,nan_row_ratio)
    # to find a score for each row and column that take into account the NaN ratio
    # we can multiply the two matrices and take the square root
    dual_ratio=np.sqrt(x*y)
    i_drop,j_drop=nan_to_drop(dual_ratio,nan_trd)
    w_row_x=np.delete(data_x,i_drop,axis=0)
    w_colrow_x=np.delete(w_row_x,j_drop,axis=1)
    w_row_y=np.delete(data_y,i_drop,axis=0)
    
    return w_colrow_x,w_row_y

dataclean_x,dataclean_y=drop_col_row_nan(x_train,y_train)

final,test=pp.clean_data(dataclean_x,x_test,1,0.1,0.1)













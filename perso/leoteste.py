
import numpy as np
import matplotlib.pyplot as plt



path="/Users/leocusumano/myCloud/EPFL/MA1/ML/Proj1/DATASETS/f_sets/"

x_train=np.load(path+"f_x_train_.npy")
x_test=np.load(path+"f_x_test_.npy")
y_train=np.load(path+"f_y_train_.npy")
test_ids=np.load(path+"f_test_ids_.npy")

nan_col_ratio=(np.isnan(x_train).sum(axis=0)/x_train.shape[0])
nan_row_ratio=(np.isnan(x_train).sum(axis=1)/x_train.shape[1])

a=plt.subplots(2)
a[1][0].hist(nan_col_ratio,color="red",bins=100)
a[1][0].set_title(r"Column NaN ratio $\eta_j$")
a[1][1].set_title(r"Row NaN ratio $\eta_i$")
a[1][1].hist(nan_row_ratio,bins=200)
plt.tight_layout()
plt.show()

x,y=np.meshgrid(nan_col_ratio,nan_row_ratio)
# to find a score for each row and column that take into account the NaN ratio, we can multiply the two matrices and take the square root
dual_ratios=np.sqrt(x*y)

plt.matshow(dual_ratios[::600,:].T,cmap="plasma")
plt.xlabel("Rapresentative rows")
plt.ylabel("Predictors")
plt.colorbar(label=r"$\sigma_{i,j}$")
plt.legend()
plt.tight_layout()
plt.show()

















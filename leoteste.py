import preprocessing as pp
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp

path="/Users/leocusumano/myCloud/EPFL/MA1/ML/Proj1/DATASETS/f_sets/"

x_train=np.load(path+"f_x_train_.npy")
x_test=np.load(path+"f_x_test_.npy")
y_train=np.load(path+"f_y_train_.npy")


x_t2,t2=pp.clean_data(x_train,x_test,1,0.3,0)
print(x_t2.shape)


def PCA(xtrain,num_axis):
    """
    This function performs PCA on the given data.
    """
    
    # Compute the covariance matrix
    cov_matrix = np.cov(xtrain, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)
    
    # Sort the eigenvalues in descending order
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    
    # Select the first num_axis eigenvectors
    eig_vectors = eig_vectors[:, :num_axis]
    
    # Project the data onto the new basis
    xtrain_pca = np.dot(xtrain, eig_vectors)
    
    explained_variance = np.zeros(eig_values.shape)
    explained_variance[0] = eig_values[0]
    for i,v in enumerate(eig_values):
        if i>0:
            explained_variance[i] = explained_variance[i-1]+v
    
    

    prct_explained_variance=explained_variance/explained_variance[-1]*100
    plt.figure()
    plt.plot(prct_explained_variance,label="Percentage of explained variance by components")
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of explained variance")
    prct90=np.where(prct_explained_variance>90)[0][0]
    plt.axvline(prct90, color='r', linestyle='--',label="90% explained variance at "+str(prct90)+" components")
    plt.legend()
    plt.show()
    
    return xtrain_pca
x_pca=PCA(x_t2,70)

print(np.isnan(x_pca).sum())
print(imp.mean_squared_error_gd(y_train,x_pca,np.zeros(x_pca.shape[1]),30,0.1))



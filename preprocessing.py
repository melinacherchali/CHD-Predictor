import numpy as np
import helpers as hlp
import matplotlib.pyplot as plt
import time


def correlation(data, threshold=0.95):
    """
    Calculate the correlation matrix and return the indices of columns that are correlated.
    """
    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=False)

    # Create a mask for columns to keep
    to_remove = set()

    # Identify indices of correlated columns
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if abs(correlation_matrix[i, j]) >= threshold:
                to_remove.add(j)  # Mark the column j for removal
                # print("Column", j, "is perfectly correlated with column", i)
    
    # Create a new array with the columns to keep
    return list(to_remove)


def columns_to_drop(data, correlation_thr=0.95, nan_thr=0.9, std_thr=0.1):
    """
    This function returns the columns to drop in the data, namely:
    - columns with high number of missing values
    - columns with low std
    - correlated columns
    
    parameters:
    data: the data
    correlation_thr: the columns with correlation score above this threshold will be dropped
    nan_thr: the columns with more than this threshold ratio of NaN values will be dropped
    std_thr: the columns with std below this threshold will be dropped
    
    returns: list of columns to drop
    
    """
    columns_to_drop = []

    # drop columns with more than 90% missing values
    nan_col = np.isnan(data).sum(axis=0)
    columns_missing_values = np.where(nan_col > (nan_thr * data.shape[0]))[0].tolist()
    columns_to_drop.extend(columns_missing_values)
    print(
        f"Number of columns with more than {nan_thr} NaN:", len(columns_missing_values)
    )

    # drop columns with std < 0.1
    std_devs = np.nanstd(data, axis=0)  # std, ignoring NaNs
    low_std_mask = std_devs < std_thr  # mask of columns with std < 0.1
    columns_constant = np.where(low_std_mask)[0].tolist()
    columns_to_drop.extend(columns_constant)
    print(f"Number of columns with std < {std_thr}:", len(columns_constant))

    # drop perfectly correlated columns
    columns_correlated = correlation(data, correlation_thr)
    columns_to_drop.extend(columns_correlated)
    print(f"Number of perfectly correlated columns:", len(columns_correlated))

    return list(set(columns_to_drop))  # removes duplicates


def clean_data(x_train, x_test, correlation_thr=0.95, nan_thr=0.9, std_thr=0.1):
    """
    This function cleans the data by dropping columns and handling missing values.
    
    returns: cleaned train and test data
    """
    to_drop = columns_to_drop(x_train, correlation_thr, nan_thr, std_thr)
    print("Columns to drop:", to_drop)
    clean_train = np.delete(x_train, to_drop, axis=1)
    clean_test = np.delete(x_test, to_drop, axis=1)

    # Handle missing values
    clean_train, clean_test = handle_nan(clean_train), handle_nan(clean_test)

    # Standardize the data
    standardized_train = (clean_train - np.mean(clean_train, axis=0)) / np.std(
        clean_train, axis=0
    )
    standardized_test = (clean_test - np.mean(clean_test, axis=0)) / np.std(
        clean_test, axis=0
    )

    # Check for correlation post processing
    columns_correlated = correlation(standardized_train)
    print(
        "Number of perfectly correlated columns after cleaning:",
        len(columns_correlated),
    )
    final_train = np.delete(standardized_train, columns_correlated, axis=1)
    final_test = np.delete(standardized_test, columns_correlated, axis=1)

    return final_train, final_test


def standardize(data):
    """
    This function standardizes the data.
    """
    assert np.isnan(data).sum() == 0, 'Data contains NaN values'
    # Standardize the data
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    
    return standardized_data


def is_column_binary(column):
    """
    This function checks if the columns are binary.
    """
    return all(np.isin(np.unique(column), [0, 1]))


def compute_mode(column):
    """
    This function calculates the mode of the data.
    """
    unique, counts = np.unique(
        column, return_counts=True
    )  # unique values and their counts
    return unique[np.argmax(counts)]  # return the value with the highest count


def handle_nan(data):
    """
    This function replaces NaN values with the mean of the column for non-binary columns and the mode for binary columns.

    """
    print("Handling NaN values...")
    cleaned_data = data.copy()
    for i in range(data.shape[1]):
        nan_ids = np.where(np.isnan(data[:, i]))  # NaN indices
        mode = compute_mode(data[:, i])
        mean = np.nanmean(data[:, i])
        if is_column_binary(data[:, i]):  # for binary columns replace NaN with the mode
            cleaned_data[nan_ids, i] = mode
        else:  # for non-binary columns replace NaN with the mean
            cleaned_data[nan_ids, i] = np.round(mean)  # round to the nearest integer
    return cleaned_data


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.
    """
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    n_train = int(np.floor(x.shape[0] * ratio))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def threshold(y_pred):
    mean_data = np.mean(y_pred)
    return np.where(y_pred >= mean_data, -1, 1)

def PCA(xtrain,num_axis,graph=False):
    """
    This function performs PCA on the given data. And returns the data projected on the new basis.
    
    parameters:
    xtrain: the input data
    num_axis: the number of dimensions to keep
    graph: boolean to plot the percentage of explained variance
    
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
    
    # Compute the percentage of explained variance
    explained_variance = np.zeros(eig_values.shape)
    explained_variance[0] = eig_values[0]
    for i,v in enumerate(eig_values):
        if i>0:
            explained_variance[i] = explained_variance[i-1]+v
            
    prct_explained_variance=explained_variance/explained_variance[-1]*100  
    
    # Find the number of components needed to explain 90% of the variance      
    prct90=np.where(prct_explained_variance>90)[0][0]
    
    if graph:
        
        plt.figure()
        plt.plot(prct_explained_variance,label="Percentage of explained variance by components")
        plt.xlabel("Number of components")
        plt.ylabel("Percentage of explained variance")
        plt.axvline(prct90, color='r', linestyle='--',label="90% explained variance at "+str(prct90)+" components")
        plt.legend()
        plt.show()
    
    return xtrain_pca


    
def i_j_rule_treshold(data,treshold):
    """
    This function returns once the columns/rows having their mean above a treshold .
    """
    col_mean=np.mean(data,axis=0)
    row_mean=np.mean(data,axis=1)
    print("Max mean NaN score rows : ",np.max(row_mean))
    print("Max mean NaN score columns : ",np.max(col_mean))
    i_drop=np.where(row_mean>treshold)
    j_drop=np.where(col_mean>treshold)
    return i_drop[0],j_drop[0]

def alternative_nan_handeling(data_x,data_y,data_x_to_predict=None,nan_trd=.5):
    """
    This function drops the rows and columns with NaN values above a certain threshold.
    The criterion is computed, for each item of the array by taking the square root of the product of the NaN ratio of the row and column.
    Criterion: NaN_score(i,j)=sqrt(NaN_ratio_row(i)*NaN_ratio_col(j))
    All the rows and columns having a single computed NaN score above the threshold are dropped.
    
    arguments:
    data_x: the input data
    data_y: the output data
    data_x_to_predict: the input data to predict
    nan_trd: the threshold for the NaN ratio
    
    returns: x_data with reduced rows and columns, y_data with reduced rows
    
    """
    assert (data_x_to_predict is None) or (data_x_to_predict.shape[1]==data_x.shape[1]), "The number of columns in the data to predict should be the same as the input data"
    if data_x_to_predict is None:
        data_x_to_predict=np.array([])
    cum_x=np.concatenate((data_x,data_x_to_predict),axis=0)
    nan_col_ratio=(np.isnan(cum_x).sum(axis=0)/data_x.shape[0])
    nan_row_ratio=(np.isnan(cum_x).sum(axis=1)/data_x.shape[1])
    x,y=np.meshgrid(nan_col_ratio,nan_row_ratio)
    # to find a score for each row and column that take into account the NaN ratio
    # we can multiply the two matrices and take the square root
    dual_ratio=np.sqrt(x*y)
    i_drop,j_drop=i_j_rule_treshold(dual_ratio,nan_trd)
    #the rows to drop should be only in the data_x avoiding the data_x_to_predict
    i_drop_=i_drop[i_drop<data_x.shape[0]]
    #return the data with the rows and columns dropped
    w_row_x=np.delete(data_x,i_drop_,axis=0)
    w_colrow_x=np.delete(w_row_x,j_drop,axis=1)
    # reduce the y data accordingly to the rows dropped
    w_row_y=np.delete(data_y,i_drop_,axis=0)
    
    print(f"Number of rows dropped because of a NaN score > {nan_trd}: ",i_drop_.shape[0])
    print(f"Number of columns dropped because of a NaN score > {nan_trd}: ",j_drop.shape[0])
    
    if data_x_to_predict.size!=0:
        # reduce the x_data_to_predict only on the columns
        w_col_x_to_predict=np.delete(data_x_to_predict,j_drop,axis=1)
        return w_colrow_x , w_row_y , w_col_x_to_predict
    
    return w_colrow_x , w_row_y

def Edited_clean_data(x_train,y_train, x_test, correlation_thr=0.95, nan_thr=0.5, std_thr=0.1):
    """
    This function cleans the data by dropping columns and handling missing values.
    
    returns: cleaned train and test data
    """
    
    # Drop rows and columns with NaN scores above a certain threshold
    x , final_y , x_toPred = alternative_nan_handeling(x_train,y_train,x_test,nan_thr)
    
    #we now only influence the columns of x and x_toPred
    
    cumX=np.concatenate((x,x_toPred),axis=0)
    
    # drop columns with std < std_thr
    std_devs = np.nanstd(cumX, axis=0)  # std, ignoring NaNs
    low_std_mask = std_devs < std_thr  # mask of columns with std < 0.1
    columns_constant = np.unique(np.where(low_std_mask)[0])
    print(f"Number of columns with std < {std_thr}:", columns_constant.shape[0])
    #we delete all the columns with std below the threshold
    x_wstd=np.delete(cumX,columns_constant,axis=1)

    # drop columns with correlation score above correlation_thr
    columns_correlated = correlation(x_wstd, correlation_thr)
    
    print(f"Number of columns with correl_coef > {correlation_thr}:", len(columns_correlated))
    
    x_wstd_wcor = np.delete(x_wstd, columns_correlated, axis=1)

    # Handle missing values
    clean_X= handle_nan(x_wstd_wcor)

    # Standardize the data
    stand_X=standardize(clean_X)
    
    
    
    # Check for correlation post processing
    columns_correlated_f = correlation(stand_X,correlation_thr)
    print(f"Number of columns with corr_coef> {correlation_thr} after cleaning:",len(columns_correlated_f))
    
    final_X = np.delete(stand_X, columns_correlated_f, axis=1)
    
    #return the cleaned data by deconcatenating the x and x_toPred
    fi_x=final_X[:x.shape[0],:]
    fi_x_toPred=final_X[x.shape[0]:,:]
    
    assert np.isnan(final_X).sum()==0, "The data should not contain NaN values"
    assert fi_x.shape[1]==fi_x_toPred.shape[1], "The number of columns in the cleaned data should be the same for the data to predict"
    assert fi_x.shape[0]==final_y.shape[0], "The number of rows in the cleaned data should be the same as the output data"
    
    
    print("The data has been cleaned and standardized")
    print("The cleaned x-data has the following shape: ",fi_x.shape)
    print("The cleaned y-data has the following shape: ",final_y.shape)
    print("The cleaned x-data-to-predict has the following shape: ",fi_x_toPred.shape)
    
    
    return fi_x,fi_x_toPred,final_y

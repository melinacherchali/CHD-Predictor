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


def standardize(data):
    """
    This function standardizes the data.
    """
    assert np.isnan(data).sum() == 0, 'Data contains NaN values'
    # Standardize the data
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    
    return standardized_data


def normalize(data):
    """
    This function normalizes the data.
    """
    assert np.isnan(data).sum() == 0, 'Data contains NaN values'
    # Normalize the data
    normalized_data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    
    return normalized_data


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


def replace_nan(data):
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

    
def nan_score_filter(data, treshold):
    """
    This function returns once the columns/rows having their median above a treshold .
    """
    col_med=np.median(data,axis=0)
    row_med=np.median(data,axis=1)
    
    print("Max median NaN score rows : ", np.max(row_med))
    print("Max median NaN score columns : ", np.max(col_med))
    
    rows_to_drop=np.where(row_med>treshold)
    cols_to_drop=np.where(col_med>treshold)
    
    return rows_to_drop[0],cols_to_drop[0]


def drop_nan(x_train, y_train, x_test=None, nan_thr=.5):
    """
    This function drops the rows and columns with NaN values above a certain threshold.
    The criterion is computed, for each item of the array by taking the square root of the product of the NaN ratio of the row and column.
    Criterion: NaN_score(i,j) = sqrt(NaN_ratio_row(i) * NaN_ratio_col(j))
    All the rows and columns having a single computed NaN score above the threshold are dropped.
    
    arguments:
    data_x: the input data
    data_y: the output data
    data_x_to_predict: the input data to predict
    nan_trd: the threshold for the NaN ratio
    
    returns: x_data with reduced rows and columns, y_data with reduced rows
    """
    
    # check if the number of columns in the test data is the same as the train data
    assert (x_test is None) or (x_test.shape[1] == x_train.shape[1])
    
    if x_test is None:
        x_test=np.array([])
        
        
    cum_x = np.concatenate((x_train, x_test),axis=0) 
    
    # compute the NaN ratio for each row and column
    nan_col_ratio = np.isnan(cum_x).sum(axis=0) / x_train.shape[0]
    nan_row_ratio = np.isnan(cum_x).sum(axis=1) / x_train.shape[1]
    
    # create a meshgrid of the two arrays to compute the NaN score
    x, y = np.meshgrid(nan_col_ratio, nan_row_ratio) 
    
    # compute the NaN score
    nan_score = np.sqrt(x * y) # the NaN score for each row and column
    
    # identify rows and columns to drop based on NaN score threshold
    rows_to_drop, cols_to_drop = nan_score_filter(nan_score, nan_thr) 
    
    # the rows to drop should be only in the train data avoiding (not test data)
    rows_to_drop = rows_to_drop[rows_to_drop < x_train.shape[0]]
    
    # apply the drops on train data
    x_train_clean = np.delete(x_train,rows_to_drop, axis=0)
    x_train_clean = np.delete(x_train_clean,cols_to_drop, axis=1)
    y_train_clean = np.delete(y_train,rows_to_drop, axis=0)
    
    print(f"Number of rows dropped because of a NaN score > {nan_thr}: ", rows_to_drop.shape[0])
    print(f"Number of columns dropped because of a NaN score > {nan_thr}: ", cols_to_drop.shape[0])
    
    if x_test.size != 0:
        # reduce the test data only on the columns
        x_test_clean = np.delete(x_test, cols_to_drop, axis=1)
        return x_train_clean , y_train_clean , x_test_clean
    
    return x_train_clean , y_train_clean


def clean_data(x_train_, y_train_, x_test_, correlation_thr = 0.95, nan_thr = 0.5, std_thr = 0.1):
    """
    This function cleans the data by dropping columns and handling missing values.
    
    returns: cleaned train and test data
    """
    
    # Drop rows and columns with NaN scores above a certain threshold
    x_train , y_train , x_test = drop_nan(x_train_, y_train_, x_test_, nan_thr)
        
    X = np.concatenate((x_train, x_test), axis=0)
    
    # Drop columns with std < std_thr
    std_devs = np.nanstd(X, axis=0)                         # std, ignoring NaNs
    low_std_mask = std_devs < std_thr                       # mask of columns with std < 0.1
    columns_constant = np.unique(np.where(low_std_mask)[0]) # columns with std < 0.1, we keep only unique values
    print(f"Number of columns with std < {std_thr}:", columns_constant.shape[0])
    x_wstd = np.delete(X, columns_constant, axis=1)    

    # Drop columns with correlation score above correlation_thr
    columns_correlated = correlation(x_wstd, correlation_thr)
    print(f"Number of columns with correl_coef > {correlation_thr}:", len(columns_correlated))
    x_wstd_wcor = np.delete(x_wstd, columns_correlated, axis=1)

    # Handle missing values
    clean_X = replace_nan(x_wstd_wcor)

    # Clip outliers at the specified percentiles
    for col in range(clean_X.shape[1]):
        lower_clip = np.percentile(clean_X[:, col], 5)
        upper_clip = np.percentile(clean_X[:, col], 95)
        clean_X[:, col] = np.clip(clean_X[:, col], lower_clip, upper_clip)
    print(f"Data clipped between {5}th and {95}th percentiles")
    
    # drop columns with std < 0.1
    std_devs = np.nanstd(clean_X, axis=0)                   # std, ignoring NaNs
    low_std_mask = std_devs < std_thr                       # mask of columns with std < 0.1
    columns_constant = np.unique(np.where(low_std_mask)[0]) # columns with std < 0.1, we keep only unique values
    print(f"Number of columns with std < {std_thr} after cleaning:", columns_constant.shape[0])
    clean_X = np.delete(clean_X, columns_constant, axis=1)

    # Standardize the data
    stand_X = standardize(clean_X)
    # norm_X = normalize(clean_X)
    
    # Check for correlation post processing
    columns_correlated_f = correlation(stand_X, correlation_thr)
    print(f"Number of columns with corr_coef> {correlation_thr} after cleaning:", len(columns_correlated_f))
    final_X = np.delete(stand_X, columns_correlated_f, axis=1)
    
    
    # Return the cleaned data by deconcatenating the x_train and x_test
    cleaned_x_train = final_X[:x_train.shape[0],:]
    cleaned_x_test = final_X[x_train.shape[0]:,:]
    
    assert np.isnan(final_X).sum() == 0                          # The data should not contain NaN values
    assert cleaned_x_train.shape[1] == cleaned_x_test.shape[1]   # The number of columns in the train data should be the same for the test data 
    assert cleaned_x_train.shape[0] == y_train.shape[0]          # The number of rows in the cleaned data should be the same as the output data
    
    
    print("The data has been cleaned and standardized")
    print("The cleaned x-data has the following shape: ", cleaned_x_train.shape)
    print("The cleaned y-data has the following shape: ", y_train.shape)
    print("The cleaned x-data-to-predict has the following shape: ", cleaned_x_test.shape)
    
    
    return cleaned_x_train, cleaned_x_test, y_train




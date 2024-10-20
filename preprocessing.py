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
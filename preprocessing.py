import numpy as np 
import helpers as hlp


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

def columns_to_drop(data):
    """
    This function returns the columns to drop in the data, namely:
    - columns with high number of missing values
    - columns with low std 
    - correlated columns
    """
    columns_to_drop = []
    
    # drop columns with more than 90% missing values
    thr = 0.9
    nan_col = np.isnan(data).sum(axis=0)
    columns_missing_values = np.where(nan_col > (thr * data.shape[0]))[0].tolist()
    columns_to_drop.extend(columns_missing_values)
    print('Number of columns with more than 90% NaN:', len(columns_missing_values))

    # drop columns with std < 0.1
    std_devs = np.nanstd(data, axis=0)  # std, ignoring NaNs
    low_std_mask = std_devs < 0.1 # mask of columns with std < 0.1
    columns_constant = np.where(low_std_mask)[0].tolist()
    columns_to_drop.extend(columns_constant)
    print('Number of columns with std < 0.1:', len(columns_constant))
    
    # drop perfectly correlated columns
    thr = 0.95
    columns_correlated = correlation(data, thr)
    columns_to_drop.extend(columns_correlated)
    print('Number of perfectly correlated columns:', len(columns_correlated))

    return list(set(columns_to_drop)) # removes duplicates


def clean_data(data):
    """
    This function cleans the data by dropping columns and handling missing values.
    """
    to_drop = columns_to_drop(data)
    print('Columns to drop:', to_drop)
    clean_data = np.delete(data, to_drop, axis=1)
    
    # replace missing values with the mean of the column
    means = np.nanmean(clean_data, axis=0)
    nan_ids = np.where(np.isnan(clean_data)) # NaN indices

    # Fill NaN values with the corresponding column means
    clean_data[nan_ids] = np.take(means, nan_ids[1])
    
    # Standardize the data
    standardized_data = (clean_data - np.mean(clean_data, axis=0)) / np.std(clean_data, axis=0)
    
    # Check for correlation post processing 
    columns_correlated = correlation(standardized_data)
    print('Number of perfectly correlated columns after cleaning:', len(columns_correlated))
    final_data = np.delete(standardized_data, columns_correlated, axis=1)
    
    return final_data


def standardize(data):
    """
    This function standardizes the data.
    """
    # Standardize the data
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    return standardized_data


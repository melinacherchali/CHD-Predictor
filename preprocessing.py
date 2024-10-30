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
    All the rows and columns having a median NaN score above the threshold are dropped.
    
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


def clean_data_final(x_train_, y_train_, x_test_, correlation_thr = 0.95, nan_thr = 0.5, std_thr = 0.1):
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
    
    # drop columns with std < std_thr after nan replacement
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



def index_remplace_9 (x_train_features):
    _CHISPNC = x_train_features.index('_CHISPNC')
    _RFHYPE5 = x_train_features.index('_RFHYPE5')
    _HCVU651 = x_train_features.index('_HCVU651')
    _CHOLCHK = x_train_features.index('_CHOLCHK')
    _RFCHOL = x_train_features.index('_RFCHOL')
    _LTASTH1 = x_train_features.index('_LTASTH1')
    _CASTHM1 = x_train_features.index('_CASTHM1')
    _ASTHMS1 = x_train_features.index('_ASTHMS1')
    _RACE = x_train_features.index('_RACE')
    _RACEG21 = x_train_features.index('_RACEG21')
    _RACEGR3 = x_train_features.index('_RACEGR3')
    WTKG3 = x_train_features.index('WTKG3')
    _RFBMI5 = x_train_features.index('_RFBMI5')
    _CHLDCNT = x_train_features.index('_CHLDCNT')
    _EDUCAG = x_train_features.index('_EDUCAG')

    _INCOMG = x_train_features.index('_INCOMG')
    _SMOKER3 = x_train_features.index('_SMOKER3')
    _RFSMOK3 = x_train_features.index('_RFSMOK3')
    _RFBING5 = x_train_features.index('_RFBING5')
    _RFDRHV5 = x_train_features.index('_RFDRHV5')
    _TOTINDA = x_train_features.index('_TOTINDA')
    _PAMISS1_ = x_train_features.index('PAMISS1_')
    _PACAT1 = x_train_features.index('_PACAT1')
    _PAINDX1 = x_train_features.index('_PAINDX1')
    _PA150R2 = x_train_features.index('_PA150R2')
    _PA300R2 = x_train_features.index('_PA300R2')
    _PA30021 = x_train_features.index('_PA30021')
    _PASTRNG = x_train_features.index('_PASTRNG')
    _PAREC1 = x_train_features.index('_PAREC1')
    _PASTAE1 = x_train_features.index('_PASTAE1')
    _RFSEAT2 = x_train_features.index('_RFSEAT2')
    _RFSEAT3 = x_train_features.index('_RFSEAT3')
    _FLSHOT6 = x_train_features.index('_FLSHOT6')
    _PNEUMO2 = x_train_features.index('_PNEUMO2')
    _AIDTST3 = x_train_features.index('_AIDTST3')

    list_index_remplace_9 = [
        _CHISPNC, _RFHYPE5, _HCVU651, _CHOLCHK, _RFCHOL, _LTASTH1, _CASTHM1, _ASTHMS1, _RACE, _RACEG21, _RACEGR3, WTKG3, 
        _RFBMI5, _CHLDCNT, _EDUCAG, _INCOMG, _SMOKER3, _RFSMOK3, _RFBING5, _RFDRHV5, _TOTINDA, _PAMISS1_, _PACAT1, _PAINDX1, 
        _PA150R2, _PA300R2, _PA30021, _PASTRNG, _PAREC1, _PASTAE1, _RFSEAT2, _RFSEAT3, _FLSHOT6, _PNEUMO2, _AIDTST3
    ]
    return list_index_remplace_9

def index_remplace_7(x_train_features):
    LANDLINE = x_train_features.index('LANDLINE')
    HHADULT = x_train_features.index('HHADULT')
    GENHLTH = x_train_features.index('GENHLTH')
    HLTHPLN1 = x_train_features.index('HLTHPLN1')
    PERSDOC2 = x_train_features.index('PERSDOC2')
    MEDCOST = x_train_features.index('MEDCOST')
    CHECKUP1 = x_train_features.index('CHECKUP1')
    BPHIGH4 = x_train_features.index('BPHIGH4')
    BPMEDS = x_train_features.index('BPMEDS')
    BLOODCHO = x_train_features.index('BLOODCHO')
    CHOLCHK = x_train_features.index('CHOLCHK')
    TOLDHI2 = x_train_features.index('TOLDHI2')
    CVDSTRK3 = x_train_features.index('CVDSTRK3')
    ASTHMA3 = x_train_features.index('ASTHMA3')
    ASTHNOW = x_train_features.index('ASTHNOW')
    CHCSCNCR = x_train_features.index('CHCSCNCR')
    CHCOCNCR = x_train_features.index('CHCOCNCR')
    CHCCOPD1 = x_train_features.index('CHCCOPD1')
    HAVARTH3 = x_train_features.index('HAVARTH3')
    ADDEPEV2 = x_train_features.index('ADDEPEV2')
    CHCKIDNY = x_train_features.index('CHCKIDNY')
    DIABETE3 = x_train_features.index('DIABETE3')
    RENTHOM1 = x_train_features.index('RENTHOM1')
    NUMHHOL2 = x_train_features.index('NUMHHOL2')
    NUMPHON2 = x_train_features.index('NUMPHON2')
    CPDEMO1 = x_train_features.index('CPDEMO1')
    VETERAN3 = x_train_features.index('VETERAN3')
    INTERNET = x_train_features.index('INTERNET')
    PREGNANT = x_train_features.index('PREGNANT') # 7 don't know
    QLACTLM2 = x_train_features.index('QLACTLM2') # 7 don't know
    USEEQUIP = x_train_features.index('USEEQUIP') # 7 don't know
    BLIND = x_train_features.index('BLIND') # 7 don't know
    DECIDE = x_train_features.index('DECIDE') # 7 don't know

    DIFFWALK = x_train_features.index('DIFFWALK') # 7 don't know
    DIFFDRES = x_train_features.index('DIFFDRES') # 7 don't know

    DIFFALON = x_train_features.index('DIFFALON') # 7 don't know
    SMOKE100 = x_train_features.index('SMOKE100') # 7 don't know
    SMOKDAY2 = x_train_features.index('SMOKDAY2') # 7 don't know
    STOPSMK2 = x_train_features.index('STOPSMK2') # 7 don't know
    USENOW3 = x_train_features.index('USENOW3') # 7 don't know
    EXERANY2 = x_train_features.index('EXERANY2') # 7 don't know
    LMTJOIN3 = x_train_features.index('LMTJOIN3') # 7 don't know
    ARTHDIS2 = x_train_features.index('ARTHDIS2') # 7 don't know
    ARTHSOCL = x_train_features.index('ARTHSOCL') # 7 don't know
    SEATBELT = x_train_features.index('SEATBELT') # 7 don't know
    FLUSHOT6 = x_train_features.index('FLUSHOT6') # 7 don't know
    PNEUVAC3 = x_train_features.index('PNEUVAC3') # 7 don't know
    HIVTST6 = x_train_features.index('HIVTST6') # 7 don't know
    PDIABTST = x_train_features.index('PDIABTST') # 7 don't know
    PREDIAB1 = x_train_features.index('PREDIAB1') # 7 don't know
    FEETCHK2 = x_train_features.index('FEETCHK2') # 7 don't know


    EYEEXAM = x_train_features.index('EYEEXAM') # 7 don't know
    DIABEYE = x_train_features.index('DIABEYE') # 7 don't know
    DIABEDU = x_train_features.index('DIABEDU') # 7 don't know
    CAREGIV1 = x_train_features.index('CAREGIV1') # 7 don't know
    CRGVLNG1 =  x_train_features.index('CRGVLNG1') # 7 don't know
    CRGVHRS1 = x_train_features.index('CRGVHRS1') # 7 don't know

    CRGVPERS = x_train_features.index('CRGVPERS') # 7 don't know
    CRGVHOUS = x_train_features.index('CRGVHOUS') # 7 don't know
    CRGVMST2 = x_train_features.index('CRGVMST2') # 7 don't know
    CRGVEXPT = x_train_features.index('CRGVEXPT') # 7 don't know

    VIDFCLT2 = x_train_features.index('VIDFCLT2') # 7 don't know
    VIREDIF3 = x_train_features.index('VIREDIF3') # 7 don't know
    VIPRFVS2 = x_train_features.index('VIPRFVS2') # 7 don't know
    VIEYEXM2 = x_train_features.index('VIEYEXM2') # 7 don't know
    VIINSUR2 = x_train_features.index('VIINSUR2') # 7 don't know
    VICTRCT4 = x_train_features.index('VICTRCT4') # 7 don't know
    VIGLUMA2 = x_train_features.index('VIGLUMA2') # 7 don't know
    VIMACDG2 = x_train_features.index('VIMACDG2') # 7 don't know
    CIMEMLOS = x_train_features.index('CIMEMLOS') # 7 don't know
    CDHOUSE = x_train_features.index('CDHOUSE') # 7 don't know
    CDASSIST = x_train_features.index('CDASSIST') # 7 don't know
    CDHELP = x_train_features.index('CDHELP') # 7 don't know
    CDSOCIAL = x_train_features.index('CDSOCIAL') # 7 don't know
    CDDISCUS = x_train_features.index('CDDISCUS') # 7 don't know
    WTCHSALT = x_train_features.index('WTCHSALT') # 7 don't know
    ASYMPTOM = x_train_features.index('ASYMPTOM') # 7 don't know
    ASNOSLEP = x_train_features.index('ASNOSLEP') # 7 don't know
    ASTHMED3 = x_train_features.index('ASTHMED3') # 7 don't know
    ASINHALR = x_train_features.index('ASINHALR') # 7 don't know
    HAREHAB1 = x_train_features.index('HAREHAB1') # 7 don't know
    STREHAB1 = x_train_features.index('STREHAB1') # 7 don't know
    CVDASPRN = x_train_features.index('CVDASPRN') # 7 don't know
    ASPUNSAF = x_train_features.index('ASPUNSAF') # 7 don't know
    RLIVPAIN = x_train_features.index('RLIVPAIN') # 7 don't know
    RDUCHART = x_train_features.index('RDUCHART') # 7 don't know
    RDUCSTRK = x_train_features.index('RDUCSTRK') # 7 don't know
    ARTTODAY = x_train_features.index('ARTTODAY') # 7 don't know
    ARTHWGT = x_train_features.index('ARTHWGT') # 7 don't know
    ARTHEXER = x_train_features.index('ARTHEXER') # 7 don't know
    ARTHEDU = x_train_features.index('ARTHEDU') # 7 don't know
    TETANUS = x_train_features.index('TETANUS') # 7 don't know
    HPVADVC2 = x_train_features.index('HPVADVC2') # 7 don't know
    SHINGLE2 = x_train_features.index('SHINGLE2') # 7 don't know
    HADMAM = x_train_features.index('HADMAM') # 7 don't know
    HOWLONG = x_train_features.index('HOWLONG') # 7 don't know
    HADPAP2 = x_train_features.index('HADPAP2') # 7 don't know
    LASTPAP2 = x_train_features.index('LASTPAP2') # 7 don't know
    HPVTEST = x_train_features.index('HPVTEST') # 7 don't know
    HPLSTTST = x_train_features.index('HPLSTTST') # 7 don't know
    HADHYST2 = x_train_features.index('HADHYST2') # 7 don't know
    PROFEXAM = x_train_features.index('PROFEXAM') # 7 don't know
    LENGEXAM = x_train_features.index('LENGEXAM') # 7 don't know
    BLDSTOOL = x_train_features.index('BLDSTOOL') # 7 don't know
    LSTBLDS3  = x_train_features.index('LSTBLDS3') # 7 don't know
    HADSIGM3 = x_train_features.index('HADSIGM3') # 7 don't know
    HADSGCO1 = x_train_features.index('HADSGCO1') # 7 don't know
    LASTSIG3 = x_train_features.index('LASTSIG3') # 7 don't know
    PCPSAAD2 = x_train_features.index('PCPSAAD2') # 7 don't know
    PCPSADI1 = x_train_features.index('PCPSADI1') # 7 don't know
    PCPSARE1 = x_train_features.index('PCPSARE1') # 7 don't know
    PSATEST1 = x_train_features.index('PSATEST1') # 7 don't know

    DRADVISE = x_train_features.index('DRADVISE') # 7 don't know
    ASATTACK = x_train_features.index('ASATTACK') # 7 don't know

    PSATIME = x_train_features.index('PSATIME') # 7 don't know
    PCPSARS1 = x_train_features.index('PCPSARS1') # 7 don't know
    PCDMDECN = x_train_features.index('PCDMDECN') # 7 don't know
    SCNTMNY1 = x_train_features.index('SCNTMNY1') # 7 don't know
    SCNTMEL1 = x_train_features.index('SCNTMEL1') # 7 don't know
    SCNTPAID = x_train_features.index('SCNTPAID') # 7 don't know
    SCNTLPAD = x_train_features.index('SCNTLPAD') # 7 don't know


    SXORIENT = x_train_features.index('SXORIENT') # 7 don't know
    TRNSGNDR = x_train_features.index('TRNSGNDR') # 7 don't know
    RCSRLTN2 = x_train_features.index('RCSRLTN2') # 7 don't know
    CASTHDX2 = x_train_features.index('CASTHDX2') # 7 don't know
    CASTHNO2  = x_train_features.index('CASTHNO2') # 7 don't know
    EMTSUPRT = x_train_features.index('EMTSUPRT') # 7 don't know
    LSATISFY = x_train_features.index('LSATISFY') # 7 don't know

    MISTMNT = x_train_features.index('MISTMNT') # 7 don't know
    ADANXEV = x_train_features.index('ADANXEV') # 7 don't know

    list_index_remplace_7 = [
        LANDLINE, HHADULT, GENHLTH, HLTHPLN1, PERSDOC2, MEDCOST, CHECKUP1, BPHIGH4, BPMEDS, BLOODCHO, 
        CHOLCHK, TOLDHI2, CVDSTRK3, ASTHMA3, ASTHNOW, CHCSCNCR, CHCOCNCR, CHCCOPD1, 
        HAVARTH3, ADDEPEV2, CHCKIDNY, DIABETE3, RENTHOM1, NUMHHOL2, NUMPHON2, CPDEMO1, VETERAN3, INTERNET, 
        PREGNANT, QLACTLM2, USEEQUIP, BLIND, DECIDE, DIFFWALK, DIFFDRES, DIFFALON, SMOKE100, SMOKDAY2, 
        STOPSMK2, USENOW3, EXERANY2, LMTJOIN3, ARTHDIS2, ARTHSOCL, SEATBELT, FLUSHOT6, PNEUVAC3, HIVTST6, 
        PDIABTST, PREDIAB1, FEETCHK2, EYEEXAM, DIABEYE, DIABEDU, CAREGIV1, CRGVLNG1, CRGVHRS1, CRGVPERS, 
        CRGVHOUS, CRGVMST2, CRGVEXPT, VIDFCLT2, VIREDIF3, VIPRFVS2, VIEYEXM2, VIINSUR2, VICTRCT4, VIGLUMA2, 
        VIMACDG2, CIMEMLOS, CDHOUSE, CDASSIST, CDHELP, CDSOCIAL, CDDISCUS, WTCHSALT, ASYMPTOM, ASNOSLEP, 
        ASTHMED3, ASINHALR, HAREHAB1, STREHAB1, CVDASPRN, ASPUNSAF, RLIVPAIN, RDUCHART, RDUCSTRK, ARTTODAY, 
        ARTHWGT, ARTHEXER, ARTHEDU, TETANUS, HPVADVC2, SHINGLE2, HADMAM, HOWLONG, HADPAP2, LASTPAP2, 
        HPVTEST, HPLSTTST, HADHYST2, PROFEXAM, LENGEXAM, BLDSTOOL, LSTBLDS3, HADSIGM3, HADSGCO1, LASTSIG3, 
        PCPSAAD2, PCPSADI1, PCPSARE1, PSATEST1, DRADVISE, ASATTACK, PSATIME, PCPSARS1, PCDMDECN, SCNTMNY1, 
        SCNTMEL1, SCNTPAID, SCNTLPAD, SXORIENT, TRNSGNDR, RCSRLTN2, CASTHDX2, CASTHNO2, EMTSUPRT, LSATISFY, 
        MISTMNT, ADANXEV
    ]
    return list_index_remplace_7

def index_remplace_77(x_train_features):
    # 77 is don't know
    PHYSHLTH = x_train_features.index('PHYSHLTH') #77 don't know 
    MENTHLTH = x_train_features.index('MENTHLTH') #77 don't know
    POORHLTH = x_train_features.index('POORHLTH') #77 don't know
    INCOME2 = x_train_features.index('INCOME2') # 77 don't know 
    LASTSMK2 = x_train_features.index('LASTSMK2') # 77 don't know
    AVEDRNK2 = x_train_features.index('AVEDRNK2') # 77 don't know
    DRNK3GE5 = x_train_features.index('DRNK3GE5') # 77 don't know
    MAXDRNKS = x_train_features.index('MAXDRNKS') # 77 don't know
    EXRACT11 = x_train_features.index('EXRACT11') # 77 don't know
    EXRACT21 = x_train_features.index('EXRACT21') # 77 don't know
    JOINPAIN = x_train_features.index('JOINPAIN') # 77 don't know
    IMFVPLAC = x_train_features.index('IMFVPLAC') # 77 don't know
    DOCTDIAB = x_train_features.index('DOCTDIAB') # 77 don't know
    CHKHEMO3 = x_train_features.index('CHKHEMO3') # 77 don't know
    FEETCHK = x_train_features.index('FEETCHK') # 77 don't know
    CRGVPRB1 = x_train_features.index('CRGVPRB1') # 77 don't know
    VINOCRE2 = x_train_features.index('VINOCRE2') # 77 don't know
    HPVADSHT = x_train_features.index('HPVADSHT') # 77 don't know
    ADPLEASR = x_train_features.index('ADPLEASR') # 77 don't know
    ADDOWN = x_train_features.index('ADDOWN') # 77 don't know
    ADSLEEP = x_train_features.index('ADSLEEP') # 77 don't know
    ADENERGY = x_train_features.index('ADENERGY') # 77 don't know
    ADEAT1 = x_train_features.index('ADEAT1') # 77 don't know
    ADFAIL = x_train_features.index('ADFAIL') # 77 don't know
    ADTHINK = x_train_features.index('ADTHINK') # 77 don't know
    ADMOVE = x_train_features.index('ADMOVE') # 77 don't know

    list_index_remplace_77 = [PHYSHLTH, MENTHLTH, POORHLTH, INCOME2, LASTSMK2, AVEDRNK2, DRNK3GE5, MAXDRNKS, EXRACT11, EXRACT21,
                        JOINPAIN, IMFVPLAC, DOCTDIAB, CHKHEMO3, FEETCHK, CRGVPRB1, VINOCRE2, HPVADSHT, ADPLEASR,
                        ADDOWN, ADSLEEP, ADENERGY, ADEAT1, ADFAIL, ADTHINK, ADMOVE]
    return list_index_remplace_77

def remplace(list_index,x_train,x_test):
    for (index) in (list_index):
        x_train[:, index-1] = np.where(x_train[:, index-1] == 9, np.nan, x_train[:, index-1]) 
        x_test[:, index-1] = np.where(x_test[:, index-1] == 9, np.nan, x_test[:, index-1]) 
    return x_train, x_test



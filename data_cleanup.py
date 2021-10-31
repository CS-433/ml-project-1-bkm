# -*- coding: utf-8 -*-
import numpy as np
import logging as log

def undefined_to_nans(tx, nan_value = -999):
    """Replaces values in a provided numpy matrix with NaN

        Parameters:
        tx (numpy.ndarray): Matrix in which values should be replaced
        nan_value (int, default: -999): Number that represents undefined values in tx

        Returns:
        numpy.ndarray: Matrix in which occurences of nan_value are replaced with numpy.nan

    """
    tx[tx == nan_value] = np.nan
    return tx

def nans_to_means(tx):
    """Replaces NaN values with column mean

        Parameters:
        tx (numpy.ndarray): Matrix in which values should be replaced

        Returns:
        numpy.ndarray: Matrix in which occurences of numpy.nan are replaced with column mean

    """
    col_mean = np.nanmean(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_mean, inds[1])
    return tx

def nans_to_medians(tx):
    """Replaces NaN values with column median

        Parameters:
        tx (numpy.ndarray): Matrix in which values should be replaced

        Returns:
        numpy.ndarray: Matrix in which occurences of numpy.nan are replaced with column median

    """
    col_median = np.nanmedian(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_median, inds[1])
    return tx

def remove_zscore_outliers(tx):
    """Replace all values with z-score outside of bounds (-3,3) with numpy.nan
        
        Parameters:
        tx (numpy.ndarray): Matrix in which values should be replaced

        Returns:
        numpy.ndarray: Matrix in which values with z-score outside of bounds (-3,3) are replaced with numpy.nan

    """
    col_mean = np.nanmean(tx, axis=0)
    std_mean = np.nanstd(tx, axis=0)
    z_score = np.zeros(tx.shape)
    for col in range(29):
        z_score[:,col] = (tx[:,col] - col_mean[col])/std_mean[col]
    tx[z_score >= 3] = np.nan
    tx[z_score <=-3] = np.nan
    return tx

def remove_nans(tx, y):
    """Drop all rows where any value is a numpy.nan

        Parameters:
        tx (numpy.ndarray): Matrix that may contain numpy.nan values
        y (numpy.ndarray): Label array related to tx

        Returns:
        numpy.ndarray: Matrix tx in which rows with numpy.nan values are dropped
        numpy.ndarray: Array y with same rows dropped as in tx

    """
    mask = ~np.isnan(tx).any(axis=1)
    return tx[mask], y[mask]

def nan_features_to_zero(tx, threshold=0.8):
    """Set column values to 0 if proportion of numpy.nan values in column exceeds threshold

        Parameters:
        tx (numpy.ndarray): Matrix in which values should be replaced
        threshold (float, default: 0.8): Threshold of proportion of count of numpy.nan values 
            over count of all values in the column

        Returns:
        numpy.ndarray: Matrix in which columns with proportion of numpy.nan values 
            over count of all values exceeding threshold have all values replaced with 0

    """
    nan_features = list()
    for i in range(tx.shape[1]):
        nans_ratio = np.count_nonzero(np.isnan(tx[:,i]))/tx.shape[0]
      
        if nans_ratio>threshold: 
            nan_features.append(i)
                    
    tx[:,nan_features]=0
        
    return tx

def add_bias(tx):
    """Add bias term to a matrix

        Parameters:
        tx (numpy.ndarray): Matrix to be modified

        Returns:
        numpy.ndarray: Matrix tx with additional bias column added

    """
    return np.column_stack((tx,np.ones(len(tx))))

def split_on_jets(tx, y=[]):
    """Split train data based on number of jets in an observation

        Parameters:
        tx (numpy.ndarray): Feature matrix to be split
        y (numpy.ndarray, default: []): Label array to be split

        Returns:
        list(numpy.ndarray): List of 4 subsets of tx matrix divided on 
            number of jets from observation (jet count feature removed)
        list(numpy.ndarray): List of 4 subsets of y array divided on
            number of jets from observation

    """
    jets = tx[:,22]
    tx = np.delete(tx, 22, 1) 
    tx_0 = tx[jets == 0]
    tx_1 = tx[jets == 1]
    tx_2 = tx[jets == 2]
    tx_3 = tx[jets == 3]
    y_0 = y[jets == 0]
    y_1 = y[jets == 1]
    y_2 = y[jets == 2]
    y_3 = y[jets == 3]
    return [tx_0, tx_1, tx_2, tx_3], [y_0, y_1, y_2, y_3] 

def split_on_jets_test(tx, idx):
    """Split test data based on number of jets in an observation

        Parameters:
        tx (numpy.matrix): Feature matrix to be split
        idx (numpy.matrix): Identifier array to be split

        Returns:
        list(numpy.ndarray): List of 4 subsets of tx matrix divided on 
            number of jets from observation (jet count feature removed)
        list(numpy.ndarray): List of 4 subsets of idx array divided on
            number of jets from observation

    """
    jets = tx[:,22]
    tx = np.delete(tx, 22, 1) 
    tx_0 = tx[jets == 0]
    tx_1 = tx[jets == 1]
    tx_2 = tx[jets == 2]
    tx_3 = tx[jets == 3]
    idx_0 = idx[jets == 0]
    idx_1 = idx[jets == 1]
    idx_2 = idx[jets == 2]
    idx_3 = idx[jets == 3]
    return [tx_0, tx_1, tx_2, tx_3], [idx_0, idx_1, idx_2, idx_3] 
   
def standardize(tx):
    """Standarize values in a matrix column-wise

        Parameters:
        tx (numpy.ndarray): Matrix to be standarized

        Returns:
        numpy.ndarray: Column-wise standarized matrix tx

    """
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)

def preprocess_train(tx, y):
    """Preprocess train data

        Parameters:
        tx (numpy.ndarray): Feature matrix to be preprocessed
        y (numpy.ndarray): Label array to be preprocessed

        Returns:
        list(numpy.ndarray): List of 4 preprocessed subsets of tx
        list(numpy.ndarray): List of 4 preprocessed subsets of y    

    """
    xs, ys = split_on_jets(tx, y)
    for i in range(4):
        xs[i] = undefined_to_nans(xs[i], nan_value = -999)
        xs[i] = nan_features_to_zero(xs[i], threshold=0.8)
        xs[i] = nans_to_medians(xs[i])
        xs[i] = remove_zscore_outliers(xs[i])
        xs[i] = nans_to_medians(xs[i])
        xs[i] = add_bias(xs[i])
        ys[i][ys[i] == -1] = 0
        log.info(f'x_{i} shape: {xs[i].shape}, y_{i} shape: {ys[i].shape}')
    return xs, ys
        
def preprocess_test(tx, idx):
    """Preprocess test data

        Parameters:
        tx (numpy.ndarray): Feature matrix to be preprocessed
        idx (numpy.ndarray): Identifier array to be preprocessed

        Returns:
        list(numpy.ndarray): List of 4 preprocessed subsets of tx
        list(numpy.ndarray): List of 4 preprocessed subsets of idx    

    """
    xs, idx = split_on_jets(tx, idx)
    for i in range(4):
        xs[i] = undefined_to_nans(xs[i], nan_value = -999)
        xs[i] = nan_features_to_zero(xs[i], threshold=0.8)
        xs[i] = nans_to_medians(xs[i])
        xs[i] = remove_zscore_outliers(xs[i])
        xs[i] = nans_to_medians(xs[i])
        xs[i] = add_bias(xs[i])
        log.info(f'x_{i} shape: {xs[i].shape}')
    return xs, idx
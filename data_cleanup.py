# -*- coding: utf-8 -*-
import numpy as np

## Removing outliers
def undefined_to_nans(tx, nan_value = -999):
    tx[tx == nan_value] = np.nan
    return tx

def nans_to_means(tx):
    col_mean = np.nanmean(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_mean, inds[1])
    return tx

def nans_to_medians(tx):
    col_median = np.nanmedian(tx, axis=0)
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(col_median, inds[1])
    return tx

def remove_zscore_outliers(tx):
    col_mean = np.nanmean(tx, axis=0)
    std_mean = np.nanstd(tx, axis=0)
    z_score = np.zeros(tx.shape)
    for col in range(29):
        z_score[:,col] = (tx[:,col] - col_mean[col])/std_mean[col]
    tx[z_score >=3] = np.nan
    tx[z_score <=-3] = np.nan
    return tx

def remove_nans(tx, y):
    mask = ~np.isnan(tx).any(axis=1)
    return tx[mask], y[mask]

def nan_features_to_zero(tx, threshold=0.8):
    nan_features = list()
    for i in range(tx.shape[1]):
        nans_ratio = np.count_nonzero(np.isnan(tx[:,i]))/tx.shape[0]
      
        if nans_ratio>threshold: 
            nan_features.append(i)
                    
    tx[:,nan_features]=0
        
    return tx

def add_bias(tx):
    return np.column_stack((tx,np.ones(len(tx))))


def split_on_jets(tx, y=[]):
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
    return (tx - np.mean(tx, axis=0)) / np.std(tx, axis=0)

def preprocess_train(tx, y):
    xs, ys = split_on_jets(tx, y)
    for i in range(4):
        xs[i] = undefined_to_nans(xs[i], nan_value = -999)
        xs[i] = nan_features_to_zero(xs[i], threshold=0.8)
        xs[i] = nans_to_medians(xs[i])
        xs[i] = remove_zscore_outliers(xs[i])
        xs[i] = nans_to_medians(xs[i])
        xs[i] = add_bias(xs[i])
        ys[i][ys[i] == -1] = 0
        print(f'x_{i} shape: {xs[i].shape}, y_{i} shape: {ys[i].shape}')
    return xs, ys
        

def preprocess_test(tx, idx):
    xs, idx = split_on_jets(tx, idx)
    for i in range(4):
        xs[i] = undefined_to_nans(xs[i], nan_value = -999)
        xs[i] = nan_features_to_zero(xs[i], threshold=0.8)
        xs[i] = nans_to_medians(xs[i])
        xs[i] = remove_zscore_outliers(xs[i])
        xs[i] = nans_to_medians(xs[i])
        xs[i] = add_bias(xs[i])
        print(f'x_{i} shape: {xs[i].shape}')
    return xs, idx
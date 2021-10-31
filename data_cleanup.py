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
    for col in range(30):
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
    return np.column_stack((tX,np.ones(len(tX))))


def split_on_jets(tX, y):
    tx_0 = tX[np.where(tX[:,22] == 0)]
    tx_1 = tX[np.where(tX[:,22] == 1)]
    tx_2 = tX[np.where(tX[:,22] == 2)]
    tx_3 = tX[np.where(tX[:,22] == 3)]
    y_0 = tX[np.where(tX[:,22] == 0)]
    y_1 = tX[np.where(tX[:,22] == 1)]
    y_2 = tX[np.where(tX[:,22] == 2)]
    y_3 = tX[np.where(tX[:,22] == 3)]
    return tx_0, tx_1, tx_2, tx_3, y_0, y_1, y_2, y_3 


def preprocess_train(tX, y):
    pass


def preprocess_test(tX):
    pass
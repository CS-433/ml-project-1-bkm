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


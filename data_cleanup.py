# -*- coding: utf-8 -*-
import numpy as np

## Removing outliers
def undefined_to_nans(tX, nan_value = -999):
    tX[tX == nan_value] = np.nan
    return tX

def nans_to_means(tX):
    col_mean = np.nanmean(tX, axis=0)
    inds = np.where(np.isnan(tX))
    tX[inds] = np.take(col_mean, inds[1])
    return tX

def remove_zscore_outliers(tX):
    col_mean = np.nanmean(tX, axis=0)
    std_mean = np.nanstd(tX, axis=0)
    z_score = np.zeros(tX.shape)
    for col in range(30):
        z_score[:,col] = (tX[:,col] - col_mean[col])/std_mean[col]
    tX[z_score >=3] = np.nan
    tX[z_score <=-3] = np.nan
    return tX

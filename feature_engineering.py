#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[11]:


DATA_TRAIN_PATH='./data/train.csv' 

Y1, X1, ids= load_csv_data(DATA_TRAIN_PATH)
Y1=Y1.reshape((Y1.shape[0],1))


# In[12]:


test=(np.abs(np.corrcoef(X1.T))>0.9)&(np.abs(np.corrcoef(X1.T))<1)
np.where(test)


# In[13]:


toRemoove=[4,5,6,12,26,27,28]


# In[14]:


def build_polynomial_features(x, degree):
    """ builds polynomial features"""
    phi = np.zeros((len(x), degree + 1))
    for i in range(degree+1):
        phi[:, i] = x**i 
    return phi


# In[15]:


def build_log_feature(x):
    return np.log(x)


# In[16]:


def build_exp_feature(x):
    return np.exp(x)


# In[17]:


def feature_normalization(X):
    means=np.mean(X,axis=0).reshape((X.shape[0],1))
    stds=np.std(X,axis=0).reshape((X.shape[0],1))
    
    return (X-means)/stds, means, stds
                                


# In[18]:


def feature_denormalization(X,means,stds):
    return X*stds+means


# In[ ]:





# In[ ]:





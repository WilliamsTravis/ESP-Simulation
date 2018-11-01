# -*- coding: utf-8 -*-
"""

Random walk
Created on Wed Oct 31 11:54:04 2018

@author: User
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
os.chdir("C:/users/user/github/esp_simulation")\


# In[] Helper functions


def plttr(data):
    '''
    works for vectors
    '''
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.show()


def randomWalk(length, randomPerc, fmin, fmax, start, end):
    '''
    https://stackoverflow.com/questions/46954510/random-walk-series-between-
    start-end-values-and-within-minimum-maximum-limits
    '''
    data_np = (np.random.random(length) - randomPerc).cumsum()
    data_np *= (fmax - fmin) / (data_np.max() - data_np.min())
    data_np += np.linspace(start - data_np[0], end - data_np[-1], len(data_np))
    plttr(data_np)
    return data_np


def vT(vector):
    '''Transpose is tricky for 1Xn vectors'''
    vT = np.array(np.matrix(vector).T)
    return vT


# In[] Our sample esp evolutions
files = glob.glob(os.path.join('data', "*"))
files = [f for f in files if "historical" not in f]
dolc_files = [f for f in files if "DOLC2" in f]
dfs = {f[-8:-4]: pd.read_csv(f) for f in dolc_files}  # data frames by year
dfs = {key: dfs[key][75: 295] for key in dfs.keys()}  # days with data
history = pd.read_csv(os.path.join('data', 'DOLC2_historical.csv'))

# Perhaps set start at the average streamflow value?
start = np.mean([pd.unique(dfs[key]['Observed Total']) for key in dfs.keys()])

# Set the end to the desired final streamflow value, q
# providing control over the severity of the situation
end = np.min([pd.unique(dfs[key]['Observed Total']) for key in dfs.keys()])

# Maximum value could be set to a somewhat larger value than the max ever
fmax = 1.2 * max(history['vol'])

# In[] Create a simple random walk to a particular point
data = randomWalk(length=len(dfs['2017']),
                  randomPerc=.5,
                  fmin=0,
                  fmax=fmax,
                  start=start,
                  end=end)

plttr(data)

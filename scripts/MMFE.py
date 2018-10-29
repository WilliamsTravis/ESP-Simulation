# -*- coding: utf-8 -*-
"""
Try to recreate the Martingal Model of Forecast Evolution


Created on Thu Oct 25 11:54:58 2018

@author: Travis Williams
"""
# In[]
import glob
import numpy as np
import os
import pandas as pd
import random
import scipy
os.chdir("c:\\users\\user\\github\\ESP_simulation")

# In[] get sample esps
files = glob.glob("data\\*")
files = [f for f in files if "MPHC2" in f and "historical" not in f]
ref_dfs = [pd.read_csv(f) for f in files]

for i in range(len(ref_dfs)):
    ref_dfs[i]= ref_dfs[i][80:300]  # These only occur from Dec to Jul

# In[]

# H is the lead time of the forecast, H for Horizon
ref_df = ref_dfs[len(ref_dfs)-1]  # latest
H = len(ref_df) - 1

# q is the ultimate streamflow observation
totals = ref_df['Observed Total'].dropna()
q = float(pd.unique(totals))

# t is the time the forecast is made and s is the time it attempts to predict
t = 1
s = H + 1  # The realized time period, s, is one period after the forecast horizon... i think

# The first forecast is tricky to simulate. Let's use average q for now
f1 = np.average([r['Observed Total'].dropna().tolist()[0] for r in ref_dfs])


# In[] Simulating errors
# what is the average error in this set?
qs = [r['Observed Total'].dropna().tolist()[0] for r in ref_dfs]
fs = [r['ESP 50'].dropna().tolist() for r in ref_dfs]
es = [np.array(fs[i]) - qs[i] for i in range(len(fs))]


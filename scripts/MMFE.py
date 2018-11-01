"""
Experimenting with random walks and Martingal Methods of Forecast Evolution

Following these papers (as best I can :):

    Zhao, T., Zhao, J., Yang, D., & Wang, H. (2013). Generalized martingale
        model of the uncertainty evolution of streamflow forecasts. Advances
        in Water Resources, 57, 41–51. ttps://doi.org/10.1016/j.advwatres.
        2013.03.008

    Zhao, T., & Zhao, J. (2014). Forecast-skill-based simulation of streamflow
        forecasts. Advances in Water Resources, 71, 55–64. https://doi.org/10.
        1016/j.advwatres.2014.05.011


Created on Mon Oct 29 12:36:50 2018

@author: User
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
os.chdir("C:/users/user/github/esp_simulation")


# In[] Helper functions
def plttr(data):
    '''works for vectors'''
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.show()


def vT(vector):
    '''Transpose is tricky for 1 X n vectors'''
    vT = np.array(np.matrix(vector).T)
    return vT


# In[] Our sample esp evolutions
files = glob.glob(os.path.join('data', "*"))
files = [f for f in files if "historical" not in f]
dolc_files = [f for f in files if "DOLC2" in f]
dfs = {f[-8:-4]: pd.read_csv(f) for f in dolc_files}  # data frames by year
dfs = {key: dfs[key][75: 295] for key in dfs.keys()}  # days with data
history = pd.read_csv(os.path.join('data', 'DOLC2_historical.csv'))


# In[] Quantifying the current forecast skill
# A sample forecast
df = dfs['2017']

# These are the forecasts that the sample made
fs = np.array(df['ESP 50'])

# To quantify skill we need a measure of observed streamflow variability.
# In our case, the variability of a single year's streamflow would be
# misleading. Variability of ultimate summer q across all years is perhaps
# more appropriate
qs_i = history['vol']

# According to "Forecast-skill-based simulations of streamflow forecasts",
# skill (Coefficient of Prediction (Cp)) is described as this:
Cp = 1 - (np.nanvar(fs) / np.var(qs_i))   # Voila a measure of skill! 0.84

# Inversely, this is the variance assoication with a lower skill score: .75
var_fs = (1 - .75) * np.var(qs_i)

# These forecast errors are assumed to have these four qualities:
# 1: They sum up to zero
# 2: They are Gaussian (An assumption of the model, at least)
# 3: They are temporally independent
# 4: They are stationary, they do not change with the number of total steps

# There are missing forecasts
for i in range(len(fs)):
    if np.isnan(fs[i]):
        fs[i] = np.nanmean(fs[i-3: i+3])  # fill in missed steps

# Looks like this
plttr(fs)

# The target streamflow observation
q = float(pd.unique(df['Observed Total']))

# These are the errors the sample made
errors = np.array([f - q for f in fs])
plttr(errors)  # same shape as the forecast

# These are the steps (changes) that the sample took (improvements in papers)
steps = np.diff(fs)
plttr(steps)

# Now, if steps follow an unbiased gaussian distribution we can 'characterize'
# it with the variance-covariance matrix.
plt.hist(x=steps, bins='auto')  # (It's not really normal)

# This is the variance covariance matrix of the improvements
vcv = (steps*vT(steps))/len(steps)

# We cannot decompose this with the Cholesky Decomposition...negatives :/

# What if we did this:


def simForecast(forecast):
    '''
    Based on the characteristics of a known ensemble streamflow prediction
        timeseries, we would like to simulate a random forecast. Then,
        eventually, specify improvements to certain parameters that determine
        skill.
    '''
    # There are missing forecasts
    for i in range(len(forecast)):
        if np.isnan(forecast[i]):
            forecast[i] = np.nanmean(forecast[i-3: i+3])  # fill in missed steps
            
    # The final forecast is the target streamflow (q)
    q = forecast[-1]
    
    # 1: Variation in forecast steps taken looks like the correct
    # progression of uncertainty
    steps = np.diff(forecast)
    plttr(steps)

    # The progression of errors look like the progression of forecasts
    errors = np.array([f - q for f in fs])
    plttr(errors)  # same shape as the forecast

    # 2: So if we take a moving window of standard deviation of the steps...
    variations = [list(steps[i-3:i]) for i in range(3, len(steps))]
    sds = np.array([np.std(v) for v in variations])
    plttr(sds)

    # 3: And use this to simulate improvements in errors
    simprovements = np.array([np.random.normal(0, sd) for sd in sds])          # here we could reduce each sd by a parameter?
    plttr(simprovements)
 
    # While the errors themselves are a random walk toward the final?

    # 4: We choose a starting number, a random one with a certain range around q
    pct = .5 # so the number would fall with +- 50% of ultimate q (parameter)
    low = 1 - pct
    high = 1 + pct
    start = random.uniform(q * low, q * high)
    
    # 5: We get our initial error
    e1 = start - q

    # 6: We iteratively add modeled improvements to the next error
    errs = [e1]
    for i in range(len(simprovements)):
        err = errs[i] + simprovements[i]
        errs.append(err)
    plttr(errs)

    # 7: Now we add our vector of errors to q
    fsim = q + np.array(errs)

    # Plot
    plttr(forecast)
    plttr(fsim)
    return fsim


fsim = simForecast(fs)










# In[] Stuff for later sorry
# list of cumulative variation in improvements at each step
variations = [list(steps[:i]) for i in range(2, len(steps))]
vrs = np.array([np.var(v) for v in variations])
plttr(vrs)

# or...

# List of cumulative variation in errors at each step
err_variations = [list(errors[:i]) for i in range(2, len(errors))]
ervrs = np.array([np.var(ev) for ev in err_variations])
plttr(ervrs)

# Now if we simulated errors along a normal distributions using these variances
sterrs = np.array([np.std(ev) for ev in err_variations])
plttr(sterrs)
simerrs = [np.random.normal(0, sterr) for sterr in sterrs]
plttr(simerrs)


# ... 



# At this point we create a the vcv matrix, but it's not making sense to me...
# switching to Zhao & Zhao (2014)

# It turns out not to be as simple, let's try it step by step
vrs_t = np.array(np.matrix(vrs).T)
vcv = (vrs * vrs_t)/len(vrs)

# What if I just used the steps?
vcv = (steps * vT(steps)) / len(steps)

# Choleski decomposition?
# the vcv must == its transpose
(vcv == vcv.T).all()  # check

# There must not be any negative eigenvalues
eigens = np.linalg.eig(vcv)
(eigens[0] > 0).all()  # Shoot, darn

# This won't work then
np.linalg.cholesky(vcv)


# Now, they say we can use this to simulate an actual forecast? I believe we
# use these figures to simulate the variation between steps (improvements)
# taken from a chosen starting point.

# Errors are described as a normal distribution of errors around the mean
# error multiplied by a random gaussian vector...
# Checking the sample errors...
plttr(np.histogram(errors, bins=100)[0])  # sure, kinda
merr = np.nanmean(errors)
sterr = np.nanstd(errors)

# list of cumulative variation in errors at each step
variations = [list(errors[:i]) for i in range(2, len(errors))]
vrs = np.array([np.var(v) for v in variations])
plttr(vrs)

# this could also be calculated as a row matrix mutliplied by it's transpose...
# vcv = vrs * np.array([[v] for v in vrs]) # or np.matrix(vrs).T # this is probzbly wrong
# vcv = vcv / len(vrs)
# vcv = np.cov(vrs, rowvar=False)








plttr(vcv)  # similar shape as the cumulative variations.
(np.sqrt(np.diagonal(vcv)) == vrs).all()

# Choleski decomposition?
# the vcv must == its transpose
(vcv == vcv.T).all()  # check

# There must not be any negative eigenvalues
eigens = np.linalg.eig(vcv)
(eigens[0] > 0).all()  # Shoot, darn

# This won't work then
np.linalg.cholesky(vcv)


# # Simulating our own
# serrs = np.random.normal(0, sterr, len(steps)-1)
# plttr(np.histogram(serrs, bins=100)[0])  # sure, kinda

# # It will ultimately look like this i think
# sim_errors = serrs * np.array([[v] for v in vrs])



# sim_errors = np.diagonal(sim_errors)

# simulation = q + sim_errors
# plttr(simulation) # Nope



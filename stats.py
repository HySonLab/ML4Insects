"""
Hurst Exponent
"""
import numpy as np

def hurst_dfa(ts,min_lag = 2,max_lag= 100):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(min_lag, max_lag)

    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    m = np.polyfit(np.log10(lags), np.log10(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return m[0]


def rescaled_range(ts):
    m = np.mean(ts)
    for i in range(len(ts)):
        ts[i] -= m
    z = np.sum(ts)
    r = np.max(z) - np.min(z)
    s = np.std(ts,ddof=0)
    return r/s

def hurst_rs(ts):
    n_splits = 16
    splits = []
    l = len(ts)//n_splits
    
    for n in range(n_splits):
        splits.append(ts[l*n:l*(n+1)])
    
    rs = []
    for n in range(n_splits):
        rs.append(rescaled_range(splits[n]))
    rng = np.arange(1,n_splits+1)
    m = np.polyfit(np.log(rng),np.log(rs),1)
    return m[0]
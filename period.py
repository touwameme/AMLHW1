import pandas as pd
import numpy as np
import math
import numpy as np
from scipy.fftpack import fft, ifft,fftfreq
import matplotlib.pyplot as plt
import seaborn
import scipy.signal as signal
from statsmodels.tsa.stattools import acf
print(findperiod(Ax))
def findperiod(ts):
    period = 0
    fft_series=fft(ts)
    power = np.abs(fft_series)
    sample_freq = fftfreq(fft_series.size)
    pos_mask = np.where(sample_freq>0)
    freqs= sample_freq[pos_mask]
    powers = power[pos_mask]
    top_k_seasons  =3
    top_k_idxs = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
    top_k_power = powers[top_k_idxs]
    fft_periods = (1 / freqs[top_k_idxs]).astype(int)
    print(f"top_k_power: {top_k_power}")
    print(f"fft_periods: {fft_periods}")
    tmpscore = -10
    for lag in fft_periods:
        # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
        acf_score = acf(ts, nlags=lag)[-1]
        if acf_score>tmpscore:
            tmpscore = acf_score
            period =lag
        print(f"lag: {lag} fft acf: {acf_score}")
    return period

def wavelettransform(ts,period):
    i=0
    res=[]
    while(i+period<len(ts)):
        tmpts = ts[i:i+period]
        


    
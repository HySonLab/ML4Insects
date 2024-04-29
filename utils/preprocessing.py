import numpy as np
import pywt
from copy import deepcopy as dc

# PREPROCESSING TECHNIQUES
def downsampling(wave_array, coef = 10):
    '''
        Downsample the signal by average 
    '''
    new_length = len(wave_array)//coef
    downsampled_wave = []
    for i in range(new_length):
        downsampled_wave.append(np.mean(wave_array[i*10:(i+1)*10]))
    return np.array(downsampled_wave)

def quantile_filter(wave_array, k = 1.5, frame_length = 10, sr = 100):
    '''
    Input:
        wave: wave signal of class np.ndarray
    Output:
        new signal whose outliers are set to zeros
    '''
    copy_array = dc(wave_array)
    n_frame = len(wave_array)//(frame_length*sr)
    for i in range(n_frame):
        idx = np.arange(i*frame_length*sr, (i+1)*frame_length*sr)
        w = copy_array[idx]
        Q1 = np.quantile(w,0.25)
        Q3 = np.quantile(w,0.75)
        IQR = abs(Q1-Q3) 
        for i in range(len(w)):
            if (w[i] <= Q1-k*IQR):
                w[i] = Q1-k*IQR
            elif (w[i] >= Q3+k*IQR):
                w[i] = Q3+k*IQR
        copy_array[idx] = w
    return copy_array

"""
Wavelet transforms
"""
def wavelet_reconstruction(wave, wavelet, level, threshold = 'global'):

    coeffs = pywt.wavedec(wave, wavelet, level=level)
    all_coeffs = np.concatenate([l for l in coeffs])

    if threshold == 'global':
        N = len(wave)
        sigma = np.median(np.abs(all_coeffs))/0.6745
        t = sigma*np.sqrt(2*np.log(N))

    elif threshold == 'std':
        std = np.std(all_coeffs,ddof = 1)
        t = 1.5*std
        
    elif threshold == 'ratio':
        t = 0.05*np.max(all_coeffs)
    
    else:
        t = threshold
        
    coeffs_thresholded = [pywt.threshold(c, t, mode='soft') for c in coeffs]
    denoised = pywt.waverec(coeffs_thresholded, wavelet)

    return denoised

import numpy as np
from tqdm import tqdm
import numpy as np
import pywt
from copy import deepcopy as dc

'''
    Adapted from 
        https://github.com/uchidalab/time_series_augmentation
    
'''
def shifting(x, sigma = 1):
    return x + np.random.normal(loc = 0, scale = sigma)*np.ones_like(x)

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=(x.shape[0], x.shape[1]))

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=x.shape[0])
    return np.multiply(x.T, factor).T

def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=x.shape[0]).astype(int)
    window_ends = (window_starts + warp_size).astype(int)
            
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        start_seg = pat[:window_starts[i]]
        window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i]])
        end_seg = pat[window_ends[i]:]
        warped = np.concatenate((start_seg, window_seg, end_seg))                
        ret[i,:] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
    return ret

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[-1]))
    rotate_axis = np.arange(x.shape[-1])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1]-2, num_segs[i]-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[-1]))
    warp_steps = (np.ones((x.shape[-1],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[-1])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[-1]))
    warp_steps = (np.ones((x.shape[-1],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[-1]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret



# PREPROCESSING
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

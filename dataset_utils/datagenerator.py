import numpy as np
import pandas as pd
import librosa
from .datahelper import create_map, ana_labels, encoded_labels
# from utils.augmentation import quantile_filter
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

def generate_sliding_windows_single(   recording, 
                                        ana_file, 
                                        window_size:int, 
                                        hop_length: int, 
                                        method= 'raw', 
                                        outlier_filter: bool = False, 
                                        scale: bool = True, 
                                        pad_and_slice = True, task = 'train'):
    '''
        ----------
        Arguments
        ----------
            recording: input signal
            ana_file: corresponding analysis file 
            window_size: length of the sliding window
            hop_length: sliding length
            method: signal processing method. Must be one of 'raw', 'fft' or 'spec_img'.
            pad_and_slice: when waveform is shorter than window size, pad the window to 2*window_size and adjust shorter hop length
        --------
        Return
        --------
            (data, label) where
                data: concatenated multidimension arrays of shape (number of windows, shape of the features)
                label: corresponding labels
    '''
    # preprocessing 
    if outlier_filter == True:
        recording = quantile_filter(recording)
    if scale == True:
        # recording = minmax_softscale(recording)
        scaler = MinMaxScaler() # Scale the data to (0,1)
        recording = scaler.fit_transform(recording.reshape(-1,1)).squeeze(1)

    if ana_file is not None:
        ana = dc(ana_file)
        ana['time'] = ana['time'].apply(lambda x: int(x*100))
        ana['time'] = ana['time'].astype(int)

    d = []; l = []
    
    if task == 'train':
        dummy_hop = hop_length     
        for i in range(0,len(ana)-1):

            start = ana['time'][i]
            end = ana['time'][i+1]
            wave_type  = ana['label'][i]
        
            if end-start > window_size: #if longer than window_size, segment into windows 
                pass

            else:#if shorter than window_size, => pad and slice
                if pad_and_slice == True:
                    start, end = (start+end)//2 - window_size, (start+end)//2 + window_size
                    hop_length = 128
                else: 
                    start, end = (start+end)//2 - window_size//2, (start+end)//2 + window_size//2

            n_window = ((end-start)-window_size)//hop_length + 1
            for k in range(n_window):
                if start + k*hop_length + window_size > len(recording):
                    break
                idx = np.arange(start + k*hop_length, start + k*hop_length + window_size)
                window = recording[idx]
                features = extract_features(window, method, window_size)
                d.append(features)
                l.append(wave_type)
            hop_length = dummy_hop

        data = np.stack([w for w in d])
        label = np.array(l)

        return data, label

    elif task == 'test':
        
        n_windows = (len(recording)-window_size)//hop_length + 1
        for i in range(n_windows):
            idx = np.arange(i*hop_length, i*hop_length + window_size)
            window = recording[idx]
            features = extract_features(window, method, window_size)
            d.append(features)

        d = np.array(d)

        if ana_file is not None:
            dense_labels = np.concatenate([[ana['label'].iloc[i]] * (ana['time'].iloc[i+1]-ana['time'].iloc[i])
                            for i in range(len(ana)-1)]) 
            map = create_map(ana_labels, encoded_labels)
            dense_labels = pd.Series(dense_labels).map(map).to_numpy()
            return d, dense_labels
        else:
            return d

# def extract_features(window, method, window_size):
#     if method == 'fft':
#         features = (np.abs(librosa.stft(window,n_fft=window_size,center=False))/np.sqrt(window_size)).ravel()
#     elif method == 'spec_img':
#         features = librosa.amplitude_to_db(np.abs(librosa.stft(window, n_fft=128, center = False, hop_length=14))/np.sqrt(window_size),ref=np.max)
#     elif method == 'raw':
#         features = window
#     else:
#         raise RuntimeError ("Param 'method' should be one of 'fft',  'spec_img' or 'raw'.")
#     return features

def extract_features(windows, method, window_size):
    if method == 'raw':
        features = windows
    elif method == 'fft':
        features = (np.abs(librosa.stft(windows,n_fft=window_size,center=False))/np.sqrt(window_size)).squeeze(-1)
    elif method == 'spectrogram':
        features = librosa.amplitude_to_db(np.abs(librosa.stft(windows, n_fft=128, center = False, hop_length=14))/np.sqrt(window_size),ref=np.max)
    elif method == 'gaf':
        gaf = GAF(image_size=64, method = 'summation', overlapping=True)
        features = gaf.transform(windows)
    elif method == 'wavelet':
        a3, d3, d2, d1 = pywt.wavedec(windows,'sym4',level = 3)
        n = windows.shape[0]
        d1_shape = d1.shape[1]
        a3 = interpolate(a3, d1_shape)
        d3 = interpolate(d3, d1_shape)
        d2 = interpolate(d2, d1_shape)

        features = np.stack([np.stack([a3[i],d3[i],d2[i],d1[i]]) for i in range(a3.shape[0])])

    elif method == 'scalogram':
        torch_wt = WaveletTransformTorch(0.01, 9/64, cuda = True)
        features = np.abs(torch_wt.cwt(windows))
        resized = []
        for i in range(features.shape[0]):
            im = features[i]
            resized_scalogram = cv2.resize(im, (im.shape[0], im.shape[0]), interpolation = cv2.INTER_LINEAR)
            resized.append(resized_scalogram)
        features = np.stack(resized)
    else:
        raise RuntimeError ("Param 'method' should be one of 'fft',  'spectrogram' or 'raw'.")
    return features

def interpolate(array, n_interp):
    ndim = len(array.shape)
    if ndim == 1:
        return np.interp(np.arange(0, n_interp), np.arange(0, len(array)), array)
    else:
        n = array.shape[0]
        interp_array = []
        for i in range(n):
            tmp = np.interp(np.arange(0, n_interp), np.arange(0, len(array[i,:])), array[i,:])
            interp_array.append(tmp)
        return np.array(interp_array)
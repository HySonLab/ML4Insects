import numpy as np
import pandas as pd
import librosa
from .datahelper import read_signal, format_data, get_filename, get_dataset_group, create_map, ana_labels, encoded_labels
from utils.preprocessing import quantile_filter
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
from tqdm import tqdm
# from utils.augmentation import *


def generate_sliding_windows(   wave_array, 
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
            wave_array: input signal
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
        wave_array = quantile_filter(wave_array)
    if scale == True:
        # wave_array = minmax_softscale(wave_array)
        scaler = MinMaxScaler() # Scale the data to (0,1)
        wave_array = scaler.fit_transform(wave_array.reshape(-1,1)).squeeze(1)

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
                if start + k*hop_length + window_size > len(wave_array):
                    break
                idx = np.arange(start + k*hop_length, start + k*hop_length + window_size)
                slice = wave_array[idx]
                features = extract_features(slice, method, window_size)
                d.append(features)
                l.append(wave_type)
            hop_length = dummy_hop

        data = np.stack([w for w in d])
        label = np.array(l)

        return data, label

    elif task == 'test':
        
        n_windows = (len(wave_array)-window_size)//hop_length + 1
        for i in range(n_windows):
            idx = np.arange(i*hop_length, i*hop_length + window_size)
            slice = wave_array[idx]
            features = extract_features(slice, method, window_size)
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

def generate_inputs(   dataset_name, 
                                window_size = 1024, 
                                hop_length = 1024, 
                                method= 'raw', 
                                outlier_filter: bool = False, 
                                scale: bool = True, 
                                pad_and_slice = True, 
                                verbose = False):
    '''
        ----------
        Arguments
        ----------
            config: configuration files containing necessary info
            verbose: if True, print descriptions
        --------
        Return
        --------
            d: dictionaries of training/testing data with keys {'data', 'label'}
    '''

    subdatasets = get_dataset_group(dataset_name)

    count = 0
    d = []; l = []  
    for subset in subdatasets:
        print(f'Sub-dataset {subset}.')
        all_recordings = get_filename(subset)
        for n in tqdm(all_recordings):

            df, ana = read_signal(n)
            features, labels = generate_sliding_windows(df, ana, window_size, hop_length, method, outlier_filter, scale, True, 'train')
            d.append(features); l.append(labels)
            count+=1

    d = np.concatenate([f for f in d])
    l = np.concatenate([lab for lab in l])
    
    d = format_data(d, l)

    if verbose == True:
        print(f'Read {count} recordings')
        print(f'Signal processing method: {method} | Outliers filtering: {str(outlier_filter)} | Scale: {str(scale)}')
        cl, c = np.unique(l, return_counts=True)
        print('Class distribution (label:ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(l),2)}' for i in range(len(cl))))
        print(f'Labels map (from:to): {{1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}')

    return d

def extract_features(window, method, window_size):
    if method == 'fft':
        features = (np.abs(librosa.stft(window,n_fft=window_size,center=False))/np.sqrt(window_size)).ravel()
    elif method == 'spec_img':
        features = librosa.amplitude_to_db(np.abs(librosa.stft(window, n_fft=128, center = False, hop_length=14))/np.sqrt(window_size),ref=np.max)
    elif method == 'raw':
        features = window
    else:
        raise RuntimeError ("Param 'method' should be one of 'fft',  'spec_img' or 'raw'.")
    return features

# class ts_aug:
#     def __init__(self, data, labels, label_to_augment, seed = 28):
        
#         self.label_to_augment = label_to_augment 
#         idx = np.where(labels == label_to_augment)[0]
#         self.data = data[idx]
#         self.labels = labels[idx]
#         self.n_obs = self.data.shape[0]
#         self.N = data.shape[0]
#         self.seed = seed
#         np.random.seed(self.seed)

#     def augment(self, augment_methods = ['shift', 'scale', 'jitter', 'window_warp'], n_samples = None):
#         # np.random.seed(self.seed)
#         if n_samples is None:
#             n_samples = int(0.01*self.N)
#             self.n_samples = n_samples
    
#         augmented_samples = []
#         for method in augment_methods:
#             tmp = self.generate_augmented_samples(method, n_samples)
#             augment_samples.append(tmp)
#         augmeted_samples = np.concatenate([s for s in augmented_samples])
#         return augmented_samples 

#     def generate_augmented_samples(self, method, n_samples = None):
#         # np.random.seed(self.seed)
#         idx = np.random.choice(self.n_obs, size = n_samples, replace = True)
#         augmented = self.data[idx]
#         # self.aug_sample = self.data[idx]
#         if method == 'shift':
#             augmented = shifting(augmented, 1)
#         elif method == 'scale':
#             augmented = scaling(augmented, 0.2)
#         elif method == 'jitter':
#             augmented = jitter(augmented, 0.005)
#         elif method == 'window_warp':
#             augmented = window_warp(augmented, window_ratio=0.3, scales = [0.25, 0.5, 1.5, 2])
#         return np.array(augmented)

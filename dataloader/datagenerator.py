import numpy as np
import librosa
import pywt
from .datahelper import read_signal, format_data
from utils.preprocessing import quantile_outlier_filtering
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

'''
    Changelog 15.Feb.2024:
        + Add instructions
        + hop_length is now configurable
'''

def average_pool(array, pool_size=2):
    pool = []
    len_pool = len(array)//pool_size
    for i in range(len_pool):
        pool.append(np.mean(array[i*pool_size: (i+1)*pool_size]))
    return np.array(pool)

def GASF_GADF(arr):
    array = arr.reshape(-1,1)
    max = array.max()
    min = array.min()
    if max > min:
        normed_array = (2*array-max-min)/(max - min)
    else:
        normed_array = array/len(array)
    sin = np.sqrt(np.ones((len(array),1))-normed_array**2)
    coscos = normed_array@normed_array.T
    sinsin = sin@sin.T
    sincos = sin@array.T
    cossin = array@sin.T
    return np.array(coscos - sinsin, sincos - cossin)

def generate_signal_dictionary(data_names: list, data_splits: dict, outlier_filter: bool = False, scale: bool = False):
    '''
        Input: 
            data_names: list of names of data folders to read
            data_splits: dict of train/test names 
            outlier_filter: if True, imputate the outliers by median
            scale: if True, scale to [0,1] by substracting min and divide by range (max - min)
        Output:
            dict containing (key, value) = (recording name, [signal, analysis])
    '''
    data = {}; data_test = {}
    if isinstance(data_names, str):
        data_names = [data_names]
    for n in data_names:
        train, test = data_splits[n]
        for i in range(len(train)):
            # Read data table and analysis file
            data[train[i]] = read_signal(train[i])
            # preprocessing 
            if outlier_filter == True:
                data[train[i]][0] = quantile_outlier_filtering(data[train[i]][0])
            if scale == True:
                scaler = MinMaxScaler() # Scale the data to (0,1)
                data[train[i]][0] = scaler.fit_transform(data[train[i]][0].reshape(-1,1)).squeeze(1)
        for i in range(len(test)):
            # Read data table and analysis file
            data_test[test[i]] = read_signal(test[i])
            # preprocessing
            if outlier_filter == True:
                data_test[test[i]][0] = quantile_outlier_filtering(data_test[test[i]][0])
            if scale == True:
                scaler = MinMaxScaler() # Scale the data to (0,1)
                data_test[test[i]][0] = scaler.fit_transform(data_test[test[i]][0].reshape(-1,1)).squeeze(1) 
    print(f'There are {len(data)} recordings used for training and {len(data_test)} recordings used for testing')
    return data, data_test

def generate_input(wave_array, ana_file, window_size:int, hop_length: int, method= 'raw'):
    '''
    Input:
        wave_array: input signal
        ana_file: corresponding analysis file 
        window_size: length of the sliding window
        hop_length: sliding length
        method: signal processing method. Must be one of 'raw', 'fft' or 'spec_img'.
    Output:
        (data, label) where
            data: concatenated multidimension arrays of shape (number of windows, shape of the features)
            label: corresponding labels
    '''
    d = []
    l = []
    dummy_hop = hop_length
    ana = dc(ana_file)
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    for i in range(0,len(ana)-1):

        start = ana['time'][i]
        end = ana['time'][i+1]
        wave_type  = ana['label'][i]

        if end-start > window_size: #if longer than window_size, segment into windows 
            pass
        else:#if shorter than window_size, pad both sides to get 2*window_size around the center hop length is reduced
            start, end = (start+end)//2 - window_size, (start+end)//2 + window_size
            hop_length = 128

        n_window = ((end-start)-window_size)//hop_length + 1
        for k in range(n_window):
            if start + k*hop_length + window_size > len(wave_array):
                break
            idx = np.arange(start + k*hop_length, start + k*hop_length + window_size)
            slice = wave_array[idx]
            if method == 'fft':
                fft_coef = (np.abs(librosa.stft(slice, n_fft = window_size, center=False))/np.sqrt(window_size)).ravel()
                d.append(fft_coef)
            elif method == 'dwt':
                dwt_coef = pywt.wavedec(slice, 'sym8', level = 2)[0]
                d.append(dwt_coef)
            elif method == 'spec_img':
                im = librosa.amplitude_to_db(np.abs(librosa.stft(slice, n_fft = 128, hop_length = 14, center = False))/np.sqrt(window_size), ref=np.max)
                d.append(im)
            elif method == 'raw':
                d.append(slice)
            # elif method == 'gramian':
            #     im = GASF_GADF(slice)
            #     d.append(im)
            else:
                raise RuntimeError ("Param 'method' should be one of 'fft', 'dwt', 'spec_img' or 'raw'.")
            
            l.append(wave_type)
        hop_length = dummy_hop

    data = np.stack([w for w in d])
    label = np.array(l)
        
    return data, label

def generate_data(data_names, data_splits, config, verbose = False):
    '''
    Input: 
        data_names: list of names of data folders to read
        data_splits: dict of train/test names 
        config: configuration files containing necessary info
        verbose: if True, print descriptions
    Output:
        (train, test): dictionaries of training/testing data with keys {'data', 'label'}
    '''
    window_size = config.window_size
    hop_length = config.hop_length
    method = config.method
    scale = config.scale
    outlier_filter = config.outlier_filter
    # multistage = config.multistage
    dict_train, dict_test = generate_signal_dictionary(data_names, data_splits, outlier_filter, scale)
    d = []; l = []    
    for filename in dict_train.keys(): 
        df = dict_train[filename][0]; ana = dict_train[filename][1] 
        features, labels = generate_input(df, ana, window_size = window_size, hop_length= hop_length, method = method)
        d.append(features); l.append(labels)
    df_train = np.concatenate([f for f in d])
    label_train = np.concatenate([lab for lab in l])

    d = []; l = []    
    for filename in dict_test.keys(): 
        df = dict_test[filename][0]; ana = dict_test[filename][1] 
        features, labels = generate_input(df, ana, window_size = window_size, hop_length= hop_length, method = method)
        d.append(features); l.append(labels)
    df_test = np.concatenate([f for f in d])
    label_test = np.concatenate([lab for lab in l])

    if verbose == True:
        print(f'Signal processing method: {method} | Outliers filtering: {str(outlier_filter)} | Scale: {str(scale)}')
        print(f'Train/test shape: {df_train.shape}/{df_test.shape}')
        cl, c = np.unique(label_train,return_counts=True)
        print('Train distribution (label/ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(label_train),2)}' for i in range(len(cl))))
        cl, c = np.unique(label_test,return_counts=True)
        print('Test distribution (label/ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(label_test),2)}' for i in range(len(cl))))
        print(f'Labels map (from/to): {{1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}')
    
    train = format_data(df_train,label_train)
    test = format_data(df_test,label_test)            
    return train, test

##################################################################################################
def generate_test_data(wave_array, ana_file = None, window_size:int = 1024, hop_length:int = 256, method:str = 'raw'):
    '''
    Input:
        wave_array: input signal
        ana_file: corresponding analysis file (if given, then also returns labels) 
        window_size: length of the sliding window
        hop_length: sliding length
        method: signal processing method. Must be one of 'raw', 'fft' or 'spec_img'.
    Output:        
        (data, label) where
            data: concatenated multidimension arrays of shape (number of windows, shape of the features)
            label: corresponding labels (if ana_file is given)  
    '''
    
    if ana is not None:
        ana = dc(ana_file)
        ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
        dense_labels = np.concatenate([[ana['label'].iloc[i]] * (ana['time'].iloc[i+1]-ana['time'].iloc[i])
                        for i in range(len(ana)-1)])

    n_windows = (len(wave_array)-window_size)//hop_length + 1
    d = []
    l = []
    for i in range(n_windows):
        idx = np.arange(i*hop_length, i*hop_length + window_size)
        slice = wave_array[idx]
        if method == 'fft':
            fft_coef = (np.abs(librosa.stft(slice,n_fft=window_size,center=False))/np.sqrt(window_size)).ravel()
            d.append(fft_coef)
        elif method == 'dwt':
            dwt_coef = pywt.wavedec(slice, 'sym8', level = 2)[0]
            d.append(dwt_coef)
        elif method == 'spec_img':
            im = librosa.amplitude_to_db(np.abs(librosa.stft(slice, n_fft=128, center = False, hop_length=14))/np.sqrt(window_size),ref=np.max)
            d.append(im)
        elif method == 'raw':
            d.append(slice)
        # elif method == 'gramian':
        #     im = GASF_GADF(slice)
        #     d.append(im)
        else:
            raise RuntimeError ("Param 'method' should be one of 'fft', 'dwt', 'spec_img' or 'raw'.")
        if ana is not None:
            tmp_label = dense_labels[i*hop_length: i*hop_length+window_size]
            cl, c = np.unique(tmp_label, return_counts=True)
            l.append(cl[np.argmax(c)])
    d = np.array(d)
    
    if ana is not None:
        l = np.array(l) 
        X = format_data(d,l)
        return X
    else: 
        return d
        
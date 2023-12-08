import pandas as pd
import numpy as np
import pywt 
import librosa

original_labels = {1:'np',2:'c',4:'e1',5:'e2',6:'f',7:'g',8:'pd'}

def read_wave(filename,extension):
    d = []
    for i in range(len(extension)):
        x = pd.read_csv(filename + extension[i],low_memory = False,delimiter=";",header = None,usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)
    return data

def read_analysis(analysis_filename):
    ana = pd.read_csv(analysis_filename,encoding='utf-16',delimiter = '\t',header = None,usecols=[0,1])
    ana.columns = ['label','time']
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    ana.drop_duplicates(subset='time',inplace=True)
    ana.index = np.arange(len(ana))
    return ana.astype(int)

def get_index(ana):
    '''
    Input.
        ana: analysis file
    Output.
        index: dictionary containing intervals of all wave types found in the analysis file 
            1 ~ np, 2 ~ c, 4 ~ e1, 5 ~ e2, 6 ~ f, 7 ~ g, 8 ~ pd
    '''
    index = {}
    n = len(ana)
    for i in range(0,n-1):
        start, end = ana.loc[i:i+1,'time'].tolist()
        if ana.loc[i,'label'] == 1:
            try:
                index['np'].append([int(start),int(end)])
            except:
                index['np'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 2:
            try:
                index['c'].append([int(start),int(end)])
            except:
                index['c'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 4:
            try:
                index['e1'].append([int(start),int(end)])
            except:
                index['e1'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 5:
            try:
                index['e2'].append([int(start),int(end)])
            except:
                index['e2'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 6:
            try:
                index['f'].append([int(start),int(end)])
            except:
                index['f'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 7:
            try:
                index['g'].append([int(start),int(end)])
            except:
                index['g'] = [[int(start),int(end)]]
        elif ana.loc[i,'label'] == 8:
            try:
                index['pd'].append([int(start),int(end)])
            except:
                index['pd'] = [[int(start),int(end)]]
    return index 

def downsampling(wave_array,ana = None,coef = 10):
    new_length = len(wave_array)//coef

    downsampled_wave = []
    for i in range(new_length):
        downsampled_wave.append(np.mean(wave_array[i*10:(i+1)*10]))
        
    if ana is None:
        return np.array(downsampled_wave)
    else:
        downsampled_ana = pd.concat([ana.loc[:,'label'],ana.loc[:,'time'].apply(lambda x: x//10)],axis=1)
        return np.array(downsampled_wave) , downsampled_ana

def outlier_filtering(wave_array,ana = {},option = 'whole'):
    '''
    Outliers are defined to be the values which is out side of the range [Q1-1.5*IQR, Q3+1.5*IQR]
    Input:
        wave: wave signal of class np.ndarray
        ana: analysis file 
        option: 
            'whole' - apply outlier filter to each of the waves given by ana
            'indiv' - use when wave_array is a single wave
    Output:
        wave_array without outliers
    '''
    if option == 'whole':
        new_wave = []
        new_ana_time = [0]

        index = 0

        for i in range(len(ana)-1):

            start = ana.iloc[i,1].item()
            end = ana.iloc[i+1,1].item()

            wave_segment = wave_array[start:end]
            
            Q1 = np.quantile(wave_segment,0.25)
            Q3 = np.quantile(wave_segment,0.75)
            IQR = abs(Q1-Q3)

            filtered_wave = wave_segment[(wave_segment > Q1-1.5*IQR) & (wave_segment < Q3+1.5*IQR)]

            index += len(filtered_wave)
            new_wave.append(filtered_wave)
            new_ana_time.append(index)
        new_wave = np.concatenate(new_wave)
        new_ana = pd.DataFrame({'label':ana['label'],'time':new_ana_time})
        
        return [new_wave, new_ana]
    
    elif option == 'indiv':
        Q1 = np.quantile(wave_array,0.25)
        Q3 = np.quantile(wave_array,0.75)
        IQR = abs(Q1-Q3) 

        return wave_array[(wave_array > Q1-1.5*IQR) & (wave_array < Q3+1.5*IQR)]
        
def generate_fft_data(wave_array,ana,window_size = 1024,hop_length = None):
    '''
    Input:
        wave: wave signal of class np.ndarray 
        ana: analysis file 
        window_size: size of fft window
        hop_length: the length of which the windows will be slided
    Output:
        data: numpy array containing absolute values of fourier coefficients arranged row-wise
        label: labels corresponding to the rows
    '''
    wave_indices = get_index(ana)
    data_list = []
    label = []

    if not isinstance(hop_length,int): # if hop_length is not a number then it is set to window_size/4 by default
        hop_length = window_size//4

    for wave_type in wave_indices.keys():

        for start,end in wave_indices[wave_type]:

            if end-start > window_size: 

                window = wave_array[start:end]
                wave_stft_coef = np.abs(librosa.stft(window,n_fft = window_size,hop_length=hop_length,center = False)).transpose()
                
            else: #if shorter than window_size, pad both sides with adjacent signals 

                window = wave_array[((start+end)//2 - window_size):((start+end)//2 + window_size)]            
                wave_stft_coef = np.abs(librosa.stft(window,n_fft = window_size,hop_length=128,center = False)).transpose()

            l = np.array([wave_type]*wave_stft_coef.shape[0]).reshape(-1,1)

            data_list.append(wave_stft_coef)
            label.append(l)

    data = np.concatenate([x for x in data_list])
    label = np.concatenate([l.ravel() for l in label])

    return data, label

def generate_raw_data(wave_array,ana,window_size = 1024,hop_length = None):
    '''
    Input:
        wave: wave signal of class np.ndarray (from func read_wave())
        ana: analysis file 
        window_size: size of fft window
        hop_length: the length of which the windows will be slided
    Output:
        data: np.ndarray containing windows of size 1024 of the input wave, shifted by hop_length, arranged row-wise
        label: labels corresponding to the rows
    '''
    wave_indices = get_index(ana)
    data_list = []
    label = []

    if not isinstance(hop_length,int): # if hop_length is not a number then it is set to window_size/4 by default
        hop_length = window_size//4

    for wave_type in wave_indices.keys():

        for start,end in wave_indices[wave_type]:

            if end-start > window_size: #if longer than window_size, segment into windows 

                n_window = ((end-start)-window_size)//hop_length + 1

                for k in range(n_window):

                    idx = np.arange(start + k*hop_length,start + k*hop_length + window_size)
                    data_list.append((wave_array[idx]))
                    label.append(wave_type)

            else: #if shorter than window_size, take the middle point and spread to both sides
                #take 2*window_size points around the mean

                start, end = (start+end)//2 - window_size, (start+end)//2 + window_size
                n_window = ((end-start)-window_size)//128 + 1

                for k in range(n_window):

                    idx = np.arange(start + k*128,start + k*128 + window_size)
                    data_list.append((wave_array[idx]))
                    label.append(wave_type)

    data = np.stack([w for w in data_list])
    label = np.array(label)

    return data, label

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from copy import deepcopy as dc

def generate_model_data(data_dictionary,n = None,mode = 'raw',scale = None,window_size = 1024,hop_length = None):
    '''
    Input:
        data_dictionary: dictionary of form [wave_array,wave analysis file] containing all the input files
        n: n first input files we would like to read
        scale: scale the waveforms by MinMax or not
        window_size: size of fft/raw window
        hop_length: the length of which the windows will be slided
    Output:
        data: concatenated array of each files data
        label: concatenated labels
    '''
    filenames = [*data_dictionary.keys()]
    temp_feature_list = []
    temp_label_list = []

    if not isinstance(hop_length,int):
        hop_length = window_size//4      

    if not isinstance(n,int): #set n = number of input files by default
        n = len(filenames)

    for ind in range(n): 
        df = dc(data_dictionary[filenames[ind]][0])
        ana = data_dictionary[filenames[ind]][1] 

        if scale == 'minmax':
            scaler = MinMaxScaler()
            df = scaler.fit_transform(df.reshape(-1,1)).squeeze(1)
        elif scale == 'standard':
            scaler = StandardScaler()
            df = scaler.fit_transform(df.reshape(-1,1)).squeeze(1)
        elif scale == 'partial':
            wave_indices = get_index(ana)
            for wave_type in wave_indices.keys():
                for start,end in wave_indices[wave_type]:
                    scaler = StandardScaler()
                    df[start:end] = scaler.fit_transform(df[start:end].reshape(-1,1)).squeeze(1)

        if mode == 'raw':
            features, labels = generate_raw_data(df,ana,window_size=window_size,hop_length=hop_length)

        elif mode == 'fft':
            features, labels= generate_fft_data(df,ana,window_size=window_size,hop_length=hop_length)

        temp_feature_list.append(features)
        temp_label_list.append(labels)


    data = np.concatenate([f for f in temp_feature_list])
    label = np.concatenate([l for l in temp_label_list])

    print(f'Included files: {filenames[:n]}')
    print(f'Model data shape: {data.shape},label shape: {label.shape}')

    return data, label

def extract_sample(wave_array,ana,wave_type,which):
    wave_indices = get_index(ana)
    start,end = wave_indices[wave_type][which]
    return wave_array[start:end]

class numeric_encoder():
    def __init__(self):
        self.labels_dict = {'np':0,'c':1,'e1':2,'e2':3,'f':4,'pd':5,'g':6} 

    def fit_transform(self,labels):
        self.labels = labels
        self.classes = np.unique(labels)
        self.n_classes = len(self.classes)

        new_labels = []
        for i in range(len(labels)):
            l = self.labels_dict[labels[i].item()]
            new_labels.append(l)
        self.new_labels = np.array(new_labels,dtype=np.int32)
        return self.new_labels
    
    def reverse_label(self,labels):
        self.reverse_labels_dict = {0:'np',1:'c',2:'e1',3:'e2',4:'f',5:'pd',6:'g'}
        reverse_labels = []
        for i in range(len(labels)):
            l = self.reverse_labels_dict[labels[i].item()]
            reverse_labels.append(l)
        return reverse_labels      
    

"""
Wavelet transforms
"""

def get_wavelet_coefficients(wave, wavelet = 'sym4', n_level = 3):

    cA, cD = pywt.dwt(wave,wavelet)
    low_freq = [cA]
    high_freq = [cD]
    for n in range(n_level-1):
        cA, cD = pywt.dwt(cA,wavelet)
        low_freq.append(cA)
        high_freq.append(cD)
        
    return low_freq, high_freq

def average_energy(low_freq, high_freq):

    average_low_frequency_energy = []
    average_high_frequency_energy = []

    n_level = len(low_freq)

    for i in range(n_level):
        average_low_frequency_energy.append(np.sum(low_freq[i]**2)/(2*len(low_freq[0])))
        average_high_frequency_energy.append(np.sum(high_freq[i]**2)/(2*len(high_freq[0])))

    return average_low_frequency_energy, average_high_frequency_energy

def wavelet_denoising(wave, wavelet, n_level, threshold = 'global'):

    coeffs = pywt.wavedec(wave, wavelet, level=n_level)
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

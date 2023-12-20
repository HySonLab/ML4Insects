import pandas as pd
import numpy as np
import pywt 
import librosa
import os

original_labels = {1:'np',2:'c',4:'e1',5:'e2',6:'f',7:'g',8:'pd'}

path = os.getcwd()
extension = ['.A01','.A02','.A03','.A04','.A05','.A06','.A07','.A08']
labels_dict = {'np':0,'c':1,'e1':2,'e2':3,'f':4,'pd':5,'g':6}
reverse_labels_dict = {0:'np',1:'c',2:'e1',3:'e2',4:'f',5:'pd',6:'g'}

def get_filename(name):
    os.chdir(f'{path}\\{name}')
    files_list = os.listdir()
    unique = []
    for name in files_list:
        name = name.split('.')[0]
        unique.append(name)
    unique = np.unique(unique)
    os.chdir(path)
    return unique

def read_signal(filename: str) -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    s = filename.split('_')
    dir = s[0]
    
    os.chdir(os.path.join(path,dir))
    d = []
    for i in range(len(extension)):
        x = pd.read_csv(filename + extension[i],low_memory = False,delimiter=";",header = None,usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    os.chdir(os.path.join(path,dir+'_ANA'))
    ana = pd.read_csv(filename + '.ANA',encoding='utf-16',delimiter = '\t',header = None,usecols=[0,1])
    ana.columns = ['label','time']
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    ana.drop_duplicates(subset='time',inplace=True)
    ana.index = np.arange(len(ana))
    
    os.chdir(path)
    return data, ana.astype(int)

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
    '''
        Downsample the signal by taking the average 
    '''
    new_length = len(wave_array)//coef

    downsampled_wave = []
    for i in range(new_length):
        downsampled_wave.append(np.mean(wave_array[i*10:(i+1)*10]))
        
    if ana is None:
        return np.array(downsampled_wave)
    else:
        downsampled_ana = pd.concat([ana.loc[:,'label'],ana.loc[:,'time'].apply(lambda x: x//10)],axis=1)
        return np.array(downsampled_wave), downsampled_ana

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

            filtered_wave = wave_segment[(wave_segment > Q1-2.0*IQR) & (wave_segment < Q3+2.0*IQR)]

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

        return wave_array[(wave_array > Q1-2.0*IQR) & (wave_array < Q3+2.0*IQR)]

def generate_data(wave_array, ana, window_size=1024, hop_length=256, mode = 'mlp'):
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

    data_list = []
    label = []

    tmp_hop = hop_length
    for i in range(0,len(ana)-1):

        start = ana['time'][i]
        end = ana['time'][i+1]
        wave_type  = ana['label'][i]

        if end-start > window_size: #if longer than window_size, segment into windows 
            pass

        else:#if shorter than window_size, pad both sides to get 2*window_size around the center
                #hop length is reduced
            start, end = (start+end)//2 - window_size, (start+end)//2 + window_size
            hop_length = 128
        # print(wave_type,end,start,hop_length)
        # break
        n_window = ((end-start)-window_size)//hop_length + 1

        for k in range(n_window):

            idx = np.arange(start + k*hop_length,start + k*hop_length + window_size)

            if mode == 'cnn1d':
                data_list.append((wave_array[idx]))

            elif mode == 'mlp':
                coef = np.abs(librosa.stft((wave_array[idx]),n_fft=window_size,center=False)).ravel()
                data_list.append(coef)

            elif mode == 'cnn2d':
                im = librosa.amplitude_to_db(np.abs(librosa.stft((wave_array[idx]) ,n_fft=128,center = False,hop_length=14)),ref=np.max)
                data_list.append(im)
            
            else:
                data_list.append((wave_array[idx]))

            label.append(wave_type)

        hop_length = tmp_hop

    data = np.stack([w for w in data_list])
    label = np.array(label)
    label = pd.Series(label).map({1: 0, 2 : 1, 4 : 2, 5 : 3, 6 : 4, 7 : 6, 8 : 5}).to_numpy()

    if mode == 'transformer':
        transfomer_data = []
        transformer_label = []

        n = len(data)//10
        for i in range(n):
            transfomer_data.append(data[i*10:(i+1)*10,:])
            transformer_label.append(label[i*10:(i+1)*10])
        transfomer_data = np.array(transfomer_data)
        data = transfomer_data
        label = transformer_label
        
    return data, label

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy as dc

def generate_model_data(data_dictionary,n = None,mode = 'mlp',scale = None,window_size = 1024,hop_length = None,verbose = False):
    '''
    Input:
        data_dictionary: dictionary of form [wave_array,wave analysis file] containing all the input files
        n: n first input files we would like to read
        scale: scale the waveforms by MinMax or not
        window_size: size of fft/raw window
        hop_length: the length of which the windows will be slided
        verbose: print description
    Output:
        data: concatenated array of each files data
        label: concatenated labels
    '''
    filenames = [*data_dictionary.keys()]
    dat = []
    lab = []

    if not isinstance(hop_length,int):
        hop_length = window_size//4      

    if not isinstance(n,int): #set n = number of input files by default
        n = len(filenames)

    for ind in range(n): 
        
        df = dc(data_dictionary[filenames[ind]][0])
        ana = data_dictionary[filenames[ind]][1] 

        if scale == True:
            scaler = MinMaxScaler() # Scale the data to (0,1)
            df = scaler.fit_transform(df.reshape(-1,1)).squeeze(1)

        features, labels = generate_data(df,ana,window_size=window_size,hop_length=hop_length,mode=mode)

        dat.append(features)
        lab.append(labels)

    df = np.concatenate([f for f in dat])
    label = np.concatenate([l for l in lab])

    if verbose == True:
        print(f'Mode: {mode}')
        print(f'Scale option: {str(scale)}')
        print(f'Model data shape: {df.shape}, label shape: {label.shape}')
        print(f'Class distribution: ',np.unique(label,return_counts=True))
        print('Class encoding: ', labels_dict)
            
    return df, label

def extract_sample(wave_array,ana,wave_type,which):
    '''
        Extract one waveform sample from the whole signal
    '''
    wave_indices = get_index(ana)
    start,end = wave_indices[wave_type][which]
    return wave_array[start:end]

class numeric_encoder():
    '''
        This class provides methods that convert labels from string to integer
    '''
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

def get_wavelet_coefficients(array, wavelet = 'sym4', n_level = 3):
    '''
        Input: 
            array: signal of the form np.ndarray
            wavelet: one of PyWavelet wavelets
            n_level: number of resolution
        Output: 
            tuple: (low_freq, high_freq) 
            low_freq: approximation coefficients of resolution 1 to n
            high_freq: detail coefficients of resolution 1 to n
    '''
    cA, cD = pywt.dwt(array,wavelet)
    low_freq = [cA]
    high_freq = [cD]
    for n in range(n_level-1):
        cA, cD = pywt.dwt(cA,wavelet)
        low_freq.append(cA)
        high_freq.append(cD)
        
    return low_freq, high_freq

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

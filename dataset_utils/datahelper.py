import os
import pandas as pd 
import glob
from easydict import EasyDict 
from copy import deepcopy as dc

import torch
from datetime import date

# ============================= Dataset =============================
datasets = {
            'zt': ['0zt','8zt','16zt'], 
            'ct': ['0ct','8ct','16ct'],
            'BCOA-Wheat': ['BCOA-Wheat'],
            'Cannabis-hemp': ['non-viruliferous-hemp', 'viruliferous-hemp'],
            'Cannabis-potato': ['non-viruliferous-potato', 'viruliferous-potato'],
            'ArabidopsisGPA': ['ArabidopsisGPA'],
            'wheatrnai': ['wheatrnai'], 
            'SBA-HostSurface': ['SBA-HostSurface'],
            'SBA-Rag5': ['SBA-Rag5'],
            'sorghumaphid': ['sorghumaphid'], 
            }

def get_dataset_group(group):
    if group == 1 or group == 'BCOA1': # BCOA1
        # return datasets['zt']
        return datasets['zt'] + datasets['ct']
    elif group == 2 or group == 'BCOA2': # BCOA2
        return datasets['BCOA-Wheat']  
    elif group == 3 or group == 'CA': # Cannabis aphid
        return datasets['Cannabis-hemp'] + datasets['Cannabis-potato']  
    elif group == 4 or group == 'GPA': # GPA
        return datasets['ArabidopsisGPA']
    elif group == 5 or group == 'SA': # Soybean
        return datasets['SBA-Rag5']
        # return datasets['SBA-Rag5'] + datasets['SBA-HostSurface']
    elif group == 6 or group == 'SorgA': # Sorghum
        return datasets['sorghumaphid']
    elif group == 7 or group == 'RNAi': # Wheat RNAi
        return datasets['wheatrnai']
    elif group == 99 or group == 'combined':
        all_names = []
        for i in range(1,6):
            all_names += get_dataset_group(i)
        return all_names
    else:
        try:
            return datasets[group]
        except:
            raise ValueError('Unsupported value.')

# ============================= Label map =============================
waveform_labels = ['np', 'c', 'e1', 'e2', 'f', 'g', 'pd']
ana_labels = [1, 2, 4, 5, 6, 7, 8]
encoded_labels = [0, 1, 2, 3, 4, 5, 6]

def create_map(list1, list2):
    map = {}
    for i in range(len(list1)):
        map[list1[i]] = list2[i]
    return map

# ============================= Format =============================
def format_data(data, label, arch = None, seq_length = None):
    label = pd.Series(label).map(create_map(ana_labels, encoded_labels)).to_numpy()
    return EasyDict({'data':data,'label':label})

# ============================= Read input =============================
import numpy as np 
import os 
wd = os.getcwd()

def get_filename(name, data_path = '../data'):
    filenames = os.listdir(f'{data_path}/{name}_ANA')
    unique = []
    for name in filenames:
        name = name[:-4]
        unique.append(name)
    return unique
    
def read_signal(filename: str, data_path = '../data') -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    # Read signals
    d = []
    dataset_name = filename.split('_')[0]
    dir = os.listdir(f'{data_path}/{dataset_name}')
    rec_components = glob.glob(f'{data_path}/{dataset_name}/{filename}.*')

    # Read the signal (.A0x) files
    for file_path in rec_components:
        x = pd.read_csv(file_path, low_memory = False, delimiter=";", header = None, usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    # Read the analysis (.ANA) files
    try:
        ana_path = f'{data_path}/{dataset_name}_ANA/{filename}.ANA'
        ana = pd.read_csv(ana_path, encoding='utf-16', delimiter = '\t',header = None, usecols=[0,1])
        ana.columns = ['label','time']
        ana = ana[(ana['label'] != 9) & (ana['label'] != 10) & (ana['label'] != 11)]
        ana.drop_duplicates(subset='time',inplace=True)
        ana.index = [i for i in range(len(ana))]
    except:
        ana = None 
    return data, ana

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
                index['np'].append([start,end])
            except:
                index['np'] = [[start,end]]
        elif ana.loc[i,'label'] == 2:
            try:
                index['c'].append([start,end])
            except:
                index['c'] = [[start,end]]
        elif ana.loc[i,'label'] == 4:
            try:
                index['e1'].append([start,end])
            except:
                index['e1'] = [[start,end]]
        elif ana.loc[i,'label'] == 5:
            try:
                index['e2'].append([start,end])
            except:
                index['e2'] = [[start,end]]
        elif ana.loc[i,'label'] == 6:
            try:
                index['f'].append([start,end])
            except:
                index['f'] = [[start,end]]
        elif ana.loc[i,'label'] == 7:
            try:
                index['g'].append([start,end])
            except:
                index['g'] = [[start,end]]
        elif ana.loc[i,'label'] == 8:
            try:
                index['pd'].append([start,end])
            except:
                index['pd'] = [[start,end]]
    return index 

def extract_sample(wave_array,ana_file,wave_type,which):
    '''
        Extract one waveform sample from the whole signal
    '''
    ana = dc(ana_file)
    ana['time'] = ana['time'].apply(lambda x: x*100)
    ana['time'] = ana['time'].astype(int)
    wave_indices = get_index(ana)
    start,end = wave_indices[wave_type][which]
    return wave_array[start:end]




# ============================== Draft =====================================
# ============================= Checkpoint =============================
# def save_checkpoint(models, config, name = None):
#     n_models = len(models)
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
#     if not os.path.exists(f'checkpoints/{models[0].__type__}'):
#         os.makedirs(f'checkpoints/{models[0].__type__}')
#     dir = f'checkpoints/{models[0].__type__}'
#     if n_models == 1:
#         level = '1stage'
#     elif n_models == 2:
#         level = '2stage'
#     for n in range(n_models):
#         torch.save(models[n], dir + f'/{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth')
#         print(f'Parameters saved! "{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth".')

# def get_train_test_filenames(train_ratio = None, n_test = None, name = None, random_state = 10):
#     np.random.seed(random_state)
#     if name is None:
#         list_dir = [d for d in os.listdir('./data/') if not d.endswith('_ANA') if '.' not in d]
#     else:
#         if isinstance(name, list):
#             list_dir = name
#     splits = {}
#     for name in list_dir: 
#         if not os.path.exists(f'./data/{name}_ANA'):
#             continue
#         recording_names = get_filename(name)
#         np.random.shuffle(recording_names)
#         if (train_ratio is not None):
#             n = int(train_ratio*len(recording_names))
#             train_name = recording_names[:n]
#             test_name = recording_names[n:]
#         elif n_test is not None:
#             n_train = len(recording_names) - n_test
#             n = min(n_train, len(recording_names)-1)
#             train_name = recording_names[:n]
#             test_name = recording_names[n:n+n_test]       
#         # print(n, name)     
#         splits[name] = (train_name, test_name)
#     return splits
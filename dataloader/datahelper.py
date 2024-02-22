import os
import pandas as pd 
from easydict import EasyDict 
from copy import deepcopy as dc

import torch
from datetime import date
# ============================= Label map =============================
path = os.getcwd()

waveform_labels = ['np', 'c', 'e1', 'e2', 'f', 'g', 'pd']
ana_labels = [1, 2, 4, 5, 6, 7, 8]
encoded_labels = [0, 1, 2, 3, 4, 5, 6]
def create_maps(list1, list2):
    map = {}
    for i in range(len(list1)):
        map[list1[i]] = list2[i]
    return map

# ============================= Format =============================
# Two levels split for 2-level classifier
def format_data(data, label, multistage = False):
    if multistage == False: 
        label = pd.Series(label).map(create_maps(ana_labels, encoded_labels)).to_numpy()
        return EasyDict({'data':data,'label':label})
    
    elif multistage == True:
        # Stage1
        stage1_data = data
        stage1_label = dc(label).tolist()
        for i in range(len(label)):
            if (label[i] == 4) or (label[i] == 5):
                stage1_label[i] = 'e1e2'
            elif (label[i] == 2) or (label[i] == 8):
                stage1_label[i] = 'cpd'
            else: #otherwise label 2
                stage1_label[i] = 'other'
        stage1_label = pd.Series(stage1_label).map({'e1e2':0, 'cpd':1, 'other':2}).to_numpy()

        #Stage2
        stage2_data_e1e2 = data[stage1_label == 0]
        stage2_label_e1e2 = pd.Series(label[stage1_label == 0]).map({4:0, 5:1}).to_numpy()

        stage2_data_cpd = data[stage1_label == 1]
        stage2_label_cpd = pd.Series(label[stage1_label == 1]).map({2:0, 8:1}).to_numpy()

        stage2_data_others = data[stage1_label == 2]
        stage2_label_others = pd.Series(dc(label[stage1_label == 2])).map({1: 0, 6: 1, 7: 2}).to_numpy()

        return EasyDict({'stage1':{'data':stage1_data, 'label': stage1_label},
                        'stage21':{'data':stage2_data_e1e2,'label':  stage2_label_e1e2},
                        'stage22':{'data':stage2_data_cpd, 'label':stage2_label_cpd},
                        'stage23':{'data':stage2_data_others, 'label':stage2_label_others}})
   
    else:
        raise RuntimeError('Param "multistage" is boolean - True or False.')

# ============================= Read input =============================

def read_signal(filename: str, signal_only = False) -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    # Read signals
    d = []
    source = filename.split('_')[0]
    readme = pd.read_csv(os.path.join(path,'data','readme.csv'),index_col='name')
    n_extension = readme.loc[source, 'recording_hour']
    extension = [f'.A0{n+1}' for n in range(n_extension)]
    
    # Read the signal (.A0x) files
    for i in range(len(extension)):
        file_path = os.path.join(path,'data', source, filename + extension[i])
        x = pd.read_csv(file_path,low_memory = False,delimiter=";",header = None, usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    # Read the analysis (.ANA) files
    if signal_only == False:
        ana_file_path = os.path.join(path,'data',f'{source}_ANA', filename + '.ANA')
        ana = pd.read_csv(ana_file_path, encoding='utf-16', delimiter = '\t',header = None, usecols=[0,1])
        ana.columns = ['label','time']
        ana = ana[(ana['label'] != 9) & (ana['label'] != 10) & (ana['label'] != 11)]
        ana.drop_duplicates(subset='time',inplace=True)
        ana.index = [i for i in range(len(ana))]
        return [data, ana.astype(int)]
    return data

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

def extract_sample(wave_array,ana,wave_type,which):
    '''
        Extract one waveform sample from the whole signal
    '''
    wave_indices = get_index(ana)
    start,end = wave_indices[wave_type][which]
    return wave_array[start:end]



# ============================= Checkpoint =============================
def save_checkpoint(models, config, name = None):
    n_models = len(models)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(f'checkpoints/{models[0].__type__}'):
        os.makedirs(f'checkpoints/{models[0].__type__}')
    dir = f'checkpoints/{models[0].__type__}'
    if n_models == 1:
        level = '1stage'
    elif n_models == 2:
        level = '2stage'
    for n in range(n_models):
        torch.save(models[n], dir + f'/{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth')
        print(f'Parameters saved! "{models[0].__type__}.{config.method}.{level}.{date.today()}.model{n+1}.pth".')




#============================ DRAFT  ============================
# def get_two_levels_split(data,label):
#     '''
#         Output: EasyDict of two level data and labels
#     '''
#     binary_label = []
#     for i in range(len(label)):
#         if label[i] == 5:
#             binary_label.append(1)
#         else:
#             binary_label.append(0)
#     binary_label = np.array(binary_label)

#     # Multilabels for level 2 classifier
#     none2_label = label[label != 5]
#     none2_label = pd.Series(none2_label).map({1: 0, 2: 1, 4: 2, 6: 3, 7: 4, 8: 5}).to_numpy()

#     return EasyDict({'level1':{'data':data, 'label': binary_label},
#                     'level2':{'data':data[label != 5],'label': none2_label}})       
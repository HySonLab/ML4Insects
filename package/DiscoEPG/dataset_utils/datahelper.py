import os
import pandas as pd 
import glob
from easydict import EasyDict 
from copy import deepcopy as dc

import torch
from datetime import date
import warnings
# ============================= Label map =============================
waveform_labels = ['NP', 'C', 'E1', 'E2', 'F', 'G', 'pd']
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

def read_signal(filename: str, data_path = '../data', dataset_name = None) -> tuple:
    '''
        Input: 
            filename: name of the recording
        Output:
            tuple of data signal and analysis dataframe 
    '''
    # Read signals
    d = []
    # dataset_name = filename.split('_')[0]
    dir = os.listdir(f'{data_path}/{dataset_name}')
    rec_components = glob.glob(f'{data_path}/{dataset_name}/{filename}.*')
    if len(rec_components) == 0:
        raise RuntimeError(f'No recording named {filename} found at {data_path}/{dataset_name}.')
    # Read the signal (.A0x) files
    for file_path in rec_components:
        # try: # Reading in binary format (*.D0x)
        fid = open(file_path, 'rb')
        fid.readline();  fid.readline();  fid.readline();
        x = pd.DataFrame(np.fromfile(fid, dtype=np.float32), columns=[1])
        fid.close()

        # except: # Reading in ASCII format (*.A0x)
        #     x = pd.read_csv(file_path, low_memory = False, delimiter=";", header = None, usecols=[1])
        d.append(x)
    data = pd.concat(d)
    data = data.to_numpy().squeeze(1)

    # Read the analysis (.ANA) files
    ana_path = f'{data_path}/{dataset_name}_ANA/{filename}.ANA'
    if os.path.exists(ana_path):
        try:
            ana = pd.read_csv(ana_path, encoding='utf-16', delimiter = '\t',header = None, usecols=[0,1])
        except:
            ana = pd.read_csv(ana_path, delimiter = '\t',header = None, usecols=[0,1])
        ana.columns = ['label','time']
        ana = ana[(ana['label'] != 9) & (ana['label'] != 10) & (ana['label'] != 11)]
        ana.drop_duplicates(subset='time',inplace=True)
        ana.index = [i for i in range(len(ana))]
        if (ana['time'].iloc[-1]) != (data.shape[0]/100.):
            len_ana = (ana['time'].iloc[-1])
            len_data = (data.shape[0]/100.)
            print("{} - Segmentation is given up to {:.2f} while the recoring's length is {:.2f}.".format(filename, len_ana, len_data))
    else:
        print(f'{filename} - No annotation (*.ANA) was found.')
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
                index['NP'].append([start,end])
            except:
                index['NP'] = [[start,end]]
        elif ana.loc[i,'label'] == 2:
            try:
                index['C'].append([start,end])
            except:
                index['C'] = [[start,end]]
        elif ana.loc[i,'label'] == 3:
            try:
                index['E1e'].append([start,end])
            except:
                index['E1e'] = [[start,end]]
        elif ana.loc[i,'label'] == 4:
            try:
                index['E1'].append([start,end])
            except:
                index['E1'] = [[start,end]]
        elif ana.loc[i,'label'] == 5:
            try:
                index['E2'].append([start,end])
            except:
                index['E2'] = [[start,end]]
        elif ana.loc[i,'label'] == 6:
            try:
                index['F'].append([start,end])
            except:
                index['F'] = [[start,end]]
        elif ana.loc[i,'label'] == 7:
            try:
                index['G'].append([start,end])
            except:
                index['G'] = [[start,end]]
        elif ana.loc[i,'label'] == 8:
            try:
                index['pd'].append([start,end])
            except:
                index['pd'] = [[start,end]]
        elif ana.loc[i,'label'] == 9:
            try:
                index['pd-S-II-2'].append([start,end])
            except:
                index['pd-S-II-2'] = [[start,end]]
        elif ana.loc[i,'label'] == 10:
            try:
                index['pd-S-II-3'].append([start,end])
            except:
                index['pd-S-II-3'] = [[start,end]]
        elif ana.loc[i,'label'] == 11:
            try:
                index['D'].append([start,end])
            except:
                index['D'] = [[start,end]]
        # elif ana.loc[i,'label'] == 11:
        #     try:
        #         index['pd-L'].append([start,end])
        #     except:
        #         index['pd-L'] = [[start,end]]
        # elif ana.loc[i,'label'] == 13:
        #     try:
        #         index['pd-L-II-2'].append([start,end])
        #     except:
        #         index['pd-L-II-2'] = [[start,end]]
        # elif ana.loc[i,'label'] == 14:
        #     try:
        #         index['pd-L-II-2'].append([start,end])
        #     except:
        #         index['pd-L-II-2'] = [[start,end]]
        
    return index 

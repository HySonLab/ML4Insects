import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew
from utils.stats import calculate_statistics
from sklearn.model_selection import train_test_split

from .datagenerator import generate_input_data

def calculate_features(df):
    d = []
    for i in range(len(df)):
        feats_vector = []
        coef = pywt.wavedec(df[i,:],'sym4',level = 3)
        for i in range(len(coef)):
            feats = calculate_statistics(coef[i])
            feats_vector += feats
        d.append(feats_vector)
    d = np.stack([f for f in d])
    return d

def compute_features_matrix(dataset_name, window_size = 1024, hop_length = 1024):

    d = generate_input_data(   dataset_name = dataset_name, 
                                    window_size = window_size, 
                                    hop_length = hop_length, 
                                    method = 'raw', 
                                    outlier_filter = False, 
                                    scale = True, 
                                    pad_and_slice = True, 
                                    verbose = False)
    data, labels = d['data'], d['label']

    # Read data
    X_train, y_train, X_test, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 28, stratify = labels)
    t0 = time.perf_counter() # Time counter

    # Compute features
    X_train = calculate_features(X_train)
    X_test = calculate_features(X_test)
    t1 = time.perf_counter() # Time counter

    # Save as .csv
    os.makedirs('./dataML', exist_ok = True)
    pd.DataFrame(y_train).to_csv(f'./dataML/Label_{dataset_name}_train.csv',header = None, index= None)
    pd.DataFrame(y_test).to_csv(f'./dataML/Label_{dataset_name}_test.csv',header = None, index= None)
    pd.DataFrame(X_train).to_csv(f'./dataML/Data_{dataset_name}_train.csv',header = None, index= None)
    pd.DataFrame(X_test).to_csv(f'./dataML/Data_{dataset_name}_test.csv',header = None, index= None)
    
    print(f'Dataset {dataset_name}. Elapsed computation time: {t1-t0}')

    f = open('./log/features_computation_time.txt', 'a')
    f.writelines([f'Dataset {dataset_name}. Elapsed features computation time: {t1-t0}\n'])

def read_dataset_csv(dataset_name, data_path = '../dataML'):
    columns = []
    for i in range(4):
        columns += [f'n5_{i}', f'n25_{i}', f'n75_{i}', f'n95_{i}', f'median_{i}', 
                    f'mean_{i}', f'std_{i}', f'var_{i}', f'rms_{i}', f'sk_{i}', f'zcr_{i}', f'en_{i}', f'perm_en_{i}']
    X_train = pd.read_csv(f'{data_path}/Data_{dataset_name}_train.csv',header = None)
    X_test = pd.read_csv(f'{data_path}/Data_{dataset_name}_test.csv',header = None)
    y_train =  pd.read_csv(f'{data_path}/Label_{dataset_name}_train.csv',header = None)
    y_test =  pd.read_csv(f'{data_path}/Label_{dataset_name}_test.csv',header = None)
    
    X_train.columns = columns
    X_test.columns = columns
    return X_train, X_test, y_train, y_test

def read_combined_dataset_csv():
    X_train_combined = []
    X_test_combined = []
    y_train_combined = []
    y_test_combined = []
    for i in range(1, 6):
        X_train, X_test, y_train, y_test = read_dataset_csv(dataset_name = i)
        X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
        X_train_combined.append(X_train)
        X_test_combined.append(X_test)
        y_train_combined.append(y_train)
        y_test_combined.append(y_test)
    X_train_combined = np.concatenate(X_train_combined)
    X_test_combined = np.concatenate(X_test_combined)
    y_train_combined = np.concatenate(y_train_combined)
    y_test_combined = np.concatenate(y_test_combined)
    return X_train_combined, X_test_combined, y_train_combined, y_test_combined

def read_dataset_from_config(config):
    if (config.dataset_name != 99) and (config.dataset_name != 'combined'):
        return read_dataset_csv(config.dataset_name)
    else: 
        return read_combined_dataset_csv()
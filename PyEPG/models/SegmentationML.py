from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew

from copy import deepcopy as dc
import matplotlib.pyplot as plt

import time
import datetime
from easydict import EasyDict
import os
from tqdm import tqdm 

from .Dataset import EPGDataset
from ..dataset_utils.datagenerator import generate_sliding_windows_single
from ..dataset_utils.datahelper import read_signal
from ..utils import metrics, visualization
from ..utils.stats import calculate_statistics

# ======================= MODEL ===========================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_MLmodel(name):
    if name == 'RF':
        return RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=28)
    elif name == 'XGB':
        return XGBClassifier(n_jobs = -1, n_estimators = 300, max_depth = 6, eta = 0.3)
    elif name == 'SVC':
        return SVC(class_weight = 'balanced', kernel = 'linear')
    elif name == 'LogReg':
        return LogisticRegression(n_jobs = -1, penalty= 'l1', solver = 'saga', warm_start=True)
    else:
        raise ValueError("Unsupported model. Must be one of ['XGB', 'RF', 'SVC', 'LogReg']")

# For Grid Search on XGB
XGB_params_grid = {'eta': [0.01, 0.1, 0.2, 0.3], 'n_estimators': [50,100, 200, 300],'max_depth': [3,4,5,6]}

class EPGSegmentML():
    def __init__(self, config, inference = False):

        # Dataset
        self.data_path = config.data_path 
        self.dataset_name = config.dataset_name
        self.dataset = EPGDataset(self.data_path, self.dataset_name)

        # Model/ optimizers
        self.config = config
        self.device = config.device        
        self.model = get_MLmodel(config.arch)

        # Configs for inputs
        self.window_size = config.window_size
        self.hop_length = config.hop_length
        self.scope = config.scope
        self.method = config.method 
        self.scale = config.scale 

        # Environment
        self.random_state = 28
        self._is_model_trained = False
        self._is_pretrained = False

        # Result
        self.classification_result_ = EasyDict({})

    def reset(self):
        self.model = get_MLmodel(config.arch)
        self.classification_result_ = EasyDict({})

    def get_traindata(self, test_size = 0.2):
        if os.path.exists(f'{self.data_path}/dataML/Label_{self.dataset_name}_train.csv'):
            print('Warning. Training data existed.')
            inp = input('Continue? (Y/N)')
            if inp == 'Y':
                pass
            elif inp == 'N':
                return 
        self.dataset.generate_sliding_windows(  window_size     = self.window_size, 
                                                hop_length      = self.hop_length, 
                                                scale           = self.scale, 
                                                method          = 'raw', 
                                                # outlier_filter = False, 
                                                # pad_and_slice = True, 
                                                verbose         = True)
        data, labels = self.dataset.windows, self.dataset.labels

        # Compute features
        t0 = time.perf_counter() # Time counter
        
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 28, stratify = labels)

        print('Computing training features matrices ...')
        X_train = calculate_features(X_train, method = self.method)

        print('Computing testing features matrices ...')
        X_test = calculate_features(X_test, method = self.method)

        t1 = time.perf_counter() # Time counter

        print(f'Dataset {self.dataset_name}. Elapsed computation time: {t1-t0}')

        # Save as .csv
        print('Saving datasets to .csv ...')
        os.makedirs(f'{self.data_path}/dataML', exist_ok = True)
        pd.DataFrame(y_train).to_csv(f'{self.data_path}/dataML/Label_{self.dataset_name}_train.csv',header = None, index= None)
        pd.DataFrame(y_test).to_csv(f'{self.data_path}/dataML/Label_{self.dataset_name}_test.csv',header = None, index= None)
        pd.DataFrame(X_train).to_csv(f'{self.data_path}/dataML/Data_{self.dataset_name}_train.csv',header = None, index= None)
        pd.DataFrame(X_test).to_csv(f'{self.data_path}/dataML/Data_{self.dataset_name}_test.csv',header = None, index= None)
        
        # f = open(f'{self.data_path}/log/features_computation_time.txt', 'w')
        # f.writelines([f'Dataset {self.dataset_name}. Elapsed features computation time: {t1-t0}\n'])
            
    def fit(self, X_train, y_train):
        
        self.X_train = X_train
        self.y_train = y_train

        print('Training...')
        t0 = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        t1 = time.perf_counter()
        training_time = round(t1 - t0, 3)
        self.classification_result_['training_time'] = training_time
        print(f'Finished training. Elapsed time: {training_time} (s)')

        self._is_model_trained = True 

    def predict(self, X_test, y_test):

        if self._is_model_trained == False:
            raise Warning('Model is not trained.')

        # Prediction
        print('Predicting ...')
        self.pred_proba = self.model.predict_proba(X_test)
        pred_windows = np.argmax(self.pred_proba, axis = -1)
        # Scoring
        results = metrics.scoring(y_test, pred_windows)
        scores = results['scores']
        cf = results['confusion_matrix']
        self.classification_result_['class_acc'] = [cf[i,i] for i in range(cf.shape[0])]
        self.classification_result_['test_scores'] = scores
        self.classification_result_['test_confusion_matrix'] = cf
        
        print(f"Accuracy: {scores['accuracy']}, f1: {scores['f1']}")
        print('Finished testing.')

    def cross_validate(self, cv = 10, verbose = 0, keep_best_model = True):
        scores = ['accuracy','f1_macro','precision_macro','recall_macro']
        self.cv_results = cross_validate(self.model, self.X_train, self.y_train, scoring = scores, cv=cv, n_jobs=-1,verbose = verbose, return_estimator= True)
        cv_scores = list(self.cv_results.keys())
        cv_scores.remove('estimator')
        summary = {}
        for key in cv_scores:
            summary[key] = [np.round(np.mean(self.cv_results[key]),2), np.round(np.std(self.cv_results[key]),2)]
        best_estimator_index = np.argmax(self.cv_results['test_accuracy'])
        best_estimator = self.cv_results['estimator'][best_estimator_index]
        if keep_best_model == True:
            self.model = best_estimator 
        self.cv_summary = summary

    def write_cv_log(self):

        date = str(datetime.datetime.now())[:-7]
        os.makedirs(f'log/{self.config.arch}', exist_ok = True)
        with open(f'log/{self.config.arch}/cv_results.txt','a') as f:
            f.writelines([f'Date: {date} | Model: {self.config.arch} | Dataset: {self.config.dataset_name}\n'])
            cv_summary_txt = [f'{key}: {self.cv_summary[key][0]} +- {self.cv_summary[key][1]}\n' for key in self.cv_summary.keys()]
            f.writelines(cv_summary_txt)

    def write_test_result(self):
        date = str(datetime.datetime.now())[:-7]
        os.makedirs(f'log/{self.config.arch}', exist_ok = True)        
        with open(f'log/{self.config.arch}/test_results.txt','a') as f:
            f.writelines([f'Date: {date} | Model: {self.config.arch} | Dataset: {self.config.dataset_name}\n'])
            f.writelines([f"{s}: {self.classification_result_['test_scores'][s]} \n" for s in self.classification_result_['test_scores'].keys()])  
            f.writelines(f'Class accuracy: {self.classification_result_["class_acc"]}\n')  

    def segment(self, recording_name, is_FS = False, return_score = True):

        # Prepare data
        print('Generating segmentation ...')
        self.recording_name = recording_name
        self.recording, self.ana = read_signal(recording_name)
        test_hop_length = self.hop_length // self.scope
        data = generate_sliding_windows_single(self.recording, self.ana, 
                                            window_size = self.window_size, 
                                            hop_length = test_hop_length, 
                                            method = 'raw', 
                                            scale = self.scale, 
                                            task = 'test')
        if self.ana is not None:
            self.input = calculate_features(data[0], method = self.config.method)
            self.true_segmentation = data[1]
        else:
            self.input = calculate_features(input, method = self.config.method)
        selected_features = [5,6,9,10,11,12,17,18,22,23,24,25,30,31,35,36,37,38,43,44,48,49,50,51]
        if is_FS == True:
            self.input = self.input[:, selected_features]
        # Predict
        

        self.pred_segm_proba = self.model.predict_proba(self.input)
        pred_windows = np.argmax(self.pred_segm_proba, axis = -1)

        # Generate segmentation
        pred_segmentation = []
        for i in range(len(pred_windows)):
            pred_segmentation.extend([pred_windows[i]]*test_hop_length)
        pred_segmentation = np.array(pred_segmentation)
        pred_segmentation = extend(pred_segmentation, self.true_segmentation)

        # Calculate overlapping rate
        accuracy = accuracy_score(self.true_segmentation, pred_segmentation) 
        self.overlap_rate = accuracy
        print(f"Overlapping rate: {accuracy}")

        # map to ground_truth labels  
        self.pred_segmentation = pd.Series(pred_segmentation).map({0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}).to_numpy() 
        self.pred_ana = to_ana(self.pred_segmentation) 
        
        if return_score == True:
            return self.pred_ana, self.overlap_rate
        else:
            return self.pred_ana

    def save_analysis(self, name: str = ''):

        os.makedirs('./prediction/ANA', exist_ok = True)
        dir = os.listdir('./prediction/ANA')
        index = len(dir) + 1
        if name == '':
            self.pred_ana.to_csv(f'./prediction/ANA/{self.recording_name}.ANA',sep = '\t',header = None,index=None)
        else:
            self.pred_ana.to_csv(f'./prediction/ANA/{name}.ANA',sep = '\t',header = None,index=None)

    def plot_pred_proba(self, r: tuple = None, ax = None):
        visualization.plot_pred_proba(self.pred_segm_proba, self.config.hop_length, self.config.scope, r, ax)

    def plot_segmentation(self, which = 'pred_vs_gt', savefig = False): 
        visualization.plot_gt_vs_pred_segmentation(self.recording, self.ana, self.pred_ana, which, savefig, name = self.recording_name)
        
    def interactive_plot(self, which = 'prediction', smoothen = False):
        if which == 'prediction':
            visualization.interactive_visualization(self.wave_array, self.predicted_analysis, smoothen, title = which)
        elif which == 'ground_truth':
           visualization.interactive_visualization(self.wave_array, self.ana, smoothen, title = which)
        else:
            raise RuntimeError("Must input either 'prediction' or 'ground_truth' ")

### Some utilities function      ##########################
def calculate_features(df, method):
    d = []
    for i in tqdm(range(len(df))):
        feats_vector = []
        if method == 'wavelet':
            coef = pywt.wavedec(df[i,:],'sym4',level = 3)
            for i in range(len(coef)):
                feats = calculate_statistics(coef[i])
                feats_vector += feats
        elif method == 'fft':
            n_fft = df[i,:].shape[0]
            coef = (np.abs(librosa.stft(df[i, :],n_fft=n_fft,center=False))/np.sqrt(n_fft)).ravel()
            feats_vector += calculate_statistics(coef)
        elif method == 'raw':
            feats_vector += calculate_statistics(df[i,:])
        else: 
            raise ValueError(f"Undefined method {str(method)}. Must be 'wavelet','fft' or 'raw'.")
        d.append(feats_vector)
    d = np.stack([f for f in d])
    return d

def to_ana(segmentation):
    # create *.ANA file
    ana_time = [0]
    ana_label = []
    tmp_label = segmentation[0]
    for i in range(1, len(segmentation)):
        if segmentation[i] != segmentation[i-1]: 
            ana_label.append(tmp_label)
            ana_time.append(i/100)
            tmp_label = segmentation[i]
    ana_label += [segmentation[i], 99]
    ana_time += [28800]
    pred_ana = pd.DataFrame({'time': ana_time, 'label': ana_label})
    
    return pred_ana

def repeat(array, coef):
    shape = array.shape 
    if len(shape) == 3:
        if shape[1] == 1:
            tmp = array.reshape(([shape[0],shape[2]]))
    n = len(array)
    repeated = []
    for i in range(n):
        repeated.extend([array[i]] * coef)
    return np.array(repeated)

def extend(arr1, target):
    '''
        Repeat the last axis of arr1 until 
        as long as target array (or reach target length)
    '''
    if isinstance(target, np.ndarray):
        ext_len = len(target) - len(arr1)
    else: 
        ext_len = target - len(arr1)
    if ext_len < 0:
        raise RuntimeError('Cannot repeat. Target length is less than input length')
    if len(arr1.shape) == 1: 
        extension = np.repeat(arr1[-1], ext_len)
    else:
        extension = np.tile(arr1[-1,:], (ext_len,1))
    return np.concatenate([arr1, extension])

def read_dataset_csv(dataset_name, data_path):
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
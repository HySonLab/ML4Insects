from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from utils.metrics import scoring

from copy import deepcopy as dc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from easydict import EasyDict
import datetime 
import os

from dataset_utils.MLdatagenerator import calculate_features
from dataset_utils.datahelper import read_signal
from dataset_utils.datagenerator import generate_sliding_windows
from utils import visualization

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

class EPGS_ML():
    def __init__(self, config, task = 'train', random_state = 28):

        self.config = config
        self.model = get_MLmodel(config.model)
        self.dataset_name = config.dataset_name 
        
        self.result_ = EasyDict({})
        self.task = task 

        self.random_state = random_state
        self._is_model_trained = False
        self._is_pretrained = False

    def reset(self):
        self.model = dc(self.model_copy)
        self.result_ = EasyDict({})
    
    def fit(self, X_train, y_train):
        
        self.X_train = X_train
        self.y_train = y_train

        print('Training...')
        t0 = time.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        t1 = time.perf_counter()
        training_time = round(t1 - t0, 3)
        self.result_['training_time'] = training_time
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
        results = scoring(y_test, pred_windows)
        scores = results['scores']
        cf = results['confusion_matrix']
        self.result_['class_acc'] = [cf[i,i] for i in range(cf.shape[0])]
        self.result_['test_scores'] = scores
        self.result_['test_confusion_matrix'] = cf
        
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
        os.makedirs(f'log/{self.config.model}', exist_ok = True)
        with open(f'log/{self.config.model}/cv_results.txt','a') as f:
            f.writelines([f'Date: {date} | Model: {self.config.model} | Dataset: {self.config.dataset_name}\n'])
            cv_summary_txt = [f'{key}: {self.cv_summary[key][0]} +- {self.cv_summary[key][1]}\n' for key in self.cv_summary.keys()]
            f.writelines(cv_summary_txt)

    def write_test_result(self):
        date = str(datetime.datetime.now())[:-7]
        os.makedirs(f'log/{self.config.model}', exist_ok = True)        
        with open(f'log/{self.config.model}/test_results.txt','a') as f:
            f.writelines([f'Date: {date} | Model: {self.config.model} | Dataset: {self.config.dataset_name}\n'])
            f.writelines([f"{s}: {self.result_['test_scores'][s]} \n" for s in self.result_['test_scores'].keys()])  
            f.writelines(f'Class accuracy: {self.result_["class_acc"]}\n')  

    def segment(self, recording_name, return_score = True, verbose = False):

        # Prepare data
        print('Preparing data...') if verbose == True else None
        self.recording_name = recording_name
        self.recording, self.ana = read_signal(recording_name)
        test_hop_length = self.config.hop_length // self.config.scope
        data = generate_sliding_windows(self.recording, self.ana, 
                                            window_size = self.config.window_size, 
                                            hop_length = test_hop_length, 
                                            method = self.config.method, 
                                            scale = self.config.scale, 
                                            task = 'test')
        if self.ana is not None:
            self.input = calculate_features(data[0])
            self.true_segmentation = data[1]
        else:
            self.input = calculate_features(input)

        # Predict
        print('Generating segmentation ...') if verbose == True else None

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
        self.overlapping_rate = accuracy
        # self.scores['top-2_accuracy'] = top_k_accuracy(self.pred_segm_proba, self.true_segmentation)
        print(f"Overlapping rate: {accuracy}")
        print('Finished.') if verbose == True else None

        # map to ground_truth labels  
        self.pred_segmentation = pd.Series(pred_segmentation).map({0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}).to_numpy() 
        self.pred_ana = to_ana(self.pred_segmentation) 

        return self.pred_ana, self.overlapping_rate

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

# def aggregate_to_analysis():
#     print('Aggregating predictions...') if verbose == True else None

#     # Write results in form of analysis files
#     n_windows = len(pred_windows)//self.config.scope
#     time = [0] # Make time marks
#     for i in range(n_windows):
#         time.append((self.config.window_size+i*self.config.scope*self.hop_length)/100)

#     agg_pred = [] # aggregating consecutive predictions
#     for i in range(n_windows):
#         cl,c = np.unique(pred_windows[i*self.config.scope:(i+1)*self.config.scope], return_counts=True)
#         agg_label = cl[np.argmax(c)]
#         agg_pred.append(agg_label)

#     pred_windows = np.append([agg_pred[0]], agg_pred) # merge the predicted labels

#     ana_label = [] # analysis file
#     ana_time = [time[0]]

#     pin = 0 # Merge intervals
#     for i in range(n_windows):
#         if pred_windows[i] != pred_windows[pin]:
#             ana_label.append(pred_windows[pin])
#             ana_time.append(time[i])
#             pin = i

#     ana_label.append(pred_windows[n_windows-1])

#     ana_time.append(time[i])
#     ana_label += [12]

#     self.predicted_analysis = pd.DataFrame({'label':ana_label,'time':ana_time})


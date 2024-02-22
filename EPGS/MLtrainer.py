from sklearn.model_selection import train_test_split, cross_validate
from utils import scoring
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle 
import time as t
from dataloader.MLDataGenerator import get_feature_matrix
from easydict import EasyDict

import datetime 
import os

# ======================= MODEL ===========================
class MLTrainer():
    def __init__(self, clf):
        self.model = clf
        self.model_copy = dc(self.model)
        self.results_ = EasyDict({})
        
    def reset(self):
        self.model = dc(self.model_copy)
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train(self, verbose = False):
        # Training model
        print('Training...') if verbose == True else None
        start = t.perf_counter()
        self.model.fit(self.X_train, self.y_train)
        end = t.perf_counter()
        training_time = round(end-start,3)
        self.results_['training_time'] = training_time
        print('Training time:', training_time) if verbose == True else None

    def predict(self, X_test, y_test, verbose = False):
        # Prediction
        print('Predicting ...') if verbose == True else None
        y_pred = self.model.predict(X_test)

        # Scoring
        results = scoring(y_test, y_pred)
        self.results_['class_acc'] = [round(self.cf[i,i],2) for i in range(self.cf.shape[0])]
        self.results_['test_scores'] = results['scores']
        self.results_['test_confusion_matrix'] = results['confusion_matrix']
        print('Finished testing.') if verbose == True else None
    
    def cross_validate(self, cv = 10, verbose = 0):
        scores = ['accuracy','f1_macro','precision_macro','recall_macro']
        self.cv_results = cross_validate(self.model, self.X_train, self.y_train, scoring = scores, cv=cv, n_jobs=-1,verbose = verbose, return_estimator= True)
        cv_scores = list(self.cv_results.keys())
        cv_scores.remove('estimator')
        summary = {}
        for key in cv_scores:
            summary[key] = [np.round(np.mean(self.cv_results[key]),2), np.round(np.std(self.cv_results[key]),2)]
        best_estimator_index = np.argmax(self.cv_results['test_accuracy'])
        cv_estimator = self.cv_results['estimator'][best_estimator_index]
        self.model = cv_estimator
        self.cv_summary = summary

    def write_log(self, description):
        path = os.getcwd()
        date = str(datetime.datetime.now())[:-7]
        if not os.path.exists('log/ML'):
            os.makedirs('log/ML')
        with open(os.path.join(path,'log/ML',f'session_result.txt'),'a') as f:
            f.writelines([
                        f'\n======================================================================================\n',
                        f'Date: {date} | Description: {description}\n' 
                        f'===> CV result \n'
                        ]
                        + [f'{key}: {self.cv_summary[key][0]} +- {self.cv_summary[key][1]}\n' for key in self.cv_summary.keys()]
                        )
            f.writelines([f'====> Test result \n'] + [f"{s}: {self.results_['test_scores'][s]}    " for s in self.results_['test_scores'].keys()])  
            f.writelines(f'Class accuracy: {self.results_["class_acc"]}\n')  

    # def plot_result(self, savefig = True, name = ''):
    #     # Learning curves
    #     train_loss = self.result['training_loss']
    #     val_loss = self.result['validation_loss']
    #     val_acc = self.result['validation_accuracy']

    #     # test scores
    #     test_score = self.result['test_score']
    #     scores = list(self.result['test_score'].keys())
        
    #     # Confusion matrices
    #     cf = dc(self.result['test_confusion_matrix'])
    #     cf = cf.astype(float)
    #     n_preds = np.sum(cf, axis = 0)
    #     for col in range(cf.shape[0]):
    #         for row in range(cf.shape[0]):
    #             try:
    #                 cf[row, col] = round(cf[row, col]/n_preds[col],2)
    #             except:
    #                 cf[row, col] = 0

    #     f, ax = plt.subplots(1,3,figsize = (12,4))
    #     ax[0].plot(train_loss,'r',label = 'train loss')
    #     ax[0].plot(val_loss,'b', label = 'validation_loss')
    #     ax[0].plot(val_acc,'b--', label = 'validation_accuracy')
    #     ax[0].set_xlabel('Epoch')
    #     ax[0].set_title('Traininng loss and validation accuracy')
    #     ax[0].grid()
    #     ax[0].legend()

    #     w = 0.3
    #     h = [test_score[k][0] for k in scores]
    #     plt.bar
    #     ax[1].bar(np.arange(len(scores)), h, width = w)
    #     ax[1].set_xticks(np.arange(len(scores)),scores)
    #     ax[1].set_title('Test accuracy and macro scores')
    #     ax[1].set_ylim(0,1)

    #     sns.heatmap(cf, ax = ax[2], annot= True, cmap = 'YlGn')
    #     ax[2].set_title('Confusion matrix')     
    #     ax[2].xaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
    #     ax[2].yaxis.set_ticklabels(list(utils.reverse_labels_dict.values()))
    #     plt.tight_layout()
    #     # if savefig == True:
    #     #     if not os.path.exists('log'):
    #     #         os.makedirs('log')
    #     #     d = str(datetime.date.today())
    #     #     p = os.path.join(os.getcwd(), 'log', f'{self.model.__type__}_{d}_{name}')
    #     #     plt.savefig(p)
    
def save(model,path):
    # save the model to disk
    pickle.dump(model, open(path, 'wb'))
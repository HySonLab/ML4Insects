from dataloader import datahelper, dataloader, datagenerator
from utils import doc_utils, metrics

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold

import os 
from os.path import join
import datetime
from easydict import EasyDict

class Trainer():
    def __init__(self, model, config = None, random_state = 28):
        # Trainer infos
        self.random_state = random_state
        self.device = config.device 
        self.model = model.to(self.device)
        self.config = config
        
        self.result_ = {'training_loss': [], 'validation_loss': [], 'validation_accuracy': [], 
                       'test_class_accuracy': [], 'test_score': [], 'test_confusion_matrix': []}

        if self.model.init_weights == False:
            self.initialize_weights()
      
        # Control variable
        self.data_loader_available = False
        self.data_available = False
        self.config_loaded = True if self.config is not None else False
        self.fit_config()   
        print(f'Time: {self.config.timestamp} | Architecture: {self.model.__arch__} | Version: {self.model.__version__} | Device: {self.device}')  
        
    def initialize_weights(self, nonlinearity = 'relu'):
        torch.manual_seed(self.random_state)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                layer.weight = nn.init.kaiming_normal_(layer.weight, nonlinearity= nonlinearity)
                layer.bias = nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.weight.data.fill_(1.0)
                layer.bias.data.fill_(0.0)
            
    def fit_config(self, config = None):
        if config is not None:
            self.config = config 
        self.criterion = nn.NLLLoss().to(self.device)
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config.lr)
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.lr)
        else: 
            raise RuntimeError("Missing params 'optimizer' in input config.")
        
    def fit_data(self, train, test):
        '''
            train: train data
            test:  test data
        '''
        self.X = train
        self.y = test
        self.data_available = True

    def generate_data(self, names, ratio = 0.8, random_state: int = 10):
        splits = doc_utils.get_train_test_filenames(ratio, random_state = random_state)
        train, test = datagenerator.generate_data(data_names = names, data_splits = splits, config = self.config, verbose = True)
        self.fit_data(train, test)
        
    def get_loader(self, r = 0.1, random_state = 28):
        print('Obtaning data loaders...')
        self.train_loader, self.validation_loader, self.test_loader = dataloader.get_loader(self.X, self.y, r = r, 
                                                                                 batch_size=self.config.batch_size, 
                                                                                 model_type = self.model.__type__,
                                                                                 random_state = random_state)
        self.data_loader_available = True

    def train_one_epoch(self):
        self.model.train()
        trainingloss = 0
        for x_batch,y_batch in self.train_loader:    
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model(x_batch)
            loss = self.criterion(output,y_batch.ravel())
            trainingloss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        trainingloss = trainingloss/(len(self.train_loader))
        self.result_['training_loss'].append(trainingloss)

    def validate_one_epoch(self):
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x_batch,y_batch in self.validation_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                loss = self.criterion(output,y_batch.ravel())
                validation_loss += loss.item()
                _,predicted = torch.max(output,1)

                n_samples += y_batch.size(0)
                n_correct += (predicted == y_batch.ravel()).sum().item()
        validation_loss = validation_loss/len(self.validation_loader)
        self.result_['validation_loss'].append(validation_loss)
        self.result_['validation_accuracy'].append(n_correct/n_samples)

    def test(self, verbose = True):
        print('Testing...') if verbose == True else None
        self.model.eval()
        with torch.no_grad():
            predicted_label = []
            for x_batch,y_batch in self.test_loader:
                
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                _,predicted = torch.max(output,1)

                predicted_label.append(predicted.cpu().numpy())
        
        predicted_label = np.concatenate([p for p in predicted_label])
        self.predicted_label = predicted_label

        true_label = self.y.label
        results = metrics.scoring(true_label, predicted_label)
        scores = results['scores']
        c = results['confusion_matrix']
        self.result_['test_score'] = results['scores']
        self.result_['test_confusion_matrix'] = c
        self.result_['test_class_accuracy'] = [c[i,i] for i in range(c.shape[0])]

        if verbose == True:
            print(f'Accuracy : {scores.accuracy}, Average f1: {scores.f1}') 
            print(f'Class accuracy: {self.result_["test_class_accuracy"]}')
            print('Finished testing!')

    def train(self, early_stop = True, patience = 5, min_delta = 0.01, verbose = True):

        if self.data_available == False:
            all_dataset_names = doc_utils.get_dataset_group(self.config.dataset_name)
            self.generate_data(all_dataset_names)
        if self.data_loader_available == False:
            self.get_loader()

        early_stopper = EarlyStopper(patience=patience, min_delta = min_delta) if early_stop == True else None
        print('Training...') if verbose == True else None
        for epoch in range(self.config.n_epochs):
            self.train_one_epoch()
            self.validate_one_epoch()
            if verbose == True:
                if (epoch %10 == 0) or (epoch == self.config.n_epochs - 1):
                    str1 = f"Epoch [{epoch+1}/{self.config.n_epochs}]"
                    str2 = f"Train loss: {self.result_['training_loss'][epoch]:.4f}"
                    str3 = f"Validation loss: {self.result_['validation_loss'][epoch]:.4f}"
                    str4 = f"Validation accuracy: {self.result_['validation_accuracy'][epoch]:.4f}"
                    msg = str1 + ' | ' + str2 + ' | ' + str3 + ' | ' + str4
                    print(msg) 
            if early_stop == True:
                if early_stopper.early_stop(self.result_['validation_loss'][-1]):      
                    print(f'Early stopped at epoch {epoch+1+patience} after {patience} epochs of changes less than {min_delta}. Validation loss: {self.result_["validation_loss"][-1]:.4f}')       
                    break        
        print('Finished training!') if verbose == True else None

    def reset(self):
        self.initialize_weights()
        self.result_ = {'training_loss': [], 'validation_loss': [],'validation_accuracy': [], 
                       'test_class_accuracy': [], 'test_score': [], 'test_confusion_matrix': []}
        
    def write_log(self, description):
        
        if not os.path.exists(f'./log/{self.model.__arch__}'):
            os.makedirs(f'./log/{self.model.__arch__}')
        columns = ['Date','Description', 'Version', 'Optimizer', 'Device', '#Epochs', 'Learning_rate', 'Batch_size'] \
                            + ['Train_loss', 'Val_loss', 'Val_acc', 'Test_acc', 'Test_f1', 'Test_precision', 'Test_recall'] \
                            + ['np_acc', 'c_acc', 'e1_acc', 'e2_acc', 'f_acc', 'g_acc', 'pd_acc']
        # Write scoring results in a .csv file
        session_result_path = f'./log/{self.model.__arch__}/session_result.csv'
        if os.path.exists(session_result_path):
            f = pd.read_csv(session_result_path, index_col = [0])
        else:
            f = pd.DataFrame(columns=columns)

        date = str(datetime.datetime.now())[:-7]
        infos = [date, self.config.exp_name, self.model.__version__, self.config.optimizer, self.device, self.config.n_epochs, self.config.lr, self.config.batch_size]
        train_loss = self.result_['training_loss'][-1]
        val_loss = self.result_['validation_loss'][-1]
        val_acc = self.result_['validation_accuracy'][-1]
        results = [train_loss, val_loss, val_acc] + list(self.result_['test_score'].values())

        class_acc = self.result_['test_class_accuracy']
        f = pd.concat([f, pd.DataFrame([infos+results+class_acc], columns = columns)])
        f.to_csv(session_result_path)

        # Write training log in a .txt file

        with open(os.path.join(self.config.wd,'log',f'{self.model.__arch__}', 'session_log.txt'),'a') as f:

            f.writelines([
                        f'======================================================================================\n',
                        f'Date: {date} | Description: {description} | Model version: {self.model.__version__}\n',
                        f'Optimizer: {self.config.optimizer} | Device: {self.device} | Epochs: {self.config.n_epochs} | Learning rate: {self.config.lr} | Batch size: {self.config.batch_size}\n',
                        f"Training loss: {' '.join([str(num) for num in np.round(self.result_['training_loss'],2)])}\n",
                        f"Validation loss: {' '.join([str(num) for num in np.round(self.result_['validation_loss'],2)])}\n",
                        f"Validation accuracy: {' '.join([str(num) for num in np.round(self.result_['validation_accuracy'],2)])}\n",
                        ])  
                            
    def plot_result(self, savefig = True, name = ''):
        # Learning curves
        train_loss = self.result_['training_loss']
        val_loss = self.result_['validation_loss']
        val_acc = self.result_['validation_accuracy']

        # test scores
        test_score = self.result_['test_score']
        scores = list(self.result_['test_score'].keys())
        
        # Confusion matrices
        cf = self.result_['test_confusion_matrix']

        f, ax = plt.subplots(1,3,figsize = (12,4))
        ax[0].plot(train_loss,'r',label = 'train loss')
        ax[0].plot(val_loss,'b', label = 'validation_loss')
        ax[0].plot(val_acc,'b--', label = 'validation_accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_title('Training loss and validation accuracy')
        ax[0].grid()
        ax[0].legend()

        w = 0.3
        h = [test_score[k] for k in scores]
        plt.bar
        ax[1].bar(np.arange(len(scores)), h, width = w)
        ax[1].set_xticks(np.arange(len(scores)),scores)
        ax[1].set_title('Test accuracy and weighted scores')
        ax[1].set_ylim(0,1)

        sns.heatmap(cf, ax = ax[2], annot= True, cmap = 'YlGn')
        ax[2].set_title('Confusion matrix')     
        ax[2].set_xlabel('Predicted label')
        ax[2].set_ylabel('True label')
        plt.suptitle(name)
        plt.tight_layout()

        if savefig == True:
            log_path = f'./log/{self.model.__arch__}'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            d = str(datetime.date.today())
            p = os.path.join(log_path, f'{name}_{self.model.__arch__}_{d}')
            plt.savefig(p)
        
    def save_checkpoint(self, description):
        # description: Write some description about the data used or simply an empty string
        archi = self.model.__arch__
        date = str(datetime.date.today())
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists(f'checkpoints/{archi}'):
            os.makedirs(f'checkpoints/{archi}')
        saved_name = f'arch-{self.model.__arch__}.window-{self.config.window_size}.'
        saved_name += f'method-{self.config.method}.scale-{self.config.scale}.optimizer-{self.config.optimizer}.'
        saved_name += f'epochs-{self.config.n_epochs}.lr-{self.config.lr}.batchsize-{self.config.batch_size}.'
        dir = f'checkpoints/{archi}/' + saved_name + date + '.' + description + '.json'
        torch.save(self.model, dir)
        print(f'Parameters saved to "{dir}".')

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if np.abs(validation_loss - self.min_validation_loss) > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False       


def calculate_statistics(array):
    max = np.max(array)
    min = np.min(array)
    mean = np.mean(array)
    sd = np.std(array)
    return np.array([max, min, mean, sd])

# ========================================================================================================================
class cross_validate():
    def __init__(self, model, config, n_folds):
        self.trainer = Trainer(model,config)
        self.n_folds = n_folds
        self.data_available = False 
        
    def CV(self, data_names, verbose = True):
        random_state = [10*n for n in range(self.n_folds)]
        k_fold_result_ = []
        for i in range(self.n_folds):
            print(f'=================== Fold {i+1}, random_state {random_state[i]}===================') if verbose == True else None
            self.trainer.reset()
            self.trainer.generate_data(data_names, random_state = random_state[i])
            self.trainer.train(early_stop = False, verbose = verbose)
            self.trainer.test(verbose = verbose)
            k_fold_result_.append(self.trainer.result_)
        self.k_fold_result_ = k_fold_result_
    
    def summarize(self):
        index = ['max','min','mean','sd']
        summ = {}
        acc = [self.k_fold_result_[i]['test_score']['accuracy'] for i in range(self.n_folds)]
        b_acc = [self.k_fold_result_[i]['test_score']['recall'] for i in range(self.n_folds)]
        f1 = [self.k_fold_result_[i]['test_score']['f1'] for i in range(self.n_folds)]
        summ['accuracy'] = calculate_statistics(acc)
        summ['recall'] = calculate_statistics(b_acc)
        summ['f1'] = calculate_statistics(f1) 

        cl = datahelper.waveform_labels
        class_acc_summary = np.array([self.k_fold_result_[i]['test_class_accuracy'] for i in range(self.n_folds)])

        for i in range(len(cl)):
            class_acc = class_acc_summary[:,i]
            summ[cl[i]] = calculate_statistics(class_acc)
        summ = pd.DataFrame(summ,index = index)
        summ = summ.apply(lambda x: round(x,2))
        return summ

    def write_log(self):
        summary = self.summarize()
        date = str(datetime.datetime.now())[:-7]
        cl = datahelper.waveform_labels

        with open('./log/kfold_log.txt','a') as f:
            f.writelines([
                        f'======================================================================================\n'
                        f'Date: {date} | Version: {self.trainer.model.__version__} | Device:{self.trainer.device} | Description: {self.trainer.config.dataset_name}' +
                        f' | #Epochs: {self.trainer.config.n_epochs} | Learning rate: {self.trainer.config.lr} | Batch size: {self.trainer.config.batch_size}\n',
                        f'{self.n_folds}-fold max/min/mean/sd\n',
                        f"{summary}\n"
                        ])


    def plot_summary(self, savefig = False, description = ''): #Trainer.result has train loss, valid acc, test score, test class accuracy and test confusion matrix
        # Learning curves
        train_loss = np.stack([self.k_fold_result_[i]['training_loss'] for i in range(self.n_folds)])
        train_loss_mean = np.mean(train_loss, axis = 0)
        train_loss_sd = np.std(train_loss, axis = 0)

        val_acc = np.stack([self.k_fold_result_[i]['validation_accuracy'] for i in range(self.n_folds)])
        val_acc_mean = np.mean(val_acc, axis = 0)
        val_acc_sd = np.std(val_acc, axis = 0)

        val_loss = np.stack([self.k_fold_result_[i]['validation_loss'] for i in range(self.n_folds)])
        val_loss_mean = np.mean(val_loss, axis = 0)
        val_loss_sd = np.std(val_loss, axis = 0)

        # Test scores
        scores = pd.concat([pd.DataFrame(self.k_fold_result_[i]['test_score'], index = [i]) for i in range(self.n_folds)]).to_numpy()
        # class accuracy
        class_accuracy = np.array([self.k_fold_result_[i]['test_class_accuracy'] for i in range(self.n_folds)])
        
        # Sum of confusion matrix
        sum_cf = np.zeros(self.k_fold_result_[0]['test_confusion_matrix'].shape)
        for i in range(self.n_folds):
            sum_cf +=self.k_fold_result_[i]['test_confusion_matrix']

        sum_cf = sum_cf.astype(float)
        n_preds = np.sum(sum_cf, axis = 0)
        for col in range(sum_cf.shape[0]):
            for row in range(sum_cf.shape[0]):
                try:
                    sum_cf[row, col] = round(sum_cf[row, col]/n_preds[col],2)
                except:
                    sum_cf[row, col] = 0

        f, ax = plt.subplots(1,4,figsize = (20,4))
        ax[0].set_title('Loss & validation accuracy')
        ax[0].fill_between(np.arange(len(train_loss_mean)), train_loss_mean - train_loss_sd, train_loss_mean + train_loss_sd,alpha = 0.2, color = 'r')
        ax[0].fill_between(np.arange(len(val_loss_mean)), val_loss_mean - val_loss_sd, val_loss_mean + val_loss_sd,alpha = 0.2, color = 'b')
        ax[0].fill_between(np.arange(len(val_acc_mean)), val_acc_mean - val_acc_sd, val_acc_mean + val_acc_sd,alpha = 0.2, color = 'm')
        ax[0].plot(train_loss_mean, 'r', label = 'training_loss')
        ax[0].plot(val_loss_mean, 'b', label = 'validation_loss')
        ax[0].plot(val_acc_mean, 'm--', label = 'validation_accuracy')
        ax[0].grid()
        ax[0].legend()

        ax[1].boxplot(scores)
        ax[1].set_xticks(np.arange(1,5),['Accuracy','f1','precision','recall'])
        ax[1].set_title('Test scores')
        ax[1].set_ylim(0.4,1)

        ax[2].boxplot(class_accuracy)
        ax[2].set_xticks(np.arange(1,8), datahelper.waveform_labels)
        ax[2].set_title('Class accuracy')
        ax[2].set_ylim(0.4,1) 
        
        sns.heatmap(sum_cf, ax = ax[3], annot = True, cmap = 'YlGn', cbar = False, 
                    xticklabels= datahelper.waveform_labels, yticklabels= datahelper.waveform_labels)        
        ax[3].set_title('Confusion matrix')     
        ax[3].set_xlabel('Predicted label')
        ax[3].set_ylabel('True label')

        if savefig == True:
            if not os.path.exists('log'):
                os.makedirs('log')
            d = str(datetime.date.today())
            p = os.path.join(os.getcwd(), 'log', f'{self.model.__arch__}_f"{self.n_folds}folds"_{d}_{description}')
            plt.savefig(p)


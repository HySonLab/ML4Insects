import torch 
import torch.nn as nn 
from .NN import *

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import time
import datetime 
import os 
from tqdm import tqdm 

from .Dataset import EPGDataset
from ..utils import utils, metrics, visualization
from ..dataset_utils import datagenerator, dataloader, datahelper



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(config):

    if config.method == 'raw':
        input_size = 1024
        in_channels = 1
    elif config.method == 'fft':
        input_size = 513
        in_channels = 1
    elif config.method == 'wavelet': 
        input_size = 515
        in_channels = 4
    if config.arch in ['mlp', 'cnn1d', 'resnet']:
        assert config.method in ['raw', 'fft', 'wavelet'], f'Unexpected feature of type {config.method} for model of type {config.arch}'
    if config.arch == 'mlp':
        return MLP(input_size = input_size)
    elif config.arch == 'cnn1d':
        return CNN1D(input_size = input_size)
    elif config.arch == 'resnet':
        return ResNet(input_size = input_size)
    elif config.arch == 'cnn2d':
        assert config.method in ['gaf', 'spectrogram', 'scalogram'], f'Unexpected feature of type {config.method} for model of type {config.arch}'
        if config.method == 'gaf':
            input_size = 64
        elif config.method == 'spectrogram':
            input_size = 65
        elif config.method == 'scalogram':
            input_size = 65
        return CNN2D(input_size = input_size)

class EPGSegment:
    def __init__(self, config, inference = False):
        self.config = config

        # Dataset
        if inference == False:
            print('Training mode.')
            self.data_path = config.data_path 
            self.dataset_name = config.dataset_name
            self.dataset = EPGDataset(self.data_path, self.dataset_name)
        else:
            print('Inference mode, skip loading data.')
        
        # Model/ optimizers
        self.device = device
        self.model = get_model(self.config).to(self.device)        
        self.lr = config.lr 
        self.batch_size = config.batch_size
        if config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = config.lr, momentum= 0.9, weight_decay= 0.001)   
        else:
            raise ValueError('Unsupported optimizer.')
        self.loss_fn = nn.NLLLoss().to(self.device)
        self.n_epochs = config.n_epochs

        # Configs for inputs
        self.window_size = config.window_size
        self.hop_length = config.hop_length
        self.scope = config.scope
        self.method = config.method 
        self.scale = config.scale 

        # Environment 
        self.random_state = 28
        self._is_model_trained = False 
        self._is_dataloaders_available = False

        # Result
        self.train_result_ = {  'training_loss': [], 
                                'training_accuracy': [], 
                                'validation_loss': [],
                                'validation_accuracy': [], 
                                'test_class_accuracy': [], 
                                'test_score': [], 
                                'test_confusion_matrix': [],
                                'training_time' :0, 
                                'data_processing_time':0, 
                                'per_epoch_training_time': []}

    def get_dataloaders(self, r = [0.7, 0.2, 0.1]):
        print('Obtaining dataloders ...')
        self.dataset.generate_sliding_windows(  window_size = self.window_size, 
                                                hop_length  = self.hop_length, 
                                                method      = self.config.method,
                                                scale       = self.scale,
                                                verbose     = True)
        data, labels = self.dataset.windows, self.dataset.labels
        if 'cnn' in self.model.__type__ :
            if self.config.method != 'wavelet':
                unsqueeze = True
        self.train_loader, self.val_loader, self.test_loader = dataloader.get_loaders(data, labels, 
                                                                                        r = [0.7, 0.2, 0.1], 
                                                                                        batch_size=self.batch_size, 
                                                                                        unsqueeze = unsqueeze,
                                                                                        random_state = self.random_state)
        self._is_dataloaders_available = True 
        
    #######################################
    ######### TRAINING ML MODELS ##########
    #######################################
    def train_epoch(self):
        self.model.train()
        t0 = time.perf_counter()
        training_loss = 0; n_correct = 0; n_samples = 0

        for x_batch,y_batch in self.train_loader:    
            
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            output = self.model(x_batch)
            predicted = torch.argmax(output.detach(),-1)
            n_samples += len(y_batch.ravel())
            n_correct += (predicted == y_batch).sum().item()     
            
            loss = self.loss_fn(output,y_batch.ravel())
            training_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            # _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
        training_loss = training_loss/(len(self.train_loader))
        accuracy = n_correct/n_samples
        self.train_result_['per_epoch_training_time'].append(time.perf_counter()-t0)
        self.train_result_['training_loss'].append(training_loss)
        self.train_result_['training_accuracy'].append(accuracy)
        return training_loss, accuracy 

    def evaluate(self, task, verbose = True):
        # For Task 1: Classifying an input segment of fixed size (default = 1024)
        self.model.eval()
        if task == 'validate':
            loader = self.val_loader 
        elif task == 'test': 
            loader = self.test_loader
        else:
            raise ValueError("task is either 'validate' or 'test'.")

        mean_loss = 0
        with torch.no_grad():
            pred_windows = []
            true_label = []
            t0 = time.perf_counter()
            for x_batch,y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                output = self.model(x_batch)
                loss = self.loss_fn(output,y_batch.ravel())
                mean_loss += loss.item()

                predicted = torch.argmax(output, -1)
                pred_windows.extend(predicted.ravel().cpu().numpy())
                true_label.extend(y_batch.ravel().cpu().numpy())

        mean_loss = mean_loss/len(loader)
        
        results = metrics.scoring(true_label, pred_windows)
        scores = results['scores']
        c = results['confusion_matrix']

        if task == 'validate': 
            self.train_result_['validation_loss'].append(mean_loss)
            self.train_result_['validation_accuracy'].append(scores['accuracy'])

        elif task == 'test':
            self.train_result_['test_score'] = scores
            self.train_result_['test_confusion_matrix'] = c
            self.train_result_['test_class_accuracy'] = [np.round(c[i,i],2) for i in range(c.shape[0])]
            self.pred_windows = pred_windows
            if verbose == True:
                print(f'Accuracy : {scores.accuracy}, Average f1: {scores.f1}') 
                print(f'Class accuracy: {self.train_result_["test_class_accuracy"]}')
                print('Finished testing!')
                
        return mean_loss, scores['accuracy']

    def train(self, early_stop = True, patience = 5, min_delta = 0.01, verbose = True):
        # Initialization 
        if self._is_dataloaders_available == False:
            self.get_dataloaders()
        early_stopper = EarlyStopper(patience=patience, min_delta = min_delta) if early_stop == True else None
        print('Training...') if verbose == True else None
        self.train_result_['early_stopping_epoch'] = self.n_epochs
        _is_early_stopped = False

        # Training loop
        t0 = time.perf_counter()
        for epoch in tqdm(range(self.n_epochs), desc = 'Training'):
            
            tr_loss, tr_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(task = 'validate')

            if verbose == True:
                if (epoch %10 == 0) or (epoch == self.n_epochs - 1):
                    print(f"Epoch [{epoch+1}/{self.n_epochs}] | Train loss: {tr_loss:.4f} | Val. loss: {val_loss:.4f} | Train acc: {tr_acc:.4f} | Val. acc: {val_acc:.4f}") 

            if early_stop == True:
                if early_stopper.early_stop(self.train_result_['validation_loss'][-1]):   
                    if _is_early_stopped == False:   
                        print(f'Early stopping occured at epoch {epoch+1+patience} after {patience} epochs of changes less than {min_delta} in validation accuracy. Validation loss: {self.train_result_["validation_loss"][-1]:.4f}')       
                        # Test and save the checkpoint at early stopped epochs
                        _, _ = self.evaluate(task = 'test')
                        self.save_checkpoint(f'early_stopped_{epoch}')
                        self.train_result_['early_stopping_epoch'] = epoch   
                        _is_early_stopped = True 

        self.train_result_['training_time'] = time.perf_counter() - t0       
        self._is_model_trained = True
        print('Finished training!') if verbose == True else None

    #######################################
    ########## EPG SEGMENTATION ###########
    #######################################

    def segment(self, recording_name, verbose = False):
        # For Task 2: Output an analysis file (segmentation)
        if (self._is_model_trained == False):
            raise Warning('Model is not trained.')
        self.recording_name = recording_name
        if isinstance(recording_name, str):
            self.recording, self.ana = self.dataset.loadRec(recName = recording_name)
            print(f'File name: {recording_name}')
        elif isinstance(recording_name, int):
            dat = self.dataset[recording_name]
            self.recording, self.ana = dat['recording'], dat['ana']
            print(f'File name: {dat["name"]}')
        if self.ana is None:
            print('Ground-truth annotation was not detected.')
        test_hop_length = self.window_size//self.scope

        self.model.eval()
        
        # Initial segmentation
        print('Preparing data...') if verbose == True else None
        data = datagenerator.generate_sliding_windows_single(self.recording, self.ana, 
                                                        window_size = self.window_size,
                                                        hop_length = test_hop_length, 
                                                        method = self.method, 
                                                        scale = self.scale, 
                                                        task = 'test')

        if self.ana is not None:
            self.input, self.true_segmentation = data
        else: 
            self.input = data
        # Reshape and convert to torch.tensor
        input = torch.from_numpy(self.input).float().to(self.device)
        if self.model.__type__ == 'cnn':
            if self.config.method != 'wavelet':
                input = input.unsqueeze(1)

        # Predict each segment
        print('Generating segmentation ...') if verbose == True else None
        out = []
        for i in range(input.shape[0]):
            out.append(self.model(input[i:(i+1)]).detach().cpu().numpy())
        
        self.log_pred_proba = np.array(out)
        pred_windows = np.argmax(out, axis=-1).ravel()

        # Generate segmentation
        pred_segmentation = []
        for i in range(len(pred_windows)):
            pred_segmentation.extend([pred_windows[i]]*test_hop_length)
        pred_segmentation = np.array(pred_segmentation)
        pred_segmentation = extend(pred_segmentation, self.true_segmentation)

        # Scoring
        if self.ana is not None: 
            self.overlap_rate = np.mean(self.true_segmentation == pred_segmentation)
            print(f'Overlap rate: {self.overlap_rate}') if verbose == True else None

        # map to ground_truth labels  
        self.pred_segmentation = pd.Series(pred_segmentation).map({0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}).to_numpy() 
        self.pred_ana = to_ana(self.pred_segmentation) 

        return self.pred_ana 

    #################################################
    ########## PLOT/SAVE RESULT UTILITIES ###########
    #################################################

    def save_analysis(self, name: str = ''):

        os.makedirs('./prediction/ANA', exist_ok = True)
        dir = os.listdir('./prediction/ANA')
        index = len(dir) + 1
        if name == '':
            save_name = f'./prediction/ANA/Untitled_{index}.ANA'
        else:
            save_name = f'./prediction/ANA/{name}.ANA'
        self.pred_ana.to_csv(save_name,sep = '\t',header = None,index=None)
        print(f'Analysis saved to {save_name}')      
    def write_train_log(self):
        utils.write_training_log(self.model, self.config, self.train_result_)
                            
    def plot_train_result(self, savefig = False):
        utils.plot_training_result(self.model, self.config, self.train_result_, savefig)
        
    def save_checkpoint(self, name = ''):
        date = str(datetime.date.today())
        os.makedirs(f'./checkpoints', exist_ok = True)

        if name == '':
            saved_name = f'arch-{self.model.__arch__}.version-{self.model.__version__}.window-{self.config.window_size}.'
            saved_name += f'method-{self.config.method}.scale-{self.config.scale}.optimizer-{self.config.optimizer}.'
            saved_name += f'epochs-{self.config.n_epochs}.lr-{self.config.lr}.batchsize-{self.config.batch_size}'
            save_path = f'./checkpoints/{self.model.__arch__}/{saved_name}.{date}.{self.config.exp_name}.json'
        else:
            save_path = f'./checkpoints/{self.model.__arch__}/{name}'
        torch.save(self.model, save_path)
        print(f'Parameters saved to {save_path}.')

    def plot_segmentation(self, which = 'pred_vs_gt', savefig = False, name: str = ''): 
        visualization.plot_gt_vs_pred_segmentation(self.recording, self.ana, self.pred_ana, which, savefig)
                
    def plot_interactive(self, which = 'prediction', smoothen = False):
        if which == 'prediction':
            visualization.interactive_visualization(self.recording, self.pred_ana, smoothen, title = which)
        elif which == 'ground_truth':
           visualization.interactive_visualization(self.recording, self.ana, smoothen, title = which)
        else:
            raise ValueError("Param which must be either 'prediction' or 'ground_truth'.")

    def reset(self):
        self._is_model_trained = False 
        self.train_result_ = {'training_loss': [], 'training_accuracy': [], 'validation_loss': [], 'validation_accuracy': [], 
                       'test_class_accuracy': [], 'test_score': [], 'test_confusion_matrix': [],
                       'training_time' :0, 'data_processing_time':0, 'per_epoch_training_time': []}
        self.model = get_model(self.config).to(self.device)       

    def load_checkpoint(self, path):
        try:
            self.model = torch.load(f'./checkpoints/{path}')
        except:
            self.model = torch.load(f'./checkpoints/{self.config.arch}/{path}')
        self._is_model_trained = True
        print('Checkpoint loaded.')
  
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
from utils import scoring
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataloader.datagenerator import generate_test_data
from MLDataGenerator import get_feature_matrix
import os


from easydict import EasyDict ### We should create a more systematic way for this
default_config = EasyDict({'window_size': 1024, 'scope': 4, 'method': 'raw'})

class MLClassifier():
    def __init__(self, clf, config = default_config):
        self.model = clf
        self.config = config
        self.hop_length = configs.window_size//configs.scope
        self.method = config.method
        print('Configurations loaded.')

    def process_input(self, wave_array, ana = None):
        # Prepare data
        print('Preparing data...')
        self.wave_array = wave_array
        self.ana = ana
        input = generate_test_data(wave_array, ana, window_size = self.config.window_size, 
                                                                    hop_length = self.hop_length, method = self.config.method)
        self.input = get_feature_matrix(input.data)
        self.true_label = input.label

    def predict(self, verbose = False):
        print('Predicting...') if verbose == True else None
        predicted_label = self.model.predict(self.input)

        #map to original labels  
        
        self.predicted_label = predicted_label 

        results = scoring(y_test, y_pred)
        self.cf = results['confusion_matrix']
        self.scores = results['scores']
        self.class_acc = [self.cf[i,i] for i in range(self.cf.shape[0])]
        
        print(f"Accuracy: {self.scores['accuracy']}, f1: {self.scores['f1']}")

        predicted_label = pd.Series(predicted_label).map({0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 5}).to_numpy() 
        print('Aggregating predictions...') if verbose == True else None
        # Write results in form of analysis files
        n_windows = len(predicted_label)//self.config.scope
        time = [0] # Make time marks
        for i in range(n_windows):
            time.append((self.config.window_size+i*self.config.scope*self.hop_length)/100)

        agg_pred = [] # aggregating consecutive predictions
        for i in range(n_windows):
            cl,c = np.unique(predicted_label[i*self.config.scope:(i+1)*self.config.scope], return_counts=True)
            agg_label = cl[np.argmax(c)]
            agg_pred.append(agg_label)

        predicted_label = np.append([agg_pred[0]], agg_pred) # merge the predicted labels

        ana_label = [] # analysis file
        ana_time = [time[0]]

        pin = 0 # Merge intervals
        for i in range(n_windows):
            if predicted_label[i] != predicted_label[pin]:
                ana_label.append(predicted_label[pin])
                ana_time.append(time[i])
                pin = i

        ana_label.append(predicted_label[n_windows-1])

        ana_time.append(time[i])
        ana_label += [12]

        self.predicted_analysis = pd.DataFrame({'label':ana_label,'time':ana_time})
        print('Finished.') if verbose == True else None
        return self.predicted_analysis
    

# def plot_result(result_dict):
#     _,((ax1,ax2,ax3,ax4)) = plt.subplots(1,4,figsize = (16,3),sharex = True,sharey = True)

#     model_name = ['DecTree','LR','SVC','RF','GB','Ada','XGB']
#     n_model = len(model_name)

#     xtick = np.arange(0,n_model)
#     w = 0.2

#     r = result_dict
#     ax1.bar(xtick+w*np.ones(n_model),r['Balanced_acc'],width = w)
#     ax2.bar(xtick+w*np.ones(n_model),r['f1'],width = w)
#     ax3.bar(xtick+w*np.ones(n_model),r['precision'],width = w)
#     ax4.bar(xtick+w*np.ones(n_model),r['recall'],width = w)

#     ax1.set_title('Balanced Accuracy')
#     ax2.set_title('f1')
#     ax3.set_title('Precision')
#     ax4.set_title('Recall')
#     ax1.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax2.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax3.set_xticks(ticks = xtick,labels = model_name,rotation = 30)
#     ax4.set_xticks(ticks = xtick,labels = model_name,rotation = 30)

    

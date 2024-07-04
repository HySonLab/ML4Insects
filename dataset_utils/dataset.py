import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

from dataset_utils.datahelper import get_dataset_group, read_signal, get_filename, get_index, extract_sample
from dataset_utils.datagenerator import generate_inputs
from utils.visualization import visualize_signal, interactive_visualization

class EPGDataset:
    def __init__(
                self, 
                data_path = '../data', 
                dataset_name = 'SA',
                ):

        self.data_path = data_path
        self.dataset_name = dataset_name
        self._format_recNames()

        self.recNames = os.listdir(f'{self.data_path}/{self.dataset_name}')
        self.recNames = set([x[:-4] for x in self.recNames])
        t = time.perf_counter()
        self.recordings = []
        print('Loading data ...')
        for id, recording_name in enumerate(tqdm(self.recNames)):
            recording, ana = read_signal(recording_name, data_path=self.data_path)
            self.recordings.append({'id': id,
                                    'name': recording_name,
                                    'recording': recording,
                                    'ana':ana    
                                    })
        print(f'Done! Elapsed: {time.perf_counter() - t} s')
        
    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        return self.recordings[idx]

    def _format_recNames(self):
        recNames = os.listdir(f'{self.data_path}/{self.dataset_name}')
        prefix = f'{self.dataset_name}_'
        for name in recNames:
            if not name.startswith(prefix):
                newname = f'{prefix}{name}'
                os.rename(f'{self.data_path}/{self.dataset_name}/{name}',f'{self.data_path}/{self.dataset_name}/{newname}')
        anaNames = os.listdir(f'{self.data_path}/{self.dataset_name}_ANA')
        for name in anaNames:
            if not name.startswith(prefix):
                newname = f'{prefix}{name}'
                os.rename(f'{self.data_path}/{self.dataset_name}_ANA/{name}',f'{self.data_path}/{self.dataset_name}_ANA/{newname}')
        
    def plot(self, idx, mode = 'static', smoothen = False):
        recording, ana = self.recordings[idx]['recording'], self.recordings[idx]['ana']
        plt.figure(figsize = (18,3))
        if mode == 'static':
            visualize_signal(recording, ana, title = self.recordings[idx]['name'])
        elif mode == 'interactive':
            interactive_visualization(recording, ana, smoothen= smoothen, title = self.recordings[idx]['name'])

    def generate_sliding_windows(self, window_size = 1024, hop_length = 1024, method = 'raw', scale = True):

        print('Generating sliding windows ...')
        d = generate_inputs(self.data_path, self.dataset_name, window_size, hop_length, method, verbose = True)
        self.windows, self.labels = d['data'], d['label']
        self.waveforms, self.distributions = np.unique(self.labels, return_counts= True)
        n = len(self.waveforms)
        self.distributions = [round(self.distributions[i]/len(self.labels),2) for i in range(n)]
        self.label_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}

    def plot_windows(self):
        pass     

    def getRecordingParams(self, idx, ana = None):
        recData = self.recordings[idx]
        recName = recData['name']
        recId = recData['id']
        recRecording = recData['recording']
        if ana is None:
            recAna = recData['ana']
            # print(recAna)
        else:
            recAna = ana
        waveformIndices = get_index(recAna)
        
        params = {}
        
        for k in waveformIndices.keys():
            duration = 0
            for start, end in waveformIndices[k]:
                duration += end - start
            count = len(waveformIndices[k])
            params[k] = [count, duration]
        
        params = pd.DataFrame(params, index = ['count', 'duration'])
        return params

    def datasetSummary(self):

        durations = {'np': [], 'c': [], 'e1': [], 'e2': [], 'f': [], 'g': [], 'pd': []}
        counts = {'np': 0, 'c': 0, 'e1': 0, 'e2': 0, 'f': 0, 'g': 0, 'pd': 0}
        total_length = 0
        n = len(self.recordings)
        
        for i in range(n):
            # print(self.recordings[i])
            recAna = self.recordings[i]['ana']
            waveform_intervals = get_index(recAna)
            for waveform in waveform_intervals.keys():
                for interval in waveform_intervals[waveform]:
                    start, end = interval
                    durations[waveform].append(end - start)
                    counts[waveform] += 1
            total_length += recAna.iloc[-1]['time']
        self.durations = durations
        
        # stats = {'np': [], 'c': [], 'e1': [], 'e2': [], 'f': [], 'g': [], 'pd': []}
        stats = []
        for waveform in durations.keys():
            count = counts[waveform]
            ratio = round(np.sum(durations[waveform])/total_length,3)
            mean = round(np.mean(durations[waveform]),3)
            std = round(np.std(durations[waveform]),3)
            max = round(np.max(durations[waveform]),3)
            min = round(np.min(durations[waveform]),3)
            median = round(np.median(durations[waveform]),3)
            Q1 = round(np.quantile(durations[waveform],0.25),3)
            Q3 = round(np.quantile(durations[waveform],0.75),3)

            stats.append([count, ratio, mean, std, max, min, median, Q1, Q3])
        
        self.statistics = pd.DataFrame(stats)
        self.statistics.columns = ['count', 'ratio', 'mean', 'std', 'max', 'min', 'median', 'Q1', 'Q3']
        self.statistics.index = ['np', 'c', 'e1', 'e2', 'f', 'g', 'pd']
        return self.statistics
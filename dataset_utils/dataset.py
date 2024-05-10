import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
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
        self.subdatasets = get_dataset_group(dataset_name)

        all_recordings = []
        for subset in self.subdatasets:
            all_recordings += get_filename(subset) 

        self.recordings = []
        print('Reading ...')
        for recording_name in tqdm(all_recordings):
            recording, ana = read_signal(recording_name, data_path=self.data_path)
            self.recordings.append((recording_name, recording, ana))

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        return self.recordings[idx]

    def plot(self, recording_name, title = '', mode = 'static', smoothen = False):
        if isinstance(recording_name, int):
            recording_name = self.recordings[recording_name]
        recording, ana = read_signal(recording_name, data_path=self.data_path)
        plt.figure(figsize = (18,3))
        if mode == 'static':
            visualize_signal(recording, ana, title = title)
        elif mode == 'interactive':
            interactive_visualization(recording, ana, smoothen= smoothen, title = title)

    def generate_sliding_windows(self, window_size = 1024, hop_length = 1024, method = 'raw', scale = True):

        print('Generating sliding windows ...')
        d = generate_inputs(self.dataset_name, window_size, hop_length, method)
        self.windows, self.labels = d['data'], d['label']
        self.waveforms, self.distributions = np.unique(self.labels, return_counts= True)
        n = len(self.waveforms)
        self.distributions = [round(self.distributions[i]/len(self.labels),2) for i in range(n)]
        self.label_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}

        print(f'Total: {len(self.recordings)} recordings.')
        print(f'Signal processing method: {method} | Scale: {str(scale)}.')
        n = len(self.waveforms)
        print('Class distribution (label:ratio): ' + ', '.join(f'{self.waveforms[i]}: {self.distributions[i]}' for i in range(n)))
        print(f'Labels map (from:to): {self.label_map}')

    def plot_windows(self):
        pass     
    
    def stats(self):

        durations = {'np': [], 'c': [], 'e1': [], 'e2': [], 'f': [], 'g': [], 'pd': []}
        counts = {'np': 0, 'c': 0, 'e1': 0, 'e2': 0, 'f': 0, 'g': 0, 'pd': 0}
        total_length = 0
        n = len(self.recordings)
        for i in range(n):
            ana = self.recordings[i][2]
            waveform_intervals = get_index(ana)
            for waveform in waveform_intervals.keys():
                for interval in waveform_intervals[waveform]:
                    start, end = interval
                    durations[waveform].append(end - start)
                    # if waveform == 'pd':
                    #     if end-start > 100:
                    #         print(filename)
                    counts[waveform] += 1
            total_length += ana.iloc[-1]['time']
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
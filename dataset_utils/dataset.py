import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from dataset_utils.datahelper import format_data, read_signal, get_index
from dataset_utils.datagenerator import generate_sliding_windows_single
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
        self.recNames = sorted(set([x[:-4] for x in self.recNames]))
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
        formated = 0
        for name in recNames:
            if not name.startswith(prefix):
                formated = 1
                newname = f'{prefix}{name}'
                os.rename(f'{self.data_path}/{self.dataset_name}/{name}',f'{self.data_path}/{self.dataset_name}/{newname}')
        anaNames = os.listdir(f'{self.data_path}/{self.dataset_name}_ANA')
        for name in anaNames:
            if not name.startswith(prefix):
                newname = f'{prefix}{name}'
                os.rename(f'{self.data_path}/{self.dataset_name}_ANA/{name}',f'{self.data_path}/{self.dataset_name}_ANA/{newname}')
        if formated == 1:
            print('Filenames formated.')
    def plot(self, idx, mode = 'static', smoothen = False):
        recording, ana = self.recordings[idx]['recording'], self.recordings[idx]['ana']
        plt.figure(figsize = (18,3))
        if mode == 'static':
            visualize_signal(recording, ana, title = self.recordings[idx]['name'])
        elif mode == 'interactive':
            interactive_visualization(recording, ana, smoothen= smoothen, title = self.recordings[idx]['name'])

    def generate_sliding_windows(   self,
                                    window_size = 1024, 
                                    hop_length = 1024, 
                                    method= 'raw', 
                                    outlier_filter: bool = False, 
                                    scale: bool = True, 
                                    pad_and_slice = True, 
                                    verbose = True):
        '''
            ----------
            Arguments
            ----------
                config: configuration files containing necessary info
                verbose: if True, print descriptions
            --------
            Return
            --------
                d: dictionaries of training/testing data with keys {'data', 'label'}
        '''
        print('Generating sliding windows ...')
        count = 0
        d = []; l = []  
        
        for rec in tqdm(self.recordings):
            recRecording = rec['recording']
            recAna = rec['ana']
            features, labels = generate_sliding_windows_single(recRecording, recAna, window_size, hop_length, 
                                                                method, outlier_filter, scale, True, 'train')
            d.append(features); l.append(labels)
            count+=1

        d = np.concatenate([f for f in d])
        l = np.concatenate([lab for lab in l])
        
        d = format_data(d, l)

        if verbose == True:
            print(f'Total: {count} recordings')
            print(f'Signal processing method: {method} | Scale: {str(scale)}')
            cl, c = np.unique(l, return_counts=True)
            print('Class distribution (label:ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(l),2)}' for i in range(len(cl))))
            print(f'Labels map (from:to): {{1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}')

        self.windows, self.labels = d['data'], d['label']
        self.waveforms, self.distributions = np.unique(self.labels, return_counts= True)
        n = len(self.waveforms)
        self.distributions = [round(self.distributions[i]/len(self.labels),2) for i in range(n)]
        self.label_map = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}

        return self.windows, self.labels 

    def plot_windows(self):
        pass     

    def getRecordingParams(self, idx, ana = None):
        
        if isinstance(idx, int):
            idx = [idx]
        self.params = []
        self.params_log = []
        for recIndex in idx:
            recData = self.recordings[recIndex]
            recName = recData['name']
            recId = recData['id']
            recRecording = recData['recording']
            
            if ana is None:
                recAna = recData['ana']
                # print(recAna)
            else:
                recAna = ana
            recTime = recAna.iloc[-1,1]//3600
            waveformIndices = get_index(recAna)
            allwaveformids = set(recAna.loc[:,'label'])
            params = {}
            params['n_recorded_hour'] = recTime
            ################# Epidermis and mesophyl #################
            # Time from start of the experiment to 1st probe
            params['t_to_1st_probe'] = waveformIndices['C'][0][0]
            # Average number of Pd per probe
            num_probes = 0
            num_pds = (recAna.loc[:,'label'] == 8).sum()
            all_probes = get_probe(recAna)
            num_probes = len(all_probes)
            params['avg_num_pd_per_C'] = num_probes/num_pds
            
            # Duration of nonprobe period before the 1st E, 
            # Time from the start of the experiment to 1st E
            if 4 not in allwaveformids:
                params['duration_non_probe_to_1st_E'] = 0
                params['t_1st_E'] = 0
            else:
                _t_to_1st_E = waveformIndices['E1'][0][0]
                E1_anaIdx = list(recAna.loc[:,'time']).index(_t_to_1st_E)
                nonprobe_duration = 0
                for i in range(0, E1_anaIdx):
                    if recAna.loc[i,'label'] != 2 and recAna.loc[i,'label'] != 8:
                        nonprobe_duration += recAna.loc[i,'time']
                params['duration_non_probe_to_1st_E'] = num_probes/num_pds  
                params['t_to_1st_E'] = waveformIndices['E1'][0][0]
            # Duration of the second nonprobe period
            # Duration of the shortest C wave before E1
            # Total duration of F, Total duration of F during the 1st, 2nd, ..., 8th h, 
            # Number of F, Number of F during the 1st, 2nd, ..., 8th h, Mean duration of F
            if 6 not in allwaveformids:
                params['total_F'] = 0
                params['mean_F'] = 0
                params['num_F'] = 0
                for i in range(1, int(recTime) + 1):
                    params[f'num_F_h{i}'] = 0
                    params[f'total_F_h{i}'] = 0
            else:
                params['total_F'] = float(np.sum([x[1] - x[0] for x in waveformIndices['F']]).astype(np.float32))
                params['num_F'] = len(waveformIndices['F'])
                params['mean_F'] = params['total_F'] / params['num_F']
                for i in range(1, int(recTime) + 1):
                    F_hour_i = [x for x in waveformIndices['F'] if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
                    params[f'num_F_h{i}'] = len(F_hour_i)
                    params[f'total_F_h{i}'] = float(np.sum([x[1] - x[0] for x in F_hour_i]).astype(np.float32))
            # Number of pd, Mean duration of pd
            if 8 not in allwaveformids:
                params['total_pd'] = 0
                params['mean_pd'] = 0
                params['num_pd'] = 0
            else:
                params['total_pd'] = float(np.sum([x[1] - x[0] for x in waveformIndices['pd']]).astype(np.float32))
                params['num_pd'] = len(waveformIndices['pd'])
                params['mean_pd'] = params['total_pd'] / params['num_pd']
            # Number of probes to the 1st E1
            # Time from the end of the last pd to the end of the probe
            # Time from 1st probe to 1st E
            if 2 not in allwaveformids or 4 not in allwaveformids:
                params['t_1st_probe_to_1st_E'] = 0
            else:
                params['t_1st_probe_to_1st_E'] = waveformIndices['E1'][0][0] - waveformIndices['C'][0][1]
            
            # Time from 1st probe to 1st pd
            if 8 not in allwaveformids:
                params['t_start_of_1st_probe_to_1st_pd'] = 0
            else:
                params['t_start_of_1st_probe_to_1st_pd'] = waveformIndices['pd'][0][0] - waveformIndices['C'][0][0]
            
            # Duration of 1st probe, Duration of 2nd probe
            params['t_1st_probe'] = all_probes[0][1] - all_probes[0][0]
            if len(all_probes) <= 1:
                params['t_2nd_probe'] = 0
            else:    
                params['t_2nd_probe'] = all_probes[1][1] - all_probes[1][0]

            # Total duration of C, Number of C, Mean duration of C 
            params['total_C'] = float(np.sum([x[1] - x[0] for x in waveformIndices['C']]).astype(np.float32))
            params['num_C'] = len(waveformIndices['C'])
            params['mean_C'] = params['total_C'] / params['num_C']

            # Total duration of NP, Number of NP, Mean duration of NP, Number of NP during the 1st, 2nd, ..., 8th h
            # Total duration of NP during the 1st, 2nd, ..., 8th h
            if 1 not in allwaveformids:
                params['total_NP'] = 0
                params['num_NP'] = 0
                params['mean_NP'] = 0    
                for i in range(1, int(recTime) + 1):
                    params[f'total_NP_h{i}'] = 0                         
            else:
                params['total_NP'] = float(np.sum([x[1] - x[0] for x in waveformIndices['NP']]).astype(np.float32))
                params['num_NP'] = len(waveformIndices['NP'])
                params['mean_NP'] = params['total_pd'] / params['num_NP']
                for i in range(1, int(recTime) + 1):
                    NP_hour_i = [x for x in waveformIndices['NP'] if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
                    params[f'total_NP_h{i}'] = float(np.sum([x[1] - x[0] for x in NP_hour_i]).astype(np.float32))
                # params[f'num_NP_h{i}'] = len(NP_hour_i)

            # Number of short probes (<3 min)
            short_probes = [x for x in all_probes if x[1] - x[0] < 180 and x[1] - x[0] > 0]
            params['num_short_probes'] = len(short_probes)
            # Number of very short probes (<1 min)
            very_short_probes = [x for x in all_probes if x[1] - x[0] < 60 and x[1] - x[0] > 0]
            params['num_very_short_probes'] = len(very_short_probes)     
            # Number of E1 (I calculated E1 instead of E1e since E1e not yet defined in our study)
            if 4 not in allwaveformids:
                params['num_E1'] = 0
            else:
                params['num_E1'] = len(waveformIndices['E1'])

            # Number of probes, Number of probes during the 1st, 2nd, ..., 8th h, Total probing time
            params['num_probes'] = len(all_probes)
            for i in range(1, int(recTime) + 1):
                probes_hour_i = [x for x in all_probes if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
                params[f'num_probes_h{i}'] = len(probes_hour_i)
  
            params['total_probes'] =   float(np.sum([x[1] - x[0] for x in all_probes]).astype(np.float32))

            # Number of sustained E2 (I dont understand what this is ) 
            # Time from the start of E1 to the end of the EPG record (Z)
            if 4 not in allwaveformids:
                params['t_start_E1_to_end'] = 0
            else:
                params['t_start_E1_to_end'] = len(recRecording)//100 - waveformIndices['E1'][0][0]
            # Time from the start of E2 to the end of the EPG record (Z)
            if 5 not in allwaveformids:
                params['t_start_E2_to_end'] = 0
            else:
                params['t_start_E2_to_end'] = len(recRecording)//100 - waveformIndices['E2'][0][0]
            # Time from the end of the last pd to the end of the EPG record (Z)
            if 8 not in allwaveformids:
                params['t_last_pd_to_end'] = 0
            else:
                params['t_last_pd_to_end'] = len(recRecording)//100 - waveformIndices['pd'][-1][0]        
            # Time from the 1st probe to 1st E2
            if 2 not in allwaveformids or 5 not in allwaveformids:
                params['t_1st_probe_to_1st_E2'] = 0
            else:
                params['t_1st_probe_to_1st_E2'] = waveformIndices['E2'][0][0] - waveformIndices['C'][0][1]
            # Time from start of the experiment to 1st E2
            if 5 not in allwaveformids:
                params['t_1st_E2'] = 0
            else:
                params['t_1st_E2'] = waveformIndices['E2'][0][0]

            ################# Xylem ################# 
            # Total duration of G, Number of G, Mean duration of G
            if 7 not in allwaveformids: 
                params['total_G'] = 0
                params['mean_G'] = 0
                params['num_G'] = 0
            else:
                params['total_G'] = float(np.sum([x[1] - x[0] for x in waveformIndices['G']]).astype(np.float32))
                params['mean_G'] = float(np.mean([x[1] - x[0] for x in waveformIndices['G']]).astype(np.float32))
                params['num_G'] = len(waveformIndices['G'])

            ################# Phloem ################# 
            # Total duration of E1, Number of E1, Mean duration of E1
            if 4 not in allwaveformids: 
                params['total_E1'] = 0
                params['mean_E1'] = 0
                params['num_E1'] = 0
            else:
                params['total_E1'] = float(np.sum([x[1] - x[0] for x in waveformIndices['E1']]).astype(np.float32))
                params['mean_E1'] = float(np.mean([x[1] - x[0] for x in waveformIndices['E1']]).astype(np.float32))
                params['num_E1'] = len(waveformIndices['E1'])
            # Total duration of E2, Number of E2, Mean duration of E2
            if 5 not in allwaveformids: 
                params['total_E2'] = 0
                params['mean_E2'] = 0
                params['num_E2'] = 0
            else:
                params['total_E2'] = float(np.sum([x[1] - x[0] for x in waveformIndices['E2']]).astype(np.float32))
                params['mean_E2'] = float(np.mean([x[1] - x[0] for x in waveformIndices['E2']]).astype(np.float32))
                params['num_E2'] = len(waveformIndices['E2'])
            params['total_E'] = params['total_E1'] + params['total_E2']

            for key in params.keys():
                params[key] = [params[key]]
            params['name'] = [recName]
            params = pd.DataFrame(params)
            params.set_index('name', inplace = True)
            self.params.append(params)
        self.params = pd.concat(self.params)
        return self.params
    def datasetSummary(self):

        durations = {'NP': [], 'C': [], 'E1': [], 'E2': [], 'F': [], 'G': [], 'pd': []}
        counts = {'NP': 0, 'C': 0, 'E1': 0, 'E2': 0, 'F': 0, 'G': 0, 'pd': 0}
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
        
        # stats = {'NP': [], 'C': [], 'E1': [], 'E2': [], 'F': [], 'G': [], 'pd': []}
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
        self.statistics.index = ['NP', 'C', 'E1', 'E2', 'F', 'G', 'pd']
        return self.statistics


def get_probe(recAna):
    probes = []
    # Average number of Pd per probe
    label = recAna.loc[0,'label']
    pos = recAna.loc[0,'time']
    if label == 2: #control variable 
        is_C = True
        num_probes += 1
    else: 
        is_C = False
    for i in range(1, len(recAna)):
        next_label = recAna.loc[i,'label']
        if is_C == True:
            if next_label == 2 or next_label == 8:
                continue
            else:
                probe_end = recAna.loc[i,'time']
                probes.append([probe_start, probe_end])
                is_C = False
        else:
            if next_label == 2 or next_label == 8:
                is_C = True 
                probe_start = recAna.loc[i,'time']
            else:
                continue
    return probes 

def get_E(recAna):
    probes = []
    # Average number of Pd per probe
    label = recAna.loc[0,'label']
    pos = recAna.loc[0,'time']
    if label == 4: #control variable 
        is_C = True
        num_probes += 1
    else: 
        is_C = False
    for i in range(1, len(recAna)):
        next_label = recAna.loc[i,'label']
        if is_C == True:
            if next_label == 4 or next_label == 5:
                continue
            else:
                probe_end = recAna.loc[i,'time']
                probe.append([probe_start, probe_end])
                is_C = False
        else:
            if next_label == 4 or next_label == 5:
                is_C = True 
                probe_start = recAna.loc[i,'time']
            else:
                continue
    return probes 

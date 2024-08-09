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
from ipywidgets import widgets, interact 

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

    def getRecordingParams(self, idx = None, ana = None, export_xlsx = True):
        
        if idx == None:
            idx = [i for i in range(len(self.recordings))]
        else:
            if isinstance(idx, int):
                idx = [idx]
        self.params = []
        self.error_log = {}
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
            allWaveformIds = set(recAna.loc[:,'label'])
            self.containedWaveforms = list(waveformIndices.keys())
            error_log = []
            params = {}
            params['recTime'] = recTime
            params['waveforms'] = ', '.join(self.containedWaveforms)
            ############ NP #############

            if 1 not in allWaveformIds:
                params['s_NP']  = 0 
                params['n_NP']    = 0 
                params['a_NP']   = 0   
                params['m_NP'] = 0   
                params['mx_NP']    = 0     
                # for i in range(1, int(recTime) + 1):
                #     params[f's_NP_h{i}'] = 0     
                                  
            else:
                NP_durations        = [x[1] - x[0] for x in waveformIndices['NP']]
                params['s_NP']  = float(np.sum(NP_durations).astype(np.float32))
                params['n_NP']    = len(waveformIndices['NP'])
                params['a_NP']   = float(np.mean(NP_durations).astype(np.float32))
                params['m_NP'] = float(np.median(NP_durations).astype(np.float32))  
                params['mx_NP']    = float(np.max(NP_durations).astype(np.float32))
                # for i in range(1, int(recTime) + 1):
                #     NP_hour_i = [x for x in waveformIndices['NP'] if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
                #     params[f's_NP_h{i}'] = float(np.sum([x[1] - x[0] for x in NP_hour_i]).astype(np.float32))
                # # params[f'n_NP_h{i}'] = len(NP_hour_i)
                if len(NP_durations) < 2:
                    params['t_2nd_NP'] = 0 
                else:
                    params['t_2nd_NP'] = NP_durations[1]
                
            ############ C, pd, probes #############

            probes_locations = get_stage(recAna, stage = 'probe')
            probes_durations        = [x[1] - x[0] for x in probes_locations]
            params['s_probes']  =   float(np.sum(probes_durations).astype(np.float32))
            params['n_probes']    =   len(probes_durations)
            params['a_probes']   =   float(np.mean(probes_durations).astype(np.float32))
            params['m_probes'] =   float(np.median(probes_durations).astype(np.float32))

            # Total duration of C, Number of C, Mean duration of C 
            C_durations         = [x[1] - x[0] for x in waveformIndices['C']]
            params['s_C']   = float(np.sum(C_durations).astype(np.float32))
            params['n_C']     = len(C_durations)
            params['a_C']    = float(np.mean(C_durations).astype(np.float32))
            params['m_C']  = float(np.median(C_durations).astype(np.float32))
            params['%C_in_probe']  = params['s_C'] / params['s_probes']

            # Time from start of the experiment to 1st probe
            params['t_to_1st_probe'] = waveformIndices['C'][0][0]

            # Duration of 1st probe, Duration of 2nd probe
            params['t_1st_probe'] = probes_locations[0][1] - probes_locations[0][0]
            if len(probes_locations) <= 1:
                params['t_2nd_probe'] = 0    
            else:    
                params['t_2nd_probe'] = probes_locations[1][1] - probes_locations[1][0]

            # Number of short probes (<3 min)
            short_probes = [x for x in probes_locations if x[1] - x[0] < 180 and x[1] - x[0] > 0]
            params['n_short_probes'] = len(short_probes)
            # Number of very short probes (<1 min)
            very_short_probes = [x for x in probes_locations if x[1] - x[0] < 60 and x[1] - x[0] > 0]
            params['n_very_short_probes'] = len(very_short_probes)     
            # Number of probes, Number of probes during the 1st, 2nd, ..., 8th h, Total probing time
            
            # for i in range(1, int(recTime) + 1):
            #     probes_hour_i              = [x for x in probes_locations if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
            #     params[f'n_probes_h{i}'] = len(probes_hour_i)


            # Number of pd, Mean duration of pd
            if 8 not in allWaveformIds:
                params['s_pd']  = 0 
                params['a_pd']   = 0 
                params['n_pd']    = 0 
                params['m_pd'] = 0 
                params['t_1st_pd']  = 0 

                # params['t_last_pd_to_end'] = 0  # Time from the end of the last pd to the end of the EPG record (Z)

                   
            else:
                pd_durations        = [x[1] - x[0] for x in waveformIndices['pd']]
                params['s_pd']  = float(np.sum(pd_durations).astype(np.float32))
                params['n_pd']    = len(pd_durations)
                params['a_pd']   = float(np.mean(pd_durations).astype(np.float32))
                params['m_pd'] = float(np.median(pd_durations).astype(np.float32))
                params['t_1st_pd']  = waveformIndices['pd'][0][0]
            #     params['t_last_pd_to_end'] = len(recRecording)//100 - waveformIndices['pd'][-1][0] 
            # Number of probes to the 1st E1
            # Time from the end of the last pd to the end of the probe
            
            if 2 not in allWaveformIds or 4 not in allWaveformIds:
                params['t_1st_probe_to_1st_E1'] = 0  # Time from 1st probe to 1st E
                params['t_1st_probe_to_1st_E2'] = 0  # Time from the 1st probe to 1st E2
            else:
                params['t_1st_probe_to_1st_E1'] = waveformIndices['E1'][0][0] - waveformIndices['C'][0][1]
                params['t_1st_probe_to_1st_E2'] = waveformIndices['E2'][0][0] - waveformIndices['C'][0][1]
                

            # Time from 1st probe to 1st pd
            if 8 not in allWaveformIds:
                params['t_start_of_1st_probe_to_1st_pd'] = 0 
            else:
                params['t_start_of_1st_probe_to_1st_pd'] = waveformIndices['pd'][0][0] - waveformIndices['C'][0][0]
                        

            # Time from the start of E2 to the end of the EPG record (Z)
            if 5 not in allWaveformIds:
                params['t_start_E2_to_end'] = 0 
            else:
                params['t_start_E2_to_end'] = len(recRecording)//100 - waveformIndices['E2'][0][0]

            # Time from start of the experiment to 1st E2
            if 5 not in allWaveformIds:
                params['t_1st_E2'] = 0 
            else:
                params['t_1st_E2'] = waveformIndices['E2'][0][0]

            ############ F #############
            # Total duration of F, Total duration of F during the 1st, 2nd, ..., 8th h, 
            # Number of F, Number of F during the 1st, 2nd, ..., 8th h, Mean duration of F
            if 6 not in allWaveformIds:
                params['s_F']   = 0 
                params['a_F']    = 0 
                params['n_F']     = 0 
                params['m_F']  = 0 
                params['%F_in_probe']  = 0 
                # for i in range(1, int(recTime) + 1):
                #     params[f'n_F_h{i}'] = 0 
                #     params[f's_F_h{i}'] = 0 
            else:
                F_durations         = [x[1] - x[0] for x in waveformIndices['F']]
                params['s_F']   = float(np.sum(F_durations).astype(np.float32))
                params['n_F']     = len(F_durations)
                params['a_F']    = float(np.mean(F_durations).astype(np.float32))
                params['m_F']  = float(np.median(F_durations).astype(np.float32))
                params['%F_in_probe']  = params['s_F'] / params['s_probes']
                # for i in range(1, int(recTime) + 1):
                #     F_hour_i = [x for x in waveformIndices['F'] if x[0] > i*3600. and x[1] <= (i+1)*3600.] 
                #     params[f'n_F_h{i}'] = len(F_hour_i)
                #     params[f's_F_h{i}'] = float(np.sum([x[1] - x[0] for x in F_hour_i]).astype(np.float32))

            ################# G ################# 
            # Total duration of G, Number of G, Mean duration of G
            if 7 not in allWaveformIds:
                params['s_G']   = 0 
                params['a_G']    = 0 
                params['n_G']     = 0 
                params['m_G']  = 0 
                params['%G_in_probe']  = 0 
                # for i in range(1, int(recTime) + 1):
                #     params[f'n_G_h{i}'] = 0 
                #     params[f's_G_h{i}'] = 0 
            else:
                G_durations         = [x[1] - x[0] for x in waveformIndices['G']]
                params['s_G']   = float(np.sum(G_durations).astype(np.float32))
                params['n_G']     = len(G_durations)
                params['a_G']    = float(np.mean(G_durations).astype(np.float32))
                params['m_G']  = float(np.median(G_durations).astype(np.float32))
                params['%G_in_probe']  = params['s_G'] / params['s_probes']
            ################# E, E1, E2 ################# 

            # # Number of sustained E2 (I dont understand what this is ) 
            # # Time from the start of E1 to the end of the EPG record (Z)
            # if 4 not in allWaveformIds:
            #     params['t_start_E1_to_end'] = 0 
            # else:
            #     params['t_start_E1_to_end'] = len(recRecording)//100 - waveformIndices['E1'][0][0]

            # Total duration of E1e, Number of E1e, Mean duration of E1
            if 3 not in allWaveformIds: 
                params['s_E1e']  = 0 
                params['a_E1e']   = 0 
                params['n_E1e']    = 0 
                params['m_E1e'] = 0 

            else:
                E1e_durations        = [x[1] - x[0] for x in waveformIndices['E1e']]
                params['s_E1e']  = float(np.sum(E1e_durations).astype(np.float32))
                params['n_E1e']    = len(waveformIndices['E1e'])
                params['a_E1e']   = float(np.mean(E1e_durations).astype(np.float32))
                params['m_E1e'] = float(np.median(E1e_durations).astype(np.float32))

            
            # Total duration of E1, Number of E1, Mean duration of E1
            if 4 not in allWaveformIds: 
                params['s_E1']  = 0 
                params['a_E1']   = 0 
                params['n_E1']    = 0 
                params['m_E1'] = 0 
                params['mx_E1']    = 0 
                params['%E1_in_probe'] = 0 

                params['s_sgE1']   = 0 
                params['n_sgE1']     = 0 
                params['a_sgE1']    = 0 
                params['m_sgE1']  = 0 
                params['mx_sgE1']     = 0 

                params['n_Pr>1E']          = 0 
                params['n_brPr>1E']    = 0 
                params['n_Pr.after1E1']       = 0 
                params['n_brPr.after1E1'] = 0 
                params['t_NP_to_1st_E1']                = 0 
                
                phloem_durations = []

            else:
                E1_durations        = [x[1] - x[0] for x in waveformIndices['E1']]
                params['s_E1']  = float(np.sum(E1_durations).astype(np.float32))
                params['n_E1']    = len(waveformIndices['E1'])
                params['a_E1']   = float(np.mean(E1_durations).astype(np.float32))
                params['m_E1'] = float(np.median(E1_durations).astype(np.float32))
                params['mx_E1']    = float(np.max(E1_durations).astype(np.float32))
                params['%E1_in_probe'] = params['s_E1'] / params['s_probes']
                
                # Single E1
                tmp = recAna[recAna['label'] == 4]
                singleE1_locations = []
                for i in tmp.index:
                    if recAna.loc[i-1, 'label'] != 5 and recAna.loc[i+1, 'label'] != 5:
                        singleE1_locations.append([recAna.loc[i+1, 'time'], recAna.loc[i, 'time']])
                singleE1_durations = [x[1] - x[0] for x in singleE1_locations]
                if len(singleE1_durations) == 0:
                    params['s_sgE1']   = 0 
                    params['n_sgE1']     = 0 
                    params['a_sgE1']    = 0 
                    params['m_sgE1']  = 0 
                    params['mx_sgE1']     = 0                
                else:
                    params['s_sgE1']   = float(np.sum(singleE1_durations).astype(np.float32))
                    params['n_sgE1']     = len(singleE1_durations)
                    params['a_sgE1']    = float(np.mean(singleE1_durations).astype(np.float32))
                    params['m_sgE1']  = float(np.median(singleE1_durations).astype(np.float32))
                    params['mx_sgE1']     = float(np.max(singleE1_durations).astype(np.float32))
                # fraction E1
                frE1_locations = [x for x in waveformIndices['E1'] if x not in singleE1_locations]
                frE1_durations = [x[1] - x[0] for x in frE1_locations]
                if len(frE1_durations) == 0:
                    params['s_frE1']   = 0 
                    params['n_frE1']     = 0 
                    params['a_frE1']    = 0 
                    params['m_frE1']  = 0 
                    params['mx_frE1']     = 0                
                else:
                    params['s_frE1']   = float(np.sum(frE1_durations).astype(np.float32))
                    params['n_frE1']     = len(frE1_durations)
                    params['a_frE1']    = float(np.mean(frE1_durations).astype(np.float32))
                    params['m_frE1']  = float(np.median(frE1_durations).astype(np.float32))
                    params['mx_frE1']     = float(np.max(frE1_durations).astype(np.float32))      
                      
                # E12: phases with both E1 and E2
                E12_locations = get_stage(recAna, stage = 'E12')
                E12_durations = [x[1] - x[0] for x in E12_locations]
                if len(E12_durations) == 0:
                    params['s_E12']   = 0 
                    params['n_E12']     = 0 
                    params['a_E12']    = 0 
                    params['m_E12']  = 0 
                    params['mx_E12']     = 0                
                else:
                    params['s_E12']   = float(np.sum(E12_durations).astype(np.float32))
                    params['n_E12']     = len(E12_durations)
                    params['a_E12']    = float(np.mean(E12_durations).astype(np.float32))
                    params['m_E12']  = float(np.median(E12_durations).astype(np.float32))
                    params['mx_E12']     = float(np.max(E12_durations).astype(np.float32))

                # E1 followed by E2
                E1followedE2_locations = get_stage(recAna, stage = 'E1followedE2')
                E1followedE2_durations = [x[1] - x[0] for x in E1followedE2_locations]
                if len(E1followedE2_durations) == 0:
                    params['s_E1>E2']   = 0             
                else:
                    params['s_E1>E2']   = float(np.sum(E1followedE2_durations).astype(np.float32))

                # E1 followed by sustained E2
                E1followedsE2_locations = get_stage(recAna, stage = 'E1followedsE2')
                E1followedsE2_durations = [x[1] - x[0] for x in E1followedsE2_locations]
                if len(E1followedsE2_durations) == 0:
                    params['s_E1>sE2']   = 0             
                else:
                    params['s_E1>sE2']   = float(np.sum(E1followedsE2_durations).astype(np.float32))

                firstE1                                 = waveformIndices['E1'][0]
                probes2firstE1                          = [probe for probe in probes_locations if probe[1] < firstE1[0]]
                params['n_Pr>1E']          = len(probes2firstE1)
                NP2firstE                               = [NPdur for NPdur in waveformIndices['NP'] if NPdur[1] < firstE1[0]]
                params['t_NP_to_1st_E1']                = float(np.sum(NP2firstE).astype(np.float32))

                briefprobes2firstE1                     = [probe for probe in probes_locations if probe[1] < firstE1[0] \
                                                            and probe[1] - probe[0] < 180.]
                params['n_brPr>1E']    = len(briefprobes2firstE1)

                probesafterfirstE1                      = [probe for probe in probes_locations if probe[0] > firstE1[1]]
                params['n_Pr.after1E1']       = len(probesafterfirstE1)

                briefprobesafterfirstE1                 = [probe for probe in probes_locations if probe[0] > firstE1[1] \
                                                            and probe[1] - probe[0] < 180.]
                params['n_brPr.after1E1'] = len(briefprobesafterfirstE1)
                
                phloem_locations = get_stage(recAna, stage = 'phloem')
                phloem_durations = [x[1] - x[0] for x in phloem_locations]
                try:
                    params['E1_index']     = params['s_E1'] / (params['s_E12'] + params['s_sgE1'])
                except Exception as e:  
                    params['E1_index'] = 0  
                try:
                    params['frE1_ratio']   = params['n_frE1'] / params['n_E12']
                except Exception as e:
                    params['frE1_ratio']   = 0 
                    error_log.append('frE1_ratio ' + str(e))
                    #print('frE1_ratio ' + str(e))
            if len(phloem_locations) == 0:
                params['t_1st_phloem'] = 0 
            else:
                firstphloem = phloem_locations[0]
                params['t_1st_phloem'] = firstphloem[1] - firstphloem[0]

            # Total duration of E2, Number of E2, Mean duration of E2
            if 5 not in allWaveformIds: 
                params['s_E2']  = 0 
                params['a_E2']   = 0 
                params['n_E2']    = 0 
                params['m_E2'] = 0 
                params['mx_E2']    = 0 
                params['t_1st_E2']  = 0 
                params['a_E2_per_phloem']    = 0 
                params['%E2_in_probe'] = 0 

                params['s_sE2']  = 0 
                params['n_sE2']    = 0 
                params['a_sE2']   = 0 
                params['m_sE2'] = 0 

                params['n_Pr>1E2']      = 0 
                params['n_Pr>1sE2']     = 0 
                params['n_probes_after_1st_E2']   = 0 
                params['%E2s_in_E2']   = 0 
            else:
                E2_durations        = [x[1] - x[0] for x in waveformIndices['E2']]
                params['s_E2']  = float(np.sum(E2_durations).astype(np.float32))
                params['n_E2']    = len(waveformIndices['E2'])
                params['a_E2']   = float(np.mean(E2_durations).astype(np.float32))
                params['m_E2'] = float(np.median(E2_durations).astype(np.float32))
                params['mx_E2']    = float(np.max(E2_durations).astype(np.float32))
                params['t_1st_E2']   = waveformIndices['E2'][0][0]
                params['%E2_in_probe'] = params['s_E2'] / params['s_probes']

                sustainedE2         = [x for x in waveformIndices['E2'] if x[1] - x[0] > 600.]
                sustainedE2_durations = [x[1] - x[0] for x in sustainedE2]
                params['s_sE2']  = float(np.sum(sustainedE2_durations).astype(np.float32))
                params['n_sE2']    = len(sustainedE2_durations)
                params['a_sE2']   = float(np.mean(sustainedE2_durations).astype(np.float32))
                params['m_sE2'] = float(np.median(sustainedE2_durations).astype(np.float32))
                
                firstE2                             = waveformIndices['E2'][0]
                probes2firstE2                      = [probe for probe in probes_locations if probe[1] < firstE2[0]]
                params['n_Pr>1E2']      = len(probes2firstE2)

                firstsustainedE2                    = sustainedE2[0]
                probes2firstsE2                     = [probe for probe in probes_locations if probe[1] < firstsustainedE2[0]]
                params['n_Pr>1sE2']     = len(probes2firstsE2)

                E2_2_firstsE2                       = [E2dur for E2dur in waveformIndices['E2'] if E2dur[1] < firstsustainedE2[0]]
                params['n_E2_to_1st_sE2']         = len(E2_2_firstsE2)

                probesafterfirstE2                  = [probe for probe in probes_locations if probe[0] > firstE2[1]]
                params['n_probes_after_1st_E2']   = len(probesafterfirstE2)

                if len(phloem_locations) == 0:
                    params['a_E2_per_phloem'] = 0 
                else:
                    params['a_E2_per_phloem'] = params['s_E2']/len(phloem_locations)

                params['%E2s_in_E2']   = params['s_sE2'] / params['s_E2']
                try:
                    params['E2_ratio']     = params['s_E2'] / params['n_E12']
                except Exception as e:
                    params['E2_ratio']     = 0 
                    error_log.append('E2_ratio ' + str(e))
                    #print('E2_ratio ' + str(e))

            params['s_E'] = params['s_E1'] + params['s_E2']

            try:
                params['%E2/C']        = params['s_E2'] / params['s_C']
            except Exception as e:
                params['%E2/C']        = 0 
                error_log.append('%E2/C ' + str(e))
                #print('%E2/C ' + str(e))
            
            ############## Aggregation #################
            for key in params.keys():
                params[key] = [params[key]]
            params['name'] = [recName]
            params = pd.DataFrame(params)
            params.set_index('name', inplace = True)

            self.params.append(params)
            self.error_log[recName] = error_log
        self.params = pd.concat(self.params)
        
        if export_xlsx == True:
            with pd.ExcelWriter(f'EPGParameters_{self.dataset_name}.xlsx', engine='xlsxwriter') as writer:
                self.params.to_excel(writer, sheet_name='Overall')
            print(f'Exported to {os.getcwd()}/EPGParameters_{self.dataset_name}.xlsx')

        return self.params
        
    def make_boxplot(self, params):
        pass 

    def datasetSummary(self):

        durations = {'NP': [], 'C': [], 'E1': [], 'E2': [], 'F': [], 'G': [], 'pd': []}
        counts = {'NP': 0, 'C': 0, 'E1': 0, 'E2': 0, 'F': 0, 'G': 0, 'pd': 0}
        s_length = 0
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
            s_length += recAna.iloc[-1]['time']
        self.durations = durations
        
        # stats = {'NP': [], 'C': [], 'E1': [], 'E2': [], 'F': [], 'G': [], 'pd': []}
        stats = []
        for waveform in durations.keys():
            count = counts[waveform]
            ratio = round(np.sum(durations[waveform])/s_length,3)
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



# ============================= Label map =============================
waveform_labels = ['NP', 'C', 'E1', 'E2', 'F', 'G', 'pd']
waveform_ids = [1, 2, 4, 5, 6, 7, 8]
phloem_label = ['non-phloem','non-phloem', 'phloem', 'phloem', 'non-phloem', 'non-phloem', 'non-phloem']
probe_label = ['NP', 'probe', 'probe', 'probe', 'probe', 'probe', 'probe']


def create_map(list1, list2):
    map = {}
    for i in range(len(list1)):
        map[list1[i]] = list2[i]
    return map

def get_stage(recAna, stage = 'probe'):
    if stage == 'probe':
        labelmap = create_map(waveform_ids, probe_label)
    elif stage == 'phloem' or stage == 'E12' or stage == 'E1followedE2' or stage == 'E1followedsE2':
        labelmap = create_map(waveform_ids, phloem_label)
    Ana = recAna.copy()
    Ana['stagelabel'] = Ana.loc[:,'label'].map(labelmap)
    stages = []
    is_at_stage = False
    substage_count = 0 
    for i in range(1, len(Ana)):
        if is_at_stage == False:
            if Ana.loc[i, 'stagelabel'] == stage:
                start_index = i
                stage_start = Ana.loc[i,'time']
                is_at_stage = True
        else:
            if Ana.loc[i, 'stagelabel'] != stage:
                end_index = i
                stage_end = Ana.loc[i,'time']
                if stage == 'E12':
                    n_substage = end_index - start_index
                    if n_substage > 1: # E12 is simply a phloem stage with more than 1 E1 and E2 components
                        stages.append([stage_start, stage_end])
                if stage == 'E1followedE2':
                    n_substage = end_index - start_index
                    if n_substage == 2: # This specific phase start with E1 and ends with the succeeding E2
                        stages.append([stage_start, stage_end])   
                    is_at_stage = False
                if stage == 'E1followedsE2': # E1 followed by sustained E2
                    n_substage = end_index - start_index
                    lenE2 = Ana.loc[end_index + 1,'time'] - Ana.loc[end_index,'time']
                    if n_substage == 2 and lenE2 > 600.: 
                        stages.append([stage_start, stage_end])                 
                else:
                    stages.append([stage_start, stage_end])
                is_at_stage = False 

    return stages 

def interval_intersection(a, b):
    '''
        Inputs: intervals, or list of intervals
    '''
    def _get_intersection(itv1, itv2):
        start1, end1 = itv1 
        start2, end2 = itv2 
        start, end = max(start1, start2), min(end1,end2)
        if end <= start:
            return None
        else:
            return [start, end]
    if isinstance(a, list) or isinstance(a[0], tuple):
        if isinstance(a[0], list) or isinstance(a[0], tuple):
            intersections = []
            for itv1 in a:
                for itv2 in b:
                    intsec = _get_intersection(itv1, itv2)
                    if intsec:
                        intersections.append(intsec)
            return intersections 
        elif isinstance(a[0], float) and isinstance(b[0], float):
            return _get_intersection(a, b)
    else:
        raise RuntimeError('Expect list or tuple type input')
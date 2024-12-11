import numpy as np
import pandas as pd 
from scipy import stats
import warnings

import os
from tqdm import tqdm
import time

import matplotlib.pyplot as plt 
from ipywidgets import widgets, interact 

from ..dataset_utils.datahelper import format_data, read_signal, get_index
from ..dataset_utils.datagenerator import generate_sliding_windows_single
from ..utils.visualization import visualize_signal, interactive_visualization


class EPGDataset:
    def __init__(
                self, 
                data_path = '../data', 
                dataset_name = 'SA',
                inference = False,
                ):
        '''
        A class providing various functionalities for EPG analysis, including 
        making analysis, visualization and processing data for training ML models.
        *NOTE: Argument 'inference' is a support argument for making automatic annotation
        on external (unloaded) EPG recording. It is often not important when 
        the user only wants to make analysis
        ----------
        Arguments
        ----------
            + data_path:      directory of the database
            + dataset_name:   name of the dataset
            + inference:      set inference mode
        '''        
        self.guidelines = [
                            'The recording is always started with NP',
                            'Np is always followed by C or last until the end',
                            'E2 is always preceded by E1',
                            'pd waveform is always preceded by C and followed by C or Np',
                            'The subphase sequence after pd (pdII-1) should be II-2, II-3',
                            'F is always preceded by C',
                            'F is always followed by Np or C',
                            'G is always preceded by C',
                            'G is always followed by Np or C',
                            'E1 is always followed by E1e, E2, C, or Np',
                            'The end of each recording must be marked (T, code 99)',
                            ]       
        self.waveform_labeltoid = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}
        self.waveform_idtolabel = {0: 1, 1: 2, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
        self.waveform_nametolabel = {'NP':1, 'C':2, 'E1':4, 'E2':5, 'F':6, 'G':7, 'pd':8}
        self.waveform_labeltoname = {1:'NP', 2:'C', 4:'E1', 5:'E2', 6:'F', 7:'G', 8:'pd'}
        self.data_path = data_path
        self.dataset_name = dataset_name
        if isinstance(self.dataset_name, str):
            self.dataset_name = [self.dataset_name]
        self.is_inference_mode = inference
        if inference == False:
            self.load_database()
        else:
            self.database_loaded = False
            # print(self.database_loaded)
            print('Inference mode, skip loading data.') 
    def load_database(self):
        '''
        Load the EPG database from the given directory.
        '''
        self.guideline_check_log = {}
        # Reading database 

        print(f'Found {len(self.dataset_name)} datasets: {str(self.dataset_name)[1:-1]}.')
        self.recNames = []
        self.recordings = []
        t = time.perf_counter()
        print('Loading data ...')
        for set_name in self.dataset_name:
            filenames = os.listdir(f'{self.data_path}/{set_name}')
            filenames = sorted(set([x[:-4] for x in filenames]))
            self.recNames += list(filenames)
            for id, recording_name in enumerate(tqdm(filenames, desc = set_name)):
                recording, ana = read_signal(recording_name, data_path=self.data_path, dataset_name = set_name)
                self.guideline_check_log[recording_name] = self.check_guidelines(ana)[1]
                self.recordings.append({'id': id,
                                        'name': recording_name,
                                        'treatment' : None,
                                        'recording': recording,
                                        'ana':ana,
                                        'dataset': set_name,
                                        })        
                # Check if time marks is correctly given
                if ana is not None:
                    if ana.iloc[0,1] != 0:
                        print(f'{recording_name} - Analysis starts at {ana.iloc[0,1]} instead of 0.')
                    if ana.iloc[-1,1] != len(recording)//100:
                        print(f'{recording_name} - Analysis ends at {ana.iloc[-1,1]} instead of {len(recording)/100}.')
                
        self.recording_name2id = {file['name']: file['id'] for file in self.recordings}
        self.recording_id2name = {file['id']: file['name'] for file in self.recordings}                                 
        print('Done! Elapsed: {:.2f} s'.format(time.perf_counter() - t))
        self.database_loaded = True 
        self.guideline_check_log = pd.DataFrame(self.guideline_check_log)
        self.guideline_check_log.index = self.guidelines
        self.guideline_check_log = self.guideline_check_log.transpose()
        with pd.ExcelWriter(f'Guideline_check.xlsx', engine='xlsxwriter') as writer:
            self.guideline_check_log.to_excel(writer, sheet_name = 'Guidelines')
        print(f'View guidelines checking log at {os.getcwd()}/Guideline_check.xlsx')
        
    def __len__(self):
        return len(self.recordings)

    def inference_mode(self):
        self.is_inference_mode = True

    def training_mode(self):
        self.is_inference_mode = False

    def __getitem__(self, idx):
        assert self.database_loaded == True, "Database is not loaded. Load with EPGDataset.load_database()."
        return self.recordings[idx]

    def loadRec(self, recName, data_path = None, dataset_name = None):
        ''' 
        For loading a single recording given its name. 
        Helpful for loading external data for making inference.
        Only need to provide 'data_path' and 'dataset_name' when reading external data.
        ----------
        Arguments
        ----------
            + recName:        recording's name 
            + data_path:      directory of database
            + dataset_name:   name of the dataset
        '''
        if data_path is not None or dataset_name is not None:
            assert self.is_inference_mode == True, 'Reading external data is only possible in inference mode. Run self.inference_mode() to switch to inference mode.'
        if self.is_inference_mode == True:
            # For loading data from external source
            if data_path == None:
                data_path = self.data_path
            if dataset_name == None:
                dataset_name = self.dataset_name[0]
            recording, ana = read_signal(recName, data_path=data_path, dataset_name = dataset_name)
            recording = {'name': recName,
                        'recording': recording,
                        'ana':ana,
                        'dataset': self.dataset_name,
                        }
            return recording
        else:
            # For loaded database
            if isinstance(recName, str):
                idx = self.recNames.index(recName)
                return self.__getitem__(idx)
            elif isinstance(recName, list):
                recs = []
                for name in recName:
                    idx = self.recNames.index(name)
                    recs.append(self.__getitem__(idx))
                return recs

    def plot(self, idx, 
            mode = 'static', 
            hour = None, 
            range = None, 
            width = None, 
            height = None, 
            figsize = (18,3),
            timeunit = None, 
            nticks = None,
            smoothen = False):
        '''
        Making aesthetic plots of EPG recordings.

        ----------
        Arguments
        ----------
            + idx:        recording's name (str) or index (int)
            + mode:       static or interactive
            + hour:       the specific hour to plot
            + range:      A 2-tuple (a, b) specifying the start and end of the interested period (unit: second)
            + width:      width of the plot (only applicable to static plot)
            + height:     height of the plot (only applicable to static plot)
            + timeunit:   unit of the x-axis (sec, min or hour)
            + nticks:     number of ticks for x-axis
            + smoothen:   for making less resource-consuming plot (only applicable to interactive plot)
        '''
        if isinstance(idx, str):
            recData = self.loadRec(idx)
        elif isinstance(idx, int):
            recData = self.recordings[idx]
        recording, ana = recData['recording'], recData['ana']
        if ana is None:
            print(f'No plot. No annotation (*.ANA) was found for {recData["name"]}.')
            return
        plt.figure(figsize = figsize)
        if mode == 'static':
            if timeunit is None:
                timeunit = 'sec'
            if nticks is None:
                nticks = 10
            if width is not None or height is not None:
                warnings.warn("'width' and 'height' arguments are only applicable to interactive plot.")
            if smoothen == True:
                warnings.warn("'smoothen' argument is only applicable to interactive plot.")
            visualize_signal(recording, ana, title = recData['name'], hour = hour, range = range, timeunit = timeunit, nticks = nticks)
        elif mode == 'interactive':
            if timeunit is not None or nticks is not None:
                warnings.warn("'timeunit' and 'nticks' arguments are only applicable to static plot.")
            interactive_visualization(recording, ana, smoothen= smoothen, hour = hour, range = range, 
                                        width = width, height = height, title = recData['name'])

    def generate_sliding_windows(   self,
                                    window_size = 1024, 
                                    hop_length = 1024, 
                                    method= 'raw', 
                                    outlier_filter: bool = False, 
                                    scale: bool = True, 
                                    enrich = True, 
                                    verbose = True):
        '''
        Creating data for training/ inference with ML models
        ----------
        Arguments
        ----------
            + config: configuration files containing necessary info
            + verbose: if True, print descriptions
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

            if recAna is None:
                print(f'{rec["name"]} has no analysis file.')
                continue
            else:
                features, labels = generate_sliding_windows_single(recRecording, recAna, window_size, hop_length, 
                                                                method, outlier_filter, scale, enrich, 'train')
                d.append(features); l.append(labels)
                count+=1

        d = np.concatenate([f for f in d])
        l = np.concatenate([lab for lab in l])
        
        d = format_data(d, l)

        if verbose == True:
            print(f'Processed {count}/{len(self.recordings)} recordings.')
            print(f'Signal processing method: {method} | Scale: {str(scale)}')
            cl, c = np.unique(l, return_counts=True)
            print('Class distribution (label:ratio): '+ ', '.join(f'{cl[i]}: {round(c[i]/len(l),2)}' for i in range(len(cl))))
            print(f'Labels map (from:to): {{1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}}')

        self.windows, self.labels = d['data'], d['label']
        self.waveforms, self.distributions = np.unique(self.labels, return_counts= True)
        n = len(self.waveforms)
        self.distributions = [round(self.distributions[i]/len(self.labels),2) for i in range(n)]

        return self.windows, self.labels 

    def check_guidelines(self, ana):
        '''
        Check 11 guidelines
        ----------
        Arguments
        ---------- 
            + ana: an ANA
        '''
        if ana is None:
            check_text = ['No ANA.']*len(self.guidelines)
            check_log = ['No ANA.']*len(self.guidelines)
            return check_text, check_log
        # 1. The recording is always started with NP
        check = [False]*len(self.guidelines)
        check_log = ['']*len(self.guidelines)
        if ana.iloc[0,0] == 1:
            check[0] = True
            check_log[0] = 'True.'
        else:
            check_log[0] = f'Starts with {self.waveform_labeltoname[ana.iloc[0,0]]} instead of NP.'
        # 2. Np is always followed by C or last until the end
        NP_idx = ana.index[ana['label'] == 1]
        next_idx = NP_idx + 1
        next_label = ana.loc[next_idx]['label']
        next_notC_idx = next_label.index[~(next_label.isin([2,12,99]))]
        if len(next_notC_idx) == 0:
            check[1] = True 
            check_log[1] = 'True.'
        else:
            check_log[1] = f'At line {str(list(next_notC_idx-1))[1:-1]}, NP is not followed by C.'
        
        # 3. E2 is always preceded by E1
        E2_idx = ana.index[ana['label'] == 5]
        prev_idx = E2_idx - 1
        prev_label = ana.loc[prev_idx]['label']
        prev_notE1_idx = prev_label.index[prev_label != 4]
        if len(prev_notE1_idx) == 0:
            check[2] = True 
            check_log[2] = 'True.'
        else:
            check_log[2] = f'At line {str(list(prev_notE1_idx+1))[1:-1]}, E2 is not preceeded by E1.'
        
        # 4. pd waveform is always preceded by C and followed by C or Np
        pd_idx = ana.index[ana['label'] == 8]
        not_prev_by_C = []
        not_follw_by_C_or_NP = []
        for idx in pd_idx:
            if idx == 0:
                not_prev_by_C.append(idx)
            elif ana.loc[idx-1,'label'] != 2:
                not_prev_by_C.append(idx)
            else:
                if ana.loc[idx+1, 'label'] != 2 and ana.loc[idx+1, 'label'] != 1:
                    not_follw_by_C_or_NP.append(idx)
        invalid_idx = not_prev_by_C + not_follw_by_C_or_NP
        text = ''
        if len(invalid_idx) != 0:
            if len(not_prev_by_C) != 0:
                text += f'At line {str(not_prev_by_C)[1:-1]}, pd is not preceeded by C. '
            if len(not_follw_by_C_or_NP) != 0:
                text += f'At line {str(not_follw_by_C_or_NP)[1:-1]}, pd is not followed by C or NP.'
            check_log[3] = text
        else:
            check[3] = True
            check_log[3] = 'True.'
        # 5. The subphase sequence after pd (pdII-1) should be II-2, II-3
        allWaveformIds = ana['label'].unique()
        if 9 not in allWaveformIds or 10 not in allWaveformIds:
            check[4] = True 
            check_log[4] = 'True. pd-II-2, pd-II-3 is not presented.'
        # 6. F is always preceded by C and 7. F is always followed by Np or C
        F_idx = ana.index[ana['label'] == 6]
        if len(F_idx) > 0:
            not_prev_by_C = []
            not_follw_by_C_or_NP = []
            for idx in F_idx:
                if idx == 0:
                    not_prev_by_C.append(idx)
                elif ana.loc[idx-1,'label'] != 2:
                    not_prev_by_C.append(idx)
                else:
                    if ana.loc[idx+1, 'label'] != 2 and ana.loc[idx+1, 'label'] != 1:
                        not_follw_by_C_or_NP.append(idx)
            if len(not_prev_by_C) != 0: # 6. F is always preceded by C
                check_log[5] = f'At line {str(not_prev_by_C)[1:-1]}, F is not preceeded by C.'
            else:
                check[5] = True
                check_log[5] = 'True.'
            if len(not_follw_by_C_or_NP) != 0: #7. F is always followed by Np or C
                check_log[6] = f'At line {str(not_follw_by_C_or_NP)[1:-1]}, F is not followed by C or NP.'
            else:
                check[6] = True
                check_log[6] = 'True.'
        else:
            check[5] = True
            check_log[5] = 'True. F is not presented.'
            check[6] = True    
            check_log[6] = 'True. F is not presented.'
        # 8. F is always preceded by C and 9. F is always followed by Np or C
        G_idx = ana.index[ana['label'] == 7]
        if len(G_idx) > 0:
            not_prev_by_C = []
            not_follw_by_C_or_NP = []
            for idx in G_idx:
                if idx == 0:
                    not_prev_by_C.append(idx)
                elif ana.loc[idx-1,'label'] != 2:
                    not_prev_by_C.append(idx)
                else:
                    if ana.loc[idx+1, 'label'] != 2 and ana.loc[idx+1, 'label'] != 1:
                        not_follw_by_C_or_NP.append(idx)
            if len(not_prev_by_C) != 0: # 8. G is always preceded by C
                check_log[7] = f'At line {str(not_prev_by_C)[1:-1]}, G is not preceeded by C.'
            else:
                check[7] = True
                check_log[7] = 'True.'
            if len(not_follw_by_C_or_NP) != 0: #9. G is always followed by Np or C
                check_log[8] = f'At line {str(not_follw_by_C_or_NP)[1:-1]}, G is not followed by C or NP.'
            else:
                check[8] = True
                check_log[8] = 'True.'
        else:
            check[7] = True
            check_log[7] = 'True. G is not presented.'
            check[8] = True  
            check_log[8] = 'True. G is not presented.'
        # 10. E1 is always followed by E1e, E2, C, or Np
        E1_idx = ana.index[ana['label'] == 4]
        next_idx = E1_idx - 1
        next_label = ana.loc[next_idx]['label']
        next_notE1eE2CNP_idx = next_label.index[~(next_label.isin([3, 5, 2, 1]))]
        if len(next_notE1eE2CNP_idx) == 0:
            check[9] = True 
            check_log[9] = 'True.'
        else:
            check_log[9] = f'At line {str(list(next_notE1eE2CNP_idx+1))[1:-1]}, E1 is not followed by E1e, E2, C, or Np.'
                
        # 11. The end of each recording must be marked (T, code 99)
        last_row = ana.iloc[-1,:]
        if last_row['label'] == 12 or last_row['label'] == 99:
            check[10] = True
            check_log[10] = 'True.'
        else:
            check_log[10] = 'The recording does not end with label 99 or 12.'
        
        check_text = [f'{i+1}. {self.guidelines[i]}: {check[i]}' for i in range(len(self.guidelines))]

        return check_text, check_log
        
    def getRecordingParams(self, 
                            recordings_ids = None, 
                            ana = None, 
                            progress = 'overall', 
                            view = 'row', 
                            export_xlsx = True,
                            separate_sheet = False):
        '''
        Calculate various EPG parameters
        ----------
        Arguments
        ----------
            + recordings_ids (list or int):  recordings' idxs
            + ana:                an ANA
            + progress:           overall, progressive or cumulative
            + view:               row or column
            + export_xlsx:        export to an Excel file
            + separate_sheet:     export parameters separately or not 
        '''
        if recordings_ids == None:
            recordings_ids = [i for i in range(len(self.recordings))]
        else:
            if isinstance(recordings_ids, int):
                recordings_ids = [recordings_ids]
        sheet_names = []
        recordingParams = []
        self.error_log = {}

        for recIndex in recordings_ids:
            
            recData = self.recordings[recIndex]
            recDataset = recData['dataset']
            recName = recData['name']
            recTreatment = recData['treatment']
            sheet_names.append(recName)
            recId = recData['id']
            recRecording = recData['recording']
            if ana is None:
                recAna = recData['ana']
                # print(recAna)
            else:
                recAna = ana
            if recAna is None:
                print(f'No annotation was found for {recName}.')
                continue
            recTime = len(recRecording)//360000
            waveformIndices = get_index(recAna)
            listWaveforms = list(waveformIndices.keys())
            error_log = []
            params = {}
            params['name'] = recName
            params['dataset'] = recDataset
            params['treatment'] = recTreatment
            params['recTime'] = recTime
            params['waveforms'] = ', '.join(listWaveforms)
            
            if progress == 'overall':

                ############ NP (non-probing) #############
                if 'NP' not in listWaveforms:
                    raise RuntimeError('No NP waveform found.')  
                    # params['s_NP']  = None
                    # params['n_NP']  = None
                    # params['a_NP']  = None  
                    # params['m_NP']  = None  
                    # params['mx_NP'] = None    
      
                # else:
                NP_durations    = [x[1] - x[0] for x in waveformIndices['NP']]
                params['s_NP']  = float(np.sum(NP_durations).astype(np.float32))
                params['n_NP']  = len(waveformIndices['NP'])
                params['a_NP']  = float(np.mean(NP_durations).astype(np.float32))
                params['m_NP']  = float(np.median(NP_durations).astype(np.float32))  
                params['mx_NP'] = float(np.max(NP_durations).astype(np.float32))

                ############ C, pd, probes ############# (Assuming that a recording always have C -> always have probes)
                if 'C' not in listWaveforms:
                    raise RuntimeError('No C waveform found.')

                probe_locations     = get_stage(recAna, stage = 'probe')
                probes_durations    = [x[1] - x[0] for x in probe_locations]
                params['s_Pr']      =   float(np.sum(probes_durations).astype(np.float32))
                params['n_Pr']      =   len(probes_durations)
                params['a_Pr']      =   float(np.mean(probes_durations).astype(np.float32))
                params['m_Pr']      =   float(np.median(probes_durations).astype(np.float32))

                # Time from start of the experiment to 1st probe
                params['t>1stPr'] = probe_locations[0][0]

                # Duration of 1st probe, Duration of 2nd probe
                params['d_1stPr'] = probe_locations[0][1] - probe_locations[0][0]
                if len(probe_locations) <= 1:
                    params['d_2ndPr'] = 0    
                    error_log.append(f'Duration of 2nd probes = 0 as number of probes = {len(probe_locations)}')
                else:    
                    params['d_2ndPr'] = probe_locations[1][1] - probe_locations[1][0]

                # Number of short probes (<3 min)
                shortprobe_locations = [x for x in probe_locations if x[1] - x[0] < 180 and x[1] - x[0] > 0]
                params['n_shortPr'] = len(shortprobe_locations)
                # Number of very short probes (<1 min)
                veryshortprobe_locations = [x for x in probe_locations if x[1] - x[0] < 60 and x[1] - x[0] > 0]
                params['n_veryshortPr'] = len(veryshortprobe_locations)     
                
                # Total duration of C, Number of C, Mean duration of C 
                C_durations         = [x[1] - x[0] for x in waveformIndices['C']]
                params['s_C']       = float(np.sum(C_durations).astype(np.float32))
                params['n_C']       = len(C_durations)
                params['a_C']       = float(np.mean(C_durations).astype(np.float32))
                params['m_C']       = float(np.median(C_durations).astype(np.float32))
                params['%probtimeinC']  = params['s_C'] / params['s_Pr']

                # Number of pd, Mean duration of pd
                if 'pd' not in listWaveforms:
                    params['s_pd']      = None
                    params['a_pd']      = None
                    params['n_pd']      = None
                    params['n_pd/minC'] = None
                    params['m_pd']      = None
                    params['t>1stpd']   = None
                    params['t>1stpd/1stPr'] = None
                else:
                    pd_durations        = [x[1] - x[0] for x in waveformIndices['pd']]
                    params['s_pd']      = float(np.sum(pd_durations).astype(np.float32))
                    params['n_pd']      = len(pd_durations)
                    params['n_pd/minC'] = params['n_pd']/(params['s_C']/60) 
                    params['a_pd']      = float(np.mean(pd_durations).astype(np.float32))
                    params['m_pd']      = float(np.median(pd_durations).astype(np.float32))
                    params['t>1stpd']   = waveformIndices['pd'][0][0]
                    params['t>1stpd/1stPr'] = waveformIndices['pd'][0][0] - probe_locations[0][0]
                            
                ############ F #############
                # Total duration of F, Total duration of F during the 1st, 2nd, ..., 8th h
                if 'F' not in listWaveforms:
                    error_log.append('No F waveform found.')
                    params['s_F']       = None
                    params['a_F']       = None
                    params['n_F']       = None
                    params['m_F']       = None
                    params['%probtimeinF']  = None

                else:
                    F_durations         = [x[1] - x[0] for x in waveformIndices['F']]
                    params['s_F']       = float(np.sum(F_durations).astype(np.float32))
                    params['n_F']       = len(F_durations)
                    params['a_F']       = float(np.mean(F_durations).astype(np.float32))
                    params['m_F']       = float(np.median(F_durations).astype(np.float32))
                    params['%probtimeinF']  = params['s_F'] / params['s_Pr']

                ################# G ################# 
                # Total duration of G, Number of G, Mean duration of G
                if 'G' not in listWaveforms:
                    error_log.append('No G waveform found.')
                    params['s_G']       = None
                    params['a_G']       = None
                    params['n_G']       = None
                    params['m_G']       = None
                    params['%probtimeinG']  = None
                else:
                    G_durations         = [x[1] - x[0] for x in waveformIndices['G']]
                    params['s_G']       = float(np.sum(G_durations).astype(np.float32))
                    params['n_G']       = len(G_durations)
                    params['a_G']       = float(np.mean(G_durations).astype(np.float32))
                    params['m_G']       = float(np.median(G_durations).astype(np.float32))
                    params['%probtimeinG']  = params['s_G'] / params['s_Pr']
                ################# E, E1, E2 ################# 

                # Total duration of E1e, Number of E1e, Mean duration of E1
                if 'E1e' not in listWaveforms: 
                    error_log.append('No E1e waveform found.')
                    params['s_E1e']     = None
                    params['a_E1e']     = None
                    params['n_E1e']     = None
                    params['m_E1e']     = None

                else:
                    E1e_durations       = [x[1] - x[0] for x in waveformIndices['E1e']]
                    params['s_E1e']     = float(np.sum(E1e_durations).astype(np.float32))
                    params['n_E1e']     = len(waveformIndices['E1e'])
                    params['a_E1e']     = float(np.mean(E1e_durations).astype(np.float32))
                    params['m_E1e']     = float(np.median(E1e_durations).astype(np.float32))

                
                # Total duration of E1, Number of E1, Mean duration of E1
                if 'E1' not in listWaveforms: 
                    # E1
                    error_log.append('No E1 waveform found.')
                    params['s_E1']      = None
                    params['a_E1']      = None
                    params['n_E1']      = None
                    params['m_E1']      = None
                    params['mx_E1']     = None
                    params['%probtimeinE1'] = None
                    # Single E1
                    params['s_sgE1']    = None
                    params['n_sgE1']    = None
                    params['a_sgE1']    = None
                    params['m_sgE1']    = None
                    params['mx_sgE1']   = None
                    # Fraction E1
                    params['s_frE1']    = None
                    params['n_frE1']    = None
                    params['a_frE1']    = None
                    params['m_frE1']    = None
                    params['mx_frE1']   = None   
                    # Number of probes/NP before first E1
                    params['n_Pr>1stE1']        = None
                    params['n_brPr>1stE1']      = None
                    params['n_Pr.after.1stE1']  = None
                    params['n_brPr.after.1stE1']= None
                    params['s_NP>1stE1']        = None
                    params['t>1stE']            = None # Time from 1st probe to 1st E
                    params['d_1st_E']           = None
                    params['tPr>1stE/1stPr']    = None

                else:
                    phloem_locations = get_stage(recAna, stage = 'phloem')
                    # E1
                    E1_durations        = [x[1] - x[0] for x in waveformIndices['E1']]
                    params['s_E1']      = float(np.sum(E1_durations).astype(np.float32))
                    params['n_E1']      = len(waveformIndices['E1'])
                    params['a_E1']      = float(np.mean(E1_durations).astype(np.float32))
                    params['m_E1']      = float(np.median(E1_durations).astype(np.float32))
                    params['mx_E1']     = float(np.max(E1_durations).astype(np.float32))
                    params['%probtimeinE1'] = params['s_E1'] / params['s_Pr']
                    # Single E1
                    tmp = recAna[recAna['label'] == 4]
                    singleE1_locations = []
                    for i in tmp.index:
                        if recAna.loc[i-1, 'label'] != 5 and recAna.loc[i+1, 'label'] != 5:
                            singleE1_locations.append([recAna.loc[i+1, 'time'], recAna.loc[i, 'time']])
                    singleE1_durations = [x[1] - x[0] for x in singleE1_locations]
                    if len(singleE1_durations) == 0:
                        params['s_sgE1']    = 0 
                        params['n_sgE1']    = 0 
                        params['a_sgE1']    = 0 
                        params['m_sgE1']    = 0 
                        params['mx_sgE1']   = 0                
                    else:
                        params['s_sgE1']    = float(np.sum(singleE1_durations).astype(np.float32))
                        params['n_sgE1']    = len(singleE1_durations)
                        params['a_sgE1']    = float(np.mean(singleE1_durations).astype(np.float32))
                        params['m_sgE1']    = float(np.median(singleE1_durations).astype(np.float32))
                        params['mx_sgE1']   = float(np.max(singleE1_durations).astype(np.float32))
                    
                    # fraction E1
                    frE1_locations = [x for x in waveformIndices['E1'] if x not in singleE1_locations]
                    frE1_durations = [x[1] - x[0] for x in frE1_locations]
                    if len(frE1_durations) == 0:
                        params['s_frE1']    = 0 
                        params['n_frE1']    = 0 
                        params['a_frE1']    = 0 
                        params['m_frE1']    = 0 
                        params['mx_frE1']   = 0                
                    else:
                        params['s_frE1']    = float(np.sum(frE1_durations).astype(np.float32))
                        params['n_frE1']    = len(frE1_durations)
                        params['a_frE1']    = float(np.mean(frE1_durations).astype(np.float32))
                        params['m_frE1']    = float(np.median(frE1_durations).astype(np.float32))
                        params['mx_frE1']   = float(np.max(frE1_durations).astype(np.float32))      

                    firstE1                                 = waveformIndices['E1'][0]
                    probes2firstE1                          = [probe for probe in probe_locations if probe[1] < firstE1[0]]
                    params['n_Pr>1stE1']                    = len(probes2firstE1)

                    shortprobes2firstE1                     = [probe for probe in shortprobe_locations if probe[1] < firstE1[0]]
                    params['n_brPr>1stE1']                  = len(shortprobes2firstE1)

                    probesafterfirstE1                      = [probe for probe in probe_locations if probe[0] > firstE1[1]]
                    params['n_Pr.after.1stE1']              = len(probesafterfirstE1)

                    shortprobesafterfirstE1                 = [probe for probe in shortprobe_locations if probe[0] > firstE1[1]]
                    params['n_brPr.after.1stE1']            = len(shortprobesafterfirstE1)

                    NP2firstE                               = [NPdur for NPdur in waveformIndices['NP'] if NPdur[1] < firstE1[0]]
                    params['s_NP>1stE1']                    = float(np.sum(NP2firstE).astype(np.float32))

                    # # phloem_durations = [x[1] - x[0] for x in phloem_locations]
                    # initialE1s = []
                    # for phlo_loc in phloem_locations:
                    #     filt = recAna['time'] == phlo_loc[0]
                    #     initialE1_idx = recAna[filt].index
                    #     d_initialE1 = recAna.loc[initialE1_idx + 1, 'time'] - recAna.loc[initialE1_idx, 'time']
                    #     initialE1s.append(d_initialE1)
                    # if len(initialE1s) > 0:
                    #     params['a_initialE1'] = float(np.mean(initialE1s))
                    # else:
                    #     params['a_initialE1'] = 0

                    params['t>1stE'] = waveformIndices['E1'][0][0] - probe_locations[0][0]   
                                   
                    firstphloem = phloem_locations[0]
                    params['d_1st_E'] = firstphloem[1] - firstphloem[0]

                    for prob_loc in probe_locations:
                        subAna = recAna[(recAna['time'] >= prob_loc[0]) & (recAna['time'] <= prob_loc[1])]
                        E_loc = subAna[subAna['label'] == 4].index
                        if len(E_loc) > 0:
                            params['tPr>1stE/1stPr'] = float(subAna.loc[E_loc[0], 'time']) - prob_loc[0]
                            break 

                if 'E2' not in listWaveforms: 
                    # Total duration of E2, Number of E2, Mean duration of E2
                    params['s_E2']      = None
                    params['a_E2']      = None
                    params['n_E2']      = None
                    params['m_E2']      = None
                    params['mx_E2']     = None
                    params['t>1stE2']   = None # Time from the 1st probe to 1st E2
                    params['d_1st_E2']  = None
                    params['a_E2_per_phloem']   = None
                    params['%probtimeinE2']     = None
                    # sustained E2
                    params['s_sE2']     = None
                    params['n_sE2']     = None
                    params['a_sE2']     = None
                    params['m_sE2']     = None
                    params['t>1stsE2']  = None# Time from the 1st probe to 1st E2
                    # E12 
                    params['s_E12']     = None
                    params['n_E12']     = None
                    params['a_E12']     = None
                    params['m_E12']     = None
                    params['mx_E12']    = None       
                    params['t>1stE12']  = None # Time from 1st probe to 1st E12 
                    # E1 followed by E2/sE2
                    params['d_E1follwedbyE2']   = None  
                    params['d_E1follwedbysE2']  = None   
                    # Miscs
                    params['n_Pr>1stE2']        = None
                    params['n_Pr>1stsE2']       = None
                    params['n_E2>1stsE2']       = None
                    params['n_Pr.after.1stsE2'] = None
                    
                    params['%_sE2']             = None
                    params['tPr>1stE2/1stPr']   = None
                    params['tPr>1stsE2/1stPr']  = None 

                    params['E2_index']          = None
                    params['E1_index']          = None
                    params['frE1_ratio']        = None
                    params['%phloem_ph_fail']   = None
                    params['E2/C_ratio']        = None

                else:
                    # E2
                    E2_durations        = [x[1] - x[0] for x in waveformIndices['E2']]
                    params['s_E2']      = float(np.sum(E2_durations).astype(np.float32))
                    params['n_E2']      = len(waveformIndices['E2'])
                    params['a_E2']      = float(np.mean(E2_durations).astype(np.float32))
                    params['m_E2']      = float(np.median(E2_durations).astype(np.float32))
                    params['mx_E2']     = float(np.max(E2_durations).astype(np.float32))
                    params['t>1stE2']   = waveformIndices['E2'][0][0] - probe_locations[0][0]
                    params['d_1st_E2']  = E2_durations[0]
                    params['%probtimeinE2'] = params['s_E2'] / params['s_Pr']
                    if len(phloem_locations) == 0:
                        params['a_E2_per_phloem'] = 0 
                    else:
                        params['a_E2_per_phloem'] = params['s_E2']/len(phloem_locations)
                    # sustained E2
                    sustainedE2_locations   = [x for x in waveformIndices['E2'] if x[1] - x[0] > 600.]
                    sustainedE2_durations   = [x[1] - x[0] for x in sustainedE2_locations]
                    if len(sustainedE2_durations) == 0:
                        params['s_sE2']         = 0
                        params['n_sE2']         = 0
                        params['a_sE2']         = 0
                        params['m_sE2']         = 0
                        params['t>1stsE2']      = 0
                        params['n_Pr>1stsE2'] = 0
                        params['n_E2>1stsE2'] = 0
                        params['n_Pr.after.1stsE2'] = 0 
                        params['tPr>1stsE2/1stPr'] = 0   
                    else:
                        params['s_sE2']         = float(np.sum(sustainedE2_durations).astype(np.float32))
                        params['n_sE2']         = len(sustainedE2_durations)
                        params['a_sE2']         = float(np.mean(sustainedE2_durations).astype(np.float32))
                        params['m_sE2']         = float(np.median(sustainedE2_durations).astype(np.float32))
                        params['t>1stsE2']      = sustainedE2_locations[0][0] - probe_locations[0][0]
                        firstsustainedE2                = sustainedE2_locations[0]
                        probes2firstsE2                 = [probe for probe in probe_locations if probe[1] < firstsustainedE2[0]]
                        params['n_Pr>1stsE2']           = len(probes2firstsE2)

                        E2_2_firstsE2                   = [E2dur for E2dur in waveformIndices['E2'] if E2dur[1] < firstsustainedE2[0]]
                        params['n_E2>1stsE2']           = len(E2_2_firstsE2)

                        probesafterfirstsE2             = [probe for probe in probe_locations if probe[0] > firstsustainedE2[1]]
                        params['n_Pr.after.1stsE2']     = len(probesafterfirstsE2)   
                        for prob_loc in probe_locations:
                            subAna = recAna[(recAna['time'] >= prob_loc[0]) & (recAna['time'] <= prob_loc[1])]
                            sE2_loc = [subAna[subAna['time'] == location[0]].index[0] for location in sustainedE2_locations
                                                    if len(subAna[subAna['time'] == location[0]]) > 0]
                            if len(sE2_loc) > 0:
                                params['tPr>1stsE2/1stPr'] = float(subAna.loc[sE2_loc[0], 'time']) - prob_loc[0]
                                break                                
                    # E12: phases with both E1 and E2
                    E12_locations = get_stage(recAna, stage = 'E12')
                    E12_durations = [x[1] - x[0] for x in E12_locations]
                    # print(E12_locations)
                    if len(E12_durations) == 0:
                        params['s_E12']     = 0 
                        params['n_E12']     = 0 
                        params['a_E12']     = 0 
                        params['m_E12']     = 0 
                        params['mx_E12']    = 0        
                        params['t>1stE12']  = 0  # Time from 1st probe to 1st E12        
                    else:
                        params['s_E12']     = float(np.sum(E12_durations).astype(np.float32))
                        params['n_E12']     = len(E12_durations)
                        params['a_E12']     = float(np.mean(E12_durations).astype(np.float32))
                        params['m_E12']     = float(np.median(E12_durations).astype(np.float32))
                        params['mx_E12']    = float(np.max(E12_durations).astype(np.float32))
                        params['t>1stE12']  = E12_locations[0][0] - probe_locations[0][0]

                    # E1 followed by E2
                    E1followedE2_locations = get_stage(recAna, stage = 'E1followedE2')
                    E1followedE2_durations = [x[1] - x[0] for x in E1followedE2_locations]
                    if len(E1followedE2_durations) == 0:
                        params['d_E1follwedbyE2']   = 0             
                    else:
                        params['d_E1follwedbyE2']   = float(np.mean(E1followedE2_durations).astype(np.float32))
                    # E1 followed by sustained E2
                    E1followedsE2_locations = get_stage(recAna, stage = 'E1followedsE2')
                    E1followedsE2_durations = [x[1] - x[0] for x in E1followedsE2_locations]
                    if len(E1followedsE2_durations) == 0:
                        params['d_E1follwedbysE2']  = 0             
                    else:
                        params['d_E1follwedbysE2']  = float(np.mean(E1followedsE2_durations).astype(np.float32))

                    firstE2                         = waveformIndices['E2'][0]
                    probes2firstE2                  = [probe for probe in probe_locations if probe[1] < firstE2[0]]
                    params['n_Pr>1stE2']            = len(probes2firstE2)



                    if params['n_E12'] == 0:
                        params['%_sE2']     = np.nan
                        error_log.append('%_sE2 is NaN as n_E12 = 0')
                    else:
                        if params['n_sE2'] == 0:
                            params['%_sE2'] = 0
                        else:
                            params['%_sE2'] = params['s_E2'] / params['n_E12']

                    for prob_loc in probe_locations:
                        subAna = recAna[(recAna['time'] >= prob_loc[0]) & (recAna['time'] <= prob_loc[1])]
                        E2_loc = subAna[subAna['label'] == 5].index
                        if len(E2_loc) > 0:
                            params['tPr>1stE2/1stPr'] = float(subAna.loc[E2_loc[0], 'time']) - prob_loc[0]
                            break
                                     
                    params['E2_index']          = params['s_E2'] / (recTime*3600 - waveformIndices['E2'][0][0])
                    params['E1_index']          = np.round((params['s_E1'] / (params['s_E12'] + params['s_sgE1']))*100, 2)
                    if params['n_E12'] == 0:
                        params['frE1_ratio']    = np.nan
                        error_log.append('frE1_ratio is NaN as n_E12 = 0' )
                    else:
                        params['frE1_ratio']    = params['n_frE1'] / params['n_E12']
                    
                    params['%phloem_ph_fail']   = np.round((params['n_sgE1'] / (params['n_E12'] + params['n_sgE1']))*100, 2)
                    params['E2/C_ratio']        = np.round((params['s_E2'] / params['s_C'])*100, 2)

                for key in params.keys():
                    params[key] = [params[key]]
                params = pd.DataFrame(params)
                params.set_index('name', inplace = True)

                recordingParams.append(params)
                self.error_log[recName] = error_log

            else:
                from copy import deepcopy as dc
                progress_params = []
                for h in range(recTime):
                    params_hour = {}
                    params_hour['name'] = recName
                    params_hour['dataset'] = recDataset
                    if progress == 'discrete':
                        start, end = h*3600, (h+1)*3600
                        params_hour['Interval'] = f'Hour{h+1}'
                    elif progress == 'cumulative':
                        start, end = 0, (h+1)*3600
                        params_hour['Interval'] = f'Hour0-{h+1}'
                    else: 
                        raise ValueError("Param 'progress' must be 'overall', 'discrete' or 'cumulative'.")
                    subAna = recAna[(recAna['time'] >= start) & (recAna['time'] <= end)]
                    if len(subAna) == 0: # No data, match the data from the closest previous location
                        prev_row_idx = recAna.index[recAna['time'] >= start] - 1
                        prev_row = recAna.loc[prev_row_idx]
                        start_row = dc(prev_row)
                        start_row['time'] = start
                        end_row = dc(prev_row)
                        end_row['time'] = end
                        subAna = pd.concat([start_row, end_row], axis = 0)
                    else: # PAD DATA TO GET THE FULL LENGTH
                        if subAna.iloc[0]['time'] != start: # PAD HEAD
                            prev_row_idx = subAna.index[0] - 1
                            first_row = dc(recAna.loc[[prev_row_idx]])
                            first_row['time'] = start 
                            subAna = pd.concat([first_row, subAna], axis = 0)
                        if subAna.iloc[-1]['time'] != end: # PAD TAIL
                            last_row = dc(recAna.iloc[[-1]])
                            last_row.iloc[-1] = [99, end - 1]
                            subAna = pd.concat([subAna, last_row], axis = 0)
                    subAna.reset_index(inplace = True, drop = True)
                    waveformIndices = get_index(subAna)
                    listWaveforms = list(waveformIndices.keys())
                    params_hour['waveforms'] = ', '.join(listWaveforms)
                    ############ NP #############
                    if 'NP' not in listWaveforms:
                        params_hour['s_NP']  = None 
                        params_hour['n_NP']  = None 
                        params_hour['a_NP']  = None   
                        params_hour['m_NP']  = None              
                    else:
                        NP_durations        = [x[1] - x[0] for x in waveformIndices['NP']]
                        params_hour['s_NP']  = float(np.sum(NP_durations).astype(np.float32))
                        params_hour['n_NP']    = len(waveformIndices['NP'])
                        params_hour['a_NP']   = float(np.mean(NP_durations).astype(np.float32))
                        params_hour['m_NP'] = float(np.median(NP_durations).astype(np.float32))  

                    ############ C, pd #############
                    # Total duration of C, Number of C, Mean duration of C 
                    if 'C' not in listWaveforms:
                        params_hour['s_C']  = None 
                        params_hour['n_C']    = None 
                        params_hour['a_C']   = None   
                        params_hour['m_C'] = None              
                    else:
                        C_durations         = [x[1] - x[0] for x in waveformIndices['C']]
                        params_hour['s_C']   = float(np.sum(C_durations).astype(np.float32))
                        params_hour['n_C']     = len(C_durations)
                        params_hour['a_C']    = float(np.mean(C_durations).astype(np.float32))
                        params_hour['m_C']  = float(np.median(C_durations).astype(np.float32))

                    # Number of pd, Mean duration of pd
                    if 'pd' not in listWaveforms:
                        params_hour['s_pd']  = None 
                        params_hour['a_pd']   = None 
                        params_hour['n_pd']    = None 
                        params_hour['m_pd'] = None 

                    else:
                        pd_durations        = [x[1] - x[0] for x in waveformIndices['pd']]
                        params_hour['s_pd']  = float(np.sum(pd_durations).astype(np.float32))
                        params_hour['n_pd']    = len(pd_durations)
                        params_hour['a_pd']   = float(np.mean(pd_durations).astype(np.float32))
                        params_hour['m_pd'] = float(np.median(pd_durations).astype(np.float32))
    
                    ############ F #############
                    # Total duration of F, Total duration of F during the 1st, 2nd, ..., 8th h, 
                    # Number of F, Number of F during the 1st, 2nd, ..., 8th h, Mean duration of F
                    if 'F' not in listWaveforms:
                        params_hour['s_F']   = None 
                        params_hour['a_F']    = None 
                        params_hour['n_F']     = None 
                        params_hour['m_F']  = None 
                    else:
                        F_durations         = [x[1] - x[0] for x in waveformIndices['F']]
                        params_hour['s_F']   = float(np.sum(F_durations).astype(np.float32))
                        params_hour['n_F']     = len(F_durations)
                        params_hour['a_F']    = float(np.mean(F_durations).astype(np.float32))
                        params_hour['m_F']  = float(np.median(F_durations).astype(np.float32))

                    ################# G ################# 
                    # Total duration of G, Number of G, Mean duration of G
                    if 'G' not in listWaveforms:
                        params_hour['s_G']   = None 
                        params_hour['a_G']    = None 
                        params_hour['n_G']     = None 
                        params_hour['m_G']  = None 
                    else:
                        G_durations         = [x[1] - x[0] for x in waveformIndices['G']]
                        params_hour['s_G']   = float(np.sum(G_durations).astype(np.float32))
                        params_hour['n_G']     = len(G_durations)
                        params_hour['a_G']    = float(np.mean(G_durations).astype(np.float32))
                        params_hour['m_G']  = float(np.median(G_durations).astype(np.float32))

                    ################# E, E1, E2 ################# 
                    if 'E1' not in listWaveforms:
                        params_hour['s_E1']   = None 
                        params_hour['a_E1']    = None 
                        params_hour['n_E1']     = None 
                        params_hour['m_E1']  = None                
                    else:
                        E1_durations        = [x[1] - x[0] for x in waveformIndices['E1']]
                        params_hour['s_E1']  = float(np.sum(E1_durations).astype(np.float32))
                        params_hour['n_E1']    = len(waveformIndices['E1'])
                        params_hour['a_E1']   = float(np.mean(E1_durations).astype(np.float32))
                        params_hour['m_E1'] = float(np.median(E1_durations).astype(np.float32))

                    if 'E2' not in listWaveforms:
                        params_hour['s_E2']   = None 
                        params_hour['a_E2']    = None 
                        params_hour['n_E2']     = None 
                        params_hour['m_E2']  = None                        
                        params_hour['s_sE2']   = None 
                        params_hour['a_sE2']    = None 
                        params_hour['n_sE2']     = None 
                        params_hour['m_sE2']  = None                
                    else:
                        E2_durations        = [x[1] - x[0] for x in waveformIndices['E2']]
                        params_hour['s_E2']  = float(np.sum(E2_durations).astype(np.float32))
                        params_hour['n_E2']    = len(waveformIndices['E2'])
                        params_hour['a_E2']   = float(np.mean(E2_durations).astype(np.float32))
                        params_hour['m_E2'] = float(np.median(E2_durations).astype(np.float32))

                        sustainedE2_locations         = [x for x in waveformIndices['E2'] if x[1] - x[0] > 600.]
                        sustainedE2_durations = [x[1] - x[0] for x in sustainedE2_locations]
                        if len(sustainedE2_durations) == 0:
                            params_hour['s_sE2']   = 0 
                            params_hour['a_sE2']    = 0 
                            params_hour['n_sE2']     = 0 
                            params_hour['m_sE2']  = 0                               
                        else:
                            params_hour['s_sE2']  = float(np.sum(sustainedE2_durations).astype(np.float32))
                            params_hour['n_sE2']    = len(sustainedE2_durations)
                            params_hour['a_sE2']   = float(np.mean(sustainedE2_durations).astype(np.float32))
                            params_hour['m_sE2'] = float(np.median(sustainedE2_durations).astype(np.float32))

                    for key in params_hour.keys():
                        params_hour[key] = [params_hour[key]]
                    params_hour = pd.DataFrame(params_hour)
                    params_hour.set_index('name', inplace = True)
                    progress_params.append(params_hour)
                progress_params = pd.concat(progress_params)
                recordingParams.append(progress_params)
        self.recordingParams = pd.concat(recordingParams)
        ############## Aggregation #################
        if len(recordingParams) == 0:
            return 
        else:
            
            if separate_sheet == True:
                if export_xlsx == True:
                    with pd.ExcelWriter(f'EPGParameters_{progress}.xlsx', engine='xlsxwriter') as writer:
                        for params, name in zip(recordingParams, sheet_names):
                            assert view in ['row', 'column'], "Params 'view' must be 'row' or 'column'."
                            if view == 'column':
                                description = pd.DataFrame([variable_descriptions[x] for x in params.columns]).transpose()
                                description.columns = params.columns
                                description.index = ['Description']
                                params = pd.concat([description, params])
                                params = params.transpose()
                                params.index.name = 'Variables'                                
                            params.to_excel(writer, sheet_name=name)
                    print(f'Exported to {os.getcwd()}/EPGParameters_{progress}.xlsx')
            else:
                recordingParams = pd.concat(recordingParams)
                assert view in ['row', 'column'], "Params 'view' must be 'row' or 'column'."
                if view == 'column':
                    description = pd.DataFrame([variable_descriptions[x] for x in recordingParams.columns]).transpose()
                    description.columns = recordingParams.columns
                    description.index = ['Description']
                    recordingParams = pd.concat([description, recordingParams])
                    recordingParams = recordingParams.transpose()
                    recordingParams.index.name = 'Variables'
 
                if export_xlsx == True:
                    with pd.ExcelWriter(f'EPGParameters_{progress}.xlsx', engine='xlsxwriter') as writer:
                        recordingParams.to_excel(writer, sheet_name = progress)
                    print(f'Exported to {os.getcwd()}/EPGParameters_{progress}.xlsx')
        return recordingParams            

    def assign_treatments(self, recordings_ids: list, treatment_id = 0):
        '''
        Assigning treatment ID to loaded recordings.
        ----------
        Arguments
        ----------        
            + recordings_ids: a list of recording
            + treatment_id: ID of the treatment group that you would like to assign
        '''
        assert self.database_loaded == True, "Database is not loaded. Load with EPGDataset.load_database()."
        if not isinstance(recordings_ids, list):
            recordings_ids = [recordings_ids]
        for file in recordings_ids:
            if isinstance(file, str):
                idx = self.recording_name2id[file]
            else:
                idx = file
            assert self.recordings_ids[idx]['id'] == idx, f'Index mismatch at {idx}'
            self.recordings_ids[idx]['treatment'] = treatment_id

    def statistical_analysis(self, parameters, treatment_ids, test, *args):
        ''' 
        perform statistical analysis
        ----------
        Arguments
        ----------   
            + parameters: which parameter to perform statistical analysis
            ...        
        '''
        assert not all(x is None for x in self.recordingParams['treatment']), 'No treatment was assigned.'
        samples = []
        for id in treatment_ids:
            a = self.recordingParams[self.recordingParams['treatment'] == id]
            a = a[parameters]
            samples.append(a)
        n_samples = len(treatment_ids)
        # except:
        #     raise RuntimeError('Please calculate the parameters first.')
        result = {}
        if not isinstance(test, list):
            test = [test]
        for type in test:
            if type == 'ttest':
                assert n_samples == 2, f't-test only works for 2 samples, not {n_samples}.'
                res = stats.ttest_ind(samples[0], samples[1], *args)
                
            elif type == 'wilcoxon':
                assert n_samples == 2, f'Wilcoxon signed-rank test only works for 2 samples, not {n_samples}.'
                res = stats.wilcoxon(samples[0], samples[1], *args)
                    
            elif type == 'ANOVA':
                assert n_samples > 2, f'ANOVA only works for 3 or more samples, not {n_samples}.'
                anova_res = stats.f_oneway(samples, *args)
                tukeyhsd_res = stats.tukey_hsd(samples, *args)
                res = [anova_res, tukeyhsd_res]
            elif type == 'kruskal':
                assert n_samples > 2, f'kruskal-wallis test only works for 3 or more samples, not {n_samples}.'
                res = stats.kruskal(samples, *args)
            result[type] = res    
        return result

    def make_boxplot(self, parameters, treatment_ids, figsize = (6,6), max_ncol = 4):
        assert not all(x is None for x in self.recordingParams['treatment']), 'No treatment was assigned.'
        if not isinstance(parameters, list):
            parameters = [parameters]
        samples = {}
        for id in treatment_ids:
            a = self.recordingParams[self.recordingParams['treatment'] == id]
            a = a[parameters]
            samples[id] = a
        if len(parameters) > max_ncol:
            ncol = max_ncol
            nrow = (len(parameters)//max_ncol) + 1
        else:
            ncol = len(parameters) 
            nrow = 1
        f, ax = plt.subplots(nrow, ncol, figsize = figsize)

        for i, param in enumerate(parameters):
            df = []
            for id in treatment_ids:
                tmp = samples[id][[param]]
                tmp.index = [i for i in range(len(tmp))]
                df.append(tmp)
            df = pd.concat(df, axis=1, ignore_index=True)
            df.columns = treatment_ids
            if nrow == 1:
                df.plot(kind='box', title=param, showmeans=True, ax=ax[i])
                ax[i].grid('on')
                ax[i].set_xticks(np.arange(1, len(df.columns)+1), [f'Group#{x}' for x in treatment_ids])
            else:
                df.plot(kind='box', title=param, showmeans=True, ax=ax[i//ncol, i%ncol])
                ax[i//ncol, i%ncol].grid('on')
                ax[i//ncol, i%ncol].set_xticks(np.arange(1, len(df.columns)+1),[f'Group#{x}' for x in treatment_ids])
        for j in range(i+1, nrow*ncol):
            f.delaxes(ax[j//ncol, j%ncol])
        plt.tight_layout()
        plt.show()

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
        index_col = []
        for waveform in durations.keys():
            if len(self.durations[waveform]) > 0:
                index_col.append(waveform)
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
            else:
                print(f'No waveform {waveform} was found in dataset.')
        self.statistics = pd.DataFrame(stats)
        self.statistics.columns = ['count', 'ratio', 'mean', 'std', 'max', 'min', 'median', 'Q1', 'Q3']
        self.statistics.index = index_col
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
        superstage = 'probe'
    elif stage == 'phloem' or stage == 'E12' or stage == 'E1followedE2' or stage == 'E1followedsE2':
        labelmap = create_map(waveform_ids, phloem_label)
        superstage = 'phloem'
    Ana = recAna.copy()
    Ana['stagelabel'] = Ana.loc[:,'label'].map(labelmap)
    stages = []
    is_at_stage = False
    substage_count = 0 
    for i in range(1, len(Ana)):
        if is_at_stage == False:
            if Ana.loc[i, 'stagelabel'] == superstage:
                start_index = i
                stage_start = Ana.loc[i,'time']
                is_at_stage = True
        else:
            if Ana.loc[i, 'stagelabel'] != superstage:
                end_index = i
                stage_end = Ana.loc[i,'time']
                if stage == 'E12':
                    n_substage = end_index - start_index
                    # print(Ana)
                    if n_substage > 1: # E12 is simply a phloem stage with more than 1 E1 and E2 components
                        stages.append([stage_start, stage_end])
                elif stage == 'E1followedE2':
                    n_substage = end_index - start_index
                    if n_substage == 2: # This specific phase start with E1 and ends with the succeeding E2
                        stages.append([stage_start, stage_end])   
                    is_at_stage = False
                elif stage == 'E1followedsE2': # E1 followed by sustained E2
                    n_substage = end_index - start_index
                    # print(end_index, start_index)
                    lenE2 = Ana.loc[end_index,'time'] - Ana.loc[end_index-1,'time']
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


variable_descriptions = {
    ### meta 
    'dataset': 'Name of the dataset',
    'recTime': 'Total recording time',
    'waveforms': 'Observed waveforms',
    'treatment': 'Treatment group ID',
    ### Non-probing
    'n_NP': 'Number of NP periods',
    'a_NP': 'Average duration of NP periods',
    'm_NP': 'Median duration of NP periods',
    's_NP': 'Sum duration of NP periods',
    'mx_NP': 'Max duration of NP',
    ### probing
    'n_Pr': 'Number of probes',
    'a_Pr': 'Average probe duration',
    'm_Pr': 'Median probe duration',
    's_Pr': 'Sum duration of all probes',
    'd_1stPr': 'Duration of 1st probe',
    'd_2ndPr': 'Duration of 2nd probe', ## ???
    'n_shortPr': 'Number of short probes (< 3 min)',
    'n_veryshortPr': 'Number of very short probes (< 1 min)', ## ???
    ### pathway
    'n_C': 'Number of C periods',
    'a_C': 'Average duration of C periods',
    'm_C': 'Median duration of C periods',
    's_C': 'Sum duration of C periods',
    ### Derailed stylet mechanics
    'n_F': 'Number of F periods',
    'a_F': 'Average duration of F periods',
    'm_F': 'Median duration of F periods',
    's_F': 'Sum duration of F periods',
    ### Xylem
    'n_G': 'Number of G periods',
    'a_G': 'Average duration of G periods',
    'm_G': 'Median duration of G periods',
    's_G': 'Sum duration of G periods',
    ### Pathway salivation 
    'n_E1e': 'Number of E1e periods',
    'a_E1e': 'Average duration of E1e periods',
    'm_E1e': 'Median duration of E1e periods',
    's_E1e': 'Sum duration of E1e periods',
    ### Phloem-phase
    'd_1st_E': 'Duration of the 1st phloem phase',
    'n_sgE1': 'Number of single E1 periods',
    'a_sgE1': 'Average duration of single E1 periods',
    'm_sgE1': 'Median duration of single E1 periods',
    's_sgE1': 'Sum duration of single E1 periods',
    'mx_sgE1': 'Max duration of single E1 periods',
##### NON-SEQUENTIAL VARIABLES #####
    'n_frE1': 'Number of fraction E1 periods',
    'a_frE1': 'Average duration of fraction E1 periods',
    'm_frE1': 'Median duration of fraction E1 periods',
    's_frE1': 'Sum duration of fraction E1 periods',
    'mx_frE1': 'Max duration of fraction E1 periods',

    'n_E1': 'Number of E1 periods',
    'a_E1': 'Average duration of E1 periods',
    'm_E1': 'Median duration of E1 periods',
    's_E1': 'Sum duration of E1 periods',
    'mx_E1': 'Max duration of E1 periods',

    'n_E12': 'Number of E12 periods',
    'a_E12': 'Average duration of E12 periods',
    'm_E12': 'Median duration of E12 periods',
    's_E12': 'Sum duration of E12 periods',
    'mx_E12': 'Max duration of E12 periods',

    'n_E2': 'Number of E2 periods',
    'a_E2': 'Average duration of E2 periods',
    'm_E2': 'Median duration of E2 periods',
    's_E2': 'Sum duration of E2 periods',
    'mx_E2': 'Max duration of E2 periods',
    'a_E2_per_phloem': 'Average duration of E2 per phloem phase',
    'd_1st_E2': 'Duration of 1st E2 in the recording',
    'n_sE2': 'Number of sustained E2 periods',
    'a_sE2': 'Average duration of sustained E2 periods',
    'm_sE2': 'Median duration of sustained E2 periods',
    's_sE2': 'Sum duration of sustained E2 periods',


#### SEQUENTIAL VARIABLES ####
    ### probing
    't>1stPr': 'Time to 1st probe from start of recording',
    ### Phloem phase
    't>1stE': 'Time from the 1st probe to the start of 1st E1',
    't>1stE12': 'Time from the 1st probe to the start of 1st E12',
    't>1stE2': 'Time from the 1st probe to the start of 1st E2',
    't>1stsE2': 'Time from the 1st probe to the start of 1st sE2',
    'tPr>1stE/1stPr': 'Time from the begining of that probe to 1st E',
    'tPr>1stE2/1stPr': 'Time from the begining of that probe to 1st E2',
    'tPr>1stsE2/1stPr': 'Time from the begining of that probe to 1st sE2',

    'n_Pr>1stE1': 'Number of probe before 1st E1',
    'n_brPr>1stE1': 'Number of short probe before 1st E1',
    'n_Pr>1stE2': 'Number of probe before 1st E2',
    'n_Pr>1stsE2': 'Number of probe before 1st sE2',
    'n_E2>1stsE2': 'Number of E2 before 1st sE2',
    'n_Pr.after.1stE1': 'Number of probe after 1st E1',
    'n_brPr.after.1stE1': 'Number of short probe after 1st E1',
    'n_Pr.after.1stsE2': 'Number of probe after 1st sE2',
    ### sequential variables
    'd_E1follwedbyE2': 'Duration of E1 follwed by E2',
    'd_E1follwedbysE2': 'Duration of E1 follwed by sustained E2',
    'E2/C_ratio': 'E2/C ratio',
    'E1_index': 'E1 index',
    'frE1_ratio': 'fraction E1 ratio',
    'E2_index': 'E2 index',
    '%probtimeinC': '% of probe duration spent in C',
    '%probtimeinF': '% of probe duration spent in F',
    '%probtimeinG': '% of probe duration spent in G',
    '%probtimeinE1': '% of probe duration spent in E1',
    '%probtimeinE2': '% of probe duration spent in E2',
    '%_sE2': '% of E2s that are sustained E2s',
    # Standard potential drops (pd)
    'n_pd/minC': 'Number of pds per minute of pathwave phase',
    'n_pd': 'Number of pd periods',
    'a_pd': 'Average duration of pd periods',
    'm_pd': 'Median duration of pd periods',
    's_pd': 'Sum duration of pd periods',
    't>1stpd': 'Time to 1st pd from start of recording',
    't>1stpd/1stPr': 'Time to 1st pd from the start of 1st probe',

    's_NP>1stE1': 'Sum of duration of NP before 1st E1',
    'a_initialE1': 'Average duration of initial E1',
    '%phloem_ph_fail': '% of phloem phases that fail to achieve ingestion',
}
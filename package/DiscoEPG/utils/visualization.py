import numpy as np
import pandas as pd 
import librosa
import pywt 

from copy import deepcopy as dc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler

from ..dataset_utils.datahelper import get_index


c = {'NP':'darkgoldenrod',    'C':'lightgreen',    'E1':'skyblue',    'E2':'deeppink',  'F':'crimson',   'G':'darkblue',   'pd':'olive'}

#######################
## Display waveforms ##
#######################
def get_position_label(waveform_indices, position):
    for wave_type in waveform_indices:
        for start, end in waveform_indices[wave_type]:
            if (start <= position) & (position <= end):
                return wave_type
            
def visualize_signal(recording, ana_file, hour = None, range = None, ax = None, title = '', timeunit = 'sec', nticks = 10):

    plt.rcParams.update({'font.size': 14})
    '''
    Input:
        :param recording: wave signal of class np.ndarray
        :param ana: analysis file corresponding to the recording
        :param wave_type: 'whole' or str of {'NP', 'C', 'E1', 'E2', 'F', 'G', 'pd'}
        :param sr: sampling rate. Default = 100Hz.
    Output:
        plot of the full wave (wave types) with colors

    NOTE: By default, sampling rate = 100
    '''    
    sr = 100
    time_axis = np.linspace(0, len(recording)/sr, len(recording))
    recording = dc(recording)
    ana = dc(ana_file)
    ana['time'] = ana['time'].apply(lambda x: x*sr)
    ana['time'] = ana['time'].astype(int)
    if hour is not None or range is not None:
        if hour is not None: # TO MAKE HOURLY PLOT
            assert isinstance(hour, int), "Param 'hour' must be int."
            assert range is None, 'Can plot either by hour or specified range.'
            start, end = hour*360000, (hour+1)*360000
        elif range is not None:
            assert (isinstance(range, list) or isinstance(range, tuple)), "Param 'range' must be a list or tuple of length 2."
            assert hour is None, 'Can plot either by hour or specified range.'
            assert len(range) == 2, 'Range must consider only start and end location.'
            assert range[0] < range[1], 'Start time must be smaller than end time.'
            start, end = range
            start = int(start*100)
            end = int(end*100)
        recording = recording[start: end]
        # Create a new ana which fit the hours
        tmp = ana[ana['time'] >= start]
        tmp = tmp[tmp['time'] < end]
        if len(tmp) == 0: # No data, match the data from the closest previous location
            prev_row_idx  = ana[ana['time'] >= start].index[0] - 1
            prev_label = ana.loc[prev_row_idx, 'label']
            tmp = pd.DataFrame({'label': [prev_label, prev_label], 'time': (start, end)})
        else: # PAD DATA TO GET THE FULL LENGTH
            if tmp.iloc[0]['time'] != start: # PAD HEAD
                prev_row_idx = tmp.index[0] - 1
                first_row = dc(ana.loc[[prev_row_idx]])
                first_row['time'] = start 
                tmp = pd.concat([first_row, tmp], axis = 0)
            if tmp.iloc[-1]['time'] != end: # PAD TAIL
                last_row = dc(tmp.iloc[[-1]])
                last_row.iloc[-1] = [99, end - 1]
                tmp = pd.concat([tmp, last_row], axis = 0)
        tmp.reset_index(inplace = True, drop = True)
        ana = tmp
    # print(ana)
    waveform_indices = get_index(ana)

    custom_legends = []
    i = 0
    if ax == None:
        ax = plt.gca()
    for wave_type in waveform_indices.keys():
        for start, end in waveform_indices[wave_type]:
            if hour is not None:
                y = recording[start - hour*360000 : end - hour*360000]
            elif range is not None:
                y = recording[start - range[0]*100: end - range[0]*100]
            else:
                y = recording[start : end]
            x = time_axis[start:end]
            ax.plot(x, y, color = c[wave_type])
        custom_legends.append(Line2D([0], [0], color=c[wave_type], lw=4))
        i+=1
    
    xlim_min = ana['time'].min()//100
    xlim_max = ana['time'].max()//100
    scale = max((xlim_max - xlim_min)//nticks, 1)
    xlim = np.arange(xlim_min, xlim_max+1, scale)
    ax.legend(custom_legends, waveform_indices.keys(), loc = 'upper right', ncols=len(custom_legends), fontsize = 15)
    if timeunit == 'sec':
        xlabels = np.arange(xlim_min, xlim_max+1, scale)
    if timeunit == 'min':
        xlabels = np.arange(xlim_min, xlim_max+1, scale)/60
    elif timeunit == 'hour':
        xlabels = np.arange(xlim_min, xlim_max+1, scale)/3600
    # print(xlabels)
    ax.set_xticks(xlim, labels=xlabels, rotation = 45)
    ax.set_xlabel(f'Time ({timeunit}). Sampling rate: {sr} (Hz)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title(title) if isinstance(title, str) else None
    ax.grid('on')
def visualize_waveform(recording, ana_file, waveform = None, in_between = False, padding = 512, seed: int = 100, idx = None, ax = None):

    plt.rcParams.update({'font.size': 14})
    '''
    Input:
        :param recording: wave signal of class np.ndarray
        :param ana: analysis file corresponding to the recording
        :param wave_type: 'whole' or str of {'NP', 'C', 'E1', 'E2', 'F', 'G', 'pd'}
        :param sr: sampling rate. Default = 100Hz.
    Output:
        plot of the full wave (wave types) with colors
    '''    
    if seed is not None: 
        np.random.seed(seed)


    time_axis = np.linspace(0,len(recording)/sr,len(recording))
    ana = dc(ana_file)
    ana['time'] = ana['time'].apply(lambda x: x*sr)
    ana['time'] = ana['time'].astype(int)
    waveform_indices = get_index(ana)

    if waveform is not None:
        if ax == None:
            ax = plt.gca()
        wave_type = waveform
        if idx is None:
            idx = np.random.choice(np.arange(len(waveform_indices[wave_type])))
        start, end = waveform_indices[wave_type][idx]
        # pad the wave with adjacent waveform to see the relationship
        ax.plot(time_axis[start:end], recording[start:end], color = c[wave_type], label = wave_type)
        ax.legend(loc = 'upper right')      

        if in_between == True:
            start_pad = start - padding
            end_pad = end + padding
            former_label = get_position_label(waveform_indices,start_pad)
            latter_label = get_position_label(waveform_indices,end_pad)
            if former_label != None:
                ax.plot(time_axis[start_pad:start+1],recording[start_pad:start+1],
                        color = c[former_label],label = former_label)

            if latter_label != None:
                ax.plot(time_axis[end-1:end_pad],recording[end-1:end_pad],
                        color = c[latter_label], label = latter_label)
    else:
        raise ValueError("Must be one of 'NP', 'C', 'E1', 'E2', 'F', 'G' or 'pd' ")
    return plt.gcf()

### Interactive
def interactive_visualization(recording, ana_file, hour = None, range = None, width = 1000, height = 400, smoothen = False, title = ''):
    sr = 100
    time_axis = np.linspace(0, len(recording)/sr, len(recording))
    recording = dc(recording)
    ana = dc(ana_file)
    ana['time'] = ana['time'].apply(lambda x: x*sr)
    ana['time'] = ana['time'].astype(int)
    if hour is not None or range is not None:
        if isinstance(hour, int): # TO MAKE HOURLY PLOT
            assert range is None, 'Can plot either by hour or specified range.'
            start, end = hour*360000, (hour+1)*360000
        elif isinstance(range, list)  or isinstance(range, tuple):
            assert hour is None, 'Can plot either by hour or specified range.'
            assert len(range) == 2, 'Range must consider only start and end location.'
            start, end = range
            start = int(start*100)
            end = int(end*100)
        recording = recording[start: end]
        # Create a new ana which fit the hours
        tmp = ana[ana['time'] >= start]
        tmp = tmp[tmp['time'] < end]
        if len(tmp) == 0: # No data, match the data from the closest previous location
            prev_row_idx = ana.index[ana['time'] >= start] - 1
            prev_row = ana.loc[prev_row_idx]
            start_row = dc(prev_row)
            start_row['time'] = start
            end_row = dc(prev_row)
            end_row['time'] = end
            tmp = pd.concat([start_row, end_row], axis = 0)
        else: # PAD DATA TO GET THE FULL LENGTH
            if tmp.iloc[0]['time'] != start: # PAD HEAD
                prev_row_idx = tmp.index[0] - 1
                first_row = dc(ana.loc[[prev_row_idx]])
                first_row['time'] = start 
                tmp = pd.concat([first_row, tmp], axis = 0)
            if tmp.iloc[-1]['time'] != end: # PAD TAIL
                last_row = dc(tmp.iloc[[-1]])
                last_row.iloc[-1] = [99, end - 1]
                tmp = pd.concat([tmp, last_row], axis = 0)
        tmp.reset_index(inplace = True, drop = True)
        ana = tmp
        # print(ana)
    waveform_indices = get_index(ana)

    if smoothen == True:
        fig = FigureWidgetResampler(go.Figure())
    elif smoothen == False:
        fig = go.Figure()

    for wave_type in waveform_indices.keys():
        i = 0
        for start, end in waveform_indices[wave_type]:
            if isinstance(hour, int):
                y = recording[start - hour*360000 : end - hour*360000]
            else:
                y = recording[start : end]
            x = time_axis[start:end]
            if i == 0:
                fig.add_trace(go.Scatter(x = x, y =y,line=dict(color=c[wave_type]), mode = 'lines',
                                        legendgroup = wave_type, name = wave_type))   
            else:
                fig.add_trace(go.Scatter(x = x, y = y,line=dict(color=c[wave_type]),mode = 'lines',
                                        legendgroup = wave_type, name = wave_type, showlegend = False))  
            i+=1
    fig.update_layout(title_text= f"{title} Interactive viewer")
    fig.update_layout( xaxis=dict( rangeslider=dict(visible=True), type="linear"),  # Add range slider
                       yaxis=dict( fixedrange = False)  )
    fig.update_layout(
        width=width,  # Set the width (e.g., 800 pixels)
        height=height  # Set the height (e.g., 600 pixels)
    )
    fig.show()

######################################################
## Illustration of signal transformation techniques ##
######################################################
from scipy.signal import find_peaks
def draw_fft_diagrams(recording, ana_file, sr = 100, wave_type: str = None, which = 0, 
                               log = False, is_average = False, n_windows = 10):
    '''
    Input:
        recording: dataframe containing wave data - 2 columns [time,amplitude]
        ana: analysis file corresponding to the recording
        wave_type: 'whole' or type of waves in 'NP', 'C', 'E1', 'E2', 'F', 'G', 'pd'
        is_average: if True, takes the average of all fft rows
        n_windows: the number of coefficients rows to plot
    Output:
        plot of the fft coefficients
    '''    
    ana = dc(ana_file)
    ana['time'] = ana['time'].apply(lambda x: x*sr)
    ana['time'] = ana['time'].astype(int)

    waveform_indices = get_index(ana)
    waveform_indices_sub = waveform_indices[wave_type]#get wave indices
    start, end = waveform_indices_sub[which]

    if end-start < 1024:
        wave_sample = recording[(start+end)//2-1024:(start+end)//2+1024]
    else:
        wave_sample = recording[start:end]

    #short-time fourier transform
    stft_coefs = np.abs(librosa.stft(wave_sample,n_fft = 1024,center = False))/np.sqrt(1024)

    if log == True:
        stft_coefs = np.log(stft_coefs)

    freq_axis = librosa.fft_frequencies(sr = sr, n_fft = 1024)

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,2))

    if is_average == True:
        avg = np.sum(stft_coefs,axis = 1)/stft_coefs.shape[1] #average
        peaks_idx = find_peaks(avg,prominence = 0.75)[0]
        ax1.plot(freq_axis[peaks_idx], avg[peaks_idx], "xr")
        ax1.plot(freq_axis,avg,'r')
        ax1.set_title(f"log-Average of {wave_type} wave windows" if log == True else f"Average of {wave_type} wave windows")

    else: 
        for i in range(n_windows):
            ax1.plot(freq_axis, stft_coefs[:,i],color = 'r')
        ax1.set_title(f"log coefs of {n_windows} {wave_type} wave windows" if log == True else f"Frequency domain representation")
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('log - Amplitude' if log == True else 'Amplitude')

    ax2.set_xlabel('Time')
    ax2.plot(wave_sample, color = 'blue')
    ax2.set_title(f"Time domain representation")
    

###############################
## Result plotting utilities ##
###############################

def plot_pred_proba(predicted_probability, hop_length, scope, r: tuple = None, ax = None):
    '''
        Plot the predicted probability distributions
        ---------
        Argument
        ---------
            r:tuple (start,end) or None: range to plot
            ax: matplotlib.pytplot.axis object or None: axis to plot 
    '''
    test_hop_length = hop_length // scope
    pred_proba = repeat(predicted_probability, test_hop_length)
    pred_proba = extend(pred_proba, 2880000)
    
    if ax == None:
        ax = plt.gca()
    if r is not None:
        r = np.arange(r[0],r[1])
        pp = pred_proba[r]
    else:
        pp = pred_proba

    waveform = list(c.keys())
    lower = np.zeros(pp.shape[0])
    upper = np.zeros(pp.shape[0])
    x = np.arange(pp.shape[0])
    
    for i in range(len(waveform)):
        lower = upper 
        upper = upper + pp[:,i]
        plt.fill_between(x, lower, upper, alpha = 0.2, color = c[waveform[i]])
        plt.plot(x, upper, color = c[waveform[i]],linewidth = 0.4, label = waveform[i])

    plt.legend(loc = 'upper right')

def plot_gt_vs_pred_segmentation(recording, gt_ana, pred_ana = None, hour = None, range = None, 
                                    timeunit = 'sec', nticks = 10, which = 'pred_vs_gt', 
                                    name: str = '', savefig = False, save_dir = ''): 
    if save_dir == '':
        save_dir = './prediction/figures'
    if gt_ana is None:
        which = 'pred'

    if which == 'pred':
        plt.figure(figsize=(16,3))
        visualize_signal(recording, pred_ana, hour = hour, range = range)
        plt.title('Prediction')
        plt.tight_layout()
    elif which == 'gt':
        assert (gt_ana is not None), 'Ground-truth analysis not found.'
        plt.figure(figsize=(16,3))
        visualize_signal(recording, gt_ana, hour = hour, range = range)
        plt.title('Prediction')
        plt.tight_layout() 

    elif which == 'pred_vs_gt':
        assert pred_ana is not None, 'Must input prediction analysis'

        f, ax = plt.subplots(2, 1, figsize=(16,5), sharex= True)
        
        visualize_signal(recording, gt_ana, ax=ax[0], hour = hour, range = range, timeunit = timeunit, nticks = nticks)
        ax[0].text(0.85, 0.1, 'Ground-truth', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
        ax[0].set_xlabel('')
        ax[0].set_title(name)
        visualize_signal(recording, pred_ana, ax = ax[1], hour = hour, range = range, timeunit = timeunit, nticks = nticks)
        ax[1].text(0.85, 0.1, 'Predicted', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        plt.subplots_adjust(hspace = 0)

        plt.tight_layout()
    else:
        raise ValueError("Param which must be one of ['pred', 'gt', 'pred_vs_gt'].")

    if savefig == True:
        os.makedirs(save_dir, exist_ok = True)
        if name == '':
            dir = os.listdir(save_dir)
            index = len(dir) + 1
            save_path = f'{save_dir}/Untitled{index}.png'
            plt.savefig(save_path)
        else:
            save_path = f'{save_dir}/{name}.png'
            plt.savefig(save_path)
        print(f'Figure saved to {save_path}.')

#############
### Miscs ###
#############

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
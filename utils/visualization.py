import numpy as np
import librosa
import matplotlib.pyplot as plt
from dataloader.datahelper import get_index
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler

from copy import deepcopy as dc

"""
VISUALIZATION FUNCTION
"""

from matplotlib.lines import Line2D

np.random.seed(28)

c={'np':np.random.rand(3),
    'c':np.random.rand(3),
    'e1':np.random.rand(3),
    'e2':np.random.rand(3),
    'g':np.random.rand(3),
    'f':np.random.rand(3),
    'pd':np.random.rand(3)}

def get_position_label(wave_indices, position):
    for wave_type in wave_indices:
        for start, end in wave_indices[wave_type]:
            if (start <= position) & (position <= end):
                return wave_type
            
def visualize_wave(wave_array, ana_file, wave_type = 'whole', n_plots = None,sr = 100, in_between = False):
    '''
    Input:
        :param wave_array: wave signal of class np.ndarray
        :param ana: analysis file corresponding to the wave_array
        :param wave_type: 'whole' or str of {'np','c','e1','e2','g','f'}:; select the part of the signal to be plotted
        :param n_plots: if wave_type is not 'full', then n_plots determines the number of wave type plots
        :param sr: sampling rate. Default = 100Hz.
    Output:
        plot of the full wave (wave types) with colors
    '''    
    np.random.seed(28)
    c={'np':np.random.rand(3),    'c':np.random.rand(3),    'e1':np.random.rand(3),    'e2':np.random.rand(3),    'g':np.random.rand(3),    'f':np.random.rand(3),    'pd':np.random.rand(3)}
    time = np.linspace(0,len(wave_array)/sr,len(wave_array))

    ana = dc(ana_file)
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    wave_indices = utils.get_index(ana)

    if wave_type == 'whole':

        custom_legends = []
        i = 0
        for wave in wave_indices.keys():
            for start,end in wave_indices[wave]:
                plt.plot(time[start:end],wave_array[start:end],color = c[wave])
            custom_legends.append(Line2D([0], [0], color=c[wave], lw=4))
            i+=1

        plt.legend(custom_legends,wave_indices.keys(),loc = 'best',ncols=len(custom_legends))
        plt.xticks(np.arange(0,time.max(),2000), labels=np.arange(0,time.max(),2000), rotation = 45)
        plt.xlabel(f'Time (s). Sampling rate: {sr} (Hz)')
        plt.ylabel('Amplitude (V)')

    else:
        if isinstance(n_plots,int) == False:
            n_plots = len(wave_indices[wave_type])

        f,ax = plt.subplots(n_plots,1,figsize=(16,2*n_plots))
        try:
            iter(ax)
        except:
            ax = [ax]
        for i in range(n_plots):
            start, end = wave_indices[wave_type][i]
            # pad the wave with adjacent waveform to see the relationship
            ax[i].plot(time[start:end],wave_array[start:end],
                       color = c[wave_type], label = wave_type)
            if in_between == True:
                start_pad = start - 1000
                end_pad = end + 1000
                former_label = get_position_label(wave_indices,start_pad)
                latter_label = get_position_label(wave_indices,end_pad)
                if former_label != None:
                    ax[i].plot(time[start_pad:start],wave_array[start_pad:start],
                            color = c[former_label],label = former_label)

                if latter_label != None:
                    ax[i].plot(time[end:end_pad],wave_array[end:end_pad],
                            color = c[latter_label], label = latter_label)
        ax[i].legend()
        ax[i].set_xlabel(f'Time (s). Sampling rate {sr} (Hz)')

        plt.suptitle(f'{n_plots} sample plots of {wave_type} wave')

### Interactive
def interactive_visualization(wave_array, ana_file, smoothen = False, sr = 100, title = ''):
    c={'np':'darkgoldenrod',    'c':'lightgreen',    'e1':'skyblue',    'e2':'deeppink',    'g':'darkblue',    'f':'crimson',    'pd':'olive'}
    time = np.linspace(0,len(wave_array)/sr,len(wave_array))
    ana = dc(ana_file)
    ana.loc[:,'time'] = ana.loc[:,'time'].apply(lambda x: int(x*100))
    wave_indices = utils.get_index(ana)

    if smoothen == True:
        fig = FigureWidgetResampler(go.Figure())
    elif smoothen == False:
        fig = go.Figure()

    for wave in wave_indices.keys():
        i = 0
        for start,end in wave_indices[wave]:
            if i == 0:
                fig.add_trace(go.Scatter(x = time[start:end], y = wave_array[start:end],line=dict(color=c[wave]),mode = 'lines',
                                        legendgroup = wave, name = wave))   
            else:
                fig.add_trace(go.Scatter(x = time[start:end], y = wave_array[start:end],line=dict(color=c[wave]),mode = 'lines',
                                        legendgroup = wave, name = wave, showlegend = False))  
            i+=1
    fig.update_layout(title_text= f"{title} Interative viewer")
    fig.update_layout( xaxis=dict( rangeslider=dict(visible=True), type="linear" ) )# Add range slider
    fig.show()


from scipy.signal import find_peaks
def visualize_fft_coefficients(wave_array, ana, sr = 100, wave_type: str = None, which = 0, 
                               log = False, is_average = False, n_windows = 10):
    '''
    Input:
        wave_array: dataframe containing wave data - 2 columns [time,amplitude]
        ana: analysis file corresponding to the wave_array
        wave_type: 'full' or type of waves in {'np','c','e1','e2','g','f'} that will determine if the whole waveform is plotted or only a specific wave type
        is_average: if True, takes the average of all fft rows
        n_windows: the number of coefficients rows to plot
    Output:
        plot of the fft coefficients
    '''    

    wave_indices = utils.get_index(ana)

    wave_indices_sub = wave_indices[wave_type]#get wave indices
    start, end = wave_indices_sub[which]

    if end-start < 1024:
        print('Shorter than window_size. Padded both side.')
        wave_sample = wave_array[(start+end)//2-1024:(start+end)//2+1024]
    else:
        wave_sample = wave_array[start:end]

    #short-time fourier transform
    stft_coefs = np.abs(librosa.stft(wave_sample,n_fft = 1024,center = False))

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
            ax1.plot(freq_axis, stft_coefs[:,i],label = f'{i}-th window')
        ax1.set_title(f"log coefs of {n_windows} {wave_type} wave windows" if log == True else f" coefs {n_windows} {wave_type} wave windows")

    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('log - Amplitude' if log == True else 'Amplitude')
    ax2.plot(wave_sample)
    ax2.set_title(f"Waveform {wave_type}")
    
def visualize_wavelet_coefficients(low_freq,high_freq):
    '''
    Input. 
        low_freq: low frequency coefficients
        high_freq: high frequency coefficients
    Output.
        Plot of wavelet coefficients (A and D) for each layer side by side 
    '''
    if len(low_freq) != len(high_freq):
        raise RuntimeError('Must input coefficients of each resolution/layer')
    n_plot = len(low_freq)
    _,axes = plt.subplots(n_plot,2,figsize = (12,2*n_plot))
    if n_plot == 1:
        axes[0].plot(high_freq[0],label = f'{1}-th high freq')
        axes[0].legend()
        axes[1].plot(low_freq[0],label = f'{1}-th low freq')
        axes[1].legend()
    else:
        for j in range(n_plot):
            axes[j,0].plot(high_freq[j],label = f'{j+1}-th high freq')
            axes[j,0].legend()
            axes[j,1].plot(low_freq[j],label = f'{j+1}-th low freq')
            axes[j,1].legend()

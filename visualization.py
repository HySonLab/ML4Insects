import numpy as np
import librosa
import matplotlib.pyplot as plt
import preprocessing

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

def color_dict():
    return c 

def visualize_wave(wave_array,ana,wave_type = 'whole', n_plots = None,sr = 100):
    '''
    Input:
        wave_array: wave signal of class np.ndarray
        ana: analysis file corresponding to the wave_array
        wave_type: 'full' or type of waves in {'np','c','e1','e2','g','f'} that will determine if the whole waveform is plotted or only a specific wave type
        n_plots: if wave_type is not 'full', then n_plots determines the number of wave type plots
        sr: sampling rate. Default = 100Hz.
    Output:
        plot of the full wave (wave types) with colors
    '''    

    if isinstance(ana,dict):
        wave_indices = ana
    else:
        wave_indices = preprocessing.get_index(ana)

    np.random.seed(28)
    c=color_dict()
    time = np.linspace(0,len(wave_array)/sr,len(wave_array))

    if wave_type == 'whole':

        plt.figure(figsize=(16,2))
        custom_legends = []
        i = 0
        for wave in wave_indices.keys():
            for start,end in wave_indices[wave]:
                plt.plot(time[start:end],wave_array[start:end],color = c[wave])
            custom_legends.append(Line2D([0], [0], color=c[wave], lw=4))
            i+=1

        plt.legend(custom_legends,wave_indices.keys(),loc = 'upper center',ncols=len(custom_legends))
        plt.xticks(np.arange(0,time.max(),2000),rotation = 45)
        plt.xlabel(f'Time (s). Sampling rate: {sr} (Hz)')
        plt.ylabel('Amplitude (V)')

    else:
        if isinstance(n_plots,int) == False:
            n_plots = len(wave_indices[wave_type])

        f,ax = plt.subplots(n_plots,1,figsize=(16,2*n_plots))
        for i in range(n_plots):
            start, end = wave_indices[wave_type][i]
            ax[i].plot(time[start:end],wave_array[start:end],color = c[wave_type])
        ax[i].set_xlabel(f'Time (s). Sampling rate {sr} (Hz)')
        plt.suptitle(f'{n_plots} sample plots of {wave_type} wave')

from scipy.signal import find_peaks
def visualize_fft_coefficients(wave_array,ana,wave_type, which = 0,is_average = False, n_wind = 10):
    '''
    Input:
        wave_array: dataframe containing wave data - 2 columns [time,amplitude]
        ana: analysis file corresponding to the wave_array
        wave_type: 'full' or type of waves in {'np','c','e1','e2','g','f'} that will determine if the whole waveform is plotted or only a specific wave type
        is_average: if True, takes the average of all fft rows
        n_wind: the number of coefficients rows to plot
    Output:
        plot of the fft coefficients
    '''    
    wave_indices = preprocessing.get_index(ana)
    #get wave indices
    sample_wave_indices = wave_indices[wave_type]
    start, end = sample_wave_indices[which]
    if end-start < 1024:
        print('Shorter than window_size')
        wave_sample = wave_array[(start+end)//2-1024:(start+end)//2+1024]
    else:
        wave_sample = wave_array[start:end]

    #short-time fourier transform
    wave_sample_stft = np.abs(librosa.stft(wave_sample,n_fft = 1024,center = False))
    wave_sample_stft = np.log(wave_sample_stft)

    print(f'n_coefs: {wave_sample_stft.shape[0]},n_windows: {wave_sample_stft.shape[1]}')

    freq_axis = librosa.fft_frequencies(sr=100,n_fft = 1024)

    f,(ax1,ax2) = plt.subplots(1,2,figsize=(12,2))
    if is_average == True:
        avg = np.sum(wave_sample_stft,axis = 1)/wave_sample_stft.shape[1] #average
        peaks_idx = find_peaks(avg,prominence = 0.75)[0]
        ax1.plot(freq_axis[peaks_idx], avg[peaks_idx], "xr")
        ax1.plot(freq_axis,avg,'r')
        ax1.set_title(f"log-Average of {wave_type} wave windows")

    else: 
        for i in range(n_wind):
            ax1.plot(freq_axis,wave_sample_stft[:,i],label = f'{i}-th window')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('log - Amplitude')
        ax1.set_title(f"log of {n_wind} {wave_type} wave windows")
    ax2.plot(wave_sample)
    ax2.set_title(f"{wave_type} wave")
    
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

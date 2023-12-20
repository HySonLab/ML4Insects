import EntropyHub
from copy import deepcopy as dc
from scipy.signal import find_peaks
from scipy.stats import skew
import librosa
from stats import hurst_dfa
import preprocessingV1
import numpy as np
import pandas as pd

var = ['mean', 'sd', 'sk', 'zcr', 'hurst', 'energy', 'sample_entropy', 
             'permutation_entropy', 'spectral_entropy', 'spectral_centroid', 'spectral_flatness']

def get_feature_matrix(df):
    d = []
    for i in range(len(df)):

        low_freq,high_freq = preprocessingV1.get_wavelet_coefficients(df[i,:],'db4',3)
        l = low_freq[2]
        h = high_freq[2]
        
        # Whole 
        m = 3
        sample_entropy = EntropyHub.SampEn(df[i,:], m)[0][m]
        permutation_entropy = EntropyHub.PermEn(df[i,:],m)[0][m-1]
        spectral_entropy = EntropyHub.SpecEn(df[i,:])[0]
        spectral_centroid = librosa.feature.spectral_centroid(y = df[i,:],sr=100,n_fft=1024,center=False).item()
        spectral_flatness = librosa.feature.spectral_flatness(y = df[i,:],n_fft = 1024,center = False).item()

        # low_freq
        
        hurst = hurst_dfa(l)
        energy = np.mean(l**2)

        # high_freq
        zcr = librosa.feature.zero_crossing_rate(y = h,frame_length = len(h),center = False).item()
        sk = skew(h)
        mean = np.mean(h)
        sd = np.std(h)
        
        # concat
        f = [mean, sd, sk, zcr, hurst, energy, sample_entropy, 
             permutation_entropy, spectral_entropy, spectral_centroid, spectral_flatness]
        
        d.append(f)

    d = np.stack([f for f in d])
    return d

from sklearn.preprocessing import StandardScaler
def preprocess(feature_matrix,lab):
    
    fm = np.concatenate([dc(feature_matrix),lab.reshape(-1,1)],axis=1)
    fm = pd.DataFrame(fm)
    fm.dropna(inplace=True)
    fm, label = fm.iloc[:,:-1].to_numpy(),fm.iloc[:,-1].to_numpy()

    n_row,n_col = fm.shape

    scaler = StandardScaler()
    for col in range(n_col):
        fm[:,col]=scaler.fit_transform(fm[:,col].reshape(-1,1)).ravel()

    return fm, label
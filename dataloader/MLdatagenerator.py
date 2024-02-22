import librosa
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew
from utils.stats import *

def get_feature_matrix(df):
    d = []
    for i in range(len(df)):
        feats_vector = []
        coef = pywt.wavedec(df[i,:],'sym4',level = 3)
        for i in range(len(coef)):
            feats = calculate_statistics(coef[i])
            feats_vector += feats
        d.append(feats_vector)
    d = np.stack([f for f in d])
    return d

from sklearn.preprocessing import MinMaxScaler
def preprocess(feature_matrix, label):
    fm = pd.DataFrame(feature_matrix)
    fm.fillna(0, inplace = True)
    fm = fm.to_numpy()

    scaler = MinMaxScaler()
    fm = scaler.fit_transform(fm)
    label = pd.Series(label).map({1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}).to_numpy()

    return fm, label
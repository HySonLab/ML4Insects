import numpy as np 
from scipy.stats import skew
import librosa 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = std**2
    rms = np.nanmean(np.sqrt(list_values**2))
    sk = skew(list_values)
    if np.isnan(sk):
        sk = 0
    zcr = librosa.feature.zero_crossing_rate(y = list_values,frame_length = len(list_values),center = False).item()
    en = shannon_entropy(list_values)
    perm_en = permutation_entropy(list_values)
    return [n5, n25, n75, n95, median, mean, std, var, rms, sk, zcr, en, perm_en]

def calculate_spectrum_statistics(list_values, n_fft = 1024, window_size = 1024, center = False):
    # fft  
    spectrum = np.abs(librosa.stft(list_values, n_fft = n_fft,center = center)).ravel()/window_size
    power_spectrum = spectrum**2

    spec_centroid = spectral_centroid(spectrum)
    spec_entropy = shannon_entropy(power_spectrum)
    spec_flatness = spectral_flatness(power_spectrum)
    return [spec_centroid, spec_entropy, spec_flatness]

def spectral_centroid(spectrum):
    return np.sum(spectrum*fft_freq) / np.sum(spectrum)

def spectral_flatness(spectrum):
    if 0. in spectrum:
        return 0
    else: 
        n = len(spectrum)
        return np.exp(np.mean(np.log(spectrum))/n)/(np.mean(spectrum)/n)
    
def shannon_entropy(array):
    max = array.max()
    min = array.min()
    if max == min:
        return np.log(2)
    else:
        normalized_array = (array - min)/(max - min)
        entropy = 0
        for i in range(len(array)):
            if normalized_array[i] != 0:
                entropy -= normalized_array[i]*np.log(normalized_array[i])
        return entropy

def time_delay_embedding(time_series, embedding_dimension, delay=1):
    """
    Time-delayed embedding.

    Parameters
    ----------
    time_series : np.ndarray
        The input time series, shape (n_times)
    embedding_dimension : int
        The embedding dimension (order).
    delay : int
        The delay between embedded points.

    Returns
    -------
    embedded : ndarray
        The embedded time series with shape (n_times - (order - 1) * delay, order).
    """
    series_length = len(time_series)
    embedded_series = np.empty((embedding_dimension, series_length - (embedding_dimension - 1) * delay))
    for i in range(embedding_dimension):
        embedded_series[i] = time_series[i * delay : i * delay + embedded_series.shape[1]]
    return embedded_series.T

def permutation_entropy(time_series, order=3, delay=1, normalize=False):
    """
    Permutation Entropy.

    Parameters
    ----------
    time_series : list | np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy

    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf

    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf

    Notes
    -----
    Last updated (Oct 2018) by Raphael Vallat (raphaelvallat9@gmail.com):
    - Major speed improvements
    - Use of base 2 instead of base e
    - Added normalization

    Examples
    --------
    1. Permutation entropy with order 2

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> print(permutation_entropy(x, order=2))
            0.918

    2. Normalized permutation entropy with order 3

        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(permutation_entropy(x, order=3, normalize=True))
            0.589
    """
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = time_delay_embedding(x, embedding_dimension=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(np.math.factorial(order))
    return pe

# def calculate_crossings(list_values):
#     zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) &gt; 0))[0]
#     no_zero_crossings = len(zero_crossing_indices)
#     mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) &gt; np.nanmean(list_values)))[0]
#     no_mean_crossings = len(mean_crossing_indices)
#     return [no_zero_crossings, no_mean_crossings]    
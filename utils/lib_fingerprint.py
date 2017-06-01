import numpy as np
import librosa
from tqdm import *
from skimage.measure import block_reduce

n_fft = 4096
hop_length = n_fft // 4
use_logmap = True
reduce_rows = 20
reduce_cols = 1
crop_rows = -1
crop_cols = -1
start_col = 5
limit = None

window = np.hanning(n_fft)

def job(y):  
    S = librosa.stft(y, n_fft = n_fft, hop_length = hop_length, window = window)
    amp = np.abs(S)
    if reduce_rows > 1 or reduce_cols > 1:
        amp = block_reduce(amp, (reduce_rows, reduce_cols), func = np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols-amp.shape[1])), 'constant')
    amp = amp[:crop_rows, :crop_cols]
    if use_logmap:
        amp = 10*np.log10(amp)
    amp = amp[5:-1, :]
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp)
    return amp
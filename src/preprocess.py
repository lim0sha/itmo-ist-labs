import os
import re

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch

from features import extract_features_from_trial, FEATS_PER_CHANNEL


def load_trials_from_file(file_path):
    mat_data = sio.loadmat(os.path.join(file_path))
    eeg_keys = sorted(
        [(int(re.search(r'\d+', k).group()), mat_data[k])
         for k in mat_data if re.match(r'.+_eeg\d+$', k)],
        key=lambda x: x[0]
    )
    return [data for _, data in eeg_keys]


def preprocess_trial(trial, fs=200):
    trial = trial - np.mean(trial, axis=1, keepdims=True) # ремуваем тренд
    b, a = iirnotch(50 / (fs / 2), Q=30) # удаляем частоты
    trial = filtfilt(b, a, trial, axis=1)
    nyq = fs / 2
    b, a = butter(4, [0.5 / nyq, 45 / nyq], btype='band') # по нормировке
    return filtfilt(b, a, trial, axis=1)


def normalize_trial(trial): # Z ахахах avg = 0 и D = 1
    mean = np.mean(trial, axis=1, keepdims=True)
    std = np.std(trial, axis=1, keepdims=True)
    std[std == 0] = 1 # не делить на 0, если дисперсия канала = 0
    return (trial - mean) / std


def process_one_file(file_path, n_trials=15):
    trials = load_trials_from_file(file_path)
    X = np.empty((n_trials, 62 * FEATS_PER_CHANNEL))
    for i, trial in enumerate(trials):
        trial = preprocess_trial(trial)
        trial = normalize_trial(trial)
        X[i] = extract_features_from_trial(trial)
    return X


def list_session_files(data_dir):
    return sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.mat') and f != 'label.mat'
    )
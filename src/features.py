import numpy as np

FS = 200  # частота дискретизации, Гц

FREQ_BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

FEATURE_NAMES = ['mean', 'std', 'var', 'skew', 'kurtosis', 'min', 'max', 'ptp',
                 'delta', 'theta', 'alpha', 'beta', 'gamma', 'complexity']
FEATS_PER_CHANNEL = len(FEATURE_NAMES)  # 14

CH_NAMES = ['Fp1', 'Fpz', 'Fp2',
            'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
            'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3']  # монтаж SEED, 62 канала

try:
    trapezoid = np.trapezoid
except AttributeError:
    trapezoid = np.trapz


def batched_psd(signals, fs=FS, nperseg=None, noverlap=None):
    """Welch-PSD для всех каналов сразу (эквивалент scipy.signal.welch)."""
    nperseg = nperseg or fs * 2
    noverlap = noverlap or nperseg // 2
    C, N = signals.shape
    win = np.hanning(nperseg)
    step = nperseg - noverlap
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)

    seg = np.lib.stride_tricks.sliding_window_view(signals, nperseg, axis=-1)
    seg = seg[..., 0::step, :].copy()
    seg -= np.mean(seg, axis=-1, keepdims=True)  # детрендинг сегмента
    seg *= win[None, None, :]
    ft = np.fft.rfft(seg, axis=-1, n=nperseg)
    psd = np.mean(ft.real ** 2 + ft.imag ** 2, axis=1)
    psd *= 1.0 / (fs * np.sum(win ** 2))
    psd[:, 1:-1] *= 2
    return freqs, psd


def band_relative_powers(psd, freqs):
    total = trapezoid(psd, freqs, axis=1)
    total[total == 0] = 1
    powers = np.zeros((psd.shape[0], len(FREQ_BANDS)))
    for b, (fmin, fmax) in enumerate(FREQ_BANDS):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        powers[:, b] = trapezoid(psd[:, idx], freqs[idx], axis=1)
    return powers / total[:, None]


def skew_kurtosis(x, axis=-1):
    mu = np.mean(x, axis=axis, keepdims=True)
    d = x - mu
    m2 = np.mean(d ** 2, axis=axis)
    m3 = np.mean(d ** 3, axis=axis)
    m4 = np.mean(d ** 4, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return m3 / m2 ** 1.5, m4 / m2 ** 2 - 3


def hjorth_complexity(signals):
    activity = np.var(signals, axis=1)
    d = np.diff(signals, axis=1)
    mobility_num = np.var(d, axis=1)
    mobility = np.sqrt(mobility_num / np.maximum(activity, 1e-30))

    dd = np.diff(d, axis=1)
    d_mobility = np.sqrt(np.var(dd, axis=1) / np.maximum(mobility_num, 1e-30))
    return d_mobility / np.maximum(mobility, 1e-30)


def extract_features_from_trial(trial, fs=FS):
    sk, ku = skew_kurtosis(trial, axis=1)
    stat = np.column_stack([
        np.mean(trial, axis=1), np.std(trial, axis=1), np.var(trial, axis=1),
        sk, ku, np.min(trial, axis=1), np.max(trial, axis=1), np.ptp(trial, axis=1)])

    freqs, psd = batched_psd(trial, fs=fs)
    spectral = band_relative_powers(psd, freqs)

    fd = hjorth_complexity(trial)[:, None]
    return np.concatenate([stat, spectral, fd], axis=1).flatten()

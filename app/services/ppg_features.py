"""PPG autocorrelation used by the V6C stacking training pipeline."""

import numpy as np

from app.services.hrv_v4 import PoorSignalQualityError, bandpass_filter


def ppg_autocorrelation(ppg, fs: float) -> float:
    signal = np.asarray(ppg, dtype=float)
    if signal.ndim != 1 or len(signal) < 10 or not np.isfinite(fs) or fs <= 16:
        raise PoorSignalQualityError("PPG_AC needs a valid PPG signal sampled above 16 Hz.")
    good = np.isfinite(signal)
    if good.sum() < 10:
        raise PoorSignalQualityError("Not enough finite PPG samples for PPG_AC.")
    if not good.all():
        indexes = np.arange(len(signal))
        signal = np.interp(indexes, indexes[good], signal[good])
    filtered = bandpass_filter(signal, fs)
    centered = filtered - np.mean(filtered)
    count = len(centered)
    nfft = 1 << ((2 * count - 1).bit_length())
    spectrum = np.fft.rfft(centered, n=nfft)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:count] / count
    if not np.isfinite(autocorrelation[0]) or abs(autocorrelation[0]) < 1e-12:
        raise PoorSignalQualityError("PPG signal has no usable variance for PPG_AC.")
    value = float(np.mean(np.abs(autocorrelation[1:] / autocorrelation[0])))
    if not np.isfinite(value):
        raise PoorSignalQualityError("PPG_AC is not finite.")
    return value

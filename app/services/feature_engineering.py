"""Trích đặc trưng HRV cho service — chuyển sang pipeline v4.

Toàn bộ logic tính toán nằm trong `hrv_v4.py` (bản sao đồng bộ với ML Lab,
nơi model `v4.pkl` được huấn luyện). File này chỉ giữ
API cũ cho consumer/router, và bộ lọc bandpass riêng cho phần vẽ đồ thị.

Bản cũ (dò đỉnh trên sóng thô, SampEn xấp xỉ, tắt lọc sinh lý) đã bị thay:
các lựa chọn đó khớp với model leakage đời v1–v3, không khớp model v4.
"""

import numpy as np
# pyrefly: ignore[untyped-import]
from scipy.signal import butter, filtfilt

from app.services.hrv_v4 import (
    HRV_FEATURE_NAMES,
    PoorSignalQualityError,
    extract_features_from_ppg,
    extract_features_from_rr,
)
from app.services.ppg_features import ppg_autocorrelation

SUPPORTED_FEATURE_NAMES = (*HRV_FEATURE_NAMES, "PPG_AC")

__all__ = [
    "SUPPORTED_FEATURE_NAMES",
    "PoorSignalQualityError",
    "butter_bandpass_filter",
    "extract_features_from_csv_data",
    "extract_hrv_features",
]


def extract_hrv_features(valid_rr) -> dict:
    """Chuỗi khoảng nhịp (ms) -> 16 đặc trưng HRV + SQI (pipeline v4)."""
    return extract_features_from_rr(list(np.asarray(valid_rr, dtype=float)))


def extract_features_from_csv_data(time_ms_array, ppg_array, fs: float = 125.0) -> dict:
    """Raw PPG -> 16 đặc trưng HRV + SQI (pipeline v4).

    Raise PoorSignalQualityError khi tín hiệu không đủ chất lượng —
    caller nên báo "chất lượng tín hiệu kém" thay vì trả kết quả đoán bừa.
    """
    features = extract_features_from_ppg(
        np.asarray(time_ms_array, dtype=float), np.asarray(ppg_array, dtype=float), fs=fs
    )
    features["PPG_AC"] = ppg_autocorrelation(ppg_array, fs)
    return features


def butter_bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=125.0, order=3):
    """Bandpass Butterworth — chỉ dùng cho phần sinh dữ liệu đồ thị (chartData)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", output="ba")  # type: ignore
    y = filtfilt(b, a, data)
    return y

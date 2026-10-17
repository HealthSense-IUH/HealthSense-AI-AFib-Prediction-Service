"""Xử lý tín hiệu PPG và trích đặc trưng HRV — bản v4 (đồng bộ với ML Lab).

Đây là bản sao có kiểm soát (vendored copy) của pipeline huấn luyện v4 trong
repo HealthSense-ML (`src/healthsense_core/signal_processing.py` +
`hrv_features.py`). Model `v4.pkl` được huấn luyện
bằng CHÍNH các hàm này — mọi thay đổi ở đây phải đồng bộ 2 phía và chạy lại
parity test (`tests/parity_check.py`), nếu không đặc trưng đưa vào model sẽ
lệch so với lúc huấn luyện một cách im lặng.

Khác biệt so với feature_engineering cũ (đã loại bỏ):
- SampEn thật (Sample Entropy m=2, r=0.2*SD) thay cho std(diff)/SDNN.
- Dò nhịp trên tín hiệu đã lọc bandpass 0.5–8 Hz, prominence theo z-score,
  khoảng cách tối thiểu 0.27s (~220 BPM) — không bỏ sót nhịp AF nhanh.
- Lọc sinh lý NN 250–2000 ms luôn bật.
- CV là tỉ số (không nhân 100), phổ dùng Welch, độ lệch chuẩn dùng ddof=1.
- Có SQI (Signal Quality Index): tín hiệu kém -> từ chối thay vì đoán bừa.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

# ===== Hằng số đồng bộ với healthsense_core.config =====
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 8.0
BANDPASS_ORDER = 4
MIN_BEAT_DISTANCE_S = 0.27
PEAK_PROMINENCE_Z = 0.5
NN_MIN_MS = 250
NN_MAX_MS = 2000
MIN_BEATS = 10

# Dải tần chuẩn (Hz)
VLF_LOW, LF_LOW, LF_HIGH, HF_HIGH = 0.0033, 0.04, 0.15, 0.4
RESAMPLE_FS = 4.0

# 16 đặc trưng HRV mà `compute_hrv_features()` sinh ra — đồng bộ với
# `healthsense_core.config.ALL_FEATURES` bên ML Lab. Đây là TOÀN BỘ những gì
# service tính được; model nào đòi đặc trưng ngoài danh sách này thì service
# không phục vụ nổi và sẽ bị từ chối lúc nạp (xem services/prediction.py).
HRV_FEATURE_NAMES = (
    "HR_mean",
    "Mean_NN",
    "SDNN",
    "RMSSD",
    "NN50",
    "pNN50",
    "CV",
    "LF",
    "HF",
    "Total_Power",
    "LF_HF_Ratio",
    "LF_norm",
    "HF_norm",
    "SD1",
    "SD2",
    "SampEn",
)

# ===== SQI — ngưỡng chất lượng tín hiệu =====
SQI_MIN_VALID_RATIO = 0.8  # tỉ lệ khoảng NN sống sót sau lọc sinh lý
SQI_HR_MIN = 30.0
SQI_HR_MAX = 220.0
# Spectral concentration: PPG thật dồn năng lượng vào dải nhịp tim 0.5-3.5 Hz
# (đo trên 35 bệnh nhân MIMIC + thiết bị MAX30102 thật: min 0.86; nhiễu trắng
# chỉ ~0.41). Dưới ngưỡng này gần như chắc chắn là nhiễu, không phải mạch.
SQI_MIN_SPECTRAL_CONC = 0.70
SQI_HR_BAND = (0.5, 3.5)  # Hz — dải tần chứa nhịp tim 30-210 BPM


class PoorSignalQualityError(ValueError):
    """Tín hiệu không đủ chất lượng để phân loại an toàn."""


# ============================================================
# Xử lý tín hiệu (bản sao signal_processing.py v4)
# ============================================================
def bandpass_filter(ppg: np.ndarray, fs: float) -> np.ndarray:
    sos = sp_signal.butter(
        BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype="bandpass", fs=fs, output="sos"
    )
    return sp_signal.sosfiltfilt(sos, ppg)


def detect_beats(ppg_filtered: np.ndarray, fs: float) -> np.ndarray:
    """Trả về thời điểm nhịp (giây). Prominence trên z-score nên không phụ
    thuộc biên độ tuyệt đối của cảm biến (IR count hay tín hiệu chuẩn hóa)."""
    std = np.std(ppg_filtered)
    if std == 0:
        return np.array([])
    z = (ppg_filtered - np.mean(ppg_filtered)) / std
    peaks, _ = sp_signal.find_peaks(
        z, distance=int(MIN_BEAT_DISTANCE_S * fs), prominence=PEAK_PROMINENCE_Z
    )
    return peaks / fs


def beats_to_nn(beat_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nhịp -> chuỗi NN (ms) + thời điểm; chỉ lọc giới hạn sinh lý.

    Với AFib, sự bất thường của NN chính là tín hiệu bệnh lý — không lọc
    theo độ lệch median, chỉ loại giá trị phi sinh lý do lỗi dò đỉnh.
    """
    if len(beat_times_s) < 2:
        return np.array([]), np.array([])
    nn = np.diff(beat_times_s) * 1000.0
    nn_times = beat_times_s[1:]
    mask = (nn >= NN_MIN_MS) & (nn <= NN_MAX_MS)
    return nn[mask], nn_times[mask]


# ============================================================
# 16 đặc trưng HRV (bản sao hrv_features.py v4)
# ============================================================
def _time_domain(nn: np.ndarray) -> dict:
    mean_nn = np.mean(nn)
    sdnn = np.std(nn, ddof=1)
    diff = np.diff(nn)
    rmssd = np.sqrt(np.mean(diff**2)) if len(diff) else 0.0
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = nn50 / len(diff) * 100 if len(diff) else 0.0
    return {
        "HR_mean": 60000.0 / mean_nn,
        "Mean_NN": mean_nn,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "CV": sdnn / mean_nn,
    }


def _frequency_domain(nn: np.ndarray, nn_times: np.ndarray) -> dict:
    out = {
        "LF": 0.0,
        "HF": 0.0,
        "Total_Power": 0.0,
        "LF_HF_Ratio": 0.0,
        "LF_norm": 0.0,
        "HF_norm": 0.0,
    }
    if len(nn) < 4:
        return out

    t_uniform = np.arange(nn_times[0], nn_times[-1], 1.0 / RESAMPLE_FS)
    if len(t_uniform) < 8:
        return out
    interp = interp1d(nn_times, nn, kind="cubic", fill_value="extrapolate")
    nn_uniform = interp(t_uniform)
    nn_uniform = nn_uniform - np.mean(nn_uniform)

    nperseg = min(len(nn_uniform), 128)
    freqs, psd = sp_signal.welch(nn_uniform, fs=RESAMPLE_FS, nperseg=nperseg)

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    lf = band_power(LF_LOW, LF_HIGH)
    hf = band_power(LF_HIGH, HF_HIGH)
    total = band_power(VLF_LOW, HF_HIGH)

    out["LF"] = lf
    out["HF"] = hf
    out["Total_Power"] = total
    out["LF_HF_Ratio"] = lf / hf if hf > 0 else 0.0
    if lf + hf > 0:
        out["LF_norm"] = lf / (lf + hf) * 100
        out["HF_norm"] = hf / (lf + hf) * 100
    return out


def _poincare(nn: np.ndarray) -> dict:
    diff = np.diff(nn)
    if len(diff) == 0:
        return {"SD1": 0.0, "SD2": 0.0}
    sd_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    sdnn = np.std(nn, ddof=1)
    sd1 = np.sqrt(0.5) * sd_diff
    sd2_sq = 2 * sdnn**2 - 0.5 * sd_diff**2
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    return {"SD1": sd1, "SD2": sd2}


def _sample_entropy(nn: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample Entropy thật (m=2, r=0.2*SD) — O(n²), đủ nhanh cho <200 nhịp."""
    n = len(nn)
    if n < m + 2:
        return 0.0
    sd = np.std(nn, ddof=1)
    if sd == 0:
        return 0.0
    r = r_factor * sd

    def count_matches(mm: int) -> int:
        templates = np.array([nn[i : i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def compute_hrv_features(nn_ms: np.ndarray, nn_times_s: np.ndarray) -> dict:
    """Đủ 16 đặc trưng; model v4 tự chọn 13 đặc trưng nó cần qua
    feature_names_in_ (nhóm LF bị loại khi huấn luyện vì cửa sổ 30s)."""
    features: dict = {}
    features.update(_time_domain(nn_ms))
    features.update(_frequency_domain(nn_ms, nn_times_s))
    features.update(_poincare(nn_ms))
    features["SampEn"] = _sample_entropy(nn_ms)
    return {k: float(v) for k, v in features.items()}


# ============================================================
# Chỉ số hiển thị bổ sung (không đưa vào model — chỉ để app show)
# ============================================================
def _hr_range(nn_ms: np.ndarray) -> dict:
    """HR min/max trong phiên đo, dùng percentile 5/95 của nhịp tức thời
    để không bị 1 khoảng NN nhiễu kéo lệch."""
    if len(nn_ms) < 5:
        return {}
    inst_hr = 60000.0 / nn_ms
    return {
        "hrMin": round(float(np.percentile(inst_hr, 5)), 1),
        "hrMax": round(float(np.percentile(inst_hr, 95)), 1),
    }


def _stress_score(nn_ms: np.ndarray) -> dict:
    """Điểm căng thẳng 0-100 (THAM KHẢO) từ Baevsky Stress Index:
    SI = AMo / (2 * Mo * MxDMn), bin 50 ms theo chuẩn. Quy về 0-100 bằng
    thang log (SI ~30 thư giãn -> 0 điểm; SI ~1000 căng thẳng cao -> 100)."""
    if len(nn_ms) < 10:
        return {}
    bins = np.arange(nn_ms.min() - 25.0, nn_ms.max() + 75.0, 50.0)
    hist, edges = np.histogram(nn_ms, bins=bins)
    if hist.sum() == 0:
        return {}
    k = int(np.argmax(hist))
    amo = hist[k] / hist.sum() * 100.0
    mo_s = (edges[k] + 25.0) / 1000.0
    mxdmn_s = (nn_ms.max() - nn_ms.min()) / 1000.0
    if mo_s <= 0 or mxdmn_s <= 0:
        return {}
    si = amo / (2.0 * mo_s * mxdmn_s)
    score = (np.log(max(si, 1e-6)) - np.log(30.0)) / (np.log(1000.0) - np.log(30.0)) * 100.0
    return {"stressScore": int(np.clip(round(score), 0, 100))}


def _respiratory_rate(ppg_raw: np.ndarray, fs: float, duration_s: float) -> dict:
    """Nhịp thở (lần/phút) từ dao động hô hấp trong PPG (dải 0.1-0.4 Hz).
    Cần >= 30s dữ liệu để phổ đủ phân giải."""
    if duration_s < 30 or len(ppg_raw) < int(fs * 30):
        return {}
    freqs, psd = sp_signal.welch(
        ppg_raw - np.mean(ppg_raw), fs=fs, nperseg=min(len(ppg_raw), int(fs * 40))
    )
    band = (freqs >= 0.1) & (freqs <= 0.4)
    if not band.any() or psd[band].max() <= 0:
        return {}
    f_resp = float(freqs[band][np.argmax(psd[band])])
    return {"respiratoryRate": round(f_resp * 60.0, 1)}


def _perfusion_index(ppg_raw: np.ndarray, ppg_filtered: np.ndarray) -> dict:
    """Perfusion Index (%) = biên độ mạch đập (AC) / thành phần nền (DC).
    PI thấp bất thường -> tiếp xúc cảm biến kém / tưới máu yếu."""
    dc = float(np.median(ppg_raw))
    if dc <= 0:
        return {}
    ac = 2.0 * float(np.sqrt(2.0)) * float(np.std(ppg_filtered))  # xấp xỉ đỉnh-đáy
    return {"perfusionIndex": round(float(ac / dc * 100.0), 2)}


# ============================================================
# SQI + API cấp cao cho service
# ============================================================
def _spectral_concentration(ppg_filtered: np.ndarray, fs: float) -> float:
    """Phần năng lượng nằm trong dải nhịp tim (0.5-3.5 Hz) so với toàn dải
    đã lọc (0.5-8 Hz). Mạch thật gần tuần hoàn -> cao; nhiễu -> thấp."""
    freqs, psd = sp_signal.welch(ppg_filtered, fs=fs, nperseg=min(len(ppg_filtered), 1024))
    band = psd[(freqs >= SQI_HR_BAND[0]) & (freqs <= SQI_HR_BAND[1])].sum()
    total = psd[(freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)].sum()
    return float(band / total) if total > 0 else 0.0


def _check_quality(
    n_raw_intervals: int,
    nn_ms: np.ndarray,
    duration_s: float,
    spectral_conc: float | None = None,
) -> dict:
    """Đánh giá chất lượng tín hiệu. Trả về dict sqi, raise nếu quá kém."""
    n_valid = len(nn_ms)
    valid_ratio = n_valid / n_raw_intervals if n_raw_intervals > 0 else 0.0
    hr = 60000.0 / np.mean(nn_ms) if n_valid else 0.0

    reasons = []
    if spectral_conc is not None and spectral_conc < SQI_MIN_SPECTRAL_CONC:
        reasons.append(
            f"tín hiệu không có dạng sóng mạch (spectral concentration "
            f"{spectral_conc:.2f} < {SQI_MIN_SPECTRAL_CONC}) — nhiều khả năng là nhiễu"
        )
    if n_valid < MIN_BEATS:
        reasons.append(f"chỉ dò được {n_valid} nhịp hợp lệ (cần >= {MIN_BEATS})")
    if valid_ratio < SQI_MIN_VALID_RATIO:
        reasons.append(
            f"{(1 - valid_ratio) * 100:.0f}% khoảng nhịp bị loại vì phi sinh lý "
            f"(cho phép tối đa {(1 - SQI_MIN_VALID_RATIO) * 100:.0f}%)"
        )
    if n_valid >= MIN_BEATS and not (SQI_HR_MIN <= hr <= SQI_HR_MAX):
        reasons.append(f"nhịp tim trung bình {hr:.0f} BPM ngoài dải sinh lý")

    sqi = {
        "sqi_n_valid_beats": int(n_valid),
        "sqi_valid_ratio": round(float(valid_ratio), 3),
        "sqi_duration_s": round(float(duration_s), 1),
        "sqi_spectral_conc": round(spectral_conc, 3) if spectral_conc is not None else None,
        "sqi_ok": len(reasons) == 0,
    }
    if reasons:
        raise PoorSignalQualityError(
            "Chất lượng tín hiệu PPG không đủ để phân loại: " + "; ".join(reasons)
        )
    return sqi


def extract_features_from_ppg(time_ms: np.ndarray, ppg: np.ndarray, fs: float = 125.0) -> dict:
    """Raw PPG -> 16 đặc trưng HRV + chỉ số SQI. Raise PoorSignalQualityError
    nếu tín hiệu không đạt chất lượng."""
    ppg = np.asarray(ppg, dtype=np.float64)
    if np.isnan(ppg).any():
        idx = np.arange(len(ppg))
        good = ~np.isnan(ppg)
        if good.sum() < MIN_BEATS:
            raise PoorSignalQualityError("Tín hiệu PPG toàn giá trị trống (NaN).")
        ppg = np.interp(idx, idx[good], ppg[good])

    filtered = bandpass_filter(ppg, fs)
    beat_times = detect_beats(filtered, fs)
    nn_ms, nn_times = beats_to_nn(beat_times)

    duration_s = (time_ms[-1] - time_ms[0]) / 1000.0 if len(time_ms) >= 2 else 0.0
    conc = _spectral_concentration(filtered, fs)
    sqi = _check_quality(max(len(beat_times) - 1, 0), nn_ms, duration_s, spectral_conc=conc)

    features = compute_hrv_features(nn_ms, nn_times)
    features.update(sqi)
    # Chuỗi NN thô (ms) cho frontend vẽ đồ thị Poincaré (~80 giá trị/60s, nhẹ)
    features["nnIntervals"] = [round(float(v), 1) for v in nn_ms]
    # Chỉ số hiển thị bổ sung (không tham gia model)
    features.update(_hr_range(nn_ms))
    features.update(_stress_score(nn_ms))
    features.update(_respiratory_rate(ppg, fs, duration_s))
    features.update(_perfusion_index(ppg, filtered))
    return features


def extract_features_from_rr(rr_intervals_ms: list[float]) -> dict:
    """Chuỗi khoảng nhịp (ms, do client tự dò) -> 16 đặc trưng + SQI."""
    rr = np.asarray(rr_intervals_ms, dtype=np.float64)
    mask = (rr >= NN_MIN_MS) & (rr <= NN_MAX_MS)
    nn = rr[mask]
    # Thời điểm nhịp dựng lại từ cộng dồn NN (giây) — đủ cho phổ Welch
    nn_times = np.cumsum(nn) / 1000.0

    duration_s = float(np.sum(rr) / 1000.0)
    sqi = _check_quality(len(rr), nn, duration_s)

    features = compute_hrv_features(nn, nn_times)
    features.update(sqi)
    features["nnIntervals"] = [round(float(v), 1) for v in nn]
    features.update(_hr_range(nn))
    features.update(_stress_score(nn))
    return features

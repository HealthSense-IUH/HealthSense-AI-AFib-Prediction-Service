"""Chuẩn bị một bản ghi PPG trước khi phân tích, dùng chung cho consumer RabbitMQ và endpoint HTTP.

Bài báo mô tả dịch vụ suy luận như sau: ứng dụng ghi 72 giây PPG, dịch vụ bỏ 2 giây đầu (nhiễu khởi
động cảm biến), rồi chia phần còn lại thành các cửa sổ 30 s với bước 10 s. Trước đây chỉ consumer
bỏ 2 giây đầu, còn endpoint /api/predict-csv thì không, nên cùng một file CSV có thể cho hai kết
quả khác nhau. Gom về một hàm để hai đường vào luôn xử lý giống nhau.
"""

from __future__ import annotations

import numpy as np

WARMUP_MS = 2000.0  # bỏ 2 giây đầu: nhiễu khởi động cảm biến / transient
DEFAULT_FS = 125.0  # Hz, dùng khi không ước lượng được từ cột thời gian
_FS_ESTIMATE_SAMPLES = 100
_MIN_SAMPLES_AFTER_TRIM = 100


def prepare_recording(time_values, ppg_values) -> tuple[np.ndarray, np.ndarray, float]:
    """Chuẩn hoá thời gian về ms, ước lượng tần số lấy mẫu và bỏ 2 giây đầu.

    Trả về (time_ms, ppg, fs). Cột thời gian tính bằng giây (giá trị nhỏ, bước < 1) được đổi
    sang ms. fs lấy từ bước thời gian trung bình của tối đa 100 mẫu đầu.
    """
    time_ms = np.asarray(time_values, dtype=float)
    ppg = np.asarray(ppg_values, dtype=float)
    if len(time_ms) != len(ppg):
        raise ValueError("Cột thời gian và cột PPG phải có cùng số mẫu.")

    if len(time_ms) >= 2 and np.max(time_ms) < 100000 and np.mean(np.diff(time_ms)) < 1.0:
        time_ms = time_ms * 1000.0

    fs = DEFAULT_FS
    if len(time_ms) >= 2:
        dt_s = np.mean(np.diff(time_ms[:_FS_ESTIMATE_SAMPLES])) / 1000.0
        if dt_s > 0:
            fs = 1.0 / dt_s

    if len(time_ms) > 0:
        keep = time_ms >= time_ms[0] + WARMUP_MS
        if np.sum(keep) >= _MIN_SAMPLES_AFTER_TRIM:
            time_ms, ppg = time_ms[keep], ppg[keep]
        else:
            # Cột thời gian hỏng (không tăng dần): cắt theo số mẫu tương ứng 2 giây.
            cut = int(WARMUP_MS / 1000.0 * fs)
            if len(time_ms) > cut + 50:
                time_ms, ppg = time_ms[cut:], ppg[cut:]
    return time_ms, ppg, float(fs)

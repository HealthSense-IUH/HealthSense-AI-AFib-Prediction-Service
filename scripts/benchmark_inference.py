"""Đo chi phí tính toán của dịch vụ suy luận trên các bản ghi PPG thật.

Script nạp mô hình V6 qua đúng PredictionService của dịch vụ (kiểm tra SHA-256 với model card rồi
deserialise). Sau đó, với mỗi đoạn 72 s cắt từ các bản ghi, nó chạy đúng chuỗi xử lý của dịch vụ:
prepare_recording (bỏ 2 s đầu) và analyse_recording (cửa sổ 30 s / bước 10 s, cổng SQI, 14 đặc
trưng, mô hình stacking, quy tắc temporal persistence).

Số đo gồm:
- thời gian khởi động: kiểm tra SHA-256 cộng nạp mô hình;
- bộ nhớ tiến trình sau khi nạp;
- thời gian xử lý mỗi bản ghi 72 s, và riêng phần gọi mô hình cho mỗi cửa sổ;
- số cửa sổ qua và bị loại ở cổng SQI.

Kết quả phụ thuộc phần cứng; cấu hình máy được ghi kèm trong file JSON đầu ra.

Ví dụ (bản ghi MIMIC PERform AF, cột "Time" tính bằng giây và cột "PPG", 125 Hz):

    python scripts/benchmark_inference.py --model path/to/v6_stacking.pkl \
        --card path/to/healthsense_af_v6c_stacking.json \
        --recordings path/to/mimic_perform/af path/to/mimic_perform/non-af \
        --time-column Time --ppg-column PPG --out benchmark.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.hrv_v4 import PoorSignalQualityError  # noqa: E402
from app.services.prediction import PredictionService  # noqa: E402
from app.services.recording import prepare_recording  # noqa: E402
from app.services.windowing import analyse_recording, window_slices  # noqa: E402

RECORDING_S = 72.0  # thời lượng đo của ứng dụng di động


class TimedPredictor:
    """Bọc PredictionService để đo riêng thời gian gọi mô hình cho mỗi cửa sổ."""

    def __init__(self, service: PredictionService):
        self.service = service
        self.call_ms: list[float] = []

    def predict_with_base_probabilities(self, features: dict):
        start = time.perf_counter()
        result = self.service.predict_with_base_probabilities(features)
        self.call_ms.append((time.perf_counter() - start) * 1000.0)
        return result


def load_service(model: Path, card: Path) -> tuple[PredictionService, float]:
    """Nạp mô hình qua PredictionService từ một thư mục tạm (hard link, không chép 1.9 GB)."""
    workdir = Path(tempfile.mkdtemp(prefix="hs_bench_"))
    target = workdir / "v6_stacking.pkl"
    try:
        os.link(model, target)
    except OSError:
        shutil.copyfile(model, target)
    shutil.copyfile(card, workdir / "v6_stacking.json")

    service = PredictionService.__new__(PredictionService)
    service.model = None
    service.is_model_loaded = False
    service.active_model_file = target.name
    service.model_version = ""
    service.expected_features = []
    service._models_dir = str(workdir)
    start = time.perf_counter()
    if not service.load_model(target.name):
        raise SystemExit("Không nạp được mô hình (xem log).")
    return service, time.perf_counter() - start


def excerpts(path: Path, time_col: str, ppg_col: str, max_excerpts: int):
    frame = pd.read_csv(path, usecols=[time_col, ppg_col])
    t = frame[time_col].to_numpy(dtype=float)
    ppg = frame[ppg_col].to_numpy(dtype=float)
    fs = 1.0 / float(np.median(np.diff(t[:1000])))
    if fs < 1.0:  # cột thời gian tính bằng ms
        fs *= 1000.0
    n = int(round(RECORDING_S * fs))
    for k in range(min(max_excerpts, len(ppg) // n)):
        sl = slice(k * n, (k + 1) * n)
        yield k, t[sl], ppg[sl]


def cpu_name() -> str:
    """Tên CPU đọc từ hệ điều hành (platform.processor() trên Windows chỉ cho họ CPU)."""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def summarise(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr.max()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--time-column", default="Time")
    parser.add_argument("--ppg-column", default="PPG")
    parser.add_argument("--max-excerpts", type=int, default=4, help="số đoạn 72 s mỗi bản ghi")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    service, startup_s = load_service(args.model, args.card)
    rss_after = proc.memory_info().rss
    predictor = TimedPredictor(service)

    rows = []
    files = sorted(p for d in args.recordings for p in d.glob("*.csv"))
    for path in files:
        for k, t, ppg in excerpts(path, args.time_column, args.ppg_column, args.max_excerpts):
            start = time.perf_counter()
            time_ms, ppg_trim, fs = prepare_recording(t, ppg)
            try:
                analysis = analyse_recording(time_ms, ppg_trim, fs, predictor)
                accepted, rejected = analysis["acceptedWindows"], analysis["rejectedWindows"]
            except PoorSignalQualityError:  # không cửa sổ nào qua cổng SQI
                accepted, rejected = 0, len(window_slices(len(ppg_trim), fs))
            # Không ghi quyết định cảnh báo: đây là phép đo chi phí, và bản ghi dùng để đo có thể
            # nằm trong dữ liệu huấn luyện (như MIMIC PERform AF), nên kết quả không phải độ chính xác.
            rows.append(
                {
                    "recording": path.parent.name + "/" + path.stem,
                    "excerpt": k,
                    "fs_hz": round(fs, 2),
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                    "accepted_windows": accepted,
                    "rejected_windows": rejected,
                }
            )

    windows_total = sum(r["accepted_windows"] + r["rejected_windows"] for r in rows)
    accepted_total = sum(r["accepted_windows"] for r in rows)
    vm = psutil.virtual_memory()
    result = {
        "environment": {
            "platform": platform.platform(),
            "processor": cpu_name(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gb": round(vm.total / 1024**3, 1),
            "python": platform.python_version(),
            "packages": {
                name: __import__(name).__version__
                for name in ("numpy", "scipy", "sklearn", "xgboost", "joblib", "pandas")
            },
        },
        "model": {
            "file_size_mb": round(args.model.stat().st_size / 1024**2, 1),
            "startup_s_sha256_and_load": round(startup_s, 2),
            "rss_increase_mb": round((rss_after - rss_before) / 1024**2, 1),
            "rss_after_load_mb": round(rss_after / 1024**2, 1),
        },
        "recordings": len(files),
        "excerpts": len(rows),
        "recording_seconds": RECORDING_S,
        "latency_ms_per_72s_recording": summarise([r["latency_ms"] for r in rows]),
        "model_call_ms_per_window": summarise(predictor.call_ms),
        "windows_total": windows_total,
        "windows_accepted": accepted_total,
        "window_acceptance_rate": round(accepted_total / windows_total, 4)
        if windows_total
        else None,
        "excerpts_without_accepted_window": sum(r["accepted_windows"] == 0 for r in rows),
        "rows": [{**r, "latency_ms": round(r["latency_ms"], 1)} for r in rows],
    }
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    keys = ("model", "excerpts", "latency_ms_per_72s_recording", "model_call_ms_per_window")
    print(json.dumps({k: result[k] for k in keys}, indent=1))
    print("window acceptance rate:", result["window_acceptance_rate"])


if __name__ == "__main__":
    main()

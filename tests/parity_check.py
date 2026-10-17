# ruff: noqa: E402  (bắt buộc chèn sys.path trước khi import 2 package cần so sánh)
"""Parity check: đặc trưng + dự đoán của AI AFib Prediction Service phải khớp ML Lab.

Model v4 được huấn luyện bằng code trong HealthSense-ML; service dùng bản
vendored (app/services/hrv_v4.py). Test này bảo đảm 2 bản cho ra CÙNG kết quả
trên dữ liệu thật — nếu fail thì KHÔNG được deploy.

Chạy (cần repo HealthSense-ML nằm cạnh repo này, có venv với sklearn/xgboost):
    python tests/parity_check.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICE_ROOT = os.path.dirname(HERE)
ML_LAB = os.path.join(os.path.dirname(SERVICE_ROOT), "HealthSense-MachineLearning-Lab")

sys.path.insert(0, SERVICE_ROOT)
sys.path.insert(0, os.path.join(ML_LAB, "src"))
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import pandas as pd

# --- Bản gốc ML Lab ---
from healthsense_core import config as lab_config
from healthsense_core.hrv_features import compute_hrv_features as lab_features
from healthsense_core.signal_processing import extract_nn_series as lab_nn

# --- Bản service (vendored) ---
from app.services.hrv_v4 import extract_features_from_ppg


def main() -> int:
    # 1. Lấy 60s PPG thật từ một bệnh nhân MIMIC
    sample = os.path.join(lab_config.RAW_DIR, "af", "mimic_perform_af_003_data.csv")
    df = pd.read_csv(sample, nrows=60 * 125)
    time_ms = df["Time"].to_numpy(float) * 1000.0
    ppg = df["PPG"].to_numpy(float)

    # 2. Đặc trưng: service vs lab
    svc = extract_features_from_ppg(time_ms, ppg, fs=125.0)
    nn, nn_t = lab_nn(ppg)
    lab = lab_features(nn, nn_t)

    print("So sánh 16 đặc trưng (service vs ML Lab):")
    max_rel = 0.0
    for k, lab_v in lab.items():
        svc_v = svc[k]
        rel = abs(svc_v - lab_v) / (abs(lab_v) + 1e-9)
        max_rel = max(max_rel, rel)
        flag = "OK " if rel < 1e-9 else "LỆCH!"
        print(f"  {flag} {k:<12} service={svc_v:.6f}  lab={lab_v:.6f}")
    if max_rel >= 1e-9:
        print(f"\n❌ FAIL: đặc trưng lệch (max relative diff = {max_rel:.2e})")
        return 1

    # 3. Dự đoán: nạp model như service (align theo feature_names_in_)
    model = joblib.load(os.path.join(SERVICE_ROOT, "app", "models", "v4.pkl"))
    cols = [str(f) for f in model.feature_names_in_]
    missing = [c for c in cols if c not in svc]
    if missing:
        print(f"\n❌ FAIL: service thiếu đặc trưng model cần: {missing}")
        return 1

    x_svc = pd.DataFrame([{c: svc[c] for c in cols}], columns=cols)
    x_lab = pd.DataFrame([{c: lab[c] for c in cols}], columns=cols)
    p_svc = float(model.predict_proba(x_svc)[0, 1])
    p_lab = float(model.predict_proba(x_lab)[0, 1])

    print(f"\nP(AFib): service={p_svc:.6f}  lab={p_lab:.6f}")
    if abs(p_svc - p_lab) > 1e-9:
        print("❌ FAIL: xác suất dự đoán lệch")
        return 1

    print("\n✅ PARITY PASS — service tính giống hệt ML Lab, an toàn để deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Docker build smoke check for the configured model and raw-PPG inference."""

import numpy as np

from app.services.feature_engineering import extract_features_from_csv_data
from app.services.prediction import prediction_service


def main() -> None:
    if not prediction_service.is_model_loaded:
        raise RuntimeError("Configured model failed to load; refuse to publish this image.")
    fs = 125.0
    time = np.arange(30 * int(fs), dtype=float) / fs
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.15 * np.sin(2 * np.pi * 2.4 * time)
    features = extract_features_from_csv_data(time * 1000, signal, fs)
    label, probability = prediction_service.predict(features)
    if label != "AFib_Probability" or not 0 <= probability <= 1:
        raise RuntimeError("Configured model failed the inference smoke check.")
    print(
        f"Model smoke check passed: {prediction_service.active_model_file}, "
        f"{prediction_service.model_version}, {len(prediction_service.expected_features)} features."
    )


if __name__ == "__main__":
    main()

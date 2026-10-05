"""Regression checks for V6 stacking order, validation and PPG autocorrelation."""

import hashlib
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.hrv_v4 import PoorSignalQualityError, bandpass_filter
from app.services.ppg_features import ppg_autocorrelation


class BaseModel:
    feature_names_in_ = np.array(["HR_mean", "PPG_AC"])
    classes_ = np.array([0, 1])

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        if list(frame.columns) != list(self.feature_names_in_):
            raise AssertionError("Base feature order changed")
        return np.array([[1 - self.probability, self.probability]])


class MetaModel:
    steps = [("scaler", object()), ("classifier", object())]
    n_features_in_ = 2
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        self.received = np.asarray(values)
        return np.array([[0.3, 0.7]])


class V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (
            patch.dict(os.environ, {"MODEL_FILE": "unused-test-model.pkl"}),
            patch.object(logging.Logger, "exception"),
        ):
            from app.services.prediction import PredictionService

        cls.service_class = PredictionService

    def setUp(self):
        self.meta = MetaModel()
        self.bundle = {
            "features": ["HR_mean", "PPG_AC"],
            "base_models": {"first": BaseModel(0.8), "second": BaseModel(0.2)},
            "meta": self.meta,
        }
        self.service = self.service_class.__new__(self.service_class)
        self.service.model = self.bundle
        self.service.expected_features = self.bundle["features"]
        self.service.is_model_loaded = True
        self.service.active_model_file = "v6_stacking.pkl"
        self.service.model_version = "test"

    def test_bundle_order_matches_training(self):
        self.assertTrue(self.service._validate(self.bundle, "v6.pkl")[0])
        result = self.service.predict({"PPG_AC": 0.2, "HR_mean": 75})
        self.assertEqual(result, ("AFib_Probability", 0.7))
        np.testing.assert_array_equal(self.meta.received, [[0.8, 0.2]])

    def test_named_meta_features_control_order(self):
        self.meta.feature_names_in_ = np.array(["second", "first"])
        self.service.predict({"PPG_AC": 0.2, "HR_mean": 75})
        np.testing.assert_array_equal(self.meta.received, [[0.2, 0.8]])

    def test_missing_ppg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PPG_AC"):
            self.service.predict({"HR_mean": 75})

    def test_nonfinite_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.service.predict({"HR_mean": 75, "PPG_AC": np.nan})

    def test_invalid_bundle_is_rejected(self):
        self.bundle["features"] = ["unknown"]
        self.assertFalse(self.service._validate(self.bundle, "v6.pkl")[0])

    def test_base_schema_mismatch_is_rejected(self):
        self.bundle["features"] = ["PPG_AC", "HR_mean"]
        self.assertFalse(self.service._validate(self.bundle, "v6.pkl")[0])

    def test_failed_switch_keeps_active_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.service._models_dir = temporary
            path = Path(temporary) / "bad.pkl"
            path.write_bytes(b"invalid model")
            path.with_suffix(".json").write_text(
                json.dumps({"sha256": hashlib.sha256(b"other bytes").hexdigest()})
            )
            with patch.object(logging.Logger, "exception"):
                self.assertFalse(self.service.load_model("bad.pkl"))
            self.assertIs(self.service.model, self.bundle)
            self.assertEqual(self.service.active_model_file, "v6_stacking.pkl")
            self.assertTrue(self.service.is_model_loaded)

    def test_autocorrelation_matches_direct_training_formula(self):
        for fs in (50.0, 125.0):
            time = np.arange(int(30 * fs)) / fs
            ppg = np.sin(2 * np.pi * 1.2 * time) + 0.1 * np.sin(2 * np.pi * 2.4 * time)
            filtered = bandpass_filter(ppg, fs)
            centered = filtered - filtered.mean()
            direct = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
            expected = float(np.mean(np.abs(direct[1:] / direct[0])))
            self.assertAlmostEqual(ppg_autocorrelation(ppg, fs), expected, places=12)

    def test_flat_ppg_is_rejected(self):
        with self.assertRaises(PoorSignalQualityError):
            ppg_autocorrelation(np.zeros(3750), 125)

    def test_rr_request_accepts_optional_ppg_ac(self):
        from app.schemas.health_data import SensorDataRequest

        request = SensorDataRequest(rr_intervals=[800.0] * 20, ppg_ac=0.2)
        self.assertEqual(request.ppg_ac, 0.2)
        with self.assertRaises(ValueError):
            SensorDataRequest(rr_intervals=[800.0] * 20, ppg_ac=np.nan)


if __name__ == "__main__":
    unittest.main()

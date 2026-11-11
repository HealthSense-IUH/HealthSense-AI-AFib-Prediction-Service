"""Parity: AI Service feature extraction vs the ML-Lab code that produced the V6 training data.

Lab side: healthsense_core (bandpass_filter, extract_nn_series, compute_hrv_features) and the
PPG_AC definition (autocorrelation_feature) copied into every V5/V6 feature-extraction script.
Service side: app/services/hrv_v4.py and app/services/ppg_features.py.

Signals: real MIMIC PERform AF recordings from the ML-Lab repository, resampled to the rates
of the training domains and the wristband (125 Hz MIMIC, 100 Hz wristband/Liu, 50 Hz PulseWatch,
32 Hz DeepBeat), in 30 s and 60 s windows.

Needs a checkout of HealthSense-MachineLearning-Lab (src/, data/raw/mimic_perform/ and the
feature scripts in notebooks/v5/scripts and notebooks/v6/scripts). Set ML_LAB_PATH or place the
repository next to this one; otherwise the tests are skipped, unless PARITY_REQUIRED=1 (as in CI),
in which case a missing checkout fails the run.
"""

import ast
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

SERVICE_ROOT = Path(__file__).resolve().parents[1]
ML_LAB = Path(
    os.environ.get("ML_LAB_PATH", SERVICE_ROOT.parent / "HealthSense-MachineLearning-Lab")
).resolve()

LAB_FEATURE_SCRIPTS = (
    "notebooks/v5/scripts/build_cosen_ac_features.py",
    "notebooks/v5/scripts/deepbeat_frozen_v5_external.py",
    "notebooks/v5/scripts/liu_pseudo30_frozen_v5.py",
    "notebooks/v5/scripts/peakdet_pac_pvc_frozen_v5.py",
    "notebooks/v5/scripts/pulsewatch_frozen_v5_external.py",
    "notebooks/v6/scripts/extract_deepbeat_v6_features.py",
)
HRV_V6 = (
    "HR_mean", "Mean_NN", "SDNN", "RMSSD", "NN50", "pNN50", "CV",
    "HF", "Total_Power", "HF_norm", "SD1", "SD2", "SampEn",
)  # fmt: skip
RECORDS = (
    ("af", "mimic_perform_af_001"),
    ("af", "mimic_perform_af_007"),
    ("af", "mimic_perform_af_013"),
    ("non-af", "mimic_perform_non_af_002"),
    ("non-af", "mimic_perform_non_af_008"),
    ("non-af", "mimic_perform_non_af_014"),
)
NATIVE_FS = 125
RATES = (125, 100, 50, 32)
WINDOWS_S = (30, 60)
STARTS_S = (60, 600)
RTOL = 1e-9


def _lab_available() -> bool:
    return (ML_LAB / "src/healthsense_core").is_dir() and (
        ML_LAB / "data/raw/mimic_perform"
    ).is_dir()


def _resample(x: np.ndarray, fs_out: int) -> np.ndarray:
    if fs_out == NATIVE_FS:
        return x
    g = np.gcd(fs_out, NATIVE_FS)
    return sp_signal.resample_poly(x, fs_out // g, NATIVE_FS // g)


def _load_lab_autocorrelation(script: Path, bandpass_filter):
    """Compile only the autocorrelation_feature definition of a lab script (nothing else runs)."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "autocorrelation_feature"
    )
    namespace = {"np": np, "bandpass_filter": bandpass_filter, "FS": NATIVE_FS}
    module = ast.Module(body=[func], type_ignores=[])
    exec(compile(module, str(script), "exec"), namespace)  # noqa: S102 - trusted lab code
    fn = namespace["autocorrelation_feature"]
    takes_fs = len(func.args.args) > 1
    return (lambda x, fs: fn(x, fs)) if takes_fs else (lambda x, fs: fn(x))


def _close(a: float, b: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


class ParityEnvironmentTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("PARITY_REQUIRED") == "1", "parity not required here")
    def test_ml_lab_checkout_present(self):
        self.assertTrue(_lab_available(), f"ML-Lab checkout not found at {ML_LAB}")


@unittest.skipUnless(_lab_available(), "ML-Lab checkout not found (set ML_LAB_PATH)")
class ParityV6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (str(SERVICE_ROOT), str(ML_LAB / "src")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import pandas as pd
        from healthsense_core import config as lab_config
        from healthsense_core import signal_processing as lab_sp
        from healthsense_core.hrv_features import compute_hrv_features as lab_hrv

        from app.services import hrv_v4 as svc
        from app.services.ppg_features import ppg_autocorrelation

        cls.lab_config, cls.lab_sp, cls.svc = lab_config, lab_sp, svc
        cls.lab_hrv = staticmethod(lab_hrv)
        cls.svc_ac = staticmethod(ppg_autocorrelation)
        cls.lab_ac = {
            s: _load_lab_autocorrelation(ML_LAB / s, lab_sp.bandpass_filter)
            for s in LAB_FEATURE_SCRIPTS
        }
        cls.windows = []
        for group, record in RECORDS:
            path = ML_LAB / "data/raw/mimic_perform" / group / f"{record}_data.csv"
            ppg = pd.read_csv(path, usecols=["PPG"])["PPG"].to_numpy(float)
            for fs in RATES:
                x = _resample(ppg, fs)
                for win in WINDOWS_S:
                    for start in STARTS_S:
                        seg = x[start * fs : (start + win) * fs]
                        cls.windows.append((f"{record}@{fs}Hz[{start}+{win}s]", fs, seg))

    def test_signal_constants_match(self):
        for name in (
            "BANDPASS_LOW", "BANDPASS_HIGH", "BANDPASS_ORDER", "MIN_BEAT_DISTANCE_S",
            "PEAK_PROMINENCE_Z", "NN_MIN_MS", "NN_MAX_MS",
        ):  # fmt: skip
            with self.subTest(constant=name):
                self.assertEqual(getattr(self.svc, name), getattr(self.lab_config, name))

    def test_lab_ppg_ac_copies_agree(self):
        reference = LAB_FEATURE_SCRIPTS[-1]
        for label, fs, seg in self.windows:
            if fs != NATIVE_FS:  # build_cosen_ac_features.py hard-codes FS = 125
                continue
            expected = self.lab_ac[reference](seg, fs)
            for script, fn in self.lab_ac.items():
                with self.subTest(window=label, script=script):
                    self.assertTrue(_close(fn(seg, fs), expected))

    def test_filter_and_nn_series_identical(self):
        for label, fs, seg in self.windows:
            with self.subTest(window=label):
                np.testing.assert_array_equal(
                    self.svc.bandpass_filter(seg, fs), self.lab_sp.bandpass_filter(seg, fs)
                )
                svc_nn, svc_t = self.svc.beats_to_nn(
                    self.svc.detect_beats(self.svc.bandpass_filter(seg, fs), fs)
                )
                lab_nn, lab_t = self.lab_sp.extract_nn_series(seg, fs)
                np.testing.assert_array_equal(svc_nn, lab_nn)
                np.testing.assert_array_equal(svc_t, lab_t)

    def test_hrv_features_match(self):
        for label, fs, seg in self.windows:
            nn, nn_t = self.lab_sp.extract_nn_series(seg, fs)
            if len(nn) < self.svc.MIN_BEATS:
                continue
            lab = self.lab_hrv(nn, nn_t)
            svc = self.svc.compute_hrv_features(nn, nn_t)
            for feature in HRV_V6:
                with self.subTest(window=label, feature=feature):
                    self.assertTrue(
                        _close(float(svc[feature]), float(lab[feature])),
                        f"service={svc[feature]!r} lab={lab[feature]!r}",
                    )

    def test_ppg_ac_matches_lab(self):
        lab_fn = self.lab_ac[LAB_FEATURE_SCRIPTS[-1]]
        for label, fs, seg in self.windows:
            with self.subTest(window=label):
                self.assertTrue(_close(self.svc_ac(seg, fs), lab_fn(seg, fs)))

    def test_service_pipeline_matches_lab(self):
        checked = 0
        for label, fs, seg in self.windows:
            time_ms = np.arange(len(seg)) / fs * 1000.0
            try:
                svc = self.svc.extract_features_from_ppg(time_ms, seg, fs=fs)
            except self.svc.PoorSignalQualityError:
                continue  # the lab pipeline has no quality gate; nothing to compare
            nn, nn_t = self.lab_sp.extract_nn_series(seg, fs)
            lab = self.lab_hrv(nn, nn_t)
            checked += 1
            for feature in HRV_V6:
                with self.subTest(window=label, feature=feature):
                    self.assertTrue(_close(float(svc[feature]), float(lab[feature])))
        self.assertGreater(checked, len(self.windows) // 2, "too many windows rejected by SQI")


if __name__ == "__main__":
    unittest.main()

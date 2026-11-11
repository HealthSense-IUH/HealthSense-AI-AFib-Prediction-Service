"""Sliding-window inference and the temporal persistence rule (app/services/windowing.py)."""

import unittest

import numpy as np

from app.services.hrv_v4 import PoorSignalQualityError
from app.services.windowing import analyse_recording, evaluate_temporal_rule, window_slices

FS = 100.0


def pulse_signal(
    seconds: float, hr_bpm: float = 72.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth synthetic PPG with mild beat-to-beat variation; passes the quality gate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    rr = np.clip(rng.normal(60.0 / hr_bpm, 0.02, size=int(seconds * hr_bpm / 30) + 5), 0.5, 1.5)
    phase = np.interp(t, np.concatenate([[0.0], np.cumsum(rr)]), np.arange(len(rr) + 1))
    ppg = np.sin(2 * np.pi * phase) + 0.25 * np.sin(4 * np.pi * phase)
    return t * 1000.0, ppg


def windows(*specs):
    """specs: (p_meta, sigma) for accepted windows or None for a rejected window."""
    out = []
    for i, spec in enumerate(specs):
        w = {"index": i, "startS": 10.0 * i, "endS": 10.0 * i + 30.0}
        if spec is None:
            w.update(accepted=False, reason="rejected by SQI")
        else:
            w.update(accepted=True, pMeta=spec[0], sigmaDis=spec[1])
        out.append(w)
    return out


class FakePredictor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def predict_with_base_probabilities(self, features):
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output


def stacked(p: float, spread: float = 0.01):
    return p, {"ExtraTrees": p - spread, "RandomForest": p, "XGBoost": p + spread}


class WindowSlicesTest(unittest.TestCase):
    def test_seventy_seconds_give_five_windows(self):
        slices = window_slices(int(70 * FS), FS)
        self.assertEqual([s for s, _ in slices], [0, 1000, 2000, 3000, 4000])
        self.assertTrue(all(stop - start == 3000 for start, stop in slices))

    def test_recording_shorter_than_one_window(self):
        self.assertEqual(window_slices(int(29 * FS), FS), [])


class TemporalRuleTest(unittest.TestCase):
    def test_two_window_branch_alerts_at_forty_seconds(self):
        result = evaluate_temporal_rule(windows((0.95, 0.01), (0.93, 0.02), (0.2, 0.01)))
        self.assertEqual(
            result,
            {"alert": True, "alertBranch": "two_window_high_confidence", "alertTimeS": 40.0},
        )

    def test_five_window_branch_needs_five_consecutive_windows(self):
        four = evaluate_temporal_rule(windows(*[(0.85, 0.01)] * 4))
        five = evaluate_temporal_rule(windows(*[(0.85, 0.01)] * 5))
        self.assertFalse(four["alert"])
        self.assertEqual(five["alertBranch"], "five_window_persistence")
        self.assertEqual(five["alertTimeS"], 70.0)

    def test_rejected_window_resets_the_counters(self):
        result = evaluate_temporal_rule(windows((0.95, 0.01), None, (0.95, 0.01), (0.4, 0.01)))
        self.assertFalse(result["alert"])

    def test_high_disagreement_blocks_both_branches(self):
        result = evaluate_temporal_rule(windows(*[(0.95, 0.10)] * 6))
        self.assertFalse(result["alert"])


class AnalyseRecordingTest(unittest.TestCase):
    def test_clean_recording_alerts_and_averages_accepted_windows(self):
        t, ppg = pulse_signal(70.0)
        result = analyse_recording(t, ppg, FS, FakePredictor([stacked(0.95)]))
        self.assertEqual(result["acceptedWindows"], 5)
        self.assertEqual(result["rejectedWindows"], 0)
        self.assertAlmostEqual(result["meanProbability"], 0.95)
        self.assertTrue(result["temporalRuleAvailable"])
        self.assertTrue(result["alert"])
        self.assertEqual(result["alertTimeS"], 40.0)
        self.assertAlmostEqual(result["windows"][0]["sigmaDis"], 0.01 * np.sqrt(2 / 3))

    def test_mean_probability_uses_window_values(self):
        t, ppg = pulse_signal(70.0, seed=1)
        outputs = [stacked(p) for p in (0.2, 0.4, 0.6, 0.8, 1.0 - 0.01)]
        result = analyse_recording(t, ppg, FS, FakePredictor(outputs))
        self.assertAlmostEqual(result["meanProbability"], np.mean([0.2, 0.4, 0.6, 0.8, 0.99]))
        self.assertFalse(result["alert"])

    def test_white_noise_is_rejected(self):
        rng = np.random.default_rng(3)
        t = np.arange(int(70 * FS)) / FS * 1000.0
        with self.assertRaises(PoorSignalQualityError):
            analyse_recording(t, rng.standard_normal(len(t)), FS, FakePredictor([stacked(0.5)]))

    def test_short_recording_is_rejected(self):
        t, ppg = pulse_signal(20.0)
        with self.assertRaises(PoorSignalQualityError):
            analyse_recording(t, ppg, FS, FakePredictor([stacked(0.5)]))

    def test_non_stacking_model_disables_the_rule(self):
        t, ppg = pulse_signal(70.0, seed=2)
        result = analyse_recording(t, ppg, FS, FakePredictor([(0.95, {})]))
        self.assertFalse(result["temporalRuleAvailable"])
        self.assertFalse(result["alert"])


if __name__ == "__main__":
    unittest.main()

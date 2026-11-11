"""Sliding-window inference and the temporal persistence rule.

A recording is split into 30 s windows with a 10 s step. Each window goes through the
signal-quality gate and the 14-feature extraction on its own. Every accepted window gets two
values: the raw meta-learner probability P_meta, and the disagreement sigma_dis between the base
learners (population standard deviation of their probabilities).

The temporal persistence rule then raises an alert when either branch is met:
- five-window persistence branch: P_meta >= 0.80 and sigma_dis <= 0.05 in at least 5
  consecutive windows (70 s);
- two-window high-confidence branch: P_meta >= 0.90 and sigma_dis <= 0.05 in at least 2
  consecutive windows (40 s).

A window rejected by the quality gate resets both counters. The thresholds are the frozen values
of the research protocol (ML Lab results/v6/FROZEN_GLOBAL_RULE_V2.json).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.services.feature_engineering import extract_features_from_csv_data
from app.services.hrv_v4 import PoorSignalQualityError

WINDOW_S = 30.0
STEP_S = 10.0
FIVE_WINDOW_BRANCH = {
    "name": "five_window_persistence",
    "p_min": 0.80,
    "sigma_max": 0.05,
    "length": 5,
}
TWO_WINDOW_BRANCH = {
    "name": "two_window_high_confidence",
    "p_min": 0.90,
    "sigma_max": 0.05,
    "length": 2,
}


class WindowPredictor(Protocol):
    def predict_with_base_probabilities(self, features: dict) -> tuple[float, dict[str, float]]: ...


def window_slices(n_samples: int, fs: float) -> list[tuple[int, int]]:
    """Sample index ranges of all complete 30 s windows with a 10 s step."""
    n_win = int(round(WINDOW_S * fs))
    n_step = int(round(STEP_S * fs))
    if n_win <= 0 or n_step <= 0 or n_samples < n_win:
        return []
    return [(start, start + n_win) for start in range(0, n_samples - n_win + 1, n_step)]


def evaluate_temporal_rule(windows: list[dict]) -> dict:
    """Apply the two-branch persistence rule to windows in time order."""
    runs = {FIVE_WINDOW_BRANCH["name"]: 0, TWO_WINDOW_BRANCH["name"]: 0}
    for window in windows:
        sigma = window.get("sigmaDis")
        if not window.get("accepted") or sigma is None:
            runs = dict.fromkeys(runs, 0)  # gap or quality rejection resets the counters
            continue
        for branch in (TWO_WINDOW_BRANCH, FIVE_WINDOW_BRANCH):
            qualifies = window["pMeta"] >= branch["p_min"] and sigma <= branch["sigma_max"]
            runs[branch["name"]] = runs[branch["name"]] + 1 if qualifies else 0
            if runs[branch["name"]] >= branch["length"]:
                return {"alert": True, "alertBranch": branch["name"], "alertTimeS": window["endS"]}
    return {"alert": False, "alertBranch": None, "alertTimeS": None}


def analyse_recording(time_ms, ppg, fs: float, predictor: WindowPredictor) -> dict:
    """Window-level predictions, their mean and the temporal-rule decision for one recording.

    Raises PoorSignalQualityError if the recording is shorter than one window or if no window
    passes the quality gate.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    ppg = np.asarray(ppg, dtype=float)
    slices = window_slices(len(ppg), fs)
    if not slices:
        raise PoorSignalQualityError(
            f"Bản ghi quá ngắn: cần ít nhất {WINDOW_S:.0f} giây tín hiệu để phân tích."
        )

    windows = []
    for index, (start, stop) in enumerate(slices):
        window = {"index": index, "startS": round(start / fs, 2), "endS": round(stop / fs, 2)}
        try:
            features = extract_features_from_csv_data(time_ms[start:stop], ppg[start:stop], fs=fs)
            p_meta, base = predictor.predict_with_base_probabilities(features)
            sigma = float(np.std(list(base.values()))) if len(base) > 1 else None
            window.update(
                accepted=True,
                pMeta=float(p_meta),
                sigmaDis=sigma,
                baseProbabilities={name: float(p) for name, p in base.items()},
            )
        except PoorSignalQualityError as error:
            window.update(accepted=False, reason=str(error))
        windows.append(window)

    accepted = [w for w in windows if w["accepted"]]
    if not accepted:
        raise PoorSignalQualityError(
            "Không cửa sổ 30 giây nào đạt chất lượng tín hiệu; vui lòng giữ yên tay và đo lại."
        )

    decision = evaluate_temporal_rule(windows)
    return {
        "windowSeconds": WINDOW_S,
        "stepSeconds": STEP_S,
        "windows": windows,
        "acceptedWindows": len(accepted),
        "rejectedWindows": len(windows) - len(accepted),
        "meanProbability": float(np.mean([w["pMeta"] for w in accepted])),
        "temporalRuleAvailable": all(w["sigmaDis"] is not None for w in accepted),
        **decision,
    }

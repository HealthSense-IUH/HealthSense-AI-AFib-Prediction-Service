"""Model loading and inference for V6C stacking and legacy pipelines."""

import glob
import hashlib
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.services.feature_engineering import SUPPORTED_FEATURE_NAMES

logger = logging.getLogger("uvicorn.error")


class PredictionService:
    def __init__(self):
        self.model = None
        self.is_model_loaded = False
        self.active_model_file = settings.MODEL_FILE
        self.model_version = ""
        self.expected_features: list[str] = []
        self._models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        self.load_model(self.active_model_file)

    def _extract_feature_names(self, model) -> list[str]:
        if isinstance(model, dict):
            features = model.get("features", [])
            if not isinstance(features, (list, tuple, np.ndarray)):
                return []
            return [str(name) for name in features]
        if hasattr(model, "feature_names_in_"):
            return [str(name) for name in model.feature_names_in_]
        if hasattr(model, "steps") and model.steps:
            for step in (model.steps[0][1], model.steps[-1][1]):
                if hasattr(step, "feature_names_in_"):
                    return [str(name) for name in step.feature_names_in_]
        return []

    def _read_card(self, model_filename: str) -> dict:
        path = os.path.join(self._models_dir, f"{os.path.splitext(model_filename)[0]}.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as stream:
            card = json.load(stream)
        if not isinstance(card, dict):
            raise ValueError("Model card must be a JSON object.")
        return card

    def _validate(self, model, filename: str) -> tuple[bool, str]:
        names = self._extract_feature_names(model)
        if not names or len(set(names)) != len(names):
            return False, f"'{filename}' must declare unique feature names."
        unknown = [name for name in names if name not in SUPPORTED_FEATURE_NAMES]
        if unknown:
            return False, f"'{filename}' requires unsupported features: {unknown}."

        if isinstance(model, dict):
            bases = model.get("base_models")
            meta = model.get("meta")
            if not isinstance(bases, dict) or not bases:
                return False, "Stacking bundle must contain fitted base_models."
            if not (hasattr(meta, "steps") and len(meta.steps) >= 2):
                return False, "Stacking meta model must include its preprocessing pipeline."
            if not hasattr(meta, "predict_proba"):
                return False, "Stacking meta model must support predict_proba."
            if getattr(meta, "n_features_in_", None) != len(bases):
                return False, "Stacking meta input count does not match the base models."
            if list(getattr(meta, "classes_", [])) != [0, 1]:
                return False, "Stacking meta model must use binary classes [0, 1]."
            meta_names = self._extract_feature_names(meta)
            if meta_names and set(meta_names) != set(bases):
                return False, "Stacking meta feature names do not match the base models."
            for name, base in bases.items():
                if not hasattr(base, "predict_proba"):
                    return False, f"Base model '{name}' must support predict_proba."
                if self._extract_feature_names(base) != names:
                    return False, f"Base model '{name}' feature order does not match the bundle."
                if list(getattr(base, "classes_", [])) != [0, 1]:
                    return False, f"Base model '{name}' must use binary classes [0, 1]."
            return True, ""

        if not (hasattr(model, "steps") and len(getattr(model, "steps", [])) >= 2):
            return False, f"'{filename}' must include its preprocessing pipeline."
        return True, ""

    def load_model(self, model_filename: str) -> bool:
        """Only replace the active model after successful loading and validation."""
        try:
            if os.path.basename(model_filename) != model_filename:
                raise ValueError("Model must be a file name inside app/models.")
            path = os.path.join(self._models_dir, model_filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file not found: {model_filename}")
            card = self._read_card(model_filename)
            if model_filename == "v6_stacking.pkl" and not card.get("sha256"):
                raise ValueError("V6 requires its model card with an artifact SHA-256.")
            if card.get("sha256"):
                with open(path, "rb") as stream:
                    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_hash != card["sha256"]:
                    raise ValueError("Model file SHA-256 does not match its card.")
            loaded = joblib.load(path)
            ok, reason = self._validate(loaded, model_filename)
            if not ok:
                raise ValueError(reason)
            features = self._extract_feature_names(loaded)
            if isinstance(loaded, dict) and card.get("features") != features:
                raise ValueError("Stacking bundle feature order does not match its model card.")
            version = str(card.get("version") or "").strip().lstrip("vV")
            freeze_version = str(card.get("freeze_version") or "").strip()
            self.model = loaded
            self.active_model_file = model_filename
            self.model_version = (
                f"v{version}"
                if version
                else (freeze_version or os.path.splitext(model_filename)[0])
            )
            self.expected_features = features
            self.is_model_loaded = True
            logger.info(
                "Loaded model '%s' (%s, %d features)",
                model_filename,
                self.model_version,
                len(features),
            )
            return True
        except Exception:
            logger.exception("Could not load model '%s'; keeping the active model", model_filename)
            return False

    def list_available_models(self) -> list[dict]:
        paths = glob.glob(os.path.join(self._models_dir, "*.pkl")) + glob.glob(
            os.path.join(self._models_dir, "*.joblib")
        )
        result = []
        for path in sorted(paths):
            filename = os.path.basename(path)
            active = filename == self.active_model_file and self.is_model_loaded
            card = self._read_card(filename)
            features = self.expected_features if active else card.get("features")
            model_type = None
            if active:
                model_type = (
                    "StackingEnsembleBundle"
                    if isinstance(self.model, dict)
                    else type(self.model).__name__
                )
            result.append(
                {
                    "filename": filename,
                    "is_active": active,
                    "size_kb": round(os.path.getsize(path) / 1024, 2),
                    "feature_count": len(features) if features else None,
                    "expected_features": features,
                    "model_type": model_type,
                }
            )
        return result

    def predict(self, features_dict: dict) -> tuple[str, float]:
        probability, _ = self.predict_with_base_probabilities(features_dict)
        return "AFib_Probability", round(probability, 4)

    def predict_with_base_probabilities(
        self, features_dict: dict
    ) -> tuple[float, dict[str, float]]:
        """Raw AF probability and, for the V6 stacking bundle, each base learner's probability.

        The base probabilities give the disagreement sigma_dis used by the temporal persistence
        rule; non-stacking models return an empty dict.
        """
        if not self.is_model_loaded or self.model is None:
            raise RuntimeError(f"Model '{self.active_model_file}' is not loaded.")
        missing = [name for name in self.expected_features if name not in features_dict]
        if missing:
            raise ValueError(
                f"Missing required model features: {missing}. "
                "V6 needs raw PPG (CSV) or a supplied ppg_ac value; RR alone cannot provide PPG_AC."
            )
        input_df = pd.DataFrame(
            [[features_dict[name] for name in self.expected_features]],
            columns=self.expected_features,
        )
        if not np.isfinite(input_df.to_numpy(dtype=float)).all():
            raise ValueError("Model features must all be finite numbers.")

        base_output: dict[str, float] = {}
        if isinstance(self.model, dict):
            bases = self.model["base_models"]
            meta = self.model["meta"]
            base_probabilities = pd.DataFrame(
                {name: base.predict_proba(input_df)[:, 1] for name, base in bases.items()}
            )
            base_output = {name: float(base_probabilities[name].iloc[0]) for name in bases}
            meta_names = self._extract_feature_names(meta)
            meta_input = (
                base_probabilities[meta_names] if meta_names else base_probabilities.to_numpy()
            )
            probabilities = meta.predict_proba(meta_input)[0]
            classes = list(meta.classes_)
        elif hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_df)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))
        else:
            predicted = str(self.model.predict(input_df)[0]).strip().lower()
            return float(predicted in ("1", "1.0", "afib", "true")), {}

        positive = [
            index
            for index, value in enumerate(classes)
            if str(value).strip().lower() in ("1", "1.0", "afib", "true")
        ]
        if len(positive) != 1:
            raise ValueError("Model must declare exactly one AFib class.")
        probability = float(probabilities[positive[0]])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
        return probability, base_output


prediction_service = PredictionService()

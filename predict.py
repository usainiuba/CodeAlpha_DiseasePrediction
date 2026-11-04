"""Load the trained model and score one set of measurements."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "disease_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
OPTIONS_PATH = MODELS_DIR / "feature_options.json"
REPORT_PATH = MODELS_DIR / "classification_report.txt"


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def risk_band(probability):
    """Turn a probability into a plain-language band. Never a diagnosis."""
    if probability < 0.30:
        return {
            "tone": "low",
            "label": "Leans benign",
            "message": "These measurements look more like the benign cases the model "
            "learned from. This is not a diagnosis.",
        }
    if probability < 0.70:
        return {
            "tone": "mid",
            "label": "Uncertain",
            "message": "These measurements sit between the benign and malignant "
            "patterns, so the model cannot lean either way with confidence.",
        }
    return {
        "tone": "high",
        "label": "Leans malignant",
        "message": "These measurements look more like the malignant cases the model "
        "learned from. Only a clinician can confirm or rule this out.",
    }


def predict_one(model, values):
    """values is a dict {feature: number}. Returns the probabilities and the band."""
    row = pd.DataFrame([values], columns=FEATURES)
    malignant = float(model.predict_proba(row)[0, 1])
    return {"malignant": malignant, "benign": 1.0 - malignant, "band": risk_band(malignant)}


def position_rows(values, options, top_features, count=5):
    """Where each key value sits between the typical benign and malignant value.

    percent is the position on a track where 25% is the typical benign value
    and 75% is the typical malignant value. Values outside are clipped.
    """
    rows = []
    for item in top_features:
        name = item["feature"]
        spec = options[name]
        low, high = spec["benign_median"], spec["malignant_median"]
        span = spec["max"] - spec["min"]
        if span <= 0 or abs(high - low) / span < 0.02:
            continue  # the two groups look the same on this measurement
        position = (values[name] - low) / (high - low)
        position = max(-0.5, min(1.5, position))
        rows.append(
            {
                "feature": name,
                "value_text": f"{values[name]:.{spec['decimals']}f}",
                "percent": (position + 0.5) / 2 * 100,
            }
        )
        if len(rows) == count:
            break
    return rows

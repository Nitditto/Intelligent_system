"""Inference service for the house price prediction application.

CRITICAL ARCHITECTURAL RULE (Course Spec Part XV):
    This module NEVER calls .fit(), .fit_transform(), or instantiates new preprocessing
    components at runtime. The entire preprocessing pipeline (imputers, scalers, encoders)
    was fitted during model training on the training split and is loaded as-is from disk.
    Training != Inference.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C  # type: ignore

_LOCK = threading.Lock()
_PIPELINE: Any = None
_FEATURE_NAMES: Optional[list[str]] = None
_SCHEMA: Optional[dict] = None

# The model was trained on log1p(Price), so expm1 is applied at inference
_TARGET_IS_LOG1P = True


def get_pipeline():
    """Load the persisted sklearn Pipeline singleton behind a thread lock."""
    global _PIPELINE, _FEATURE_NAMES
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                if not C.MODEL_PIPELINE_PATH.exists():
                    raise FileNotFoundError(
                        f"Model pipeline artifact not found at {C.MODEL_PIPELINE_PATH}"
                    )
                _PIPELINE = joblib.load(C.MODEL_PIPELINE_PATH)

                if C.FEATURE_NAMES_PATH.exists():
                    _FEATURE_NAMES = list(joblib.load(C.FEATURE_NAMES_PATH))
                elif C.INPUT_SCHEMA_PATH.exists():
                    schema = _load_schema()
                    _FEATURE_NAMES = list(schema.get("model_features_order", []))
    return _PIPELINE


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None and C.INPUT_SCHEMA_PATH.exists():
        try:
            _SCHEMA = json.loads(C.INPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
        except Exception:
            _SCHEMA = {}
    return _SCHEMA or {}


def feature_names() -> list[str]:
    """Return the exact ordered list of features expected by the pipeline."""
    get_pipeline()
    return _FEATURE_NAMES or []


def model_name() -> str:
    """Return the name of the deployed model algorithm."""
    schema = _load_schema()
    chosen = schema.get("chosen_model")
    if chosen and not str(chosen).startswith("TODO"):
        return str(chosen)
    try:
        pipe = get_pipeline()
        last_step = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
        return type(last_step).__name__
    except Exception:
        return "RandomForestRegressor"


def is_ready() -> bool:
    """Check if model pipeline is loaded and ready for inference."""
    try:
        get_pipeline()
        return _PIPELINE is not None
    except Exception:
        return False


def _build_frame(payload: dict) -> pd.DataFrame:
    """Convert input dictionary into a 1-row pandas DataFrame.

    Auto-computes missing indicator flags when raw values are omitted,
    and fills unspecified values with np.nan so the pipeline's own SimpleImputers
    handle them without data leakage.
    """
    cols = feature_names()
    if not cols:
        return pd.DataFrame([payload])

    row: dict[str, Any] = {}
    for col in cols:
        if col == "Bedrooms_missing":
            row[col] = 1.0 if payload.get("Bedrooms") is None else 0.0
        elif col == "Bathrooms_missing":
            row[col] = 1.0 if payload.get("Bathrooms") is None else 0.0
        elif col == "Floors_missing":
            row[col] = 1.0 if payload.get("Floors") is None else 0.0
        else:
            val = payload.get(col)
            row[col] = np.nan if val is None else val

    return pd.DataFrame([row])[cols]


def predict_price(payload: dict) -> float:
    """Run model inference on raw property attributes and return estimated price in million VND."""
    pipe = get_pipeline()
    X = _build_frame(payload)
    pred = float(pipe.predict(X)[0])

    if _TARGET_IS_LOG1P:
        pred = float(np.expm1(pred))

    return max(0.0, pred * C.PRICE_UNIT_MULTIPLIER)

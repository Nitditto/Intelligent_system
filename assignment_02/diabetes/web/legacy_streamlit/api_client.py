"""Thin HTTP wrapper around the diabetes screening API for the Streamlit app."""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

API_URL = os.environ.get("DIABETES_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 30


def _get(path: str, **params) -> Any:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict, headers: Optional[dict] = None, **params) -> Any:
    r = requests.post(f"{API_URL}{path}", json=body, params=params,
                      headers=headers or {}, timeout=TIMEOUT)
    if r.status_code == 422:
        raise ValueError(r.json())
    r.raise_for_status()
    return r.json()


def health() -> dict:
    return _get("/healthz")


def model_info() -> dict:
    return _get("/model-info")


def questions() -> dict:
    return _get("/questions")


def predict(payload: dict, include: str = "explain,similar,whatif,counterfactual") -> dict:
    return _post("/predict", payload, include=include)


def history(session: str, limit: int = 50) -> dict:
    return _get("/history", session=session, limit=limit)


def metrics() -> dict:
    return _get("/metrics")


def threshold_curve(split: str = "val", grid: int = 49) -> dict:
    return _get("/threshold-curve", split=split, grid=grid)


def threshold_recommend(objective: str = "recall", target: float = 0.90) -> dict:
    return _get("/threshold-recommend", objective=objective, target=target)


def get_config() -> dict:
    return _get("/config")


def set_thresholds(moderate_cut: float, high_cut: float, admin_key: str) -> dict:
    return _post("/config/thresholds",
                 {"moderate_cut": moderate_cut, "high_cut": high_cut},
                 headers={"X-Admin-Key": admin_key})

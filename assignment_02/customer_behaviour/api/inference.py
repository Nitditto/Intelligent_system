"""Model loading and the core prediction path.

Loads the persisted pipeline once (``model/model_pipeline.joblib``), rebuilds the
feature frame with the same stateless ``build_features()`` used in the notebook,
turns P(satisfied) into a decision, and — for the deployed linear model — returns
an exact per-feature contribution breakdown (linear SHAP around the training mean).
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
from scipy.special import expit

from . import config as C
from .features import build_features

_LOCK = threading.Lock()
_PIPELINE = None
_EXPL = None            # {names, means, intercept, coef}  or  None if not linear


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                import joblib
                _PIPELINE = joblib.load(C.MODEL_PIPELINE_PATH)
    return _PIPELINE


def _explainer():
    """Loads model/feature_means.joblib once. Returns None if the deployed model
    is not linear (no coef) or the artifact is missing."""
    global _EXPL
    if _EXPL is not None:
        return _EXPL or None
    _EXPL = {}
    try:
        import joblib
        d = joblib.load(C.MODEL_MEANS_PATH)
        if "coef" in d and "intercept" in d:
            _EXPL = {
                "names": list(d["names"]),
                "means": np.asarray(d["means"], dtype=float).ravel(),
                "coef": np.asarray(d["coef"], dtype=float).ravel(),
                "intercept": float(d["intercept"]),
            }
    except Exception:
        _EXPL = {}
    return _EXPL or None


def _score(frame: pd.DataFrame) -> float:
    pipe = get_pipeline()
    if hasattr(pipe, "predict_proba"):
        return float(pipe.predict_proba(frame)[0, 1])
    return float(expit(pipe.decision_function(frame)[0]))


# --------------------------------------------------------------------------- main
def predict_one(raw_order: dict) -> dict:
    # pin the fields the wizard doesn't ask for (client value wins if supplied)
    merged = {**C.FIXED_INPUTS, **{k: v for k, v in raw_order.items() if v is not None}}
    feat = build_features(pd.DataFrame([merged]))
    p_sat = _score(feat)
    label = "satisfied" if p_sat >= C.DECISION_THRESHOLD else "dissatisfied"
    confidence = p_sat if label == "satisfied" else 1.0 - p_sat

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "p_satisfied": round(p_sat, 4),
        "threshold": C.DECISION_THRESHOLD,
        "signals": _signals(feat),
        "contributions": _contributions(feat, p_sat),
        "model": C.CHOSEN_MODEL,
        "representation": C.REPRESENTATION,
    }


def predict_batch(raw_orders: list[dict]) -> list[dict]:
    return [predict_one(o) for o in raw_orders]


# ------------------------------------------------------------------ SHAP-ish part
_NUM_PRETTY = {
    "price_total": "Order value", "freight_total": "Shipping cost",
    "payment_value_total": "Total charged", "freight_ratio": "Shipping-to-value ratio",
    "n_items": "Item count", "n_sellers": "Seller count",
    "max_installments": "Instalments", "n_payment_types": "Payment methods used",
    "product_weight_g": "Product weight", "product_desc_len": "Listing description length",
    "product_photos_qty": "Listing photo count", "comment_len": "Comment length",
    "estimated_days": "Promised delivery window",
}


def _pretty_label(name: str, feat_row: pd.Series, x_transformed: float) -> str:
    """name is a get_feature_names_out() entry like 'tab__std__delivery_delay_days'
    or 'tab__cat__category_grp_bed_bath_table' or 'txt__atrasado'."""
    if name.startswith("txt__"):
        return f'comment: "{name[5:]}"'

    parts = name.split("__")
    branch = parts[1] if len(parts) > 2 else ""
    rest = parts[-1]

    # one-hot columns
    if branch == "cat":
        for col, tpl in (("category_grp_", "Category: {}"),
                         ("customer_region_", "Region: {}"),
                         ("main_payment_type_", "Paid by {}")):
            if rest.startswith(col):
                return tpl.format(rest[len(col):].replace("infrequent_sklearn", "other"))
        return rest

    def _raw(col):
        v = feat_row.get(col)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    if rest == "delivery_delay_days":
        v = _raw("delivery_delay_days")
        if v is None:
            return "Delivery timing unknown"
        if v > 0.5:
            return f"Delivered {round(v)} days late"
        if v < -0.5:
            return f"Delivered {abs(round(v))} days early"
        return "Delivered on the promised date"
    if rest == "delivery_days":
        v = _raw("delivery_days")
        return f"Took {round(v)} days to arrive" if v is not None else "Delivery time unknown"
    if rest == "is_late":
        return "Order arrived late" if _raw("is_late") else "Order arrived on time"
    if rest == "has_comment":
        return "Customer left a comment" if _raw("has_comment") else "No comment left"

    pretty = _NUM_PRETTY.get(rest, rest.replace("_", " "))
    side = "above average" if x_transformed >= 0 else "below average"
    return f"{pretty}: {side}"


def _contributions(feat: pd.DataFrame, p_sat: float, top_k: int = 12) -> dict | None:
    ex = _explainer()
    if ex is None:
        return None
    pipe = get_pipeline()
    try:
        pre = pipe.named_steps["prep"]
        Xt = pre.transform(feat)
        x = np.asarray(Xt.todense()).ravel() if hasattr(Xt, "todense") else np.asarray(Xt).ravel()
        names, means, coef, b0 = ex["names"], ex["means"], ex["coef"], ex["intercept"]
        if not (len(x) == len(means) == len(coef) == len(names)):
            return None
    except Exception:
        return None

    phi = coef * (x - means)                       # per-feature log-odds contribution
    base_logit = b0 + float(coef @ means)
    base_p = float(expit(base_logit))

    row = feat.iloc[0]
    has_comment = bool(str(row.get(C.TEXT_COLUMN, "") or "").strip())
    idx = np.argsort(-np.abs(phi))
    items, shown = [], 0
    other = 0.0
    for j in idx:
        e = float(phi[j])
        is_txt = names[j].startswith("txt__")
        # an empty comment still gives tiny non-zero text phi (coef * -mean_j) —
        # fold those into "other" so the chart doesn't show phantom words
        if abs(e) < 1e-4 or (is_txt and not has_comment):
            other += e
            continue
        if shown < top_k:
            items.append({
                "label": _pretty_label(names[j], row, float(x[j])),
                "kind": "text" if names[j].startswith("txt__") else "tabular",
                "effect": round(e, 4),
            })
            shown += 1
        else:
            other += e

    return {
        "base_p": round(base_p, 4),
        "final_p": round(p_sat, 4),
        "dataset_base_rate": C.DATASET_BASE_RATE,
        "items": items,
        "other_effect": round(other, 4),
        "note": "bars are log-odds contributions; reference = the model's average order "
                "(class-balanced, so lower than the 79% dataset positive rate)",
    }


# ------------------------------------------------------------------------ signals
def _signals(feat: pd.DataFrame) -> dict:
    row = feat.iloc[0]
    delay = row.get("delivery_delay_days")
    ddays = row.get("delivery_days")
    return {
        "days_vs_promise": None if pd.isna(delay) else round(float(delay), 1),
        "late": bool(row.get("is_late", 0)),
        "has_comment": bool(row.get("has_comment", 0)),
        "delivery_days": None if pd.isna(ddays) else round(float(ddays), 1),
        "category_group": row.get("category_grp"),
        "customer_region": row.get("customer_region"),
    }


def model_info() -> dict:
    pipe = get_pipeline()
    schema = json.loads(C.INPUT_SCHEMA_PATH.read_text(encoding="utf-8"))
    ex = _explainer()
    return {
        "chosen_model": C.CHOSEN_MODEL,
        "representation": C.REPRESENTATION,
        "target": schema["target"],
        "decision_threshold": C.DECISION_THRESHOLD,
        "random_seed": C.RANDOM_SEED,
        "sklearn_version": C.SKLEARN_VERSION,
        "pipeline_steps": [s[0] for s in pipe.steps],
        "explainable": ex is not None,
        "n_features": len(ex["names"]) if ex else None,
        "raw_input_fields": C.RAW_INPUT_FIELDS,
        "tabular_feature_order": C.TABULAR_FEATURE_ORDER,
    }

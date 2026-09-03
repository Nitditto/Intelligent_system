"""Stateless raw-order -> feature-frame step.

This is a port of ``build_features()`` / ``clean_text()`` from notebook section 12.
It has **no fitted parameters** — every column is a deterministic function of one
raw order — so training and serving share one code path. The fitted imputers /
scaler / one-hot / TF-IDF all live inside ``model_pipeline.joblib`` and run
*after* this step.

Difference from the notebook version: the API may receive a partial order, so any
raw column the function reads is first guaranteed to exist (as all-NaN) — the
pipeline's imputers then fill it exactly as they were fitted to.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

_PUNCT = re.compile(r"[^\wÀ-ÿ\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# every raw column build_features() reads
_RAW_COLS = [
    "order_purchase_timestamp", "order_delivered_customer_date",
    "order_estimated_delivery_date", "price_total", "freight_total",
    "payment_value_total", "n_payment_types", "review_comment_message",
    "customer_state", "category", "product_weight_g", "product_desc_len",
    "product_photos_qty", "n_items", "n_sellers", "max_installments",
    "main_payment_type",
]


def clean_text(s) -> str:
    """lower-case, strip punctuation, collapse whitespace (notebook section 12)."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw order-level frame -> the exact columns the persisted pipeline expects
    (``TABULAR_FEATURE_ORDER + [TEXT_COLUMN]``)."""
    d = frame.copy()
    for col in _RAW_COLS:
        if col not in d.columns:
            d[col] = np.nan

    for col in ("order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"):
        d[col] = pd.to_datetime(d[col], errors="coerce")

    day = 86400.0
    d["delivery_days"] = (d.order_delivered_customer_date - d.order_purchase_timestamp).dt.total_seconds() / day
    d["estimated_days"] = (d.order_estimated_delivery_date - d.order_purchase_timestamp).dt.total_seconds() / day
    d["delivery_delay_days"] = (d.order_delivered_customer_date - d.order_estimated_delivery_date).dt.total_seconds() / day
    d["is_late"] = (d["delivery_delay_days"] > 0).astype(int)

    price = pd.to_numeric(d["price_total"], errors="coerce")
    freight = pd.to_numeric(d["freight_total"], errors="coerce")
    d["price_total"] = price
    d["freight_total"] = freight
    d["freight_ratio"] = (freight / price.replace(0, np.nan)).fillna(0.0)

    pay = pd.to_numeric(d["payment_value_total"], errors="coerce")
    d["payment_value_total"] = pay.fillna(price.fillna(0) + freight.fillna(0))

    d["n_payment_types"] = pd.to_numeric(d["n_payment_types"], errors="coerce").fillna(1)

    comment_raw = d["review_comment_message"]
    d["has_comment"] = comment_raw.notna().astype(int)
    d["comment_clean"] = comment_raw.map(clean_text)
    d["comment_len"] = d["comment_clean"].str.len()

    d["customer_region"] = d["customer_state"].map(C.STATE_TO_REGION).fillna("__other__")

    # notebook section 12: np.where returns "__other__" for every non-top value,
    # NaN included, so the notebook's trailing .fillna("__missing__") was a no-op.
    d["category_grp"] = np.where(d["category"].isin(C.TOP_CATEGORIES), d["category"], "__other__")

    for col in ("product_weight_g", "product_desc_len", "product_photos_qty",
                "n_items", "n_sellers", "max_installments"):
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # the one-hot encoder was fitted on string columns; a partial request can leave
    # a categorical entirely NaN (float dtype) which breaks OneHotEncoder. Force
    # object/str with a sentinel — handle_unknown="ignore" then emits an all-zero row.
    for col in ("main_payment_type", "customer_region", "category_grp"):
        d[col] = pd.Series(d[col], index=d.index).astype(object)
        d[col] = d[col].where(d[col].notna(), "__na__").astype(str)

    return d[C.TABULAR_FEATURE_ORDER + [C.TEXT_COLUMN]]

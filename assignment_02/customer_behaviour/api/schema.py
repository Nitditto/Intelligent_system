"""Pydantic request / response models for the customer-behaviour API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    """One raw order, exactly as the notebook's section-23 ``raw_order`` dict.

    Every field is optional except the three the model leans on most (order value,
    freight, and the three dates that define delivery lateness). Missing numeric
    fields are median-imputed inside the pipeline; missing categoricals fall back
    to a sentinel level.
    """
    price_total: float = Field(..., ge=0, description="Sum of item prices, BRL")
    freight_total: float = Field(..., ge=0, description="Sum of freight values, BRL")
    order_purchase_timestamp: str = Field(..., description="ISO datetime")
    order_estimated_delivery_date: str = Field(..., description="ISO datetime — promised date")
    order_delivered_customer_date: str = Field(..., description="ISO datetime — actual delivery")

    payment_value_total: Optional[float] = Field(None, ge=0)
    n_items: Optional[int] = Field(None, ge=1)
    n_sellers: Optional[int] = Field(None, ge=1)
    n_payment_types: Optional[int] = Field(None, ge=1)
    max_installments: Optional[int] = Field(None, ge=0, le=24)
    main_payment_type: Optional[str] = Field(None, examples=["credit_card"])
    customer_state: Optional[str] = Field(None, examples=["SP"], max_length=2)
    category: Optional[str] = Field(None, examples=["bed_bath_table"])
    product_weight_g: Optional[float] = Field(None, ge=0)
    product_desc_len: Optional[float] = Field(None, ge=0)
    product_photos_qty: Optional[float] = Field(None, ge=0)
    review_comment_message: Optional[str] = Field(None, description="Portuguese free text")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "price_total": 129.90, "freight_total": 18.30,
                "payment_value_total": 148.20, "n_items": 1, "n_sellers": 1,
                "max_installments": 3, "main_payment_type": "credit_card",
                "customer_state": "SP", "category": "bed_bath_table",
                "product_weight_g": 1200, "product_desc_len": 850, "product_photos_qty": 3,
                "order_purchase_timestamp": "2018-05-01 10:00:00",
                "order_estimated_delivery_date": "2018-05-20 00:00:00",
                "order_delivered_customer_date": "2018-05-31 14:00:00",
                "review_comment_message": "Produto chegou muito atrasado e a embalagem estava danificada.",
            }]
        }
    }


class BatchInput(BaseModel):
    orders: list[OrderInput]


class Prediction(BaseModel):
    prediction: str
    confidence: float
    p_satisfied: float
    threshold: float
    review_terms: Optional[dict] = None
    signals: dict
    contributions: Optional[dict] = None   # linear-SHAP breakdown (None if model not linear)
    model: str
    representation: str

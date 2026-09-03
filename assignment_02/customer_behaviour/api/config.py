"""Static configuration for the customer-behaviour API.

Filesystem paths and the small pieces of metadata the client needs to render its
form. Nothing here is fitted or learned — the fitted objects live in
``model/model_pipeline.joblib`` and the contract in ``model/input_schema.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

API_DIR = Path(__file__).resolve().parent          # customer_behaviour/api
APP_DIR = API_DIR.parent                             # customer_behaviour
MODEL_DIR = APP_DIR / "model"
DATA_DIR = APP_DIR / "data"

MODEL_PIPELINE_PATH = MODEL_DIR / "model_pipeline.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"
MODEL_MEANS_PATH = MODEL_DIR / "feature_means.joblib"
INPUT_SCHEMA_PATH = MODEL_DIR / "input_schema.json"

# decision threshold on P(satisfied); below it -> "dissatisfied" (the class we act on)
DECISION_THRESHOLD = 0.50

# share of orders that were positive (>=4 star) in the cleaned training data — shown
# as context next to the model's own (class-balanced) reference point.
DATASET_BASE_RATE = 0.79

# fields the web wizard does not ask for — filled server-side with a fixed value
# (dataset medians). Keeps the form short without changing the model contract.
FIXED_INPUTS = {
    "product_desc_len": 607,   # listing description length (chars) — median
    "n_payment_types": 1,      # almost every order uses a single payment method
}

# ------------------------------------------------------------------ schema-derived
_SCHEMA = json.loads(INPUT_SCHEMA_PATH.read_text(encoding="utf-8"))

RAW_INPUT_FIELDS: list[str] = _SCHEMA["raw_input_fields"]
TABULAR_FEATURE_ORDER: list[str] = _SCHEMA["tabular_feature_order"]
TEXT_COLUMN: str = _SCHEMA["text_column"]
TOP_CATEGORIES: list[str] = _SCHEMA["top_categories"]
STATE_TO_REGION: dict[str, str] = _SCHEMA["state_to_region"]
CHOSEN_MODEL: str = _SCHEMA["chosen_model"]
REPRESENTATION: str = _SCHEMA["representation"]
RANDOM_SEED: int = _SCHEMA["random_seed"]
SKLEARN_VERSION: str = _SCHEMA["sklearn_version"]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
BRAZIL_STATES = sorted(STATE_TO_REGION.keys())

# Portuguese example comments (with an English gloss) the client can offer as
# one-click fills for the review field.
COMMENT_EXAMPLES = [
    ["Produto excelente, chegou antes do prazo. Recomendo!",
     "Great product, arrived before the deadline. Recommend!"],
    ["Entrega rápida e produto conforme o anúncio.",
     "Fast delivery and product as advertised."],
    ["Até agora não recebi o produto.",
     "I still haven't received the product."],
    ["Veio com defeito e a caixa estava toda amassada.",
     "Came defective and the box was all crushed."],
    ["Produto diferente do que foi anunciado no site.",
     "Product different from what was advertised on the site."],
]

# The client renders its wizard from this list. Fields carry a `section` (= wizard
# step) and a plain-language `note`. `product_desc_len` and `n_payment_types` are
# not asked for — see FIXED_INPUTS.
#   type:  "number" | "choice" | "date" | "text"
FORM_FIELDS = [
    # =============================== step 1: Product ===============================
    {"field": "category", "type": "choice", "label": "Product category",
     "options": TOP_CATEGORIES + ["__other__"], "required": False, "section": "Product",
     "note": "The kind of product the order is for. Some categories (furniture, home) "
             "draw more complaints than others (watches, books)."},
    {"field": "price_total", "type": "number", "label": "Order value (R$)",
     "min": 0, "required": True, "section": "Product",
     "note": "Total price of the items, in Brazilian reais — shipping not included."},
    {"field": "n_items", "type": "number", "label": "Quantity (item units)", "min": 1,
     "required": False, "section": "Product",
     "note": "How many units are in the order (1 for a single product)."},
    {"field": "n_sellers", "type": "number", "label": "Seller count", "min": 1,
     "required": False, "section": "Product",
     "note": "How many different sellers the order is split across. More sellers = more "
             "shipments that can go wrong."},
    {"field": "product_photos_qty", "type": "number", "label": "Photo count in listing",
     "min": 0, "required": False, "section": "Product",
     "note": "How many photos the product page showed. Thin listings lead to "
             "“not as described” complaints."},
    {"field": "product_weight_g", "type": "number", "label": "Main product weight (g)",
     "min": 0, "required": False, "section": "Product",
     "note": "Weight of the main product in grams. Heavy or bulky items arrive damaged "
             "or late more often."},

    # =============================== step 2: Payment ==============================
    {"field": "main_payment_type", "type": "choice", "label": "Payment method",
     "options": PAYMENT_TYPES, "required": False, "section": "Payment",
     "note": "How the customer paid. “boleto” = Brazilian bank slip, "
             "“voucher” = store credit."},
    {"field": "max_installments", "type": "number", "label": "Instalments",
     "min": 1, "max": 24, "required": False, "section": "Payment",
     "note": "Monthly instalments the payment was split into (1 = paid in full)."},
    {"field": "freight_total", "type": "number", "label": "Shipping paid (R$)",
     "min": 0, "required": True, "section": "Payment",
     "note": "Freight cost the customer paid, in reais."},
    {"field": "payment_value_total", "type": "number", "label": "Total charged (R$)",
     "min": 0, "required": False, "section": "Payment",
     "note": "Amount actually charged (items + shipping). Leave blank and we use "
             "order value + shipping."},
    {"field": "customer_state", "type": "choice", "label": "Customer's state (Brazil)",
     "options": BRAZIL_STATES, "required": False, "section": "Payment",
     "note": "Two-letter Brazilian state code (e.g. SP = São Paulo). Used only to "
             "derive the customer's region."},

    # ========================= step 3: Delivery & review =========================
    {"field": "order_purchase_timestamp", "type": "date", "label": "Order placed on",
     "required": True, "section": "Delivery & review",
     "note": "When the customer placed the order."},
    {"field": "order_estimated_delivery_date", "type": "date", "label": "Delivery promised by",
     "required": True, "section": "Delivery & review",
     "note": "The delivery date the customer was shown at checkout."},
    {"field": "order_delivered_customer_date", "type": "date", "label": "Actually delivered on",
     "required": True, "section": "Delivery & review",
     "note": "When the order really reached the customer. Later than the promised date "
             "= “late”, and lateness is the single biggest cause of a bad review."},
    {"field": "review_comment_message", "type": "text", "label": "Review comment the customer wrote",
     "required": False, "section": "Delivery & review", "examples": COMMENT_EXAMPLES,
     "note": "The free-text comment left with the star rating (optional). The model "
             "learned from Brazilian-Portuguese comments, so Portuguese works best — "
             "other languages are accepted but barely move the prediction."},
]

WIZARD_STEPS = ["Product", "Payment", "Delivery & review"]

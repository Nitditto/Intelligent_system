"""Generate api/samples.json — a serving convenience, not a graded artifact.

Joins the 9 Olist CSVs the same way notebook section 3 does, keeps delivered
orders, then writes:

  {
    "defaults": { ...one raw order of typical (median / modal) values... },
    "examples": [ { ...raw OrderInput fields..., "_label": "...", "_true_score": 4 }, ... ]
  }

The web client uses `defaults` to pre-fill the fields Quick mode hides, and
`examples` for the "Load an example order" picker.

Run from customer_behaviour/ :   python api/make_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent / "samples.json"
SEED = 42
N_PER_SCORE = {5: 9, 4: 8, 3: 7, 2: 8, 1: 10}     # ~42 examples, skewed to the interesting end


def read_csv_smart(path, **kw) -> pd.DataFrame:
    """The Olist reviews file has circulated as both UTF-8 and Latin-1; try UTF-8,
    fall back to Latin-1, last resort replace undecodable bytes."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kw)


def load() -> pd.DataFrame:
    orders = read_csv_smart(D / "olist_orders_dataset.csv",
                            parse_dates=["order_purchase_timestamp", "order_estimated_delivery_date",
                                         "order_delivered_customer_date"])
    items = read_csv_smart(D / "olist_order_items_dataset.csv")
    pays = read_csv_smart(D / "olist_order_payments_dataset.csv")
    reviews = read_csv_smart(D / "olist_order_reviews_dataset.csv")
    products = read_csv_smart(D / "olist_products_dataset.csv")
    custs = read_csv_smart(D / "olist_customers_dataset.csv")
    catmap = read_csv_smart(D / "product_category_name_translation.csv")

    rv = (reviews.sort_values("review_answer_timestamp")
          .drop_duplicates("order_id", keep="first")
          [["order_id", "review_score", "review_comment_message"]])
    it = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"), n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"), freight_total=("freight_value", "sum"),
        main_product_id=("product_id", "first")).reset_index()
    pm = pays.groupby("order_id").agg(
        payment_value_total=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        n_payment_types=("payment_type", "nunique")).reset_index()
    main_pt = (pays.sort_values("payment_value", ascending=False)
               .drop_duplicates("order_id", keep="first")[["order_id", "payment_type"]]
               .rename(columns={"payment_type": "main_payment_type"}))
    prod = (products.merge(catmap, on="product_category_name", how="left")
            [["product_id", "product_category_name_english", "product_photos_qty",
              "product_weight_g", "product_description_lenght"]]
            .rename(columns={"product_category_name_english": "category",
                             "product_description_lenght": "product_desc_len"}))

    df = (orders.merge(rv, on="order_id", how="inner")
          .merge(it, on="order_id", how="left")
          .merge(pm, on="order_id", how="left")
          .merge(main_pt, on="order_id", how="left")
          .merge(custs[["customer_id", "customer_state"]], on="customer_id", how="left")
          .merge(prod, left_on="main_product_id", right_on="product_id", how="left"))
    df = df[(df.order_status == "delivered") & df.order_delivered_customer_date.notna()].copy()
    df["delay_days"] = (df.order_delivered_customer_date - df.order_estimated_delivery_date).dt.total_seconds() / 86400
    df["deliv_days"] = (df.order_delivered_customer_date - df.order_purchase_timestamp).dt.total_seconds() / 86400
    return df.reset_index(drop=True)


RAW_FIELDS = ["price_total", "freight_total", "payment_value_total", "n_items", "n_sellers",
              "max_installments", "n_payment_types", "main_payment_type", "customer_state",
              "category", "product_weight_g", "product_desc_len", "product_photos_qty",
              "order_purchase_timestamp", "order_estimated_delivery_date",
              "order_delivered_customer_date", "review_comment_message"]


def _clean(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 2)
    return v


def row_to_order(r: pd.Series) -> dict:
    o = {k: _clean(r.get(k)) for k in RAW_FIELDS}
    for k in ("n_items", "n_sellers", "n_payment_types", "max_installments"):
        if o[k] is not None:
            o[k] = int(round(o[k]))
    score = int(r.review_score)
    delay = r.delay_days
    when = "on time" if delay <= 0.5 else f"{round(delay)}d late"
    if delay <= -10:
        when = f"{abs(round(delay))}d early"
    has_c = isinstance(r.review_comment_message, str) and r.review_comment_message.strip() != ""
    tail = "with comment" if has_c else "no comment"
    o["_label"] = f"{'★' * score}{'☆' * (5 - score)}  ·  {when}  ·  {tail}"
    o["_true_score"] = score
    return o


def main() -> None:
    df = load()
    rng = np.random.default_rng(SEED)

    examples: list[dict] = []
    for score, n in N_PER_SCORE.items():
        pool = df[df.review_score == score]
        # prefer variety: half with a comment, half without; spread on lateness
        commented = pool[pool.review_comment_message.notna()]
        silent = pool[pool.review_comment_message.isna()]
        take_c = min(len(commented), (n + 1) // 2)
        take_s = min(len(silent), n - take_c)
        pick = pd.concat([
            commented.sample(take_c, random_state=int(rng.integers(1 << 30))) if take_c else commented.iloc[:0],
            silent.sample(take_s, random_state=int(rng.integers(1 << 30))) if take_s else silent.iloc[:0],
        ])
        examples += [row_to_order(r) for _, r in pick.iterrows()]

    # sort worst-first so the picker reads 1★ … 5★
    examples.sort(key=lambda o: (o["_true_score"], o["_label"]))

    num_defaults = {
        "price_total": round(float(df.price_total.median()), 2),
        "freight_total": round(float(df.freight_total.median()), 2),
        "payment_value_total": round(float(df.payment_value_total.median()), 2),
        "n_items": int(df.n_items.median()),
        "n_sellers": int(df.n_sellers.median()),
        "max_installments": int(df.max_installments.median()),
        "n_payment_types": int(df.n_payment_types.median()),
        "product_weight_g": round(float(df.product_weight_g.median()), 0),
        "product_desc_len": int(df.product_desc_len.median()),
        "product_photos_qty": int(df.product_photos_qty.median()),
        "main_payment_type": df.main_payment_type.mode().iat[0],
        "customer_state": df.customer_state.mode().iat[0],
        "category": df.category.mode().iat[0],
        # a neutral timeline: bought 2018-05-01, ~median delivery, a few days early
        "order_purchase_timestamp": "2018-05-01 10:00:00",
        "order_estimated_delivery_date": "2018-05-19 00:00:00",
        "order_delivered_customer_date": "2018-05-13 12:00:00",
        "review_comment_message": "",
    }

    OUT.write_text(json.dumps({"defaults": num_defaults, "examples": examples},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ·  {len(examples)} examples  ·  "
          f"score mix {dict(pd.Series([e['_true_score'] for e in examples]).value_counts().sort_index())}")


if __name__ == "__main__":
    main()

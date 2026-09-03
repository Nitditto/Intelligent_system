# House-price dataset

## Source

| Field | Value |
|---|---|
| **Name** | VN Real Estate Listings (April–September 2025) |
| **File** | `VN-real-estate-Apr-Sept-2025.csv` |
| **Rows / cols** | 236,226 rows × 28 columns (true CSV record count; `wc -l` reports 238,924 because `Description` contains newlines) |
| **Size** | ~210 MB |
| **Encoding** | UTF-8 **with BOM** — read with `encoding="utf-8-sig"` |
| **Separator** | `,` |
| **Provenance** | Scraped Vietnamese property-listing portal; listings updated Apr–Sept 2025 (`Scraped At` / `Last Updated Date`). |
| **Licence** | Scraped data — research use only; cite the source portal in the report. |

## How to obtain it

The CSV is **not committed** (210 MB, over GitHub's 100 MB limit). It must be placed at:

```
Assignment_2/house_price/data/VN-real-estate-Apr-Sept-2025.csv
```

It currently also lives at the `Intelligence-System/` repo root — copy it in:

```bash
cp ../../../VN-real-estate-Apr-Sept-2025.csv data/
```

## Target

`Price` — listing price in **million VND**. Title "2.7 tỷ" ↔ `Price` = 2700 (1 tỷ = 1,000 million VND).

## Column roles (finalised in notebook §11)

| Group | Columns | Notes |
|---|---|---|
| **Target** | `Price` | million VND; raw range corrupted (min 0, max 9.2e12) — cleaned in §8 |
| **Numeric features** | `Area`, `Width`, `Length`, `Bedrooms`, `Bathrooms`, `Floors`, `Alley Width`, `Agent Listing Count` | `Area` also corrupted (min −6, max 1e18). Missingness: `Bedrooms` 72%, `Bathrooms` 84%, `Floors` 79% |
| **Optional numeric** | `Latitude`, `Longitude` | only ~32% populated |
| **Categorical (one-hot)** | `Property Type` (7), `Position` (2 + NA), `Direction` (8 + NA), `Road Type` (4 + NA), `Province` (63), `Agent Role` (2) | |
| **Engineered from text** | `Location` → `ward`, `district` | comma-separated, inconsistent arity; parser + frequency-encoding in §13 |
| **Dropped** | `Title` (**leaks `Price`** — contains "2.7 tỷ"), `Listing ID`, `Description` (free text), `VIP Account` (constant `False`), `Agent Name` (28k unique), `Avatar`, `Property Type Slug` (duplicate of `Property Type`), `Last Updated`, `Scraped At`, `Last Updated Date` | |

## Known data-quality issues (handled in §5–§9)

- `Price <= 0` and absurd `Price` (> ~200 tỷ) → drop / cap
- `Area <= 0` and `Area` > ~10,000 m² → drop / cap
- `Bedrooms` / `Bathrooms` / `Floors` mostly missing → missing-indicator + median impute (or drop for land listings)
- `Direction`, `Road Type`, `Position`, `Alley Width` 30–70% missing → "Unknown" category / impute
- `VIP Account` constant → drop
- Duplicate listings by `Listing ID` — checked in §7

## Note on the previous dataset

An earlier version of this assignment used `vietnam_housing_dataset.csv` (~30k rows, Kaggle
"Vietnam Housing Dataset"). It has been **replaced** by this larger scrape. The old trained
models and their JSON artifacts were removed; `notebook/house_price.ipynb` regenerates
everything from this CSV.

# Assignment 03 — From Data Representation to Deep Learning

See `REQUIREMENT.md` for the assignment brief and `PLAN.md` for the report plan
(each app also has its own `<app>/PLAN.md` with the detailed dataset/model/notebook
plan).

## Datasets

The raw CSVs are **not committed to git** — each is 40-110MB, and even the ones under
GitHub's 100MB hard limit are still large enough that keeping them out of history is
the right call (`*/data/*.csv` is gitignored; the `data/` folders themselves are kept
via `.gitkeep`).

| App | Kaggle source | File(s) used | Rows | Notes |
|---|---|---|---|---|
| `diabetes` | [`spandanjit2005/brfss-diabetes-indicator-dataset`](https://www.kaggle.com/datasets/spandanjit2005/brfss-diabetes-indicator-dataset) | `csv/DATASET_2019.csv`, `csv/DATASET_2020.csv` | 815,711 (416,661 + 399,050) | 2 of the 20 available per-year files (2005–2024); concatenate before use |
| `house_price` | [`minorin2847/usa-real-estate-600k-sample`](https://www.kaggle.com/datasets/minorin2847/usa-real-estate-600k-sample) | `realtor-data-sample.csv` | 600,000 | Our own published seeded sample (`RANDOM_SEED = 42`) of [`ahmedshahriarsakib/usa-real-estate-dataset`](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) (2,226,382 rows) — credit to that dataset for the original data; we mirror the exact sample so re-runs don't need to re-sample |
| `customer_behaviour` | [`niraliivaghani/flipkart-dataset`](https://www.kaggle.com/datasets/niraliivaghani/flipkart-dataset) | `Dataset.csv` | 364,401 (full file) | Saved locally as `flipkart_reviews.csv` |

### Re-downloading the data

```bash
# requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_API_TOKEN set)
kaggle datasets download -d spandanjit2005/brfss-diabetes-indicator-dataset -f csv/DATASET_2019.csv -p diabetes/data --unzip
kaggle datasets download -d spandanjit2005/brfss-diabetes-indicator-dataset -f csv/DATASET_2020.csv -p diabetes/data --unzip

kaggle datasets download -d minorin2847/usa-real-estate-600k-sample -p house_price/data --unzip

kaggle datasets download -d niraliivaghani/flipkart-dataset -p customer_behaviour/data --unzip
```

## Why bigger datasets than assignment_02

Each app's dataset here is a genuine growth step over the one used in
assignment_02 (3.2x / 2.5x / 3.1x rows respectively), per the professor's separate
verbal requirement that each assignment's dataset be larger than the previous one's —
while staying well under a gigabyte in total. See each `<app>/PLAN.md` §Dataset for
the full reasoning behind the specific source and size chosen.

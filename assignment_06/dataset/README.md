# Datasets

Raw files are not committed (see `.gitignore`). Download from Kaggle into the folders below; subfolders created by unzipping are fine.
Each notebook writes `splits.npz` (chronological 70/15/15 indices) on first run; that file is committed.

| Code | Dataset | Download | Put these files in | Used |
|---|---|---|---|---|
| `EC` | Online Retail II, UCI (e-commerce orders and cancellations) | [kaggle.com/datasets/mashlyn/online-retail-ii-uci](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) (~95 MB) | `dataset/ec/online_retail_II.csv` (or `.xlsx`) | top 750 products, 604 open days, ~431k windows |
| `ST` | S&P 500 stock data, 5 years | [kaggle.com/datasets/camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500) (~30 MB) | `dataset/st/all_stocks_5yr.csv` | 350 tickers, daily, ~430k windows |

Checked after download: `online_retail_II.csv` 1,067,371 rows; `all_stocks_5yr.csv` 619,040 rows, 505 tickers. Only these two CSVs are used; the extras shipped in the S&P zip (`individual_stocks_5yr/`, `getSandP.py`, `merge.sh`, `__MACOSX/`) are redundant and can be deleted (all gitignored).

## Kaggle CLI

Needs `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`). Run from `assignment_06/`:

```
kaggle datasets download -d mashlyn/online-retail-ii-uci -p dataset/ec --unzip
kaggle datasets download -d camnugent/sandp500 -p dataset/st --unzip
```

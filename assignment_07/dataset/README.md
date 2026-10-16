# Datasets

Raw files are not committed (see `.gitignore`). Download from Kaggle into the folders below.

| Code | Dataset | Kaggle | Put files in |
|---|---|---|---|
| `EC` | Amazon Reviews: Unlocked Mobile Phones (PromptCloud) | [PromptCloudHQ/amazon-reviews-unlocked-mobile-phones](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones) | `dataset/ec/` |
| `ST` | S&P 500 Stocks (daily) | [andrewmvd/sp-500-stocks](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks) | `dataset/st/` |

Kaggle CLI (needs `~/.kaggle/kaggle.json`), run from `assignment_07/`:

```
kaggle datasets download -d PromptCloudHQ/amazon-reviews-unlocked-mobile-phones -p dataset/ec --unzip
kaggle datasets download -d andrewmvd/sp-500-stocks -p dataset/st --unzip
```

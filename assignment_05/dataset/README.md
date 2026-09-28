# Datasets

The raw files are not committed (see `.gitignore`). Download them from Kaggle and place
them in the folders below. File names must match; subfolders created by unzipping are fine,
because the notebooks search each folder recursively.

On the first run, each notebook writes `splits.npz` into its folder: the row indices of the
stratified 70 / 15 / 15 train / validation / test split (seed 42). That file is committed,
so every later run and both frameworks use exactly the same samples.

| Code | Dataset | Download | Put these files in | Samples used |
|---|---|---|---|---|
| `SV` | SVHN, cropped digits (Netzer et al., 2011) | [kaggle.com/datasets/sahityasetu/street-view-house-numbers-images](https://www.kaggle.com/datasets/sahityasetu/street-view-house-numbers-images) (246 MB) | `dataset/sv/train_32x32.mat`, `dataset/sv/test_32x32.mat` | 99,289 images, 32×32 RGB, 10 classes |
| `GT` | GTSRB, 32×32 pickles (Stallkamp et al., 2011) | [kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset) (124 MB) | `dataset/gt/train.p`, `valid.p`, `test.p`, `signname.csv` | 51,839 images, 32×32 RGB, 43 classes |
| `DB` | Diabetes Prediction Challenge, Playground Series S5E12 | [kaggle.com/competitions/playground-series-s5e12/data](https://www.kaggle.com/competitions/playground-series-s5e12/data) (click **Join Competition** first) | `dataset/db/train.csv` | 300,000-row stratified sample of 700,000; 24 features, binary target |

## Download options

**Browser:** open each link, click *Download*, unzip into the folder in the table.

**Kaggle CLI** (needs an API token in `~/.kaggle/kaggle.json`, or
`C:\Users\<you>\.kaggle\kaggle.json` on Windows), run from `assignment_05/`:

```
kaggle datasets download -d sahityasetu/street-view-house-numbers-images -p dataset/sv --unzip
kaggle datasets download -d harbhajansingh21/german-traffic-sign-dataset -p dataset/gt --unzip
kaggle competitions download -c playground-series-s5e12 -f train.csv -p dataset/db
```

The competition file may arrive as `train.csv.zip`; unzip it so `dataset/db/train.csv` exists.

## Notes

- `SV`: SVHN stores digit 0 as label 10; the notebook maps it to 0.
- `GT`: classes are unbalanced (about 180 to 2,000 training images per class), so macro F1 is reported.
- `DB`: Kaggle generated the Playground data synthetically from a real diabetes dataset. Only `train.csv`
  has labels, so all three splits come from it.

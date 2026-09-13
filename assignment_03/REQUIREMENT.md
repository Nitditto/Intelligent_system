# Assignment 03 — Requirements

Source material: `materials/intel_sys_dev_slide_03_deepLeaning_1.pdf` (Lecture 03 — From
Data to Deep Learning) and `materials/int_sys_dev_slide_03_basicML_deepLearning_04.09.pdf`
(Deep Learning from Scratch with NumPy). Project layout follows `assignment_02/` where the
slides don't say otherwise.

## Goal

For each of the 3 applications carried over from assignment_02, extend the existing
classical-ML pipeline with a **deep learning model built from scratch** (no Keras, no
PyTorch, no TensorFlow — NumPy only, per the lecture's explicit constraint), then evaluate
and compare **4 models total per dataset**: the 3 classical ML models already used/expanded
in assignment_02, plus the 1 NumPy deep learning model.

```
Raw data → Represent → Train (3 classical ML + 1 from-scratch DL) → Evaluate → Compare → Visualize
```

## The 3 applications (unchanged scope from assignment_02)

| # | App | Task | Representation | Deep net architecture (from slides) |
|---|---|---|---|---|
| 1 | `diabetes` | binary classification | standardized numeric feature vector, `X ∈ ℝ^{N×8}` (Pima-style: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree, age) — or the assignment_02 BRFSS feature set if kept | `8 → 16 → 8 → 1`, ReLU hidden layers, sigmoid output, binary cross-entropy |
| 2 | `house_price` | regression | numeric + encoded feature matrix, `X ∈ ℝ^{N×d}` | `d → h1 → h2 → 1`, ReLU hidden layers, linear output, MSE loss |
| 3 | `customer_behaviour` (comments) | binary classification (e.g. recommend / not, or sentiment) | TF-IDF or bag-of-words vector over review text, `X ∈ ℝ^{N×d}` (a from-scratch embedding/sequence path is optional/stretch — TF-IDF-as-input to the same dense NumPy network is sufficient) | same dense architecture family as diabetes: `d → h1 → h2 → 1`, ReLU + sigmoid, binary cross-entropy |

Regression vs. classification determines the output activation/loss per model
(sigmoid+BCE for classification, linear+MSE for regression), matching Lecture 03's
"Application 1/2/3" sections.

## Model requirements per dataset

Each of the 3 project folders must train and report **4 models**:

1–3. **Three classical machine-learning models** (scikit-learn, consistent with
   assignment_02's tooling — e.g. Logistic/Linear Regression, Random Forest, Gradient
   Boosting / SVM / KNN — pick 3 per dataset, at least one must be the model already
   deployed in assignment_02 for continuity).
4. **One deep learning model implemented from scratch with NumPy only**:
   - Manual forward propagation (`Z = XW + b`, then activation).
   - ReLU for hidden layers; sigmoid (classification) or linear (regression) output.
   - Manual loss: binary cross-entropy (classification) or MSE (regression).
   - Manual backpropagation via the chain rule (no autograd).
   - Manual gradient descent parameter updates (`W -= lr * dW`).
   - Weight init: `np.random.randn(...) * sqrt(2/fan_in)`, zero biases (He-style, per slides).
   - Train/test split + feature standardization computed on train only, applied to test
     (no leakage), matching assignment_02's rule.
   - Training loop tracks loss per epoch for the loss-curve visualization.

No TensorFlow, PyTorch, or Keras imports anywhere in the DL implementation. scikit-learn
remains allowed only for the 3 classical ML models, not for the DL model itself.

## Evaluation & comparison (all 4 models, per dataset)

Classification datasets (diabetes, customer_behaviour): Accuracy, Precision, Recall, F1,
confusion matrix (TP/TN/FP/FN) — same metric set the slides derive by hand in §37–39.

Regression dataset (house_price): MSE/RMSE, MAE, R².

Required visualizations per dataset:
- Bar chart comparing all 4 models on the primary metric (Accuracy or R²) and on
  Precision/Recall/F1 (classification) or RMSE/MAE (regression).
- Training loss curve (epoch vs. loss) for the from-scratch DL model.
- Confusion matrix heatmap for the DL model (classification datasets).
- A short written comparison: which model wins, and why (representation capacity vs.
  overfitting vs. dataset size), echoing the lecture's "Deep Learning = Function
  Composition + Representation Learning + Optimization" framing.

A top-level summary table/notebook comparing all 3 datasets × 4 models is expected in the
final report, mirroring assignment_02's "Data-representation summary" table.

## Project structure (per app, following `assignment_02/<app>/`)

```
<app>/
  data/                raw dataset (or download reference)
  notebook/            <app>.ipynb — EDA + 3 classical ML models + from-scratch NumPy DL model
                        + evaluation/comparison + visualizations, executed with outputs
  model/               model_pipeline.joblib (best classical model) + dl_weights.npz
                        (from-scratch DL weights) + feature_names.joblib + input_schema.json
  requirements.txt     numpy, pandas, scikit-learn, matplotlib/seaborn — no DL framework
report/                Assignment_03.pdf / .md — final report across all 3 apps
```

No `api/`, `web/`, or `mobile/` subfolders — see below.

**No API, web, or mobile layer is required.** Both source decks (Lecture 03 and the
NumPy-from-scratch tutorial) stop at the modeling pipeline — representation → forward
propagation → loss → backpropagation → gradient descent → evaluation — and never mention
deployment. The full deliverable for assignment_03 is notebook-level: modeling, evaluation,
comparison, and visualization. `api/`, `web/`, `mobile/` folders from assignment_02 are out
of scope unless the user explicitly asks to extend serving to the new DL model.

## Constraints checklist

- [ ] No `tensorflow`, `torch`, or `keras` imports anywhere.
- [ ] DL model's forward/backward/update implemented manually with NumPy.
- [ ] 3 classical ML models + 1 from-scratch DL model trained per dataset (4 × 3 = 12
      models total).
- [ ] Same train/test split and no-leakage preprocessing discipline as assignment_02
      (`RANDOM_SEED = 42`, scaler fit on train only).
- [ ] Evaluation metrics appropriate to task type (classification vs. regression).
- [ ] At least one comparison visualization and one training-loss visualization per
      dataset.
- [ ] Final report compares all 4 models per dataset and summarizes across all 3 datasets.

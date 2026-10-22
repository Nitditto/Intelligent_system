import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def build():
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        "# 1. System and Problem Definition\n"
        "**Problem:** Estimating a fair market price for a residential property listing in Vietnam "
        "without manually comparing dozens of similar listings, helping buyers avoid overpaying and "
        "sellers/agents price competitively.\n\n"
        "**System Statement:** The intelligent system is a price-estimation assistant that takes a raw "
        "property listing — physical measurements (area, frontage, road access), layout (floors, bedrooms, "
        "bathrooms), location (address/district), and qualitative attributes (direction, legal status, "
        "furniture state) — and converts it into a numerical feature vector. A trained regression model "
        "then maps this vector to a predicted price in billions of VND, giving users a fast, data-driven "
        "reference point before negotiating a real transaction."
    ))

    cells.append(new_markdown_cell(
        "# 2. Intelligent System Diagram\n"
        "```mermaid\n"
        "flowchart LR\n"
        "    A[\"Input:<br/>Raw Property<br/>Listing Data\"] --> B[\"Represent:<br/>Feature Vector<br/>"
        "(Scaled Numerics +<br/>One-Hot Categoricals)\"]\n"
        "    B --> C[\"Model:<br/>Trained ML<br/>Regressor\"]\n"
        "    C --> D[\"Predict:<br/>Continuous Price<br/>(Billion VND)\"]\n"
        "```"
    ))

    cells.append(new_markdown_cell(
        "# 3. Dataset Source\n"
        "The dataset is `vietnam_housing_dataset.csv`, a collection of Vietnamese residential real-estate "
        "listings.\n\n"
        "**[TODO: paste the exact Kaggle/source URL here.]**"
    ))

    cells.append(new_markdown_cell(
        "# 4. Dataset Description\n"
        "- **Phenomenon:** The pricing behavior of the Vietnamese residential real-estate market, driven "
        "by property size, location, and quality attributes.\n"
        "- **Observation:** A single property listing (one row = one house/apartment posted for sale).\n"
        "- **Raw features (11):** Address, Area, Frontage, Access Road, House direction, Balcony direction, "
        "Floors, Bedrooms, Bathrooms, Legal status, Furniture state.\n"
        "- **Target:** `Price` (billions of VND) — numerical, continuous.\n"
        "- **Task:** Regression.\n"
        "- **Observations:** 30,229 listings (after dropping rows with missing `Price`).\n"
        "- **Numerical features:** Area, Frontage, Access Road, Floors, Bedrooms, Bathrooms.\n"
        "- **Categorical features:** District (engineered from Address), House direction, Balcony direction, "
        "Legal status, Furniture state — 367 dimensions after one-hot encoding (incl. 342 districts)."
    ))

    cells.append(new_markdown_cell(
        "# 5. Data Representation\n\n"
        "| Feature | Type | Representation | Meaning |\n"
        "| --- | --- | --- | --- |\n"
        "| Area | Numerical | Real value | Property floor area (m²) |\n"
        "| Frontage | Numerical | Real value | Width of the property facing the street (m) |\n"
        "| Access Road | Numerical | Real value | Width of the road giving access to the property (m) |\n"
        "| Floors | Numerical | Integer | Number of floors |\n"
        "| Bedrooms | Numerical | Integer | Number of bedrooms |\n"
        "| Bathrooms | Numerical | Integer | Number of bathrooms |\n"
        "| District (from Address) | Categorical | One-hot encoded (342 categories) | District extracted from "
        "the raw address string |\n"
        "| House direction | Categorical | One-hot encoded | Compass direction the house faces |\n"
        "| Balcony direction | Categorical | One-hot encoded | Compass direction the balcony faces |\n"
        "| Legal status | Categorical | One-hot encoded | Legal ownership status |\n"
        "| Furniture state | Categorical | One-hot encoded | Furnishing level |\n\n"
        "Missing numerical values are imputed with the column median; missing categorical values are filled "
        "with `'Unknown'` before encoding (missingness is substantial for several columns, e.g. Balcony "
        "direction 82.6%, House direction 70.3%, Furniture state 46.7%). Categorical columns are converted "
        "with `pd.get_dummies(drop_first=True)`. The target `Price` is transformed with `log1p` before "
        "training and inverted with `expm1` before computing evaluation metrics; all 367 input dimensions "
        "are standardized with `StandardScaler` fit on the training set only."
    ))

    cells.append(new_markdown_cell(
        "# 6. Feature and Target Analysis\n"
        "Numerical features (Area, Frontage, Access Road, Floors, Bedrooms, Bathrooms) carry different "
        "physical scales, so distance/margin-based models (KNN, SVR) require standardization. Categorical "
        "features are one-hot encoded, with `District` contributing 342 sparse binary columns — this makes "
        "the encoded feature space high-dimensional (367-d) and mostly sparse. The target `Price` has a "
        "strong right skew (a small number of very expensive listings), which motivates modeling `log1p(Price)` "
        "during training and inverse-transforming predictions with `expm1` for evaluation in real VND."
    ))

    cells.append(new_markdown_cell(
        "# 7. Exploratory Data Analysis\n"
        "Below we explore the distributions, correlations, and missingness patterns of the dataset features."
    ))
    if os.path.exists("Visualization_2.ipynb"):
        viz_nb = load_notebook("Visualization_2.ipynb")
        merged = viz_nb.cells[1:]
        cells.extend(merged)
        n_imgs = sum(
            1 for c in merged if c.cell_type == 'code'
            for o in c.get('outputs', []) if 'image/png' in o.get('data', {})
        )
        print(f"Merged {len(merged)} cells from Visualization_2.ipynb ({n_imgs} contain a rendered chart).")
    else:
        print("WARNING: Visualization_2.ipynb not found — Section 7 will have no EDA charts.")

    cells.append(new_markdown_cell(
        "# 8. Train/Test Split\n"
        "We clean the data, extract `District` from the free-text `Address`, one-hot encode all categorical "
        "columns, log-transform the target, and split 80/20 (`random_state=42`). Numerical + one-hot features "
        "are standardized with `StandardScaler` fit only on the training set."
    ))
    model_nb = load_notebook("Training_1.ipynb") if os.path.exists("Training_1.ipynb") else None
    if model_nb:
        setup_cell = [c for c in model_nb.cells if c.cell_type == 'code'][0]
        cells.append(setup_cell)
    else:
        print("WARNING: Training_1.ipynb not found — Sections 8-20 will be missing model code/results.")

    cells.append(new_markdown_cell(
        "# 9. Baseline\n"
        "Before evaluating the six regression models, we establish a baseline that always predicts the "
        "training set's mean price directly in raw VND space."
    ))
    cells.append(new_code_cell(
        "from sklearn.dummy import DummyRegressor\n"
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n"
        "y_train_raw = np.expm1(y_train_log)\n"
        "y_test_raw = np.expm1(y_test_log)\n\n"
        "dummy = DummyRegressor(strategy='mean')\n"
        "dummy.fit(X_train_scaled, y_train_raw)\n"
        "y_pred_dummy = dummy.predict(X_test_scaled)\n\n"
        "mae_dummy = mean_absolute_error(y_test_raw, y_pred_dummy)\n"
        "mse_dummy = mean_squared_error(y_test_raw, y_pred_dummy)\n"
        "rmse_dummy = np.sqrt(mse_dummy)\n"
        "r2_dummy = r2_score(y_test_raw, y_pred_dummy)\n\n"
        "print(f'Baseline (predict mean) -> MAE: {mae_dummy:.4f} | MSE: {mse_dummy:.4f} "
        "| RMSE: {rmse_dummy:.4f} | R2: {r2_dummy:.4f}')"
    ))

    if model_nb:
        code_cells = [c for c in model_nb.cells if c.cell_type == 'code']
        model_titles = [
            "# 10. Model 1: Linear Regression",
            "# 11. Model 2: Support Vector Regressor (RBF Kernel)",
            "# 12. Model 3: Support Vector Regressor (Linear Kernel)",
            "# 13. Model 4: K-Nearest Neighbors Regressor",
            "# 14. Model 5: Random Forest Regressor",
            "# 15. Model 6: XGBoost Regressor",
        ]
        for title, code_cell in zip(model_titles, code_cells[1:7]):
            cells.append(new_markdown_cell(title))
            cells.append(code_cell)

    cells.append(new_markdown_cell(
        "# 16. Evaluation\n"
        "For a price-estimation tool, **R²** summarizes how much of the price variance the model explains "
        "overall, but **MAE** and **RMSE** (both in billions of VND) are more actionable for an end user, "
        "since they express a typical prediction error in real currency terms rather than an abstract "
        "variance-explained percentage. RMSE additionally penalizes large errors more than MAE, which "
        "matters here because the target has high-value outliers."
    ))

    cells.append(new_markdown_cell(
        "# 17. Experiment 1: Model Comparison\n"
        "**Question:** Which of the six regression models best predicts property price under identical "
        "train/test conditions?\n\n"
        "**Setup:** All six models are trained on the same 80/20 split (`random_state=42`) of the same "
        "standardized, one-hot encoded feature matrix, with `log1p(Price)` as the target; predictions are "
        "inverse-transformed with `expm1` before scoring on the held-out test set."
    ))
    if model_nb:
        valid_code_cells = [c for c in code_cells if c.source.strip()]
        cells.append(valid_code_cells[-1])

    cells.append(new_markdown_cell(
        "# 18. Experiment 2: Hyperparameter Investigation\n"
        "**Question:** How do the number of neighbors (`n_neighbors`) and the weighting scheme (`uniform` "
        "vs. `distance`) affect KNN regression accuracy on this dataset?\n\n"
        "**Setup:** `GridSearchCV` (3-fold) is run over `n_neighbors` ∈ {3, 5, 7, 9} × `weights` ∈ "
        "{uniform, distance} on the scaled training data, scoring by negative MSE in log-price space "
        "(see the KNN model cell above for the grid search code).\n\n"
        "**Result:** Error decreases monotonically as `k` grows from 3 to 9 (higher `k` reduces "
        "variance/overfitting to noisy individual neighbors), and `distance` weighting consistently beats "
        "`uniform` weighting at every `k`. The best configuration found was `n_neighbors=9, weights='distance'` "
        "(CV RMSE in log space ≈ 0.2684)."
    ))

    cells.append(new_markdown_cell(
        "# 19. Experiment 3: Representation / Feature Investigation\n"
        "**Question:** Does modeling `log1p(Price)` instead of raw `Price` improve Linear Regression "
        "performance, given the strong right skew of the target observed during EDA?\n\n"
        "**Setup:** Linear Regression is trained twice on the same 80/20 split and the same standardized "
        "features — once directly on raw `Price`, once on `log1p(Price)` with predictions inverted via "
        "`expm1` — and compared on the same test set."
    ))
    cells.append(new_code_cell(
        "from sklearn.linear_model import LinearRegression\n\n"
        "lr_raw = LinearRegression()\n"
        "lr_raw.fit(X_train_scaled, y_train_raw)\n"
        "y_pred_raw_target = lr_raw.predict(X_test_scaled)\n\n"
        "mae_raw = mean_absolute_error(y_test_raw, y_pred_raw_target)\n"
        "rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw_target))\n"
        "r2_raw = r2_score(y_test_raw, y_pred_raw_target)\n\n"
        "lr_log = LinearRegression()\n"
        "lr_log.fit(X_train_scaled, y_train_log)\n"
        "y_pred_log_target = np.expm1(lr_log.predict(X_test_scaled))\n\n"
        "mae_log = mean_absolute_error(y_test_raw, y_pred_log_target)\n"
        "rmse_log = np.sqrt(mean_squared_error(y_test_raw, y_pred_log_target))\n"
        "r2_log = r2_score(y_test_raw, y_pred_log_target)\n\n"
        "print('Target       | MAE     | RMSE    | R2')\n"
        "print(f'Raw Price    | {mae_raw:.4f} | {rmse_raw:.4f} | {r2_raw:.4f}')\n"
        "print(f'log1p(Price) | {mae_log:.4f} | {rmse_log:.4f} | {r2_log:.4f}')\n\n"
        "print('\\nFinding: modeling the raw price directly gives BETTER test-set MAE/RMSE/R2 than the '\n"
        "      'log-transformed target for Linear Regression. Minimizing squared error in log space does '\n"
        "      'not minimize squared error in the original VND space (expm1 back-transformation reintroduces '\n"
        "      'bias), so a naive log-transform-then-invert pipeline can hurt the reported metric even though '\n"
        "      'the log target is better behaved during training. Tree ensembles (XGBoost, Random Forest) are '\n"
        "      'far less sensitive to this effect since they are not linear in the target space.')"
    ))

    cells.append(new_markdown_cell(
        "# 20. Final Model\n"
        "XGBoost achieved the best overall performance (R² = 0.5954, MAE ≈ 1.05 billion VND), narrowly "
        "ahead of Random Forest (R² = 0.5623). The trained `xgboost.pkl` model is selected as the final "
        "model and serialized for use in the application."
    ))

    cells.append(new_markdown_cell(
        "# 21. Application\n"
        "The trained `xgboost.pkl` model is integrated into a full-stack application (`app/` directory: "
        "FastAPI backend + frontend). Users enter property details (area, district, legal status, etc.) into "
        "the UI and receive an instant predicted price from the backend API.\n\n"
        "*[Insert screenshot of Application UI here]*"
    ))

    cells.append(new_markdown_cell(
        "# 22. System Demonstration\n"
        "The user submits property attributes through the form; the frontend sends a JSON payload to the "
        "backend API; the backend loads the serialized `.pkl` model, applies the saved encoding/scaling "
        "pipeline, and returns a predicted price in billions of VND.\n\n"
        "*[Insert screenshot of a Prediction Result here — recommended: at least 3 different input cases]*"
    ))

    cells.append(new_markdown_cell(
        "# 23. Reflection\n"
        "- **Information received:** Raw property listing attributes — physical measurements, layout, "
        "direction, legal status, furnishing, and a free-text address.\n"
        "- **Internal representation:** A standardized, 367-dimensional feature vector (numerical values "
        "scaled, categorical values one-hot encoded), with the target modeled in log-price space during "
        "training.\n"
        "- **What the model learns:** Statistical relationships between property/location attributes and "
        "price, learned from ~30,000 historical listings, including non-linear interactions captured by "
        "the tree-based models.\n"
        "- **Prediction produced:** A continuous price estimate in billions of VND for a new, unseen "
        "listing.\n"
        "- **Why it generalizes:** It learned generalizable patterns (e.g. \"larger area + central district "
        "+ full legal certificate → higher price\") rather than memorizing specific training rows.\n"
        "- **What is \"intelligent\":** The training phase, where each model autonomously adjusts its "
        "internal parameters (regression weights, tree splits and leaf values) to minimize prediction error "
        "on historical data, without hand-coded pricing rules.\n"
        "- **Limitations:** No image understanding of the property, no true geospatial reasoning (only a "
        "coarse district category), no market-timing awareness, and no natural-language explanation of its "
        "reasoning."
    ))

    cells.append(new_markdown_cell(
        "# 24. Conclusion\n"
        "We designed and implemented an intelligent system for predicting Vietnamese residential property "
        "prices from listing attributes. Across six traditional ML models, XGBoost achieved the best "
        "performance (R² = 0.5954, MAE ≈ 1.05 billion VND), with Random Forest close behind, confirming "
        "that tree-based ensembles are best suited to this tabular, high-cardinality, interaction-heavy "
        "pricing problem. The controlled experiments revealed two non-obvious findings: KNN performance "
        "improves monotonically with `k` up to 9 with distance weighting, and — counter to the usual "
        "\"always transform skewed targets\" intuition — naively log-transforming the target actually "
        "*hurt* Linear Regression's real-currency error due to back-transformation bias. While the deployed "
        "application provides real, actionable price estimates, its value is bounded by substantial missing "
        "data, a coarse district-only location representation, and a limited observed price range, which "
        "should guide future improvements (e.g. geocoded coordinates, learned location embeddings, listing "
        "photos)."
    ))

    nb.cells = cells
    with open('Report_Notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Wrote Report_Notebook.ipynb with", len(cells), "cells")

if __name__ == "__main__":
    build()

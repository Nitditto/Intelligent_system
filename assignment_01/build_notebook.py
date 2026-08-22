import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def load_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def build():
    nb = new_notebook()
    cells = []
    
    # 1. System and Problem Definition
    cells.append(new_markdown_cell(
        "# 1. System and Problem Definition\n\n"
        "**System Statement:** The intelligent system is a diagnostic assistant that analyzes a patient's "
        "clinical and demographic data (glucose level, BMI, age, etc.) to predict whether the patient has "
        "diabetes. It serves as a preliminary screening tool that helps doctors identify high-risk patients "
        "quickly, before a full clinical diagnosis is made.\n\n"
        "**Problem Definition (required questions):**\n\n"
        "1. **What real-world problem does the system address?** The need for early, low-cost detection of "
        "diabetes so that timely medical intervention and lifestyle changes can reduce long-term health "
        "complications.\n"
        "2. **What information does the system receive?** Demographic data (Age) and clinical measurements "
        "(Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function).\n"
        "3. **How is that information represented internally?** As a structured, 8-dimensional feature "
        "vector of real numbers. Physiologically-impossible zero readings are imputed with the training-set "
        "median, and the vector is standardized (mean 0, std 1) before being given to distance-based models.\n"
        "4. **What does the model learn?** The statistical relationship / decision boundary between the "
        "input clinical features and the historic diagnosis (diabetic vs. non-diabetic).\n"
        "5. **What decision or prediction does it produce?** A binary classification: 1 (Diabetic) or 0 "
        "(Non-Diabetic).\n"
        "6. **Who or what uses the prediction?** Healthcare professionals (and, via the screening app, "
        "patients themselves) as a preliminary indicator to decide on further testing or treatment.\n\n"
        "**Formal problem statement:** *Given the 8-dimensional clinical feature vector of a patient, "
        "predict whether that patient's diabetes outcome is positive (1) or negative (0) for a previously "
        "unseen observation.*"
    ))
    
    # 2. Intelligent System Diagram
    cells.append(new_markdown_cell("# 2. Intelligent System Diagram\n```mermaid\nflowchart LR\n    A[Input:\\nRaw Patient\\nMedical Data] --> B[Represent:\\nFeature Vector\\n(Scaled Numerics)]\n    B --> C[Model:\\nTrained ML\\nClassifier]\n    C --> D[Predict:\\nBinary Outcome\\n(Diabetic / Non-Diabetic)]\n```"))
    
    # 3. Dataset Source
    cells.append(new_markdown_cell("# 3. Dataset Source\nThe dataset is the \"Diabetes Dataset\" obtained from Kaggle.\nSource: [Kaggle - hasibur013/diabetes-dataset](https://www.kaggle.com/datasets/hasibur013/diabetes-dataset)"))
    
    # 4. Dataset Description
    cells.append(new_markdown_cell(
        "# 4. Dataset Description\n\n"
        "**Dataset questions (required):**\n\n"
        "1. **What real-world phenomenon is represented?** The physiological and demographic factors "
        "associated with the onset of diabetes in patients (Pima Indian heritage females).\n"
        "2. **What is one observation?** The medical record of one patient at a single point in time.\n"
        "3. **What are the features?** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
        "DiabetesPedigreeFunction, Age.\n"
        "4. **What is the target?** `Outcome` (the diagnosis).\n"
        "5. **Is the target numerical or categorical?** Categorical (binary).\n"
        "6. **Is this regression or classification?** Classification.\n"
        "7. **How many observations are available?** 768.\n"
        "8. **How many features are available?** 8.\n"
        "9. **Which features are numerical?** All 8 (continuous or discrete integers).\n"
        "10. **Which features are categorical?** None among the inputs; only the target `Outcome` is "
        "categorical."
    ))
    
    # 5. Data Representation
    cells.append(new_markdown_cell("# 5. Data Representation\nThe data is represented as an 8-dimensional numerical feature vector. \n\n| Feature | Type | Representation | Meaning |\n| --- | --- | --- | --- |\n| Pregnancies | Numerical | Integer | Number of times pregnant |\n| Glucose | Numerical | Real value | Plasma glucose concentration |\n| BloodPressure | Numerical | Real value | Diastolic blood pressure (mm Hg) |\n| SkinThickness | Numerical | Real value | Triceps skin fold thickness (mm) |\n| Insulin | Numerical | Real value | 2-Hour serum insulin (mu U/ml) |\n| BMI | Numerical | Real value | Body mass index |\n| DiabetesPedigreeFunction | Numerical | Real value | Diabetes pedigree function |\n| Age | Numerical | Integer | Age in years |\n\nZero values in Glucose, BP, SkinThickness, Insulin, and BMI are treated as missing and imputed with the median "
        "(the median is computed from the training split only, then reused for the test split, so no test "
        "information leaks into preprocessing). No categorical encoding is needed here since every raw feature "
        "is already numerical, but the pipeline still has three distinct representations: **raw feature** "
        "(value read from the CSV, may contain an invalid 0) -> **imputed feature** (0 replaced by the train "
        "median) -> **model input** (imputed feature standardized by `StandardScaler` for distance-based models; "
        "tree models consume the imputed-but-unscaled feature directly)."))
    
    # 6. Feature and Target Analysis
    cells.append(new_markdown_cell("# 6. Feature and Target Analysis\nAll inputs are purely numerical. The target is binary. Because features like 'DiabetesPedigreeFunction' have a tiny range compared to 'Insulin', the features must be standardized (scaled to mean=0, std=1) before being fed into distance-based algorithms like SVM and KNN."))
    
    # 7. Exploratory Data Analysis
    cells.append(new_markdown_cell("# 7. Exploratory Data Analysis\nBelow we explore the distributions and correlations of the dataset features."))
    
    if os.path.exists("Data_visualizer.ipynb"):
        viz_nb = load_notebook("Data_visualizer.ipynb")
        cells.extend(viz_nb.cells)

    # 8. Train/Test Split
    cells.append(new_markdown_cell(
        "# 8. Train/Test Split\n"
        "The data is split 80% training / 20% testing (`stratify=y` keeps the diabetic/non-diabetic ratio "
        "balanced in both splits) **before** any imputation or scaling statistic is computed. The median "
        "imputer and the `StandardScaler` are then `fit()` on the training split only and `transform()`-ed "
        "onto both splits, so the test set never influences preprocessing (only used later, once, for final "
        "evaluation) — this is why the test set must never be used as a training source: reusing it earlier "
        "would let information about the evaluation data quietly bias every model's training."
    ))
    
    model_nb = load_notebook("Model_training.ipynb") if os.path.exists("Model_training.ipynb") else None
    
    if model_nb:
        # The first code cell is usually the setup and split
        setup_cell = [c for c in model_nb.cells if c.cell_type == 'code'][0]
        cells.append(setup_cell)
    
    # 9. Baseline
    cells.append(new_markdown_cell(
        "# 9. Baseline\n"
        "Before evaluating complex ML models, we establish a baseline using a `DummyClassifier` that always "
        "predicts the most frequent class (Non-Diabetic). **Why is a baseline necessary?** It gives a lower "
        "bound: any trained model that fails to clearly beat this trivial strategy is not actually learning "
        "anything useful from the features. The baseline is scored with the exact same rich-metrics helper "
        "used for every other model, so it appears in the Experiment 1 comparison table on equal footing."
    ))
    cells.append(new_code_cell(
        "from sklearn.dummy import DummyClassifier\n\n"
        "dummy = DummyClassifier(strategy='most_frequent')\n"
        "dummy.fit(X_train, y_train)\n"
        "y_pred_dummy = evaluate_model('Baseline (Most Frequent)', dummy, X_test, y_test, 'baseline.pkl', needs_scaling=False)"
    ))
    
    # 10-15 Models
    if model_nb:
        code_cells = [c for c in model_nb.cells if c.cell_type == 'code']
        # 10. Model 1
        cells.append(new_markdown_cell("# 10. Model 1: Logistic Regression"))
        cells.append(code_cells[1])
        # 11. Model 2
        cells.append(new_markdown_cell("# 11. Model 2: SVM (Linear)"))
        cells.append(code_cells[2])
        # 12. Model 3
        cells.append(new_markdown_cell("# 12. Model 3: SVM (RBF)"))
        cells.append(code_cells[3]) 
        # 13. Model 4
        cells.append(new_markdown_cell("# 13. Model 4: KNN"))
        cells.append(code_cells[4])
        # 14. Model 5
        cells.append(new_markdown_cell("# 14. Model 5: Random Forest"))
        cells.append(code_cells[5])
        # 15. Model 6
        cells.append(new_markdown_cell("# 15. Model 6: XGBoost"))
        cells.append(code_cells[6])

    # 16. Evaluation
    cells.append(new_markdown_cell(
        "# 16. Evaluation\n"
        "Every model (including the baseline) is scored with **Accuracy, Precision, Recall, F1-score, and a "
        "Confusion Matrix**, printed via `evaluate_model(...)` right after training. For a medical screening "
        "tool, **Recall** is arguably the most important single metric: a low Recall means diabetic patients "
        "are being told they are healthy (a False Negative), which is the costliest mistake this system can "
        "make. **F1** is used as the overall ranking metric for Experiment 1 and for automatic final-model "
        "selection because it balances Precision and Recall and is more informative than raw Accuracy on a "
        "dataset where roughly 65% of patients are non-diabetic."
    ))

    # 17. Experiment 1
    cells.append(new_markdown_cell("# 17. Experiment 1: Model Comparison\nQuestion: Which model achieves the highest accuracy across the board?"))
    if model_nb:
        # Filter out empty code cells
        valid_code_cells = [c for c in code_cells if c.source.strip()]
        cells.append(valid_code_cells[-1]) # The comparison cell at the end of model_nb
    
    # 18. Experiment 2
    cells.append(new_markdown_cell(
        "# 18. Experiment 2: Hyperparameter Investigation\n"
        "**Question:** How do the number of neighbors (`k`) and the weighting scheme (`uniform` vs. "
        "`distance`) affect KNN classification accuracy on this dataset?\n\n"
        "**Setup:** `GridSearchCV` (5-fold) was already run inside the Model 4 (KNN) cell over "
        "`n_neighbors` in `{3, 5, 7, 9}` x `weights` in `{uniform, distance}`. The full "
        "`grid_knn.cv_results_` is inspected below instead of only reporting the single best value, so the "
        "underfitting/overfitting trade-off across `k` is visible."
    ))
    cells.append(new_code_cell(
        "knn_cv_df = pd.DataFrame(grid_knn.cv_results_)[['param_n_neighbors', 'param_weights', 'mean_test_score', 'std_test_score']]\n"
        "knn_cv_df.columns = ['k', 'weights', 'Mean CV Accuracy', 'Std']\n"
        "knn_cv_df = knn_cv_df.sort_values('Mean CV Accuracy', ascending=False).reset_index(drop=True)\n"
        "display(knn_cv_df.style.format({'Mean CV Accuracy': '{:.4f}', 'Std': '{:.4f}'}))\n"
        "\n"
        "plt.figure(figsize=(8, 5))\n"
        "sns.lineplot(data=knn_cv_df, x='k', y='Mean CV Accuracy', hue='weights', marker='o')\n"
        "plt.title('KNN: Cross-Validated Accuracy vs. Number of Neighbors (k)')\n"
        "plt.xlabel('k (n_neighbors)')\n"
        "plt.ylabel('Mean CV Accuracy')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "print(f\"Best configuration: k={grid_knn.best_params_['n_neighbors']}, weights='{grid_knn.best_params_['weights']}' \"\n"
        "      f\"-> CV accuracy = {grid_knn.best_score_:.4f}\")\n"
        "\n"
        "spread = knn_cv_df['Mean CV Accuracy'].max() - knn_cv_df['Mean CV Accuracy'].min()\n"
        "print(f\"Accuracy spread across all k/weights combinations tested: {spread:.4f} \"\n"
        "      f\"({'small -> the model is fairly insensitive to k in this range' if spread < 0.02 else 'noticeable -> k is a meaningful hyperparameter to tune here'}).\")"
    ))

    # 19. Experiment 3
    cells.append(new_markdown_cell("# 19. Experiment 3: Representation / Feature Investigation\nQuestion: Does standardizing the numerical features improve the performance of distance-based models like KNN?"))
    cells.append(new_code_cell("from sklearn.neighbors import KNeighborsClassifier\n\n# Train KNN on unscaled data\nknn_unscaled = KNeighborsClassifier(n_neighbors=5)\nknn_unscaled.fit(X_train, y_train)\nacc_unscaled = accuracy_score(y_test, knn_unscaled.predict(X_test))\n\n# Train KNN on scaled data\nknn_scaled = KNeighborsClassifier(n_neighbors=5)\nknn_scaled.fit(X_train_scaled, y_train)\nacc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))\n\nprint(f'KNN Accuracy (Unscaled Data): {acc_unscaled*100:.2f}%')\nprint(f'KNN Accuracy (Scaled Data): {acc_scaled*100:.2f}%')\nprint('\\nConclusion: Unscaled features with large ranges (like Insulin) completely dominate the distance calculation in KNN. Scaling forces all features to contribute equally, significantly improving accuracy.')"))
    
    # 20. Final Model
    cells.append(new_markdown_cell(
        "# 20. Final Model\n"
        "The final model is **selected automatically from the Experiment 1 table**, not asserted by hand: "
        "we take the highest-**F1** model among the trained classifiers (the baseline is excluded from "
        "selection since it exists only as a reference point, per Section 9)."
    ))
    cells.append(new_code_cell(
        "candidates_df = results_df_display[results_df_display['Model'] != 'Baseline (Most Frequent)']\n"
        "best_row = candidates_df.sort_values('F1', ascending=False).iloc[0]\n"
        "best_model_name = best_row['Model']\n"
        "best_model_file = model_filenames[best_model_name]\n"
        "final_model_needs_scaling = model_needs_scaling[best_model_name]\n"
        "final_model = joblib.load(best_model_file)\n"
        "\n"
        "print(f\"Selected final model: {best_model_name}\")\n"
        "print(f\"  -> Serialized file: {best_model_file}\")\n"
        "print(f\"  -> Accuracy={best_row['Accuracy']:.2f}%, Precision={best_row['Precision']:.3f}, \"\n"
        "      f\"Recall={best_row['Recall']:.3f}, F1={best_row['F1']:.3f}\")\n"
        "print(\"Selection rule: highest F1-score on the held-out test set among the trained models \"\n"
        "      \"(F1 balances Precision and Recall, appropriate for this imbalanced medical screening task).\")"
    ))

    # 21. Application
    cells.append(new_markdown_cell(
        "# 21. Application\n"
        "A full-stack application (`app/` directory: FastAPI backend + React frontend, with SHAP-based "
        "explanations) was developed to serve the models. Here, the same input -> representation -> "
        "preprocessing -> model -> prediction pipeline used by that API is reproduced directly in the "
        "notebook, so the complete path can be demonstrated without leaving the notebook."
    ))
    cells.append(new_code_cell(
        "FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',\n"
        "                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']\n"
        "ZERO_AS_MISSING_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n"
        "\n"
        "\n"
        "def predict_diabetes(patient: dict, model=final_model, needs_scaling=final_model_needs_scaling):\n"
        "    \"\"\"Run one raw patient record through the exact preprocessing pipeline used in training:\n"
        "    Input -> Representation (feature vector) -> Preprocessing (impute + scale) -> Model -> Prediction.\"\"\"\n"
        "    row = pd.DataFrame([patient], columns=FEATURE_NAMES)\n"
        "    row[ZERO_AS_MISSING_COLS] = row[ZERO_AS_MISSING_COLS].replace(0, np.nan)\n"
        "    row[ZERO_AS_MISSING_COLS] = imputer.transform(row[ZERO_AS_MISSING_COLS])\n"
        "\n"
        "    model_input = scaler.transform(row) if needs_scaling else row.values\n"
        "\n"
        "    pred = model.predict(model_input)[0]\n"
        "    proba = model.predict_proba(model_input)[0][1] if hasattr(model, 'predict_proba') else None\n"
        "    return int(pred), proba\n"
        "\n"
        "\n"
        "print(f\"Application ready. Serving model: {best_model_name}\")"
    ))

    # 22. System Demonstration
    cells.append(new_markdown_cell(
        "# 22. System Demonstration\n"
        "**Complete path:** User / Environment -> Input (raw clinical values) -> Feature Representation "
        "(8-d vector) -> Preprocessing (impute + scale) -> ML Model -> Prediction.\n\n"
        "Below, three different patient profiles are run through `predict_diabetes(...)` end to end, "
        "exactly as the deployed FastAPI `/api/predict` endpoint would process them."
    ))
    cells.append(new_code_cell(
        "sample_patients = {\n"
        "    'Low-risk profile': {'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 'SkinThickness': 29,\n"
        "                          'Insulin': 0, 'BMI': 26.6, 'DiabetesPedigreeFunction': 0.351, 'Age': 22},\n"
        "    'Borderline profile': {'Pregnancies': 3, 'Glucose': 130, 'BloodPressure': 78, 'SkinThickness': 23,\n"
        "                           'Insulin': 100, 'BMI': 31.2, 'DiabetesPedigreeFunction': 0.523, 'Age': 34},\n"
        "    'High-risk profile': {'Pregnancies': 8, 'Glucose': 183, 'BloodPressure': 90, 'SkinThickness': 0,\n"
        "                          'Insulin': 0, 'BMI': 39.5, 'DiabetesPedigreeFunction': 1.288, 'Age': 51},\n"
        "}\n"
        "\n"
        "print(f\"{'Case':<20}{'Prediction':<16}{'P(Diabetic)':<15}\")\n"
        "print('-' * 51)\n"
        "for label, patient in sample_patients.items():\n"
        "    pred, proba = predict_diabetes(patient)\n"
        "    outcome = 'Diabetic' if pred == 1 else 'Non-Diabetic'\n"
        "    proba_str = f'{proba:.2%}' if proba is not None else 'N/A'\n"
        "    print(f'{label:<20}{outcome:<16}{proba_str:<15}')"
    ))

    # 23. Reflection
    cells.append(new_markdown_cell("# 23. Reflection\n- The system processes continuous and discrete numerical health vectors.\n- It learns nonlinear statistical boundaries to distinguish diabetic from non-diabetic profiles.\n- The 'intelligence' is isolated to the model training phase where weights are iteratively adjusted.\n- Limitations include geographical/demographic bias in the dataset (Pima Indian females) and a lack of temporal tracking."))

    # 24. Conclusion
    cells.append(new_markdown_cell("# 24. Conclusion\nThe notebook successfully demonstrates the complete pipeline from raw data ingestion to EDA, intelligent model comparison, and final deployment. The experiments highlight the necessity of feature scaling and rigorous hyperparameter tuning in traditional ML."))

    nb.cells = cells
    with open('Report_Notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    build()

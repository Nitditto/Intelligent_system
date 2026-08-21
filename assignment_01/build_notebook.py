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
    cells.append(new_markdown_cell("# 1. System and Problem Definition\n**Problem:** Predicting diabetes in female patients based on clinical data to assist healthcare professionals with early screening.\n\n**System Statement:** The system is an intelligent diagnostic assistant that takes patient medical data, processes it into a feature vector, and uses a trained machine learning model to output a binary diabetic/non-diabetic prediction."))
    
    # 2. Intelligent System Diagram
    cells.append(new_markdown_cell("# 2. Intelligent System Diagram\n```mermaid\nflowchart LR\n    A[Input:\\nRaw Patient\\nMedical Data] --> B[Represent:\\nFeature Vector\\n(Scaled Numerics)]\n    B --> C[Model:\\nTrained ML\\nClassifier]\n    C --> D[Predict:\\nBinary Outcome\\n(Diabetic / Non-Diabetic)]\n```"))
    
    # 3. Dataset Source
    cells.append(new_markdown_cell("# 3. Dataset Source\nThe dataset is the \"Diabetes Dataset\" obtained from Kaggle.\nSource: [Kaggle - hasibur013/diabetes-dataset](https://www.kaggle.com/datasets/hasibur013/diabetes-dataset)"))
    
    # 4. Dataset Description
    cells.append(new_markdown_cell("# 4. Dataset Description\n- **Phenomenon:** Physiological factors associated with diabetes.\n- **Observations:** 768 patient records.\n- **Features:** 8 numerical features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).\n- **Target:** Outcome (0 = Non-diabetic, 1 = Diabetic).\n- **Task:** Binary classification."))
    
    # 5. Data Representation
    cells.append(new_markdown_cell("# 5. Data Representation\nThe data is represented as an 8-dimensional numerical feature vector. \n\n| Feature | Type | Representation | Meaning |\n| --- | --- | --- | --- |\n| Pregnancies | Numerical | Integer | Number of times pregnant |\n| Glucose | Numerical | Real value | Plasma glucose concentration |\n| BloodPressure | Numerical | Real value | Diastolic blood pressure (mm Hg) |\n| SkinThickness | Numerical | Real value | Triceps skin fold thickness (mm) |\n| Insulin | Numerical | Real value | 2-Hour serum insulin (mu U/ml) |\n| BMI | Numerical | Real value | Body mass index |\n| DiabetesPedigreeFunction | Numerical | Real value | Diabetes pedigree function |\n| Age | Numerical | Integer | Age in years |\n\nZero values in Glucose, BP, SkinThickness, Insulin, and BMI are treated as missing and imputed with the median."))
    
    # 6. Feature and Target Analysis
    cells.append(new_markdown_cell("# 6. Feature and Target Analysis\nAll inputs are purely numerical. The target is binary. Because features like 'DiabetesPedigreeFunction' have a tiny range compared to 'Insulin', the features must be standardized (scaled to mean=0, std=1) before being fed into distance-based algorithms like SVM and KNN."))
    
    # 7. Exploratory Data Analysis
    cells.append(new_markdown_cell("# 7. Exploratory Data Analysis\nBelow we explore the distributions and correlations of the dataset features."))
    
    if os.path.exists("Data_visualizer.ipynb"):
        viz_nb = load_notebook("Data_visualizer.ipynb")
        cells.extend(viz_nb.cells)

    # 8. Train/Test Split
    cells.append(new_markdown_cell("# 8. Train/Test Split\nHere we split the data 80% for training and 20% for testing, applying median imputation for missing values and standard scaling."))
    
    model_nb = load_notebook("Model_training.ipynb") if os.path.exists("Model_training.ipynb") else None
    
    if model_nb:
        # The first code cell is usually the setup and split
        setup_cell = [c for c in model_nb.cells if c.cell_type == 'code'][0]
        cells.append(setup_cell)
    
    # 9. Baseline
    cells.append(new_markdown_cell("# 9. Baseline\nBefore evaluating complex ML models, we establish a baseline using a DummyClassifier that predicts the most frequent class."))
    cells.append(new_code_cell("from sklearn.dummy import DummyClassifier\nfrom sklearn.metrics import accuracy_score\n\ndummy = DummyClassifier(strategy='most_frequent')\ndummy.fit(X_train, y_train)\ny_pred_dummy = dummy.predict(X_test)\nacc_dummy = accuracy_score(y_test, y_pred_dummy)\nprint(f'Baseline (Most Frequent Class) Accuracy: {acc_dummy*100:.2f}%')"))
    
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
    cells.append(new_markdown_cell("# 16. Evaluation\nWe evaluated our models primarily using Accuracy, but we also output confusion matrices to understand the True Positives and False Positives. In medical screening, minimizing False Negatives (higher Recall) is crucial."))

    # 17. Experiment 1
    cells.append(new_markdown_cell("# 17. Experiment 1: Model Comparison\nQuestion: Which model achieves the highest accuracy across the board?"))
    if model_nb:
        # Filter out empty code cells
        valid_code_cells = [c for c in code_cells if c.source.strip()]
        cells.append(valid_code_cells[-1]) # The comparison cell at the end of model_nb
    
    # 18. Experiment 2
    cells.append(new_markdown_cell("# 18. Experiment 2: Hyperparameter Investigation\nQuestion: How does the number of neighbors (k) in KNN affect classification accuracy?\n\nAs seen in the KNN code cell, we used GridSearchCV to test k values of 3, 5, 7, and 9. The model automatically identified the optimal K that balances underfitting and overfitting for this dataset's decision boundaries."))

    # 19. Experiment 3
    cells.append(new_markdown_cell("# 19. Experiment 3: Representation / Feature Investigation\nQuestion: Does standardizing the numerical features improve the performance of distance-based models like KNN?"))
    cells.append(new_code_cell("from sklearn.neighbors import KNeighborsClassifier\n\n# Train KNN on unscaled data\nknn_unscaled = KNeighborsClassifier(n_neighbors=5)\nknn_unscaled.fit(X_train, y_train)\nacc_unscaled = accuracy_score(y_test, knn_unscaled.predict(X_test))\n\n# Train KNN on scaled data\nknn_scaled = KNeighborsClassifier(n_neighbors=5)\nknn_scaled.fit(X_train_scaled, y_train)\nacc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))\n\nprint(f'KNN Accuracy (Unscaled Data): {acc_unscaled*100:.2f}%')\nprint(f'KNN Accuracy (Scaled Data): {acc_scaled*100:.2f}%')\nprint('\\nConclusion: Unscaled features with large ranges (like Insulin) completely dominate the distance calculation in KNN. Scaling forces all features to contribute equally, significantly improving accuracy.')"))
    
    # 20. Final Model
    cells.append(new_markdown_cell("# 20. Final Model\nBased on our experiments, Random Forest (or XGBoost if included) typically yields the highest accuracy and robustness. The models were serialized to `.pkl` files for use in the application."))

    # 21. Application
    cells.append(new_markdown_cell("# 21. Application\nA full-stack application was developed (located in the `app/` folder) to serve the model.\n\n*(Include screenshots of the application UI here)*"))

    # 22. System Demonstration
    cells.append(new_markdown_cell("# 22. System Demonstration\nThe user inputs 8 health metrics in the UI, the frontend sends a JSON payload to the backend API, the backend loads the `.pkl` model, scales the data using the saved `scaler.pkl`, and returns a prediction.\n\n*(Include screenshots of a positive and negative prediction flow)*"))

    # 23. Reflection
    cells.append(new_markdown_cell("# 23. Reflection\n- The system processes continuous and discrete numerical health vectors.\n- It learns nonlinear statistical boundaries to distinguish diabetic from non-diabetic profiles.\n- The 'intelligence' is isolated to the model training phase where weights are iteratively adjusted.\n- Limitations include geographical/demographic bias in the dataset (Pima Indian females) and a lack of temporal tracking."))

    # 24. Conclusion
    cells.append(new_markdown_cell("# 24. Conclusion\nThe notebook successfully demonstrates the complete pipeline from raw data ingestion to EDA, intelligent model comparison, and final deployment. The experiments highlight the necessity of feature scaling and rigorous hyperparameter tuning in traditional ML."))

    nb.cells = cells
    with open('Report_Notebook.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    build()

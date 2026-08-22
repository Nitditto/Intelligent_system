# Assignment 1: Diabetes System Report

## 1. Introduction
This report documents the design, implementation, and evaluation of an intelligent system designed to predict the likelihood of diabetes in female patients based on their medical records. The system utilizes various traditional Machine Learning models to analyze clinical measurements and provide diagnostic assistance to healthcare professionals.

## 2. System Definition
### System Statement
The intelligent system is a diagnostic assistant that analyzes a patient's clinical and demographic data—such as glucose levels, BMI, and age—to predict whether the patient has diabetes. It serves as a preliminary screening tool to aid doctors in identifying high-risk patients quickly.

### System Diagram
```mermaid
flowchart LR
    A[Input:\nRaw Patient\nMedical Data] --> B[Represent:\nFeature Vector\n(Scaled Numerics)]
    B --> C[Model:\nTrained ML\nClassifier]
    C --> D[Predict:\nBinary Outcome\n(Diabetic / Non-Diabetic)]
```

## 3. Problem Definition
- **What real-world problem does the system address?** It addresses the need for early and accurate detection of diabetes to allow for timely medical intervention and lifestyle changes, reducing long-term health complications.
- **What information does the system receive?** The system receives demographic data (Age) and clinical measurements (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, and a Diabetes Pedigree Function score).
- **How is that information represented internally?** The information is represented as a structured feature vector of real numbers. Missing or invalid values (like 0 for BMI) are imputed with the median, and the vectors are standardized (scaled) so that all features have a mean of 0 and a standard deviation of 1.
- **What does the model learn?** The model learns the statistical correlations and decision boundaries between the input clinical features and the historic diagnosis (diabetic vs. non-diabetic).
- **What decision or prediction does it produce?** It produces a binary classification prediction: 1 (Diabetic) or 0 (Non-Diabetic).
- **Who or what uses the prediction?** Healthcare professionals, doctors, and potentially the patients themselves (via a screening app) use the prediction as a preliminary indicator to decide on further testing or treatment.

## 4. Dataset
- **Dataset Source:** Kaggle - Diabetes Dataset (https://www.kaggle.com/datasets/hasibur013/diabetes-dataset)
- **What real-world phenomenon is represented?** The physiological and demographic factors associated with the onset of diabetes in patients (specifically females of Pima Indian heritage).
- **What is one observation?** A single observation represents the medical record of one specific patient at a single point in time.
- **What are the features?** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- **What is the target?** Outcome (the diagnosis).
- **Is the target numerical or categorical?** Categorical (Binary).
- **Is this regression or classification?** Classification.
- **How many observations are available?** 768 observations.
- **How many features are available?** 8 features.
- **Which features are numerical?** All 8 features are numerical (continuous or discrete integers).
- **Which features are categorical?** None of the input features are categorical. The target (Outcome) is categorical.

## 5. Data Representation

| Feature | Type | Representation | Meaning |
| --- | --- | --- | --- |
| Pregnancies | Numerical | Integer | Number of times pregnant |
| Glucose | Numerical | Real value | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| BloodPressure | Numerical | Real value | Diastolic blood pressure (mm Hg) |
| SkinThickness | Numerical | Real value | Triceps skin fold thickness (mm) |
| Insulin | Numerical | Real value | 2-Hour serum insulin (mu U/ml) |
| BMI | Numerical | Real value | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction | Numerical | Real value | Diabetes pedigree function (genetic risk) |
| Age | Numerical | Integer | Age in years |

*(Note: There are no categorical features requiring encoding in this dataset. However, invalid `0` measurements for features like Glucose and BMI were imputed to their median, and the entire feature vector was standardized using `StandardScaler` to have a mean of 0 and variance of 1. Both the imputer and the scaler are `fit()` on the training split only and `transform()`-ed onto the test split, so no test-set information leaks into preprocessing — three distinct representations exist along the pipeline: **raw feature** (may contain an invalid 0) → **imputed feature** (0 replaced by the train-only median) → **model input** (imputed feature standardized for distance-based models; tree models use the imputed-but-unscaled feature directly).)*

## 6. Traditional ML Methods

### 1. Logistic Regression
- **What representation does it receive?** Scaled numerical feature vectors.
- **What relationship does it try to learn?** A linear relationship between the features and the log-odds of the probability of the target being class 1 (Diabetic).
- **What parameters or structures are learned?** Weights (coefficients) for each feature and a bias term (intercept).
- **What criterion guides learning?** Maximizing the likelihood of the training data (minimizing the log-loss/cross-entropy loss).
- **What assumptions does the model make?** Assumes a linear relationship between the independent variables and the log-odds of the dependent variable. Assumes minimal multicollinearity among features.
- **What are its strengths?** Fast, highly interpretable, provides probabilities rather than just classifications.
- **What are its weaknesses?** Cannot solve non-linear problems easily without feature engineering; underperforms on highly complex datasets.

### 2. Support Vector Machine (Linear Kernel)
- **What representation does it receive?** Scaled numerical feature vectors.
- **What relationship does it try to learn?** The optimal straight hyperplane that maximizes the margin of separation between the two classes.
- **What parameters or structures are learned?** The weights defining the hyperplane and the support vectors (the specific data points closest to the boundary).
- **What criterion guides learning?** Maximizing the margin while penalizing points that cross the margin (hinge loss), controlled by the regularization parameter $C$.
- **What assumptions does the model make?** Assumes the data is linearly separable (or soft-margin separable) in the feature space.
- **What are its strengths?** Effective in high-dimensional spaces; robust against outliers when properly regularized.
- **What are its weaknesses?** Scales poorly to very large datasets; linear kernel cannot capture non-linear relationships.

### 3. Support Vector Machine (RBF Kernel)
- **What representation does it receive?** Scaled numerical feature vectors.
- **What relationship does it try to learn?** A complex, non-linear boundary that separates classes by projecting data into a higher-dimensional space using the Radial Basis Function.
- **What parameters or structures are learned?** Support vectors, weights, and the influence of each support vector (gamma).
- **What criterion guides learning?** Maximizing margin in the transformed high-dimensional space.
- **What assumptions does the model make?** Assumes closer points in the transformed space belong to the same class.
- **What are its strengths?** Highly capable of modeling non-linear, complex datasets.
- **What are its weaknesses?** Harder to interpret than linear SVM; training is computationally expensive; sensitive to hyperparameter choices.

### 4. K-Nearest Neighbors (KNN)
- **What representation does it receive?** Scaled numerical feature vectors.
- **What relationship does it try to learn?** It doesn't learn a global mathematical function. It assumes similar inputs have similar outputs based on spatial distance.
- **What parameters or structures are learned?** KNN is a "lazy learner"—it essentially memorizes the training data. No parameters are explicitly learned during training.
- **What criterion guides learning?** Proximity (usually Euclidean distance). The class of a new point is determined by a majority vote of its $k$ closest neighbors.
- **What assumptions does the model make?** Assumes that data points close to each other in the feature space belong to the same class. Assumes all features contribute equally to the distance (hence scaling is mandatory).
- **What are its strengths?** Simple to understand, handles non-linear boundaries well, no complex training phase required.
- **What are its weaknesses?** Very slow at prediction time for large datasets; highly sensitive to irrelevant features and the curse of dimensionality.

### 5. Random Forest
- **What representation does it receive?** Unscaled numerical feature vectors (trees do not require scaling).
- **What relationship does it try to learn?** A complex, non-linear hierarchical decision structure by combining multiple independent decision trees.
- **What parameters or structures are learned?** An ensemble of decision trees, each containing a series of splitting rules based on feature thresholds.
- **What criterion guides learning?** Minimizing impurity (like Gini impurity or Entropy) at each split for the individual trees.
- **What assumptions does the model make?** Very few assumptions about the data distribution. Assumes combining weak learners reduces overall variance.
- **What are its strengths?** Highly accurate, robust to outliers, naturally handles non-linear relationships and interactions without requiring feature scaling.
- **What are its weaknesses?** Prone to overfitting if trees are too deep (though the ensemble mitigates this); acts as a "black box," making it hard to interpret the exact decision path.

### 6. XGBoost (Extreme Gradient Boosting)
- **What representation does it receive?** Unscaled numerical feature vectors.
- **What relationship does it try to learn?** A non-linear decision boundary built by sequentially combining weak decision trees where each tree corrects the errors of the previous ones.
- **What parameters or structures are learned?** A sequence of trees, with specific split points and leaf weights optimized via gradient descent.
- **What criterion guides learning?** Minimizing a differentiable loss function (log-loss) using gradient descent over the ensemble.
- **What assumptions does the model make?** Assumes that the errors of prior trees can be modeled and corrected by subsequent trees.
- **What are its strengths?** Often achieves state-of-the-art performance on tabular data; handles missing values natively; highly optimized.
- **What are its weaknesses?** Can easily overfit if not tuned correctly; less interpretable than single trees; computationally heavier to tune than Random Forest.

## 7. Experimental Design

### Experiment 1: Model Comparison
- **Question:** Which of the traditional machine learning models provides the best performance for classifying diabetes when trained under identical conditions?
- **Setup:** We compare Logistic Regression, SVM (Linear), SVM (RBF), KNN, Random Forest, and XGBoost using the same 80/20 train-test split. The scaled data is provided to distance-based models (LR, SVM, KNN), and unscaled data to tree models (Random Forest, XGBoost). We evaluate using Accuracy, Precision, Recall, and F1-score.

### Experiment 2: Hyperparameter Investigation
- **Question:** How do the number of neighbors ($k$) and the weighting scheme (`uniform` vs. `distance`) affect the performance of the KNN algorithm on this dataset?
- **Setup:** Using 5-fold `GridSearchCV`, we test $k \in \{3, 5, 7, 9\}$ × `weights` $\in \{$uniform, distance$\}$, then inspect the **full** `cv_results_` (not just the single best combination) to see how cross-validated accuracy moves across the whole grid.
- **Result:** Best configuration: $k=9$, `weights='uniform'` → CV accuracy = 0.7607. The accuracy spread across all 8 combinations tested was 0.0342 — noticeable enough that $k$ is a meaningful hyperparameter here, not just a knob that can be left at a default. Accuracy trends upward from $k=3$ to $k=9$ in this range, consistent with $k=3$ overfitting to a few noisy neighbors (high variance) and larger $k$ smoothing the decision boundary (higher bias, lower variance) more appropriately for this dataset's class overlap.

### Experiment 3: Representation / Feature Investigation
- **Question:** Does standardizing (scaling) the feature vectors improve the performance of distance-based models like KNN?
- **Setup:** We train a K-Nearest Neighbors classifier ($k=5$) twice: once on the raw, unscaled feature vectors, and once on the standardized feature vectors, and compare test accuracy.
- **Result:** Unscaled accuracy = **67.53%**, Scaled accuracy = **75.32%** — a **+7.8 point** jump from standardization alone. This confirms the expectation: raw features have vastly different ranges (e.g., Pedigree Function is < 1, while Insulin can be > 100), so unscaled Euclidean distance is dominated by the large-range features. Standardizing forces every feature to contribute proportionally, which is exactly why the representation choice ("Data Representation", Section 5) is not a cosmetic detail but a determinant of model quality.

## 8. Results

*Reproduced end-to-end by executing [Report_Notebook.ipynb](Report_Notebook.ipynb) (Aug 2026 run, after fixing the train/test leakage in the median-imputation step — see Section 5).*

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Baseline (Most Frequent) | 64.94% | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 70.78% | 0.600 | 0.500 | 0.545 |
| SVM (Linear) | 70.78% | 0.605 | 0.481 | 0.536 |
| SVM (RBF) | 69.48% | 0.581 | 0.463 | 0.515 |
| KNN (Best k=9, uniform) | 74.03% | 0.635 | 0.611 | 0.623 |
| **Random Forest** | **76.62%** | **0.696** | 0.593 | **0.640** |
| XGBoost | 74.68% | 0.653 | 0.593 | 0.621 |

The baseline's Precision/Recall/F1 are all **0.000** because `DummyClassifier(strategy='most_frequent')` always predicts "Non-Diabetic" — it never once identifies a diabetic patient, even though it still scores 64.94% accuracy (the non-diabetic proportion of the dataset). This is exactly why a baseline is necessary: accuracy alone would make this trivial, useless strategy look "pretty good," while Precision/Recall/F1 correctly expose that it has zero diagnostic value.

**Appropriate Metrics:** 
For a medical diagnostic tool like predicting diabetes, **Recall** is arguably the most critical metric. High recall ensures that we minimize False Negatives (telling a diabetic patient they are healthy, leading to lack of treatment). While Accuracy gives a good overall picture, **F1-score** (harmonic mean of precision and recall) is used as the ranking metric for model comparison and automatic final-model selection, since it is more informative than raw accuracy on this dataset (roughly 65% non-diabetic / 35% diabetic).

## 9. Model Comparison
Random Forest achieved the highest overall performance — Accuracy 76.62%, Precision 0.696, and the best F1-score (0.640) — and is selected as the final model. As an ensemble method, it naturally captured the non-linear relationships and interactions between features like Glucose, BMI, and Age better than the linear models (Logistic Regression and SVM Linear). KNN (k=9) and XGBoost followed closely (F1 = 0.623 and 0.621), confirming that both instance-based and boosted-tree methods are competitive on this tabular dataset once features are properly represented (imputed + scaled). SVM (RBF) was the weakest trained model (F1 = 0.515), edging out only the baseline — its non-linear kernel did not find structure beyond what the simpler linear models already captured on this dataset.

## 10. Representation Analysis
- **Why is your feature-vector representation appropriate?** The medical data naturally comes in independent tabular measurements. A fixed-length feature vector perfectly encapsulates these discrete health metrics for a single patient.
- **What information does it preserve?** It preserves the exact magnitude and statistical distribution of the clinical tests.
- **What information might it lose?** It loses temporal dynamics. A single feature vector captures a snapshot in time; it cannot show if a patient's BMI or Glucose has been rapidly increasing or decreasing over the last 5 years.
- **Could the same problem be represented as an image?** No, not effectively. However, if the data included retinal scans (to detect diabetic retinopathy) or ultrasound images of organs, image representation would be appropriate.
- **Could it be represented as a sequence?** Yes. If we had multiple visits recorded for the same patient over time, we could represent the data as a time-series sequence and use RNNs/LSTMs.
- **Could it be represented as a graph?** Yes, if we included genetic relationships (family trees) or social/environmental proximity to see if diabetes clusters in specific communities or family branches.
- **Could it be represented using learned embeddings?** If we were processing raw electronic health record (EHR) text notes from a doctor, we could learn embeddings. For standard numerical clinical features, learned embeddings are generally unnecessary.
- **What would change if the representation changed?** The choice of model would have to change drastically (e.g., CNNs for images, RNNs for sequences, GNNs for graphs). The computational cost would increase, and the interpretability of the model (knowing exactly why a prediction was made) would likely decrease.

## 11. Intelligent Application
The trained `random_forest.pkl` model (auto-selected in Section 20 of the notebook as the highest-F1 model) is integrated into a full-stack application (`app/` directory: FastAPI backend + React frontend, with SHAP-based per-feature explanations). Users enter clinical data into the UI and receive an instant prediction from the backend `/api/predict` endpoint.

The exact same pipeline (impute → scale → predict) is also demonstrated directly inside the notebook via a `predict_diabetes()` function (Section 21–22), so the complete path can be verified without needing the web UI. Three input cases were run end-to-end:

| Case | Prediction | P(Diabetic) |
| --- | --- | --- |
| Low-risk profile (young, normal glucose/BMI) | Non-Diabetic | 1.00% |
| Borderline profile (elevated glucose, mid BMI) | Diabetic | 54.00% |
| High-risk profile (high glucose, high BMI, high pedigree) | Diabetic | 85.00% |

These three cases move monotonically with the clinical risk factors, which is the qualitative sanity check expected of the system: as glucose, BMI, and pedigree function rise, the predicted diabetes probability rises with them.
*[Insert screenshot of the web Application UI + a live prediction here for the final submission]*

## 12. Limitations
- **Data Bias:** The dataset is limited to females of Pima Indian heritage. The model will likely fail to generalize accurately to males, or to populations with different genetic backgrounds and dietary habits.
- **Temporal Blindness:** As a point-in-time assessment, the model cannot factor in a patient's trajectory (e.g., recent rapid weight loss).
- **False Negatives:** The model does not achieve 100% recall, meaning it will still miss some diabetic patients, which is a significant risk in healthcare.

## 13. Reflection
- **What information does your system receive?** 8 clinical and demographic numerical measurements.
- **What is the internal representation?** A scaled, 8-dimensional feature vector.
- **What does the model learn from examples?** It learns the statistical boundaries (like decision trees or hyperplanes) that separate diabetic patients from non-diabetic patients based on historical patterns.
- **What prediction or decision does it make?** It outputs a discrete binary classification (0 or 1).
- **Why can it handle an unseen input?** Because it learned underlying generalizable patterns (e.g., "High BMI + High Glucose = High Risk") rather than strictly memorizing the exact rows of the training data.
- **What part of the system can reasonably be called “intelligent”?** The model training phase, where the algorithm autonomously adjusts its internal parameters (weights, splits) to minimize errors and discover patterns without explicit human programming of rules.
- **What limitations prevent it from being a more capable intelligent system?** It lacks contextual awareness. It cannot ask follow-up questions, it doesn't know the patient's family history (beyond the pedigree function), and it cannot explain its reasoning to the doctor in a human-readable way (lack of deep explainability).

## 14. Conclusion
We successfully designed and implemented an intelligent system to predict diabetes based on clinical features. After correcting a train/test leakage issue in the imputation step (median now computed from the training split only) and adding full Precision/Recall/F1 reporting for every model, Random Forest achieved the best held-out performance (Accuracy 76.62%, F1 0.640) and was selected automatically as the final model. The three controlled experiments showed that: (1) all six trained models clearly beat the zero-F1 baseline, (2) KNN's accuracy is meaningfully sensitive to `k` (spread of 0.034 across the grid), and (3) standardizing features improves KNN accuracy by +7.8 points — confirming that representation choices, not just model choice, materially affect outcomes. While the system demonstrates clear value as a preliminary screening tool — verified end-to-end in the notebook with three sample patients — its deployment must be accompanied by awareness of its demographic biases (Pima Indian females only), lack of temporal awareness, and the fact that the whole pipeline was built on only 768 records.

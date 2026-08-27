import os
import uuid
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models import PredictionRequest, PredictionResponse, ModelResultSummary, DetailedModelResult, SHAPFeature
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory database
results_db = {}

# Global variables for models and utilities
models = {}
scaler = None
imputer = None
model_accuracies = {}
explainers = {}
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def load_resources():
    global scaler, imputer, models, model_accuracies, explainers

    # Load dataset to evaluate accuracy and setup SHAP background
    df = pd.read_csv("diabetes_dataset.csv")

    X = df[feature_names]
    y = df["Outcome"]

    # Split BEFORE imputation (mirrors the training notebook) so preprocessing
    # statistics are never derived from the test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X_train.copy(), X_test.copy()

    # Impute invalid 0 values using the imputer fit on the training split only (same as notebook)
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    imputer = joblib.load("imputer.pkl")
    X_train[cols_to_impute] = X_train[cols_to_impute].replace(0, np.nan)
    X_test[cols_to_impute] = X_test[cols_to_impute].replace(0, np.nan)
    X_train[cols_to_impute] = imputer.transform(X_train[cols_to_impute])
    X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

    # Load scaler
    scaler = joblib.load("scaler.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Background data for KernelExplainer to speed up
    background_data = shap.sample(X_train_scaled, 50)

    # Load all predictive models (exclude preprocessing artifacts, which are not models)
    non_model_files = {"scaler.pkl", "imputer.pkl", "baseline.pkl"}
    model_files = [f for f in os.listdir(".") if f.endswith(".pkl") and f not in non_model_files]
    for f in model_files:
        model_name = f.replace(".pkl", "")
        model = joblib.load(f)
        models[model_name] = model
        
        # Compute accuracy on the test dataset
        if model_name in ["random_forest", "xgboost"]:
            preds = model.predict(X_test)
        else:
            preds = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, preds)
        model_accuracies[model_name] = acc
        
        # Setup explainer
        try:
            if model_name in ["random_forest", "xgboost"]:
                explainers[model_name] = shap.TreeExplainer(model)
            else:
                explainers[model_name] = shap.KernelExplainer(model.predict_proba, background_data)
        except Exception as e:
            print(f"Error setting up explainer for {model_name}: {e}")
            # Fallback to KernelExplainer if something fails
            if hasattr(model, "predict_proba"):
                explainers[model_name] = shap.KernelExplainer(model.predict_proba, background_data)
            else:
                explainers[model_name] = shap.KernelExplainer(model.predict, background_data)

@app.on_event("startup")
async def startup_event():
    load_resources()
    print("Loaded models and computed accuracies:", model_accuracies)

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    req_dict = req.dict()
    features = np.array([[req_dict[f] for f in feature_names]])
    scaled_features = scaler.transform(features)
    
    result_id = str(uuid.uuid4())
    results_db[result_id] = {}
    
    summary_list = []
    
    for model_name, model in models.items():
        is_tree = model_name in ["random_forest", "xgboost"]
        model_input = features if is_tree else scaled_features

        # Prediction and confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(model_input)[0]
            pred = np.argmax(proba)
            confidence = proba[pred]
        else:
            pred = model.predict(model_input)[0]
            confidence = 1.0 # fallback if no proba
            
        # SHAP Values
        explainer = explainers.get(model_name)
        shap_values_obj = None
        shap_features = []
        if explainer:
            shap_vals = explainer.shap_values(model_input)
            
            if isinstance(shap_vals, list):
                # Older SHAP versions or certain explainers return list of arrays (one per class)
                sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                # Newer SHAP versions often return a single ndarray
                if shap_vals.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    sv = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
                else:
                    # (n_samples, n_features)
                    sv = shap_vals[0]

            for i, f_name in enumerate(feature_names):
                shap_features.append(SHAPFeature(feature=f_name, value=float(sv[i])))
                
        acc = model_accuracies.get(model_name, 0.0)
        
        # Save detailed results
        results_db[result_id][model_name] = {
            "model": model_name,
            "prediction": int(pred),
            "confidence": float(confidence),
            "accuracy": float(acc),
            "shap_values": shap_features
        }
        
        summary_list.append(ModelResultSummary(
            model=model_name, accuracy=acc, prediction=int(pred), confidence=float(confidence)
        ))
    
    # Sort summaries by accuracy descending
    summary_list.sort(key=lambda x: x.accuracy, reverse=True)
    
    return PredictionResponse(id=result_id, results=summary_list)

@app.get("/api/result/{result_id}", response_model=DetailedModelResult)
async def get_result(result_id: str, model: str):
    if result_id not in results_db:
        raise HTTPException(status_code=404, detail="Result ID not found")
    if model not in results_db[result_id]:
        raise HTTPException(status_code=404, detail="Model not found for this result")
        
    return results_db[result_id][model]


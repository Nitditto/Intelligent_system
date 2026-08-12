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
model_accuracies = {}
explainers = {}
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def load_resources():
    global scaler, models, model_accuracies, explainers
    
    # Load dataset to evaluate accuracy and setup SHAP background
    df = pd.read_csv("diabetes_dataset.csv")

    # Impute invalid 0 values (same as notebook)
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col] = df[col].replace(0, df[col].median())

    X = df[feature_names]
    y = df["Outcome"]

    # Split features and target to evaluate accuracy on the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load scaler
    scaler = joblib.load("scaler.pkl")

    X_scaled = scaler.transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Background data for KernelExplainer to speed up
    background_data = shap.sample(X_scaled, 50)

    # Load all models
    model_files = [f for f in os.listdir(".") if f.endswith(".pkl") and f != "scaler.pkl"]
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
        
        summary_list.append(ModelResultSummary(model=model_name, accuracy=acc))
    
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


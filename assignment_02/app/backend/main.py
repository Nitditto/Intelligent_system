import os
import uuid
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ddgs import DDGS
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseFeatures(BaseModel):
    City: str
    District: str
    Area: float
    Frontage: float
    Access_Road: float
    House_direction: str
    Balcony_direction: str
    Floors: float
    Bedrooms: float
    Bathrooms: float
    Legal_status: str
    Furniture_state: str

class ModelResultSummary(BaseModel):
    model: str
    price: float

class PredictionResponse(BaseModel):
    id: str
    results: List[ModelResultSummary]

class SHAPFeature(BaseModel):
    feature: str
    value: float

class DetailedModelResult(BaseModel):
    model: str
    prediction: float
    shap_values: List[SHAPFeature]

results_db = {}
models = {}
scaler = None
explainers = {}
feature_cols = []
locations = {} 
median_values = {}
X_train_scaled = None

def extract_district(address):
    if pd.isna(address) or address == 'Unknown': return 'Unknown'
    parts = [part.strip() for part in str(address).split(',')]
    return parts[-2] if len(parts) >= 2 else 'Unknown'

def extract_city(address):
    if pd.isna(address) or address == 'Unknown': return 'Unknown'
    parts = [part.strip() for part in str(address).split(',')]
    return parts[-1] if len(parts) >= 1 else 'Unknown'

def load_resources():
    global scaler, models, explainers, feature_cols, locations, median_values, X_train_scaled
    
    print("Loading dataset...")
    df = pd.read_csv("vietnam_housing_dataset.csv")
    df = df.dropna(subset=['Price'])

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Price')
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        median_val = df[col].median()
        median_values[col] = median_val
        df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    df['District'] = df['Address'].apply(extract_district)
    df['City'] = df['Address'].apply(extract_city)
    
    # Build locations dict
    for city in df['City'].unique():
        districts = df[df['City'] == city]['District'].unique().tolist()
        locations[city] = districts

    X = df.drop(['Price', 'Address', 'City'], axis=1, errors='ignore') 
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_cols = X_encoded.columns.tolist()

    # Rebuild scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_encoded)
    
    background_data = shap.sample(X_train_scaled, 50)

    # Load models
    model_files = [f for f in os.listdir(".") if f.endswith(".pkl") and f not in ["features.pkl", "scaler.pkl"]]
    for f in model_files:
        model_name = f.replace(".pkl", "")
        print(f"Loading {model_name}...")
        model = joblib.load(f)
        models[model_name] = model
        
        # Setup explainer (only for tree models to be fast)
        try:
            if model_name in ["random_forest", "xgboost"]:
                explainers[model_name] = shap.TreeExplainer(model)
        except Exception as e:
            print(f"Error setting up explainer for {model_name}: {e}")

@app.on_event("startup")
async def startup_event():
    load_resources()
    print("Backend initialized.")

@app.get("/api/locations")
async def get_locations():
    return locations

from search_service import (
    build_queries, search_ddg, search_bing, search_google, 
    dedupe_results, score_result, search_images_ddg
)

@app.get("/api/search")
async def search_houses(
    city: str, district: str, price: float, 
    area: float = None, floors: float = None, bedrooms: float = None
):
    import time
    print(f"--- START SEARCH ---")
    print(f"Input: {city}, {district}, price={price}, area={area}")
    
    # Generate targeted queries
    queries = build_queries(city, district, price, area, bedrooms)
    print("Queries:", queries)
    
    all_raw_results = []
    
    # Dispatch searches
    for q in queries[:2]:  # Limit to first 2 queries to avoid too many requests
        print(f"Executing DDGS for: {q}")
        all_raw_results.extend(search_ddg(q, max_results=5))
        time.sleep(0.5)
        
    print(f"Executing Bing for: {queries[0]}")
    all_raw_results.extend(search_bing(queries[0]))
    
    # Deduplicate before scoring
    unique_results = dedupe_results(all_raw_results)
    
    # Score results
    scored_results = []
    for r in unique_results:
        s = score_result(r, district, city, predicted_price=price, target_area=area)
        if s > 0:  # Only keep results with positive scores
            scored_results.append(r)
            
    # Sort by score descending
    scored_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_results = scored_results[:3]
    
    print(f"Total raw: {len(all_raw_results)}, Unique: {len(unique_results)}, Scored positive: {len(scored_results)}")
    
    if len(top_results) == 0:
        print("No organic results passed the soft scoring filter.")
        # DO NOT fallback to dummy data per Phase 6 & 8
        return []

    # Get images for top results
    print(f"Executing Image search for: nhà đẹp {district}")
    images = search_images_ddg(f"nhà đẹp {district}", max_results=len(top_results))
    
    final_output = []
    for i, r in enumerate(top_results):
        img_url = images[i] if i < len(images) else "https://via.placeholder.com/400x300?text=House+Image"
        final_output.append({
            "title": r.get('title'),
            "link": r.get('href'),
            "snippet": r.get('body'),
            "image": img_url,
            "source": r.get('source'),
            "score": r.get('score'),
            "matched_price": r.get('matched_price'),
            "matched_area": r.get('matched_area')
        })
        
    print(f"--- END SEARCH ---")
    return final_output

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(req: HouseFeatures):
    req_dict = req.dict()
    
    # Construct a DataFrame with 1 row using median values for missing/omitted features
    row_data = {}
    for col in feature_cols:
        row_data[col] = 0.0 # default for dummies
        
    # Set numeric values
    row_data['Area'] = req.Area if req.Area > 0 else median_values.get('Area', 0)
    row_data['Frontage'] = req.Frontage if req.Frontage > 0 else median_values.get('Frontage', 0)
    row_data['Access Road'] = req.Access_Road if req.Access_Road > 0 else median_values.get('Access Road', 0)
    row_data['Floors'] = req.Floors if req.Floors > 0 else median_values.get('Floors', 0)
    row_data['Bedrooms'] = req.Bedrooms if req.Bedrooms > 0 else median_values.get('Bedrooms', 0)
    row_data['Bathrooms'] = req.Bathrooms if req.Bathrooms > 0 else median_values.get('Bathrooms', 0)
    
    # Set dummy variables if they match
    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in row_data:
            row_data[col_name] = 1.0
            
    set_dummy("House direction", req.House_direction)
    set_dummy("Balcony direction", req.Balcony_direction)
    set_dummy("Legal status", req.Legal_status)
    set_dummy("Furniture state", req.Furniture_state)
    
    # Strip common prefixes from District to match the dataset's extracted names
    clean_district = req.District.replace("Quận ", "").replace("Huyện ", "").replace("Thị xã ", "").replace("Thành phố ", "")
    set_dummy("District", clean_district)
    
    input_df = pd.DataFrame([row_data])[feature_cols] # Ensure column order
    scaled_features = scaler.transform(input_df)
    
    result_id = str(uuid.uuid4())
    results_db[result_id] = {}
    
    summary_list = []
    
    for model_name, model in models.items():
        # Tree models and SVR/LR use scaled features in this dataset based on scratch_code.py
        # Actually scratch_code.py used scaled features for ALL models
        model_input = scaled_features
        
        pred_log = model.predict(model_input)[0]
        pred_price = np.expm1(pred_log)
            
        shap_features = []
        if model_name in explainers:
            explainer = explainers[model_name]
            shap_vals = explainer.shap_values(model_input)
            
            sv = shap_vals[0]
            # SHAP might be list of arrays or array
            if isinstance(sv, list): sv = sv[0]
            if len(sv.shape) > 1: sv = sv[0] # Handle shape (n_features, )
                
            for i, f_name in enumerate(feature_cols):
                if abs(sv[i]) > 0.001: # only send significant ones to save bandwidth
                    shap_features.append(SHAPFeature(feature=f_name, value=float(sv[i])))
            
            # Sort by absolute SHAP value
            shap_features.sort(key=lambda x: abs(x.value), reverse=True)
            shap_features = shap_features[:10] # Top 10
            
        results_db[result_id][model_name] = {
            "model": model_name,
            "prediction": float(pred_price),
            "shap_values": shap_features
        }
        
        summary_list.append(ModelResultSummary(model=model_name, price=float(pred_price)))
    
    return PredictionResponse(id=result_id, results=summary_list)

@app.get("/api/result/{result_id}", response_model=DetailedModelResult)
async def get_result(result_id: str, model: str):
    if result_id not in results_db:
        raise HTTPException(status_code=404, detail="Result ID not found")
    if model not in results_db[result_id]:
        raise HTTPException(status_code=404, detail="Model not found for this result")
        
    return results_db[result_id][model]


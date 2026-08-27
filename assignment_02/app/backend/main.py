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
    confidence: float
    r2: float
    mape: float

class PredictionResponse(BaseModel):
    id: str
    results: List[ModelResultSummary]

class SHAPFeature(BaseModel):
    feature: str
    value: float

class DetailedModelResult(BaseModel):
    model: str
    prediction: float
    confidence: float
    r2: float
    mape: float
    base_value: float
    shap_values: List[SHAPFeature]

results_db = {}
models = {}
scaler = None
explainers = {}
feature_cols = []
locations = {}
median_values = {}
X_train_scaled = None
dataset_stats = {}

def extract_district(address):
    if pd.isna(address) or address == 'Unknown': return 'Unknown'
    parts = [part.strip() for part in str(address).split(',')]
    return parts[-2] if len(parts) >= 2 else 'Unknown'

def extract_city(address):
    if pd.isna(address) or address == 'Unknown': return 'Unknown'
    parts = [part.strip() for part in str(address).split(',')]
    return parts[-1] if len(parts) >= 1 else 'Unknown'

def compute_dataset_stats(raw_df, clean_df):
    total_rows = len(raw_df)
    missing_pct = {}
    for col in ['Area', 'Frontage', 'Access Road', 'House direction', 'Balcony direction', 'Floors', 'Bedrooms', 'Bathrooms', 'Legal status', 'Furniture state']:
        missing_pct[col] = round(float(raw_df[col].isna().mean() * 100), 1)

    price = clean_df['Price']
    bucket_edges = [0, 2, 4, 6, 8, 10, 12, float('inf')]
    bucket_labels = ['<2', '2-4', '4-6', '6-8', '8-10', '10-12', '12+']
    price_buckets = pd.cut(price, bins=bucket_edges, labels=bucket_labels, right=False)
    price_histogram = [
        {"range": label, "count": int((price_buckets == label).sum())}
        for label in bucket_labels
    ]

    def avg_price_by(col, top_n=None):
        grouped = clean_df[clean_df[col] != 'Unknown'].groupby(col)['Price'].agg(['mean', 'count'])
        grouped = grouped.sort_values('count', ascending=False)
        if top_n:
            grouped = grouped.head(top_n)
        return [
            {"label": idx, "avgPrice": round(float(row['mean']), 2), "count": int(row['count'])}
            for idx, row in grouped.iterrows()
        ]

    top_districts = avg_price_by('District', top_n=10)
    by_legal_status = avg_price_by('Legal status')
    by_furniture_state = avg_price_by('Furniture state')

    sample_n = min(200, len(clean_df))
    sample = clean_df[['Area', 'Price']].sample(n=sample_n, random_state=42)
    area_price_sample = [
        {"area": round(float(r.Area), 1), "price": round(float(r.Price), 2)}
        for r in sample.itertuples()
    ]

    return {
        "totalListings": total_rows,
        "featureDimensions": len(feature_cols),
        "districtCount": clean_df['District'].nunique(),
        "priceRange": {"min": round(float(price.min()), 1), "max": round(float(price.max()), 1)},
        "missingPct": missing_pct,
        "priceHistogram": price_histogram,
        "topDistricts": top_districts,
        "byLegalStatus": by_legal_status,
        "byFurnitureState": by_furniture_state,
        "areaPriceSample": area_price_sample,
    }

def load_resources():
    global scaler, models, explainers, feature_cols, locations, median_values, X_train_scaled, dataset_stats

    print("Loading dataset...")
    raw_df = pd.read_csv("vietnam_housing_dataset.csv")
    raw_df = raw_df.dropna(subset=['Price'])
    df = raw_df.copy()

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

    dataset_stats = compute_dataset_stats(raw_df, df)

    # Load models
    model_files = [f for f in os.listdir(".") if f.endswith(".pkl") and f not in ["features.pkl", "scaler.pkl"]]
    for f in model_files:
        model_name = f.replace(".pkl", "")
        print(f"Loading {model_name}...")
        model = joblib.load(f)
        models[model_name] = model

@app.on_event("startup")
async def startup_event():
    load_resources()
    print("Backend initialized.")

@app.get("/api/locations")
async def get_locations():
    return locations

@app.get("/api/dataset-stats")
async def get_dataset_stats():
    return dataset_stats

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

    final_output = []
    for r in top_results:
        title = r.get('title')
        print(f"Executing Image search for: {title}")
        imgs = search_images_ddg(title, max_results=1)
        img_url = imgs[0] if imgs else "No Images"
        
        final_output.append({
            "title": title,
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
    results_db[result_id] = {
        "model_input": scaled_features,
        "results": {}
    }
    
    summary_list = []
    
    base_conf_map = {
        "xgboost": 0.94,
        "random_forest": 0.91,
        "svr_rbf": 0.85,
        "svr_linear": 0.78,
        "linear_regression": 0.75,
        "knn": 0.72
    }

    model_r2_scores = {
        "xgboost": 0.5954,
        "svr_rbf": 0.5249,
        "knn": 0.4108,
        "svr_linear": 0.3765,
        "linear_regression": 0.3735,
        "random_forest": 0.3727
    }
    
    model_mape_accuracies = {
        "xgboost": 78.91,
        "svr_rbf": 77.54,
        "knn": 74.37,
        "svr_linear": 73.94,
        "linear_regression": 74.02,
        "random_forest": 71.51
    }
    
    for model_name, model in models.items():
        model_input = scaled_features
        
        pred_log = model.predict(model_input)[0]
        pred_price = np.expm1(pred_log)
            
        # Calculate confidence
        base_conf = base_conf_map.get(model_name, 0.80)
        price_diff = abs(pred_price - 5.87)
        adjustment = 1.0 - min(0.3, price_diff / (3.0 * 2.21))
        confidence = base_conf * adjustment
        
        r2_val = model_r2_scores.get(model_name, 0.0)
        mape_val = model_mape_accuracies.get(model_name, 0.0)
            
        # Save to DB (without shap_values computed yet)
        results_db[result_id]["results"][model_name] = {
            "model": model_name,
            "prediction": float(pred_price),
            "confidence": float(confidence),
            "r2": float(r2_val),
            "mape": float(mape_val),
            "base_value": 0.0,  # Will be populated on demand
            "shap_values": None  # Lazy load
        }
        
        summary_list.append(ModelResultSummary(
            model=model_name, 
            price=float(pred_price),
            confidence=float(confidence),
            r2=float(r2_val),
            mape=float(mape_val)
        ))
    
    return PredictionResponse(id=result_id, results=summary_list)

@app.get("/api/result/{result_id}", response_model=DetailedModelResult)
async def get_result(result_id: str, model: str):
    if result_id not in results_db:
        raise HTTPException(status_code=404, detail="Result ID not found")
    if model not in results_db[result_id]["results"]:
        raise HTTPException(status_code=404, detail="Model not found for this result")
        
    model_data = results_db[result_id]["results"][model]
    
    # Lazy load SHAP values
    if model_data["shap_values"] is None:
        model_input = results_db[result_id]["model_input"]
        shap_features = []
        
        target_model = model
            
        if target_model not in explainers:
            try:
                m = models[target_model]
                if target_model in ["random_forest", "xgboost"]:
                    explainers[target_model] = shap.TreeExplainer(m)
                elif target_model in ["linear_regression", "svr_linear"]:
                    explainers[target_model] = shap.LinearExplainer(m, X_train_scaled)
                elif target_model in ["svr_rbf", "knn"]:
                    background_summary = shap.kmeans(X_train_scaled, 10)
                    explainers[target_model] = shap.KernelExplainer(m.predict, background_summary)
            except Exception as e:
                print(f"Error setting up explainer on demand for {target_model}: {e}")
                
        if target_model in explainers:
            try:
                explainer = explainers[target_model]
                
                if isinstance(explainer, shap.KernelExplainer):
                    shap_vals = explainer.shap_values(model_input, nsamples=100)
                else:
                    shap_vals = explainer.shap_values(model_input)
                
                if isinstance(shap_vals, list):
                    sv = shap_vals[0]
                else:
                    sv = shap_vals
                
                if len(sv.shape) > 1: 
                    sv = sv[0]
                
                # Fetch base log value from explainer
                base_log = explainer.expected_value
                if hasattr(base_log, "__iter__"):
                    base_log = float(base_log[0])
                else:
                    base_log = float(base_log)
                
                base_price = float(np.expm1(base_log))
                predicted_price = model_data["prediction"]
                predicted_log = float(np.log1p(predicted_price))
                
                # Calculate multiplier to scale SHAP values from log-space to price-space
                total_log_change = predicted_log - base_log
                price_change = predicted_price - base_price
                
                if abs(total_log_change) > 1e-6:
                    multiplier = price_change / total_log_change
                else:
                    multiplier = float(np.exp(base_log))
                
                # Don't drop small values yet to keep the sum exact
                for i, f_name in enumerate(feature_cols):
                    val = float(sv[i]) * multiplier
                    shap_features.append(SHAPFeature(feature=f_name, value=val))
                        
                shap_features.sort(key=lambda x: abs(x.value), reverse=True)
                
                # Keep top 10 features, group the rest
                top_features = shap_features[:10]
                others_sum = sum(x.value for x in shap_features[10:])
                
                # Only keep features > 0.001 visually
                top_features = [x for x in top_features if abs(x.value) > 0.001]
                
                if abs(others_sum) > 0.001:
                    top_features.append(SHAPFeature(feature="Các yếu tố khác", value=others_sum))
                    
                # Add base value as a feature to force plot starting from 0.0
                top_features.append(SHAPFeature(feature="Giá trị nền (Base Value)", value=base_price))
                
                shap_features = top_features
                model_data["base_value"] = 0.0
            except Exception as e:
                print(f"Error calculating lazy SHAP for {model}: {e}")
                shap_features = []
                model_data["base_value"] = 0.0
        else:
            shap_features = []
            model_data["base_value"] = 0.0
            
        model_data["shap_values"] = shap_features
        
    return model_data


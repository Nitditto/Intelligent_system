from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class ModelResultSummary(BaseModel):
    model: str
    accuracy: float
    prediction: int
    confidence: float

class PredictionResponse(BaseModel):
    id: str
    results: List[ModelResultSummary]

class SHAPFeature(BaseModel):
    feature: str
    value: float

class DetailedModelResult(BaseModel):
    model: str
    prediction: int
    confidence: float
    accuracy: float
    shap_values: List[SHAPFeature]

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_district(address):
    if pd.isna(address) or address == 'Unknown': return 'Unknown'
    parts = [part.strip() for part in str(address).split(',')]
    return parts[-2] if len(parts) >= 2 else 'Unknown'

df = pd.read_csv('vietnam_housing_dataset.csv')
df = df.dropna(subset=['Price'])

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Price')
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

df['District'] = df['Address'].apply(extract_district)

X = df.drop(['Price', 'Address'], axis=1, errors='ignore') 
y = df['Price']
X_encoded = pd.get_dummies(X, drop_first=True)

y_log = np.log1p(df['Price'])
X_train, X_test, y_train_log, y_test_log = train_test_split(X_encoded, y_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_log)

joblib.dump(rf, 'random_forest.pkl')
print("Saved random_forest.pkl")

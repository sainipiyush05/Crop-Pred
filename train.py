import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
import os

# Paths relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "agroware_crop_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].str.replace(r"\d+", "", regex=True).str.replace("_", "").str.replace(" ", "")
    
    # Features and Target
    X = df[['N','P','K','temperature','humidity','ph','rainfall']]
    y = df['label']
    
    # Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        objective="multi:softprob",
        num_class=len(le.classes_),
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y_encoded)
    
    # Saving
    print(f"Saving artifacts to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    
    print("Training complete! Artifacts generated.")

if __name__ == "__main__":
    train()

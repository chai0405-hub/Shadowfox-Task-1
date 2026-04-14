# =========================================
# Boston Housing Model Training Script
# =========================================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# 1. FILE PATHS
# =========================================
DATA_PATH = "HousingData.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
IMPUTER_PATH = "imputer.pkl"

# =========================================
# 2. LOAD DATA
# =========================================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df

# =========================================
# 3. PREPROCESSING
# =========================================
def preprocess(df):
    df = df.copy()

    # Target column
    target = "MEDV"

    if target not in df.columns:
        raise ValueError("Target column 'MEDV' not found in dataset")

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, scaler, imputer

# =========================================
# 4. TRAIN MODEL
# =========================================
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

# =========================================
# 5. SAVE FILES
# =========================================
def save_artifacts(model, scaler, imputer):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(IMPUTER_PATH, "wb") as f:
        pickle.dump(imputer, f)

# =========================================
# 6. MAIN FUNCTION
# =========================================
def main():
    print("🚀 Starting training process...")

    try:
        # Load data
        df = load_data(DATA_PATH)
        print("✅ Dataset loaded successfully")

        # Preprocess
        X, y, scaler, imputer = preprocess(df)
        print("✅ Data preprocessing completed")

        # Train model
        model, rmse, r2 = train(X, y)
        print("✅ Model training completed")

        # Save files
        save_artifacts(model, scaler, imputer)
        print("✅ Model and preprocessors saved")

        # Results
        print("\n📊 Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")

# =========================================
# 7. RUN SCRIPT
# =========================================
if __name__ == "__main__":
    main()
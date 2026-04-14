# ===============================
# Boston House Price Prediction - Streamlit App
# Fully Working End-to-End Pipeline
# ===============================

# 1. IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 2. FILE PATHS (IMPORTANT)
# ===============================
DATA_PATH = "HousingData.csv"   # <-- Keep your dataset with this exact name
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ===============================
# 3. LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ===============================
# 4. PREPROCESSING FUNCTION
# ===============================
def preprocess_data(df):
    df = df.copy()

    # Target column
    target = "MEDV"

    # Split features & target
    X = df.drop(columns=[target])
    y = df[target]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, scaler, imputer

# ===============================
# 5. TRAIN MODEL
# ===============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choose model (RandomForest works better)
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

# ===============================
# 6. SAVE MODEL
# ===============================
def save_model(model, scaler):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

# ===============================
# 7. LOAD MODEL
# ===============================
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        return model, scaler
    return None, None

# ===============================
# 8. STREAMLIT UI
# ===============================
st.set_page_config(page_title="Boston Housing Predictor", layout="wide")

st.title("🏠 Boston House Price Prediction App")
st.write("Predict house prices using Machine Learning")

# Load dataset
if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Please upload 'HousingData.csv'")
    st.stop()

# Sidebar actions
st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Choose Action", [
    "View Data",
    "Train Model",
    "Predict"
])

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ===============================
# 9. VIEW DATA
# ===============================
if option == "View Data":
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    st.subheader("📈 Dataset Info")
    st.write(df.describe())

# ===============================
# 10. TRAIN MODEL
# ===============================
elif option == "Train Model":
    st.subheader("⚡ Training Model...")

    X, y, scaler, imputer = preprocess_data(df)

    model, rmse, r2 = train_model(X, y)

    save_model(model, scaler)

    st.success("✅ Model trained and saved successfully!")

    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

# ===============================
# 11. PREDICTION
# ===============================
elif option == "Predict":
    st.subheader("🔮 Make Prediction")

    model, scaler = load_model()

    if model is None:
        st.warning("⚠️ Train the model first!")
        st.stop()

    # Input fields
    CRIM = st.number_input("Crime Rate", value=0.1)
    ZN = st.number_input("Residential Land Zone (%)", value=0.0)
    INDUS = st.number_input("Industrial Area (%)", value=10.0)
    CHAS = st.selectbox("Charles River (0 = No, 1 = Yes)", [0, 1])
    NOX = st.number_input("Nitric Oxide", value=0.5)
    RM = st.number_input("Number of Rooms", value=6.0)
    AGE = st.number_input("Age of Property", value=50.0)
    DIS = st.number_input("Distance to Employment Centers", value=4.0)
    RAD = st.number_input("Accessibility to Highways", value=5)
    TAX = st.number_input("Property Tax Rate", value=300.0)
    PTRATIO = st.number_input("Pupil-Teacher Ratio", value=15.0)
    B = st.number_input("Black Population Index", value=390.0)
    LSTAT = st.number_input("Lower Status Population (%)", value=12.0)

    if st.button("Predict Price"):
        input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.success(f"💰 Predicted House Price: ${prediction * 1000:.2f}")

# ===============================
# 12. FOOTER
# ===============================
st.markdown("---")
st.write("Made with ❤️ using Streamlit")

# ===============================
# END OF FILE
# ===============================
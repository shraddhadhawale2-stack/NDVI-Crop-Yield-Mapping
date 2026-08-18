import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="NDVI Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 NDVI-Based Crop Yield Prediction")
st.write(
    "Predict crop yield using NDVI and agricultural "
    "and environmental factors."
)

# Load dataset
df = pd.read_csv("meteorological_observations_regression_1000 (1).csv")

st.subheader("Enter Agricultural & Environmental Details")

col1, col2 = st.columns(2)

with col1:
    ndvi = st.number_input(
        "NDVI",
        min_value=0.0,
        max_value=1.0,
        value=0.72
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=650.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=24.0
    )

    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        value=0.38
    )

with col2:
    crop_type = st.selectbox(
        "Crop Type",
        sorted(df["Crop_Type"].dropna().unique())
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].dropna().unique())
    )

    season = st.selectbox(
        "Season",
        sorted(df["Season"].dropna().unique())
    )

    irrigation = st.selectbox(
        "Irrigation Type",
        sorted(df["Irrigation_Type"].dropna().unique())
    )

st.divider()

st.info(
    "Model prediction will be connected after the trained "
    "Random Forest model is added."
)

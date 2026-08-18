import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="NDVI Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# Load trained model
model = joblib.load("crop_yield_model.pkl")

st.title("🌱 NDVI-Based Crop Yield Prediction")
st.write(
    "Predict crop yield using NDVI and agricultural "
    "and environmental factors."
)

st.subheader("Enter Agricultural & Environmental Details")

col1, col2 = st.columns(2)

with col1:
    region = st.text_input("Region", "Maharashtra")
    crop_type = st.text_input("Crop Type", "Wheat")
    season = st.text_input("Season", "Rabi")
    irrigation_type = st.text_input("Irrigation Type", "Canal")

    ndvi = st.number_input(
        "NDVI",
        min_value=0.0,
        max_value=1.0,
        value=0.72
    )

    evi = st.number_input("EVI", value=0.63)
    savi = st.number_input("SAVI", value=0.68)
    lai = st.number_input("LAI", value=4.5)

with col2:
    rainfall = st.number_input("Rainfall (mm)", value=650.0)
    temperature = st.number_input("Temperature (°C)", value=24.0)
    soil_moisture = st.number_input("Soil Moisture", value=0.38)
    fertilizer = st.number_input("Fertilizer (kg/ha)", value=100.0)
    soil_ph = st.number_input("Soil pH", value=6.8)
    organic_carbon = st.number_input(
        "Organic Carbon (%)",
        value=1.1
    )
    canopy_temperature = st.number_input(
        "Canopy Temperature (°C)",
        value=23.0
    )
    growing_days = st.number_input(
        "Growing Days",
        value=120
    )

healthy_vegetation = st.checkbox(
    "Healthy Vegetation",
    value=True
)

irrigation_available = st.checkbox(
    "Irrigation Available",
    value=True
)

if st.button("🌾 Predict Crop Yield"):

    new_data = pd.DataFrame({
        "Region": [region],
        "Crop_Type": [crop_type],
        "Season": [season],
        "Irrigation_Type": [irrigation_type],
        "NDVI": [ndvi],
        "EVI": [evi],
        "SAVI": [savi],
        "LAI": [lai],
        "Rainfall_mm": [rainfall],
        "Temperature_C": [temperature],
        "Soil_Moisture": [soil_moisture],
        "Fertilizer_kg_ha": [fertilizer],
        "Soil_pH": [soil_ph],
        "Organic_Carbon_pct": [organic_carbon],
        "Canopy_Temperature_C": [canopy_temperature],
        "Growing_Days": [growing_days],
        "Healthy_Vegetation_Flag": [healthy_vegetation],
        "Irrigation_Available": [irrigation_available]
    })

    # Feature Engineering
    new_data["NDVI_Rainfall_Interaction"] = (
        new_data["NDVI"] * new_data["Rainfall_mm"]
    )

    new_data["Temperature_SoilMoisture_Interaction"] = (
        new_data["Temperature_C"] *
        new_data["Soil_Moisture"]
    )

    new_data["Moisture_Rainfall_Ratio"] = (
        new_data["Soil_Moisture"] /
        (new_data["Rainfall_mm"] + 1)
    )

    new_data["Vegetation_Index_Average"] = (
        new_data[["NDVI", "EVI", "SAVI"]].mean(axis=1)
    )

    prediction = model.predict(new_data)

    st.success(
        f"Predicted Crop Yield: "
        f"{prediction[0]:.2f} Tons per Hectare"
    )

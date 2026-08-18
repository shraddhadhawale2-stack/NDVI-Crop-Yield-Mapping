# NDVI-Based Crop Yield Prediction and Mapping Using Machine Learning

## Project Overview

This project uses Machine Learning to predict crop yield using **NDVI (Normalized Difference Vegetation Index)** along with agricultural and environmental factors.

## Objective

The main objective is to analyze vegetation health and environmental conditions and predict crop yield using a **Random Forest Regression** model.

## Dataset

The dataset contains agricultural and environmental features such as:

- NDVI
- EVI
- SAVI
- LAI
- Rainfall
- Temperature
- Soil Moisture
- Soil pH
- Fertilizer Usage
- Crop Type
- Region
- Season
- Irrigation

### Target Variable

`Yield_tons_per_hectare`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Model

**Random Forest Regression** is used to predict crop yield.

## Project Workflow

1. Data Collection
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Engineering
5. Model Building
6. Model Evaluation
7. Crop Yield Prediction
8. Streamlit Deployment

## Evaluation Metrics

- MAE
- MSE
- RMSE
- R² Score

## Application

The trained model is integrated with a **Streamlit web application** to provide crop-yield predictions based on user-provided agricultural and environmental inputs.

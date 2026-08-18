import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Load dataset
df = pd.read_csv("meteorological_observations_regression_1000 (1).csv")


# Feature Engineering
df["NDVI_Rainfall_Interaction"] = (
    df["NDVI"] * df["Rainfall_mm"]
)

df["Temperature_SoilMoisture_Interaction"] = (
    df["Temperature_C"] * df["Soil_Moisture"]
)

df["Moisture_Rainfall_Ratio"] = (
    df["Soil_Moisture"] / (df["Rainfall_mm"] + 1)
)

df["Vegetation_Index_Average"] = (
    df[["NDVI", "EVI", "SAVI"]].mean(axis=1)
)


# Target
target = "Yield_tons_per_hectare"

df = df.dropna(subset=[target])

X = df.drop(columns=["Record_ID", target])
y = df[target]


# Identify feature types
categorical_cols = X.select_dtypes(
    include=["object"]
).columns

numerical_cols = X.select_dtypes(
    include=[np.number, "bool"]
).columns


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_cols
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        )
    ]
)


# Random Forest Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1
        )
    )
])


# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)


# Save trained model
joblib.dump(model, "crop_yield_model.pkl")

print("Model trained successfully!")
print("Model saved as crop_yield_model.pkl")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import datetime

# Load trained model
model = joblib.load("model/model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Stroke Prediction API")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Define input data schema
class StrokeInput(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    ever_married: int
    work_type: int
    residence_type: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int

# Function to log predictions
def log_prediction(input_data: StrokeInput, prediction: int):
    log_df = pd.DataFrame([{
        "timestamp": datetime.datetime.now(),
        **input_data.dict(),
        "prediction": prediction
    }])
    log_path = "logs/prediction_logs.csv"
    log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

# Prediction endpoint
@app.post("/predict")
def predict(data: StrokeInput):
    input_array = np.array([[ 
        data.gender, data.age, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    ]])
    prediction = model.predict(input_array)[0]

    # Log the prediction
    log_prediction(data, int(prediction))

    return {"stroke_prediction": int(prediction)}

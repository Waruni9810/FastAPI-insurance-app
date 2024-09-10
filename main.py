from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Define the FastAPI app
app = FastAPI(title="Insurance Premium Prediction API", version="1.0")

# Load the trained model
model_path = 'insurance_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")
model = joblib.load(model_path)

# Define the input data model using Pydantic
class InsuranceData(BaseModel):
    Age: int
    Diabetes: int
    BloodPressureProblems: int
    AnyTransplants: int
    AnyChronicDiseases: int
    Height: float
    Weight: float
    KnownAllergies: int
    HistoryOfCancerInFamily: int
    NumberOfMajorSurgeries: int

# Prediction endpoint
@app.post("/predict/", response_model=dict)
async def predict(data: InsuranceData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Calculate BMI and add it to the input data
        input_data['BMI'] = input_data['Weight'] / (input_data['Height'] / 100) ** 2
        
        # Drop Height and Weight as they are not needed after BMI calculation
        #input_data = input_data.drop(columns=['Height', 'Weight'])
       
        # Make prediction
        prediction = model.predict(input_data)
        prediction_lkr = prediction[0] * 4.35  # Convert to LKR
        
        return {"predicted_premium_lkr": round(prediction_lkr, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Insurance Premium Prediction API is running"}

from fastapi import FastAPI
import joblib
import pandas as pd


# Creating FastAPI application
app = FastAPI(
    title="Smart Energy Management API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Welcome to my first FastAPI application!"}


@app.post("/predict-demand")
def predict_demand(data: DemandInput):

    # Convert input into the exact format expected by the model
    input_df = pd.DataFrame([{
        "Household_Size": data.household_size,
        "Avg_Temperature_C_x": data.avg_temperature,
        "Has_AC": data.has_ac,
        "Avg_Humidity_Pct": data.avg_humidity,
        "Day": data.day,
        "Month": data.month,
        "Is Weekend": data.is_weekend,
        "Is_Holiday": data.is_holiday
    }])

    # Make prediction
    prediction = xgb_demand_model.predict(input_df)

    # Return JSON response
    return {
        "predicted_demand_kwh": round(float(prediction[0]), 2)
    }
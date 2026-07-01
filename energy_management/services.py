from .model_loader import xgb_demand_model, rain_prediction_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
def predict_demand(data):

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

    prediction = xgb_demand_model.predict(input_df)

    return float(prediction[0])

def predict_rain(data):
    input_df = pd.DataFrame([{
        "temperature_2m":data.temperature_2m,
        "relative_humidity_2m":data.relative_humidity_2m,
        "cloud_cover":data.cloud_cover,
        "pressure_msl":data.pressure_msl,
        "wind_speed_10m":data.wind_speed_10m,
        "month":data.month,
        "day":data.day,
        "Season":data.Season
    }])
    project_dir = Path()
    scalar_path = project_dir/"models/Standard_scalar_rainfall.pkl"
    scalar = joblib.load(scalar_path)
    scaled_Data = scalar.transform(input_df)
    rain_prediction = rain_prediction_model.predict(scaled_Data)
    probabilities = rain_prediction_model.predict_proba(scaled_Data)[0]
  
  # Grab confidence based on what class was predicted
    predicted_class = int(rain_prediction[0])
    confidence = probabilities[predicted_class]
    # print(f"Predicted Rain: {rain_prediction[0]}, Confidence: {confidence:.2f}")


    return float(predicted_class), float(confidence)
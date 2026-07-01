import joblib
from pathlib import Path
from pydantic import BaseModel

project_dir=Path()
model_dir= project_dir/'models'

#loading model
xgb_demand_model = joblib.load(model_dir/"xgb_demand_forecast_model.pkl")

rain_prediction_model = joblib.load(model_dir/"rf_rain_prediction_model.pkl")
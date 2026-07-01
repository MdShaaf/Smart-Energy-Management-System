from fastapi import FastAPI
import pandas as pd
import joblib
from .model_loader import xgb_demand_model,rain_prediction_model
from routes.energy_demand import router

from .schemas import DemandInput, RainfallInput

app = FastAPI()
app.include_router(router)

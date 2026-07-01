from fastapi import APIRouter
from energy_management.schemas import DemandInput, RainfallInput
from energy_management.services import predict_demand, predict_rain

router = APIRouter(
    prefix="/demand",
    tags=["Demand Prediction"]
)


@router.post("/predict")
def predict(data: DemandInput):

    prediction = predict_demand(data)

    return {
        "predicted_demand": prediction

    }

@router.post("/rain")
def rain_endpoint(data:RainfallInput):
    prediction = predict_rain(data)
    print(f"--- RECEIVED DATA FROM UI: {data.model_dump()} ---")
    return {
        "predicted_rain": prediction[0], "confidence": prediction[1]

    }
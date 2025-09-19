from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from predict import predict_data
import json

app = FastAPI(
    title="Wine Classifier API",
    description="Predict wine type (0, 1, 2) using 13 chemical features." \
    "And get model metrics.",
    version="1.0.0"
)

# Input model (13 features)
class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

# Output model
class WineResponse(BaseModel):
    response: int

# Health check
@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


# Prediction
@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315,
            wine_features.proline
        ]]
        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Metrics
@app.get("/metrics")
async def get_metrics():
    try:
        with open("../model/metrics.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load metrics: {e}")

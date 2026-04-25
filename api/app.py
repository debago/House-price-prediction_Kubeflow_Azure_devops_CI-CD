from fastapi import FastAPI, HTTPException
import pandas as pd
import os
import logging

from api.model_loader import get_model
from api.schemas import HouseInput
from src.config_loader import get_pipeline_config



# ---------------------------
# Load config
# ---------------------------

config = get_pipeline_config("api")


# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(
    title=config["api"]["name"],
    version=config["api"]["version"],
    description="House Price Prediction API"
)


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "env": config["env"],
        "service": config.get("api", {}).get("name", "ml-api"),
        "version": config.get("api", {}).get("version"),
        "model_uri": config.get("mlflow", {}).get("model_uri")
    }


# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(data: HouseInput):
    try:
        logger.info("📥 Received prediction request")

        # Convert request to DataFrame
        df = pd.DataFrame([data.dict()])

        # Load model (with refresh logic)
        model = get_model()

        # Perform inference
        preds = model.predict(df)

        logger.info("✅ Prediction successful")

        return {
            "prediction": preds.tolist(),
            "model_uri": config["mlflow"]["model_uri"]
        }

    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
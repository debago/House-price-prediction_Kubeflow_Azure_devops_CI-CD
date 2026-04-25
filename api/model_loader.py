import os
import time
import logging
import mlflow

from src.config_loader import get_pipeline_config

logger = logging.getLogger(__name__)

config = get_pipeline_config("api")

MODEL = None
LAST_LOADED = 0
REFRESH_INTERVAL = config["api"].get("refresh_interval", 60)  # default to 60 seconds if not set


def get_model_uri():
    # Env variable can override model URI if needed
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return model_uri

    return config["mlflow"]["model_uri"]


def get_model():
    global MODEL, LAST_LOADED

    current_time = time.time()

    if MODEL is None or (current_time - LAST_LOADED) > REFRESH_INTERVAL:
        model_uri = get_model_uri()
        logger.info(f"Loading model from: {model_uri}")

        MODEL = mlflow.pyfunc.load_model(model_uri)
        LAST_LOADED = current_time
    else:
        logger.info("Using cached model")

    return MODEL
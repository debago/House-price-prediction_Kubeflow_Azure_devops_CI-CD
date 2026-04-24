import os
import time
import mlflow
from src.utils import load_params

MODEL = None
LAST_LOADED = 0
REFRESH_INTERVAL = 60  # seconds


def get_model_uri():

    #ENV variable can also be used to override the model URI if needed
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return model_uri

    # Fallback to params.yaml
    params = load_params()
    return params["mlflow"]["model_uri"]

def get_model(model_path: str):
    global MODEL, LAST_LOADED

    current_time = time.time()
    if MODEL is None or (current_time - LAST_LOADED) > REFRESH_INTERVAL:
        model_uri = get_model_uri()
        print(f"Loading model from: {model_uri}")

        MODEL = mlflow.pyfunc.load_model(model_uri)
        LAST_LOADED = current_time
    else:
        print("Using cached model...")

    return MODEL
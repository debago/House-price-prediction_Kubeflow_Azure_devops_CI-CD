# src/config_loader.py

import os
from src.utils import load_config

def get_pipeline_config():
    env = os.getenv("ENV", "dev")
    return load_config("pipeline", env)
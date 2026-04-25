# src/config_loader.py

import os
from src.utils import load_config

def get_pipeline_config(component="pipeline"):
    ENV = os.getenv("ENV", "dev")

    config = load_config(component, ENV)

    # ✅ inject dynamically
    config["env"] = ENV

    return config
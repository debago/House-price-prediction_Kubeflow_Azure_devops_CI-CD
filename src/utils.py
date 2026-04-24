import os
import yaml
from pathlib import Path
import pandas as pd


# Load_params is now load_config, which handles multiple YAML files and merging
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_dicts(base, override):
    """Recursively merge dictionaries"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(component="pipeline", env="dev"):
    base_cfg = load_yaml(CONFIG_DIR / "base.yaml")
    comp_cfg = load_yaml(CONFIG_DIR / f"{component}.yaml")
    env_cfg = load_yaml(CONFIG_DIR / f"{env}.yaml")

    config = merge_dicts(base_cfg, comp_cfg)
    config = merge_dicts(config, env_cfg)

    return config

#------------------- Working with single params.yaml file -------------------#
# def load_params(params_path: str = "params.yaml") -> dict:
#     """
#     Load YAML configuration file.
#     """
#     with open(params_path, "r") as f:
#         return yaml.safe_load(f)

#----------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """
    Ensure parent directory exists for a file path
    or ensure directory exists if directory path is passed.
    """
    # Check if path looks like a file (has an extension) or is a directory
    # If it has a common file extension, treat it as a file path
    if "." in os.path.basename(path):
        # It's likely a file path, create parent directory
        dir_path = os.path.dirname(path)
    else:
        # It's a directory path, create it directly
        dir_path = path

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV.
    """
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"✅ Data saved at: {path}")
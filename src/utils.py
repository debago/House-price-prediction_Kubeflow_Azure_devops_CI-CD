import os
import yaml
import pandas as pd


def load_params(params_path: str = "params.yaml") -> dict:
    """
    Load YAML configuration file.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


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
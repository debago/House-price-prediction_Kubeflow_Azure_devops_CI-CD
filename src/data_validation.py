import pandas as pd
from src.utils import load_params, load_data

def validate_data():
    params = load_params()
    raw_data_path = params["paths"]["raw_data"]
    target_column = params["data"]["target_column"]

    df = load_data(raw_data_path)

    if df.empty:
        raise ValueError("Dataset is empty")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing")

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric")

    print("✅ Data validation passed")
    return True
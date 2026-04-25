import pandas as pd
from src.config_loader import get_pipeline_config
from src.utils import load_config, load_data
from src.config_loader import get_pipeline_config
import logging


logger = logging.getLogger(__name__)


def validate_data():
    logger.info("Validation Started")
    # params = load_params()
    config = get_pipeline_config("pipeline")
    raw_data_path = config["paths"]["raw_data"]
    target_column = config["data"]["target_column"]

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

    logger.info("✅ Data validation passed")

    return True
from sklearn.datasets import fetch_california_housing
from src.utils import load_config, ensure_dir, save_data
from src.config_loader import get_pipeline_config
import logging

logger = logging.getLogger(__name__)

def ingest_data():
    logger.info("Loading raw dataset")
    # params = load_params()
    config = get_pipeline_config()
    raw_data_path = config["paths"]["raw_data"]

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    save_data(df, raw_data_path)

    print(f"Shape: {df.shape}")
    logger.info(f"Data shape: {df.shape}")
    return raw_data_path
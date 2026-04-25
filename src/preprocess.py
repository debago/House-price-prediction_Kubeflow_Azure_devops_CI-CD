import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_config, load_data, save_data
from src.config_loader import get_pipeline_config
import logging


logger = logging.getLogger(__name__)


def preprocess():
    logger.info("Starting preprocessing")
    # 1️⃣ Load config
    config = get_pipeline_config("pipeline")

    raw_path = config["paths"]["raw_data"]
    train_path = config["paths"]["train_data"]
    test_path = config["paths"]["test_data"]
    target_column = config["data"]["target_column"]

    # 2️⃣ Load raw data
    df = load_data(raw_path)

    # 3️⃣ Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 4️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # 5️⃣ Recreate DataFrames
    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    # 6️⃣ Save processed data
    save_data(train_df, train_path)
    save_data(test_df, test_path)

    logger.info("Preprocessing completed")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    return train_path, test_path


if __name__ == "__main__":
    preprocess()
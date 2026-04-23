from sklearn.datasets import fetch_california_housing
from src.utils import load_params, ensure_dir, save_data

def ingest_data():
    params = load_params()
    raw_data_path = params["paths"]["raw_data"]

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    save_data(df, raw_data_path)

    print(f"Shape: {df.shape}")
    return raw_data_path
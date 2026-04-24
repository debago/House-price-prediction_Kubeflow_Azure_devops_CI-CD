import os
import pickle
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from src.utils import load_config, load_data
from src.config_loader import get_pipeline_config

def evaluate():
    # 1️⃣ Load config
    config = get_pipeline_config()

    test_path = config["paths"]["test_data"]
    model_dir = config["paths"]["model_dir"]
    target_column = config["data"]["target_column"]

    model_path = os.path.join(model_dir, "model.pkl")

    # 2️⃣ Load test data
    df = load_data(test_path)
    X_test = df.drop(target_column, axis=1)
    y_test = df[target_column]

    # 3️⃣ Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 4️⃣ Predict
    preds = model.predict(X_test)

    # 5️⃣ Metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)


    print("📊 Metrics:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")


    # 6️⃣ Log metrics
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="rf-evaluation"):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        print("✅ Metrics logged to MLflow")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2
    }


if __name__ == "__main__":
    evaluate()
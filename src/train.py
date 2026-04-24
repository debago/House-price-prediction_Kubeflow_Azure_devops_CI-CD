import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature
from src.utils import load_config, load_data, ensure_dir
from src.config_loader import get_pipeline_config
import logging

logger = logging.getLogger(__name__)
logger.info("Training started")



def train():

    logger.info("Training started")
    # 1️⃣ Load config
    config = get_pipeline_config()

    train_path = config["paths"]["train_data"]
    model_dir = config["paths"]["model_dir"]
    target_column = config["data"]["target_column"]

    # 2️⃣ Load training data
    df = load_data(train_path)
    X_train = df.drop(target_column, axis=1)
    y_train = df[target_column]

    # 3️⃣ Initialize model
    model = RandomForestRegressor(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"]
    )

    # 4️⃣ MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="rf-training"):

        # 5️⃣ Train
        model.fit(X_train, y_train)

        # 5️⃣ Infer signature
        samples_input = X_train.head(5)
        samples_output = model.predict(samples_input)
        signature = infer_signature(samples_input, samples_output)

        # 6️⃣ Save model locally
        ensure_dir(model_dir)
        model_path = os.path.join(model_dir, "model.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"✅ Model saved at: {model_path}")

        # 7️⃣ Log params
        mlflow.log_param("n_estimators", config["model"]["n_estimators"])
        mlflow.log_param("max_depth", config["model"]["max_depth"])

        # 8️⃣ Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["mlflow"]["registered_model_name"],
            signature=signature,
            input_example=samples_input

        )

        logger.info("✅ Model logged to MLflow")
        logger.info("Model Training completed")

    return model_path


if __name__ == "__main__":
    train()
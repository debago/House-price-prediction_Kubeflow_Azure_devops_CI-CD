import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature
from src.utils import load_params, load_data, ensure_dir
import logging

logger = logging.getLogger(__name__)
logger.info("Training started")



def train():

    logger.info("Training started")
    # 1️⃣ Load config
    params = load_params()

    train_path = params["paths"]["train_data"]
    model_dir = params["paths"]["model_dir"]
    target_column = params["data"]["target_column"]

    # 2️⃣ Load training data
    df = load_data(train_path)
    X_train = df.drop(target_column, axis=1)
    y_train = df[target_column]

    # 3️⃣ Initialize model
    model = RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["model"]["random_state"]
    )

    # 4️⃣ MLflow setup
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

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
        mlflow.log_param("n_estimators", params["model"]["n_estimators"])
        mlflow.log_param("max_depth", params["model"]["max_depth"])

        # 8️⃣ Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=params["mlflow"]["registered_model_name"],
            signature=signature,
            input_example=samples_input

        )

        logger.info("✅ Model logged to MLflow")
        logger.info("Model Training completed")

    return model_path


if __name__ == "__main__":
    train()
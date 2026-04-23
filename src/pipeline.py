from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate


def run_pipeline():
    print("Starting pipeline...")

    # 1️⃣ Data ingestion
    ingest_data()
    print("Data ingestion completed")

    # 2️⃣ Data validation
    validate_data()
    print("Data validation completed")

    # 3️⃣ Preprocessing
    preprocess()
    print("Preprocessing completed")

    # 4️⃣ Training
    train()
    print("Training completed")

    # 5️⃣ Evaluation
    metrics = evaluate()
    print("Evaluation completed")
    print("Final metrics:", metrics)

    return metrics


if __name__ == "__main__":
    run_pipeline()
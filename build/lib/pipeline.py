from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.preprocess import preprocess
from src.train import train
from src.evaluate import evaluate
from src.logging_config import setup_logging

logger = setup_logging()


def run_pipeline():
    logger.info("🚀 Pipeline started")

    try:
        logger.info("Step 1: Data ingestion")
        ingest_data()

        logger.info("Step 2: Data validation")
        validate_data()

        logger.info("Step 3: Preprocessing")
        preprocess()

        logger.info("Step 4: Training")
        train()

        logger.info("Step 5: Evaluation")
        metrics = evaluate()

        logger.info(f"Final metrics: {metrics}")
        logger.info("✅ Pipeline completed")

        return metrics

    except Exception:
        logger.error("❌ Pipeline failed", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()
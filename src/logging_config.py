import logging
import uuid
import os

RUN_ID = str(uuid.uuid4())

def setup_logging():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )

    class ContextFilter(logging.Filter):
        def filter(self, record):
            record.run_id = RUN_ID
            return True

    logger = logging.getLogger()
    logger.addFilter(ContextFilter())

    return logger
import logging
from datetime import datetime
from retriever import Retriever
import os

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/ingest_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting document ingestion")
    try:
        agent = Retriever()
        logger.info("Reading data.txt file")
        with open("data.txt", "r", encoding="utf-8") as f:
            txt = f.read()
        logger.debug(f"Read {len(txt)} characters from data.txt")
        
        n = agent.add_texts([txt], metadatas=[{"source": "data.txt"}])
        logger.info(f"Successfully added {n} documents")
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise

import os, requests, logging, time
from datetime import datetime
from dotenv import load_dotenv
import sys

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/ollama_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def generate(model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        end_time = time.time()

        logger.info(f"OLLAMA generate call successful for model '{model}' in {end_time - start_time:.2f}s")
        return data.get("text", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"OLLAMA generate call failed for model '{model}': {str(e)}", exc_info=True)
        return ""
print(f"OLLAMA_HOST set to '{OLLAMA_HOST}'")
print("OLLAMA client module loaded.")
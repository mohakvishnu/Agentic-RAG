import os, requests, logging, time
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

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


def generate(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stream: bool = False) -> str:
    """
    Non-stream by default. Reads 'response' from Ollama JSON.
    If stream=True, aggregates 'response' chunks.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    headers = {"Content-Type": "application/json"}

    try:
        start = time.time()
        if not stream:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            out = (data.get("response") or "").strip()
            logger.info(f"Ollama generate OK (model={model}, {time.time()-start:.2f}s, {len(out)} chars)")
            return out
        else:
            # streaming aggregation
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                buff = []
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = line.decode("utf-8")
                        obj = requests.utils.json.loads(chunk)
                        piece = obj.get("response") or ""
                        if piece:
                            buff.append(piece)
                    except Exception:
                        continue
                out = "".join(buff).strip()
                logger.info(f"Ollama stream OK (model={model}, {time.time()-start:.2f}s, {len(out)} chars)")
                return out
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama generate FAILED (model={model}): {e}", exc_info=True)
        return ""

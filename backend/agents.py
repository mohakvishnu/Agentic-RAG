import os, sqlite3, json, re, psutil, time, sys
import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from ollama_client import generate   # your local wrapper over Ollama
import uuid

# ===================== Logging =====================
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/agents_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== Config / Env =====================
load_dotenv()

SQLITE_PATH = os.getenv("SQLITE_PATH", "./rag_memory.db")

# Model registry (can be overridden by env)
MODEL = {
    "rag":      os.getenv("LLM_RAG",      "mistral:7b-instruct"),
    "code":     os.getenv("LLM_CODE",     "Qwen2.5-Coder:latest"),
    "sql":      os.getenv("LLM_SQL",      "sqlcoder:latest"),
    "fallback": os.getenv("LLM_FALLBACK", "llama3:latest"),
}

# ===================== System info (startup) =====================
process = psutil.Process()
logger.info("=== System Information ===")
logger.info(f"Python Version: {sys.version}")
logger.info(f"CPU Count: {psutil.cpu_count()}")
logger.info(f"Available Memory: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
logger.info(f"Process ID: {process.pid}")
logger.info(f"Models => rag: {MODEL['rag']} | code: {MODEL['code']} | sql: {MODEL['sql']} | fallback: {MODEL['fallback']}")
logger.info("========================")

# ===================== Helpers =====================
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"[{request_id}] Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)
            execution_time = end_time - start_time
            memory_diff = end_memory - start_memory
            logger.info(f"[{request_id}] Completed {func.__name__}")
            logger.debug(f"[{request_id}] Execution time: {execution_time:.2f}s")
            logger.debug(f"[{request_id}] Memory change: {memory_diff:.2f}MB")
            return result
        except Exception as e:
            logger.error(f"[{request_id}] Failed {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def _ensure_fenced_code(text: str, default_lang: str = "python") -> str:
    """If model forgot fences, wrap whole output in a fenced code block."""
    if "```" in (text or ""):
        return text
    clean = (text or "").strip() or "# (no content returned)"
    return f"```{default_lang}\n{clean}\n```"

def _first_lang_hint(query: str) -> Optional[str]:
    q = (query or "").lower()
    if "python" in q or " py" in q or q.endswith(".py"):
        return "python"
    if "typescript" in q or " ts" in q or q.endswith(".ts"):
        return "typescript"
    if "javascript" in q or " js" in q or q.endswith(".js"):
        return "javascript"
    if "java" in q: return "java"
    if "c++" in q or "cpp" in q: return "cpp"
    if "c#" in q: return "csharp"
    if "go " in q or q.endswith(".go"): return "go"
    if "rust" in q: return "rust"
    return None

# ===================== Memory =====================
class Memory:
    def __init__(self, path: str = SQLITE_PATH):
        logger.info(f"Initializing Memory with database path: {path}")
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure()
        logger.info("Memory initialization successful")

    def _ensure(self):
        logger.debug("Ensuring memories table exists")
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS memories(
            session_id TEXT, user TEXT, assistant TEXT,
            citations TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()
        logger.info("Memories table verified/created successfully")

    @log_execution_time
    def save(self, session_id: str, user: str, assistant: str, citations: List[Dict[str, Any]]):
        logger.info(f"Saving memory for session: {session_id}")
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memories(session_id, user, assistant, citations) VALUES(?,?,?,?)",
                    (session_id, user, assistant, json.dumps(citations)))
        self.conn.commit()
        logger.info("Memory saved successfully")

# ===================== Routing =====================
class Route(Enum):
    RAG = "RAG"
    CODE = "CODE"
    SQL  = "SQL"

class Policy:
    @staticmethod
    @log_execution_time
    def decide(query: str) -> Route:
        logger.info(f"Deciding route for query: {query}")
        q = (query or "").lower()
        route = Route.RAG
        if any(k in q for k in ["select ", "sql ", "schema", "table", "join ", "group by", "where "]):
            route = Route.SQL
        elif any(k in q for k in ["code", "function", "class", "bug", "refactor", "python", "typescript", " js ", "javascript"]):
            route = Route.CODE
        logger.info(f"Selected route: {route.value}")
        return route

# ===================== Answer Functions =====================
@log_execution_time
def answer_rag(query: str, context: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] RAG Request - Query length: {len(query)}")
    logger.debug(f"[{req_id}] Context chars: {len(context)}")
    prompt = f"""You are a helpful assistant. Use CONTEXT to answer QUESTION concisely.
    If the answer is not in the context, you can check with the configured models and get the asked questions response especially in python code.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Instructions:
    - Prefer tight bullet points.
    - If you include code, return it in fenced blocks like ```language.
    """
    start_llm = time.time()
    response = generate(MODEL["rag"], prompt)
    llm_time = time.time() - start_llm
    logger.info(f"[{req_id}] LLM took {llm_time:.2f}s")
    logger.debug(f"[{req_id}] Response tokens: {len((response or '').split())}")
    return response or ""


@log_execution_time
def answer_code(query: str, context: str) -> str:
    """Always returns Markdown with at least one fenced code block."""
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] Code Request - Query length: {len(query)}")
    lang_hint = _first_lang_hint(query) or "python"
    prompt = f"""You are a senior engineer. Based on REQUEST and any useful CONTEXT,
    produce a minimal plan and a complete working code snippet. If assumptions are needed, list them.
    Return STRICTLY in this format:

    - Plan: (3–5 bullets, one line each)
    - Code:
    ```{lang_hint}
    # code here (fully runnable)
    CONTEXT:
    {context}

    REQUEST:
    {query}
    """
    start_llm = time.time()
    response = generate(MODEL["code"], prompt)
    llm_time = time.time() - start_llm
    logger.info(f"[{req_id}] LLM took {llm_time:.2f}s")
    logger.debug(f"[{req_id}] Response contains {(response or '').count('```')} code blocks")
    # Enforce fenced block if missing
    return _ensure_fenced_code(response or "", default_lang=lang_hint)

@log_execution_time
def answer_sql(query: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] SQL Request - Query length: {len(query)}")
    schema_hint = """
    -- Tables available:
    -- memories(session_id TEXT, user TEXT, assistant TEXT, citations TEXT, ts DATETIME)
    -- docs(id TEXT PRIMARY KEY, text TEXT, source TEXT)
    """
    prompt = f"""You are SQLCoder. Produce a single SQLite query in a fenced sql block
    that satisfies the USER REQUEST. Use only existing columns. Then add a 1–2 line explanation.

    DB SCHEMA:
    {schema_hint}

    USER REQUEST:
    {query}
    """
    start_llm = time.time()
    llm = generate(MODEL["sql"], prompt)
    llm_time = time.time() - start_llm
    logger.info(f"[{req_id}] SQL generation took {llm_time:.2f}s")
    if "" not in (llm or ""): 
        llm = f"sql\n{(llm or '').strip()}\n```"
        sql = _extract_sql(llm or "")
        if not sql:
            logger.warning(f"[{req_id}] No SQL extracted from response")
        return llm

    logger.debug(f"[{req_id}] Executing SQL: {sql}")
    start_exec = time.time()
    try:
        conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        payload = {"columns": cols, "rows": rows}
        exec_time = time.time() - start_exec
        logger.info(f"[{req_id}] SQL execution took {exec_time:.2f}s, rows: {len(rows)}")
        return llm + "\n\n-- Execution Result --\n" + json.dumps(payload, indent=2, default=str)
    except Exception as e:
        logger.error(f"[{req_id}] SQL execution failed: {str(e)}", exc_info=True)
        return llm + f"\n\n[Execution Error] {e}"

def _extract_sql(text: str) -> Optional[str]:
    logger.debug("Extracting SQL query from text")
    m = re.search(r"sql\s*([\s\S]*?)", text, re.IGNORECASE)
    result = m.group(1).strip().rstrip(";") if m else None
    logger.debug(f"Extracted SQL: {result}")
    return result
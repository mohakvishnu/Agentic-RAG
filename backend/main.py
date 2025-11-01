import os, io, logging, time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import Retriever
from agents import Memory, Policy, Route, answer_rag, answer_code, answer_sql
from pypdf import PdfReader
from docx import Document
import sqlite3
from fastapi import Query


# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/api_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]

app = FastAPI(title="Agentic RAG (Lite) — Hybrid + RRF")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
memory = Memory()
logger.info("Initialized Retriever and Memory services")
logger.info("=== API Server Starting ===")
logger.info(f"CORS Origins: {CORS_ORIGINS}")

class AskRequest(BaseModel):
    query: str
    session_id: str = "default"

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    route: str

@app.get("/health")
def health():
    logger.debug("Health check requested")
    return {"status": "ok"}

# ---------- Text ingestion (quick) ----------
@app.post("/ingest")
def ingest(text: str = Body(..., embed=True)):
    logger.info("Processing text ingestion request")
    logger.debug(f"Text length: {len(text)} chars")
    try:
        n = retriever.add_texts([text], metadatas=[{"source": "api"}])
        logger.info(f"Successfully ingested {n} documents")
        return {"added": n}
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise

# ---------- File upload & ingestion ----------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    logger.info(f"Processing upload request for {len(files)} files")
    texts, metas = [], []
    
    for f in files:
        start_time = time.time()
        name = f.filename or "file"
        logger.info(f"Processing file: {name}")
        
        try:
            content = await f.read()
            ext = (name.split(".")[-1] or "").lower()
            logger.debug(f"File type: {ext}, size: {len(content)} bytes")

            if ext in ("txt", "md", "csv", "log"):
                text = content.decode("utf-8", errors="ignore")
                texts.append(text)
                metas.append({"source": name})
                logger.debug(f"Processed text file: {len(text)} chars")
            elif ext in ("pdf",):
                reader = PdfReader(io.BytesIO(content))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
                texts.append(text)
                metas.append({"source": name})
                logger.debug(f"Processed PDF: {len(pages)} pages")
            elif ext in ("docx",):
                doc = Document(io.BytesIO(content))
                text = "\n".join([p.text for p in doc.paragraphs])
                texts.append(text)
                metas.append({"source": name})
                logger.debug(f"Processed DOCX: {len(text)} chars")
            else:
                logger.warning(f"Unsupported file type: {ext}")
                metas.append({"source": name, "skipped": True})
                
            duration = time.time() - start_time
            logger.info(f"Processed {name} in {duration:.2f}s")
        except Exception as e:
            logger.error(f"Failed to process {name}: {str(e)}", exc_info=True)
            metas.append({"source": name, "error": str(e)})

    pairs = [(t, m) for t, m in zip(texts, metas) if t and not m.get("skipped")]
    if pairs:
        logger.info(f"Ingesting {len(pairs)} processed documents")
        n = retriever.add_texts([p[0] for p in pairs], metadatas=[p[1] for p in pairs])
    else:
        n = 0
        logger.warning("No valid documents to ingest")
    
    return {"uploaded": len(files), "ingested": n, "metas": metas}

# ---------- Ask: hybrid + RRF ----------
@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    req_id = f"req_{int(time.time())}"
    logger.info(f"[{req_id}] Processing ask request for session: {payload.session_id}")
    logger.debug(f"[{req_id}] Query: {payload.query}")
    
    try:
        start_time = time.time()
        docs = retriever.hybrid_search(payload.query, k_dense=6, k_sparse=6, k_rrf=60, top_k=6)
        logger.debug(f"[{req_id}] Retrieved {len(docs)} documents")
        
        context = "\n\n".join(d["text"] for d in docs)
        citations = [{"doc_id": d["id"], "meta": d.get("meta", {}), "rrf": d.get("_rrf")} for d in docs]
        
        route = Policy.decide(payload.query)
        logger.info(f"[{req_id}] Selected route: {route.value}")
        
        if route == Route.CODE:
            answer = answer_code(payload.query, context)
        elif route == Route.SQL:
            answer = answer_sql(payload.query)
        else:
            answer = answer_rag(payload.query, context)
            
        memory.save(payload.session_id, payload.query, answer, citations)
        
        duration = time.time() - start_time
        logger.info(f"[{req_id}] Request completed in {duration:.2f}s")
        
        return AskResponse(answer=answer, citations=citations, route=route.value)
    except Exception as e:
        logger.error(f"[{req_id}] Request failed: {str(e)}", exc_info=True)
        raise

@app.get("/dbinfo")
def dbinfo():
    logger.info("DB info requested")
    # Dense (Chroma)
    try:
        dense_count = retriever.count_dense()        # if your retriever exposes it
    except Exception:
        try:
            dense_count = int(retriever.col.count()) # direct Chroma collection
        except Exception as e:
            logger.warning(f"count_dense failed: {e}")
            dense_count = None

    # Sparse (SQLite)
    try:
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        sparse_count = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.warning(f"sparse count failed: {e}")
        sparse_count = None

    return {
        "vector_db": {
            "name": "Chroma",
            "collection": getattr(retriever, "collection_name", "rag_docs"),
            "path": os.getenv("CHROMA_DIR", ".chroma"),
            "doc_count": dense_count,
        },
        "sparse_db": {
            "name": "SQLite",
            "path": os.getenv("SQLITE_PATH", "./rag_memory.db"),
            "tables": ["docs", "memories"],
            "doc_count": sparse_count,
        },
    }

@app.get("/documents")
def docs(offset: int = 0, limit: int = 20):
    logger.info(f"Docs list requested offset={offset} limit={limit}")
    try:
        # If your retriever has list_docs(), use it:
        if hasattr(retriever, "list_docs"):
            data = retriever.list_docs(offset=offset, limit=limit)
            return {"offset": offset, "limit": limit, "total": data["total"], "items": data["items"]}

        # Fallback: query SQLite directly
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        total = cur.fetchone()[0]
        cur.execute(
            "SELECT id, source, substr(text,1,500) FROM docs ORDER BY rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = [{"id": r[0], "source": r[1], "snippet": r[2]} for r in cur.fetchall()]
        conn.close()
        return {"offset": offset, "limit": limit, "total": total, "items": rows}
    except Exception as e:
        logger.error(f"Docs list failed: {e}", exc_info=True)
        return {"offset": offset, "limit": limit, "total": 0, "items": [], "error": str(e)}

@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    logger.info(f"Memory requested for session_id={session_id}")
    try:
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("""
            SELECT ts, user, substr(assistant,1,400) AS assistant, citations
            FROM memories WHERE session_id=?
            ORDER BY ts DESC
        """, (session_id,))
        rows = cur.fetchall()
        conn.close()
        return {"session_id": session_id, "rows": rows}
    except Exception as e:
        logger.error(f"Memory fetch failed: {e}", exc_info=True)
        return {"session_id": session_id, "rows": [], "error": str(e)}

@app.post("/clear")
def clear_all():
    logger.warning("Clearing vector + sqlite stores (dev)")
    # Chroma (vector) – clears all collections in the path
    try:
        retriever.client.reset()
    except Exception as e:
        logger.warning(f"Chroma reset error: {e}")

    # SQLite (sparse + memories)
    try:
        conn = sqlite3.connect(os.getenv("SQLITE_PATH", "./rag_memory.db"))
        cur = conn.cursor()
        cur.execute("DELETE FROM docs")
        cur.execute("DELETE FROM memories")
        conn.commit()
        conn.close()
        return {"cleared": True}
    except Exception as e:
        logger.error(f"Clear failed: {e}", exc_info=True)
        return {"cleared": False, "error": str(e)}


@app.get("/rerank-debug")
def rerank_debug(q: str = Query(..., description="Query to debug ranking")):
    logger.info(f"Rerank debug for q='{q}'")
    try:
        dense = retriever.search_dense(q, k=6)
        sparse = retriever.search_bm25(q, k=6)
        fused  = retriever._rrf(dense, sparse, k=60)
        def slim(lst):
            return [{
                "id": d["id"],
                "src": d.get("meta", {}).get("source"),
                "rrf": d.get("_rrf"),
                "text": (d["text"][:180] + "…") if len(d["text"]) > 200 else d["text"]
            } for d in lst]
        return {"dense": slim(dense), "sparse": slim(sparse), "fused": slim(fused[:6])}
    except Exception as e:
        logger.error(f"Rerank debug failed: {e}", exc_info=True)
        return {"error": str(e)}

@app.get("/models")
def models():
    try:
        from agents import MODEL  # dict: rag/code/sql/fallback
        return { "models": MODEL }
    except Exception:
        # Fallback to defaults you’re already using implicitly
        return { "models": {
            "rag": "mistral:7b-instruct",
            "code": "Qwen2.5-Coder:latest",
            "sql":  "sqlcoder:latest",
            "fallback": "llama3:latest",
        }}


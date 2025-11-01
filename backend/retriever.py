import os, re, sqlite3, uuid
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-base-v2")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./rag_memory.db")

_word_re = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(text or "")]

class Retriever:
    """
    Dense store: Chroma
    Sparse store: SQLite (table: docs) + BM25
    Hybrid search: dense + BM25 → RRF
    """
    def __init__(self, collection: str = "rag_docs"):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        self._ensure_sqlite()
        self.collection_name = collection

    # ---------- SQLite (sparse) ----------
    def _ensure_sqlite(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS docs(
                id TEXT PRIMARY KEY,
                text TEXT,
                source TEXT
            )
        """)
        self.conn.commit()

    def _sqlite_add(self, pairs: List[Tuple[str, str, str | None]]):
        cur = self.conn.cursor()
        cur.executemany("INSERT OR REPLACE INTO docs(id, text, source) VALUES(?,?,?)", pairs)
        self.conn.commit()

    # public: list docs with pagination
    def list_docs(self, offset: int = 0, limit: int = 20) -> Dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        total = cur.fetchone()[0]
        cur.execute("SELECT id, source, substr(text,1,500) FROM docs ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = [{"id": r[0], "source": r[1], "snippet": r[2]} for r in cur.fetchall()]
        return {"total": total, "items": rows}

    # counts
    def count_sparse(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM docs")
        return cur.fetchone()[0]

    def count_dense(self) -> int:
        # Chroma 0.5.x supports .count()
        try:
            return int(self.col.count())
        except Exception:
            # fallback: approximate by fetching metadatas
            got = self.col.get(limit=1_000_000)  # large cap, fine for small demos
            return len(got.get("ids", []))

    # ---------- Dense add/search ----------
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None, ids: List[str] | None = None) -> int:
        if not texts:
            return 0
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Dense
        vecs = self.encoder.encode(texts, normalize_embeddings=True).tolist()
        self.col.add(documents=texts, embeddings=vecs, metadatas=metadatas, ids=ids)

        # Sparse
        pairs = [(ids[i], texts[i], metadatas[i].get("source")) for i in range(len(texts))]
        self._sqlite_add(pairs)
        return len(texts)

    def search_dense(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q_emb = self.encoder.encode([f"query: {query}"], normalize_embeddings=True).tolist()[0]
        res = self.col.query(query_embeddings=[q_emb], n_results=k)
        out: List[Dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else [{} for _ in ids]
        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i], "meta": metas[i]})
        return out

    # ---------- BM25 over SQLite ----------
    def _sqlite_all_docs(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT id, text, source FROM docs")
        rows = cur.fetchall()
        return [{"id": r[0], "text": r[1], "meta": {"source": r[2]} if r[2] else {}} for r in rows]

    def search_bm25(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        docs = self._sqlite_all_docs()
        if not docs:
            return []
        corpus_tokens = [tokenize(d["text"]) for d in docs]
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = tokenize(query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
        return [r[0] | {"_bm25": float(r[1])} for r in ranked]

    # ---------- RRF fusion ----------
    @staticmethod
    def _rrf(dense_list: List[Dict[str, Any]], sparse_list: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for rank, d in enumerate(dense_list, start=1):
            fused.setdefault(d["id"], d | {"_rrf": 0.0, "_dense_rank": rank})
            fused[d["id"]]["_rrf"] += 1.0 / (k + rank)
        for rank, d in enumerate(sparse_list, start=1):
            fused.setdefault(d["id"], d | {"_rrf": 0.0, "_sparse_rank": rank})
            fused[d["id"]]["_rrf"] += 1.0 / (k + rank)
        merged = list(fused.values())
        merged.sort(key=lambda x: x["_rrf"], reverse=True)
        return merged

    def hybrid_search(self, query: str, k_dense: int = 6, k_sparse: int = 6, k_rrf: int = 60, top_k: int = 6) -> List[Dict[str, Any]]:
        dense = self.search_dense(query, k_dense)
        sparse = self.search_bm25(query, k_sparse)
        fused = self._rrf(dense, sparse, k=k_rrf)
        return fused[:top_k]

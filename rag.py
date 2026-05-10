import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import chromadb  # vector store only — no embedding hook used
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@traceable(run_type="embedding", name="Gemini Embed", tags=["embedding", "gemini"])
def _embed_texts(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed a list of texts using Gemini. Returns list of float vectors."""
    embeddings = []
    for text in texts:
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.append(list(result.embeddings[0].values))
    return embeddings


class TextChunker:
    """Splits text into overlapping sentence-aware chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        text = re.sub(r"\s+", " ", text).strip()
        chunks: List[DocumentChunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            slice_ = text[start:end]

            if end < len(text):
                boundary = max(slice_.rfind(". "), slice_.rfind(".\n"))
                if boundary > self.chunk_size // 2:
                    slice_ = slice_[: boundary + 1]
                    end = start + boundary + 1

            chunk_id = hashlib.md5(slice_.encode(), usedforsecurity=False).hexdigest()[:12]
            chunks.append(
                DocumentChunk(id=chunk_id, text=slice_.strip(), metadata=metadata or {})
            )
            start = end - self.overlap
            if start >= len(text):
                break

        return chunks


class VectorStore:
    """Persistent ChromaDB vector store with Gemini embeddings."""

    COLLECTION = "research_docs"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._db = chromadb.PersistentClient(path=persist_dir)
        self._col = self._get_or_create()

    def _get_or_create(self):
        # No embedding_function — we pre-compute and pass embeddings directly
        return self._db.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @traceable(run_type="tool", name="ChromaDB Ingest", tags=["chromadb", "rag"])
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        ids = [c.id for c in chunks]
        existing = set(self._col.get(ids=ids)["ids"])
        new = [(c.id, c.text, c.metadata) for c in chunks if c.id not in existing]
        if not new:
            return 0
        nids, ndocs, nmetas = zip(*new)
        # Pre-compute embeddings ourselves — avoids ChromaDB's internal hook which can hang
        embeddings = _embed_texts(list(ndocs), task_type="RETRIEVAL_DOCUMENT")
        self._col.add(
            ids=list(nids),
            documents=list(ndocs),
            metadatas=list(nmetas),
            embeddings=embeddings,
        )
        return len(new)

    @traceable(run_type="retriever", name="ChromaDB Retrieval", tags=["chromadb", "rag"])
    def search(self, query: str, k: int = TOP_K) -> List[DocumentChunk]:
        if self.count() == 0:
            return []
        k = min(k, self.count())
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_emb = list(result.embeddings[0].values)
        results = self._col.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                DocumentChunk(id="", text=doc, metadata={**meta, "score": round(1 - dist, 3)})
            )
        return chunks

    def count(self) -> int:
        return self._col.count()

    def clear(self):
        self._db.delete_collection(self.COLLECTION)
        self._col = self._get_or_create()


class DocumentProcessor:
    """Loads text from plain files, PDFs, and URLs."""

    def load_file(self, filepath: str) -> Tuple[str, str]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(filepath), os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), os.path.basename(filepath)

    def _load_pdf(self, filepath: str) -> str:
        import PyPDF2

        parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                parts.append(page.extract_text() or "")
        return "\n\n".join(parts)

    def load_url(self, url: str) -> Tuple[str, str]:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (ResearchBot/1.0)"}
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text, url


class RAGPipeline:
    """End-to-end RAG pipeline: ingest → embed → retrieve → contextualize."""

    def __init__(self):
        self.store = VectorStore()
        self.chunker = TextChunker()
        self.processor = DocumentProcessor()

    def add_text(self, text: str, source: str = "manual") -> int:
        chunks = self.chunker.chunk(text, metadata={"source": source, "type": "text"})
        return self.store.add_chunks(chunks)

    def add_file(self, filepath: str) -> int:
        text, source = self.processor.load_file(filepath)
        chunks = self.chunker.chunk(
            text, metadata={"source": source, "type": "file", "path": filepath}
        )
        return self.store.add_chunks(chunks)

    def add_url(self, url: str) -> int:
        text, source = self.processor.load_url(url)
        chunks = self.chunker.chunk(text, metadata={"source": source, "type": "url"})
        return self.store.add_chunks(chunks)

    def search(self, query: str, k: int = TOP_K) -> List[DocumentChunk]:
        return self.store.search(query, k)

    def build_context(self, chunks: List[DocumentChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            score = chunk.metadata.get("score", "")
            header = (
                f"[Source {i}: {source}"
                + (f" | relevance={score}" if score else "")
                + "]"
            )
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def total_chunks(self) -> int:
        return self.store.count()

    def clear(self):
        self.store.clear()

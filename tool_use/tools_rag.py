"""Retrieval tool for the RAG specialist: hybrid search (BM25 + dense) over the corpus."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tool_use.registry import tool

from rag.chunking import recursive_chunks
from rag.embed import Embedder
from rag.ingest import load_corpus
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
# copia aquí el decorador y los imports de tool que uses en tools_cuota_tae.py

TOP_K = 3
WEIGHT_BM25 = 0.6
INDEX_DIR = Path(__file__).resolve().parent.parent / "rag" / "index"


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    """Built once per process: loading the embedder and the index is the slow part."""
    chunks = recursive_chunks(load_corpus())
    if not chunks:
        raise RuntimeError("rag/corpus está vacío: ejecuta `python -m rag.fetch_corpus`.")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, Embedder(), persist_dir=INDEX_DIR)
    return HybridRetriever(bm25, dense, weight_bm25=WEIGHT_BM25)


@tool
def search_knowledge(query: str) -> dict:
    """Search the credit and finance knowledge base and return the most relevant passages.

    Args:
        query: The question or keywords to look up, in Spanish.
    """
    hits = _retriever().search(query, top_k=TOP_K)
    passages = [f"[{h.chunk.source}] {h.chunk.text}" for h in hits]
    return {
        "result": "\n\n".join(passages),
        "inputs": {"query": query, "sources": [h.chunk.source for h in hits]},
    }
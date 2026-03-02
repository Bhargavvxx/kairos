# backend/app/services/reranker.py

"""
Cross-encoder reranker service.
Re-scores and re-orders retrieval results using a cross-encoder model,
which reads (query, passage) pairs jointly for much higher relevance accuracy
than the initial bi-encoder / BM25 retrieval stage.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_cross_encoder = None
_model_lock = asyncio.Lock()

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


async def _get_model(model_name: str = DEFAULT_MODEL):
    """Lazily load (and cache) the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    async with _model_lock:
        # Double-check after acquiring lock
        if _cross_encoder is not None:
            return _cross_encoder

        logger.info(f"Loading cross-encoder model: {model_name} …")
        from sentence_transformers import CrossEncoder

        loop = asyncio.get_running_loop()
        _cross_encoder = await loop.run_in_executor(
            None, lambda: CrossEncoder(model_name)
        )
        logger.info("Cross-encoder model loaded successfully.")
        return _cross_encoder


async def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    model_name: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks using the cross-encoder.

    Parameters
    ----------
    query : str
        The user query.
    chunks : list[dict]
        Each dict must contain at least a ``content`` key.
    top_k : int or None
        If given, keep only the top-k results after reranking.
    model_name : str
        HuggingFace model identifier for the cross-encoder.

    Returns
    -------
    list[dict]
        The same dicts, sorted by ``rerank_score`` (descending), with
        ``rerank_score`` and ``original_rank`` fields added.
    """
    if not chunks:
        return []

    try:
        model = await _get_model(model_name)

        # Build (query, passage) pairs
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Score in a thread so the event loop stays responsive
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, lambda: model.predict(pairs).tolist()
        )

        # Attach scores and original rank
        for idx, (chunk, score) in enumerate(zip(chunks, scores)):
            chunk["rerank_score"] = round(float(score), 4)
            chunk["original_rank"] = idx + 1

        # Sort descending by rerank_score
        chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        if top_k is not None:
            chunks = chunks[:top_k]

        return chunks

    except Exception as e:
        logger.error(f"Reranking failed, returning original order: {e}", exc_info=True)
        # Graceful fallback – return original order untouched
        for idx, chunk in enumerate(chunks):
            chunk["rerank_score"] = chunk.get("similarity_score", 0.0)
            chunk["original_rank"] = idx + 1
        return chunks

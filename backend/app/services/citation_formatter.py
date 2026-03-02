# backend/app/services/citation_formatter.py

"""
Chunk-level citation formatter.

Builds structured citation objects that the frontend can render inline or as
a numbered reference list. Each citation includes:

• ``id``          – sequential reference number (``[1]``, ``[2]``, …)
• ``chunk_id``    – the vector-store chunk identifier
• ``source``      – original filename
• ``page``        – page number (if present in metadata)
• ``snippet``     – a short text excerpt from the chunk
• ``score``       – the rerank score (or similarity score as fallback)

It also rewrites the LLM answer to inject ``[1]``, ``[2]`` markers so
the user can trace claims back to source passages.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

MAX_SNIPPET_LEN = 180  # characters


def build_citations(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a list of reranked chunks into numbered citation objects.

    Parameters
    ----------
    chunks : list[dict]
        Each chunk dict should have ``content``, ``metadata``, ``chunk_id``,
        and optionally ``rerank_score`` / ``similarity_score``.

    Returns
    -------
    list[dict]
        Citation dicts ready for the JSON response.
    """
    citations = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        content = chunk.get("content", "")

        snippet = content[:MAX_SNIPPET_LEN].strip()
        if len(content) > MAX_SNIPPET_LEN:
            # Try to cut at the last full sentence or word
            cut = snippet.rfind(". ")
            if cut == -1:
                cut = snippet.rfind(" ")
            if cut > MAX_SNIPPET_LEN // 2:
                snippet = snippet[: cut + 1]
            snippet += " …"

        citation = {
            "id": idx,
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
            "source": meta.get("filename") or meta.get("original_filename", "unknown"),
            "page": meta.get("page_number") or meta.get("page", None),
            "snippet": snippet,
            "score": round(
                chunk.get("rerank_score") if chunk.get("rerank_score") is not None
                else chunk.get("similarity_score", 0.0),
                4,
            ),
        }
        citations.append(citation)

    return citations


def build_answer_prompt_with_citations(
    query: str,
    chunks: List[Dict[str, Any]],
    graph_context: Any = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build the LLM prompt and the citation list simultaneously.

    The context block labels each chunk with ``[1]``, ``[2]``, … so the
    model can reference them in its answer.

    Returns
    -------
    (prompt_str, citations_list)
    """
    citations = build_citations(chunks)

    context_parts = []
    for cit, chunk in zip(citations, chunks):
        source_label = cit["source"]
        if cit["page"]:
            source_label += f", p.{cit['page']}"
        context_parts.append(
            f"[{cit['id']}] (Source: {source_label})\n{chunk['content']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # Optionally include graph context summary
    graph_section = ""
    if graph_context and graph_context.get("found_entities"):
        entities = [
            e.get("label", e.get("id", ""))
            for e in graph_context["found_entities"][:10]
        ]
        relationships = []
        for rel in graph_context.get("relationships", [])[:10]:
            relationships.append(
                f"{rel.get('source', '?')} —[{rel.get('label', rel.get('type', ''))}]→ {rel.get('target', '?')}"
            )
        graph_section = (
            "\n\nKNOWLEDGE GRAPH CONTEXT:\n"
            f"Entities found: {', '.join(entities)}\n"
        )
        if relationships:
            graph_section += f"Relationships: {'; '.join(relationships)}\n"

    prompt = f"""You are KAIROS, an intelligent document analysis assistant.
Answer the user's question using ONLY the numbered context passages below.
When you use information from a passage, cite it with its number, e.g. [1], [2].
If the context does not contain enough information, say so honestly.
Format your answer with markdown for readability.

CONTEXT:
{context_text}
{graph_section}
QUESTION: {query}

ANSWER (cite sources with [1], [2], etc.):"""

    return prompt, citations

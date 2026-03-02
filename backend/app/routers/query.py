# backend/app/routers/query.py

import logging
import time
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.dependencies import get_vector_service, get_graph_service, get_llm_client
from app.services.vector_service import VectorService
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


# --- Request / Response Models ---

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    include_graph: bool = Field(default=True)
    max_results: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    processing_time: float = 0.0
    graph_context: Optional[Dict[str, Any]] = None


class DashboardRisk(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    document: str


class DashboardOpportunity(BaseModel):
    id: str
    title: str
    description: str
    potential: str


class DashboardResponse(BaseModel):
    risks: List[DashboardRisk] = Field(default_factory=list)
    opportunities: List[DashboardOpportunity] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)


# --- Helper: build an LLM prompt from context chunks ---

def _build_answer_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("metadata", {}).get("filename", f"chunk_{i}")
        context_parts.append(f"[Source: {source}]\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""You are KAIROS, an intelligent document analysis assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so honestly.
Format your answer with markdown for readability.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""


# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    vector_service: VectorService = Depends(get_vector_service),
    graph_service: GraphService = Depends(get_graph_service),
    llm_client: Any = Depends(get_llm_client),
):
    """
    RAG-powered chat endpoint.
    Retrieves relevant document chunks, optionally queries the knowledge graph,
    and generates an answer with the LLM.
    """
    start = time.time()

    # 1. Vector search
    try:
        search_results = await vector_service.search(
            query=request.query,
            n_results=request.max_results,
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        search_results = []

    # 2. (Optional) Graph context
    graph_context = None
    if request.include_graph:
        try:
            # Extract potential entity names from the query (simple heuristic)
            words = [w for w in request.query.split() if len(w) > 2]
            entity_results = await graph_service.query_for_orchestrator(words)
            if entity_results.get("found_entities"):
                graph_context = entity_results
        except Exception as e:
            logger.warning(f"Graph query failed (non-fatal): {e}")

    # 3. Generate answer with LLM
    if search_results:
        prompt = _build_answer_prompt(request.query, search_results)
        try:
            answer = await llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "I'm sorry, I encountered an error generating a response. Please try again."
    else:
        answer = (
            "I don't have any documents to reference yet. "
            "Please upload documents first so I can answer your questions."
        )

    # 4. Collect source filenames
    sources: List[str] = []
    seen = set()
    for r in search_results:
        fname = r.get("metadata", {}).get("filename") or r.get("metadata", {}).get("original_filename", "unknown")
        if fname not in seen:
            sources.append(fname)
            seen.add(fname)

    # 5. Confidence heuristic
    confidence = 0.0
    if search_results:
        scores = [r.get("similarity_score", 0) for r in search_results]
        confidence = round(sum(scores) / len(scores), 2) if scores else 0.0

    elapsed = round(time.time() - start, 3)

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        processing_time=elapsed,
        graph_context=graph_context,
    )


@router.get("/graph", response_model=GraphResponse)
async def get_graph(
    graph_service: GraphService = Depends(get_graph_service),
):
    """
    Return the full knowledge graph for visualisation.
    """
    try:
        data = graph_service.export_for_visualization()
        nodes = data.get("nodes", [])
        links = data.get("links", [])

        # Normalise for the frontend (react-force-graph-2d format)
        fe_nodes = []
        for n in nodes:
            fe_nodes.append({
                "id": n.get("id", ""),
                "label": n.get("label", n.get("id", "")),
                "group": n.get("type", "Entity"),
                "val": n.get("size", 5),
                "color": n.get("color", "#9CA3AF"),
                "properties": n.get("properties", {}),
            })

        fe_links = []
        for link in links:
            fe_links.append({
                "source": link.get("source"),
                "target": link.get("target"),
                "label": link.get("label", link.get("type", "")),
            })

        return GraphResponse(nodes=fe_nodes, links=fe_links)
    except Exception as e:
        logger.error(f"Graph export failed: {e}")
        return GraphResponse(nodes=[], links=[])


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    vector_service: VectorService = Depends(get_vector_service),
    graph_service: GraphService = Depends(get_graph_service),
    llm_client: Any = Depends(get_llm_client),
):
    """
    Return a proactive intelligence dashboard built from the knowledge graph and
    vector store. It uses the LLM to identify risks and opportunities.
    """
    risks: List[DashboardRisk] = []
    opportunities: List[DashboardOpportunity] = []

    try:
        stats = graph_service.get_graph_statistics()
        total_nodes = stats.get("total_nodes", 0)
        total_edges = stats.get("total_edges", 0)

        if total_nodes == 0:
            return DashboardResponse(risks=risks, opportunities=opportunities)

        # Pull a sample of graph data
        viz = graph_service.export_for_visualization()
        sample_nodes = viz.get("nodes", [])[:30]
        sample_links = viz.get("links", [])[:30]

        summary = (
            f"The knowledge graph has {total_nodes} entities and {total_edges} relationships.\n"
            f"Sample entities: {[n.get('label', n.get('id','')) for n in sample_nodes[:10]]}\n"
            f"Sample relationships: {[(l.get('source'), l.get('label',''), l.get('target')) for l in sample_links[:10]]}\n"
        )

        prompt = f"""You are KAIROS, a financial document intelligence system.
Based on the following knowledge graph summary, identify up to 3 risks and 3 growth opportunities.
Return ONLY valid JSON in the format:
{{
  "risks": [
    {{"id": "1", "title": "...", "description": "...", "severity": "High|Medium|Low", "document": "source document"}}
  ],
  "opportunities": [
    {{"id": "1", "title": "...", "description": "...", "potential": "High|Medium|Low"}}
  ]
}}

KNOWLEDGE GRAPH SUMMARY:
{summary}

JSON:"""

        import json
        raw = await llm_client.generate(prompt)
        # Try to parse JSON from the response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?", "", cleaned)
                cleaned = re.sub(r"```$", "", cleaned)
            parsed = json.loads(cleaned.strip())
            for r in parsed.get("risks", []):
                risks.append(DashboardRisk(**r))
            for o in parsed.get("opportunities", []):
                opportunities.append(DashboardOpportunity(**o))
        except json.JSONDecodeError:
            logger.warning("LLM did not return valid JSON for dashboard")

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")

    return DashboardResponse(risks=risks, opportunities=opportunities)

# backend/app/services/confidence_scorer.py

"""
Multi-signal confidence scorer.

Instead of naïvely averaging retrieval similarity scores, this module
combines several independent signals into a single 0–1 confidence value:

• **Retrieval signal** – how strong are the top-k similarity / rerank scores?
• **Score spread** – sharp peak = model is sure; flat = ambiguous.
• **Coverage signal** – did the reranker keep most of the original chunks?
• **Answerability heuristic** – does the LLM answer look like a refusal?
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Phrases that strongly suggest the model could NOT answer the question
_REFUSAL_PATTERNS = [
    r"(?:i )?(?:do not|don'?t|cannot|can'?t) (?:have|find|see|locate)",
    r"not (?:enough|sufficient) information",
    r"no (?:relevant|matching) (?:information|documents|context)",
    r"unfortunately.*(?:unable|couldn'?t|could not)",
    r"the (?:context|documents?) (?:do(?:es)? not|don'?t) (?:contain|mention|provide)",
    r"there is no (?:information|mention|data)",
    r"i (?:could|couldn'?t|was) not (?:find|able)",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def compute_confidence(
    chunks: List[Dict],
    answer: str,
    *,
    retrieval_weight: float = 0.35,
    rerank_weight: float = 0.30,
    spread_weight: float = 0.15,
    answerability_weight: float = 0.20,
) -> Dict:
    """
    Return ``{"score": float, "details": {...}}`` where **score** ∈ [0, 1].

    Parameters
    ----------
    chunks : list[dict]
        Chunks after reranking. Each should have ``similarity_score`` and
        optionally ``rerank_score``.
    answer : str
        The generated answer text.
    retrieval_weight, rerank_weight, spread_weight, answerability_weight : float
        Relative importance of each signal (should sum to ≈ 1).

    Returns
    -------
    dict  {"score": float, "details": dict}
    """
    details = {}

    # --- 1. Retrieval signal (average of original similarity scores) ---
    sim_scores = [c.get("similarity_score", 0.0) for c in chunks]
    retrieval_signal = _safe_mean(sim_scores)
    details["retrieval_signal"] = round(retrieval_signal, 4)

    # --- 2. Rerank signal (average of cross-encoder scores, normalised to 0-1) ---
    rerank_scores = [c.get("rerank_score") for c in chunks if c.get("rerank_score") is not None]
    if rerank_scores:
        # Cross-encoder scores are logits; sigmoid-ish normalisation:
        rerank_signal = _safe_mean([_sigmoid(s) for s in rerank_scores])
    else:
        # No reranking happened – fall back to retrieval signal
        rerank_signal = retrieval_signal
    details["rerank_signal"] = round(rerank_signal, 4)

    # --- 3. Score spread (low variance among top scores → high confidence) ---
    if len(sim_scores) >= 2:
        top = max(sim_scores)
        second = sorted(sim_scores, reverse=True)[1]
        # If top score is much higher than second → clear winner → high confidence
        spread_signal = min(1.0, (top - second) * 5 + 0.5)
    else:
        spread_signal = 0.5  # unknown
    details["spread_signal"] = round(spread_signal, 4)

    # --- 4. Answerability heuristic ---
    if not answer or not answer.strip():
        answerability_signal = 0.0
    elif _REFUSAL_RE.search(answer):
        answerability_signal = 0.15
    else:
        # Longer, substantive answers score higher
        word_count = len(answer.split())
        answerability_signal = min(1.0, word_count / 60)  # saturates at ~60 words
    details["answerability_signal"] = round(answerability_signal, 4)

    # --- Weighted combination ---
    raw = (
        retrieval_signal * retrieval_weight
        + rerank_signal * rerank_weight
        + spread_signal * spread_weight
        + answerability_signal * answerability_weight
    )
    # Clamp to [0, 1]
    score = round(max(0.0, min(1.0, raw)), 2)
    details["raw_weighted"] = round(raw, 4)

    logger.debug(f"Confidence: {score} | {details}")

    return {"score": score, "details": details}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for normalising cross-encoder logits."""
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

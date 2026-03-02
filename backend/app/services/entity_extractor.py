# backend/app/services/entity_extractor.py

"""
LLM-powered entity extraction for graph context retrieval.

Replaces the naive ``query.split()`` heuristic with a lightweight LLM call
that returns structured entities, allowing the knowledge-graph service to
look up genuinely relevant nodes instead of random short words.

Falls back to a regex + stopword heuristic when the LLM is unavailable.
"""

import json
import logging
import re
from typing import Any, List

logger = logging.getLogger(__name__)

# Common English stopwords (lowercase) for the fallback extractor
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could about above after again against all am and "
    "any as at because before below between both but by down during each few for "
    "from further get got had has have he her here hers herself him himself his "
    "how i if in into is it its itself just let like me more most my myself no "
    "nor not now of off on once only or other our ours ourselves out over own "
    "quite rather really said same she should since so some still such take than "
    "that the their theirs them themselves then there these they this those through "
    "to too under until up upon us very want was we were what when where which "
    "while who whom why with without would yes yet you your yours yourself "
    "yourselves also already another back being could doing each enough even every "
    "going going got getting having however including keep keeping last long looking "
    "made make many much must neither never new next none nothing now often onto "
    "open over part per perhaps please putting rather really regarding right said "
    "seem several since so some something sometimes still such sure take tell than "
    "that the their them then there therefore these they thing think this those "
    "though through thus together too toward tried trying under unless unlikely "
    "until upon us used using usual various very want well went were what whatever "
    "whatsoever when whenever where whether which while whole whom whose why will "
    "with within without work works working would yet".split()
)


async def extract_entities_llm(
    query: str,
    llm_client: Any,
) -> List[str]:
    """
    Use the LLM to extract meaningful entities from the user query.

    Returns a list of entity strings suitable for graph lookup.
    """
    prompt = (
        "Extract the key named entities and important noun-phrases from the following question.\n"
        "Return ONLY a JSON array of strings. No explanation, no markdown fences.\n"
        "Include: person names, organisation names, policy terms, medical/legal terms, "
        "amounts, dates, product names, and domain-specific concepts.\n"
        "Exclude: generic stop-words, pronouns, determiners.\n\n"
        f'Question: "{query}"\n\n'
        "JSON array:"
    )

    try:
        raw = await llm_client.generate(prompt)
        raw = raw.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw)
            raw = re.sub(r"```$", "", raw)
            raw = raw.strip()

        entities = json.loads(raw)

        if isinstance(entities, list):
            # Deduplicate while preserving order, drop empties
            seen = set()
            result = []
            for e in entities:
                e_str = str(e).strip()
                if e_str and e_str.lower() not in seen:
                    seen.add(e_str.lower())
                    result.append(e_str)
            logger.debug(f"LLM extracted entities: {result}")
            return result

    except Exception as e:
        logger.warning(f"LLM entity extraction failed, falling back to heuristic: {e}")

    # Fallback
    return extract_entities_heuristic(query)


def extract_entities_heuristic(query: str) -> List[str]:
    """
    Regex + stopword fallback.

    Much better than the old ``[w for w in query.split() if len(w) > 2]`` because
    it strips punctuation, removes stopwords, and groups capitalised multi-word
    sequences as single entities.
    """
    entities: List[str] = []
    seen_lower = set()

    # 1. Try to capture capitalised multi-word phrases (e.g. "ICICI Lombard", "Pre Existing")
    capitalized = re.findall(r"\b(?:[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)+)\b", query)
    # Track words that are part of multi-word phrases so we don't duplicate them
    phrase_words = set()
    for phrase in capitalized:
        key = phrase.lower()
        if key not in seen_lower:
            entities.append(phrase)
            seen_lower.add(key)
            for word in phrase.split():
                phrase_words.add(word.lower())

    # 2. Tokenise remaining words, keep meaningful ones
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", query)
    for token in tokens:
        key = token.lower()
        if len(token) > 2 and key not in _STOPWORDS and key not in seen_lower and key not in phrase_words:
            entities.append(token)
            seen_lower.add(key)

    logger.debug(f"Heuristic extracted entities: {entities}")
    return entities

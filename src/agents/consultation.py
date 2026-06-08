"""
BanglaLex — Agent 1: Client Consultation Agent
================================================
Front-end intake agent.

Responsibility
--------------
Take a client's free-text legal problem (English or Bangla) and convert it
into a structured case representation that downstream agents can reason over.

Input  : raw user query string
Output : structured case dict with keys —
           language, domain, parties, location, dates,
           facts_summary, key_claims, urgency, missing_info
"""

import logging
from .base import GeminiAgent, detect_language

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior legal intake specialist for BanglaLex, an AI legal assistance
system for Bangladesh. Your job is to listen to a client's legal problem and
extract structured information for further analysis.

You handle queries in BOTH English and Bangla. Regardless of input language,
always return a valid JSON object with the exact keys listed below.

Extract the following:

{
  "language": "en" or "bn",
  "domain": one of ["land", "contract", "service", "family", "criminal", "other"],
  "parties": {
    "plaintiff": "who is bringing the complaint (the client or their side)",
    "defendant": "who the complaint is against"
  },
  "location": "district or area mentioned, or null if not stated",
  "dates": ["list of relevant dates or time periods mentioned, empty list if none"],
  "facts_summary": "2-3 sentence objective summary of the core situation",
  "key_claims": ["list of main legal grievances or claims the client has"],
  "urgency": "high" (immediate legal threat e.g. eviction tomorrow) or
             "medium" (ongoing dispute) or
             "low" (general query),
  "missing_info": ["list of important details NOT provided that would help legal analysis"]
}

Domain classification guide:
  land     → property disputes, land ownership, tenancy, eviction, boundary
  contract → employment, wages, agreements, business disputes, breach of contract
  service  → government job disputes, civil service, public sector employment
  family   → marriage, divorce, inheritance, maintenance, guardianship
  criminal → theft, fraud, assault, police complaints, FIR
  other    → anything that does not fit above categories

Return ONLY the JSON object. No preamble, no explanation.
""".strip()


# ── Agent class ────────────────────────────────────────────────────────────────

class ClientConsultationAgent(GeminiAgent):
    """
    Agent 1 — Client Consultation Agent.

    Converts a free-text legal problem into a structured case dict.

    Parameters
    ----------
    model_name  : Gemini model (inherits default from GeminiAgent)
    temperature : Low temperature keeps extraction deterministic (default 0.05)

    Examples
    --------
    >>> agent = ClientConsultationAgent()
    >>> case  = agent.process("My landlord changed the locks without any notice.")
    >>> print(case["domain"])   # "land"
    >>> print(case["urgency"])  # "high"
    """

    def __init__(self, model_name: str = None):
        super().__init__(model_name=model_name, temperature=0.05)

    def process(self, user_query: str) -> dict:
        """
        Process a raw user query into a structured case representation.

        Parameters
        ----------
        user_query : free-text legal problem in English or Bangla

        Returns
        -------
        dict with keys: language, domain, parties, location, dates,
                        facts_summary, key_claims, urgency, missing_info,
                        raw_query
        """
        logger.info("Agent 1 — processing client query …")

        # Quick language detection (used as fallback if LLM disagrees)
        detected_lang = detect_language(user_query)

        prompt = f"""{_SYSTEM_PROMPT}

---
CLIENT QUERY:
{user_query}
---

Return the JSON extraction now."""

        result = self._call_json(prompt)

        # Normalise and add raw query
        result["raw_query"] = user_query

        # Ensure language field is present (fallback to heuristic)
        if "language" not in result or result["language"] not in ("en", "bn"):
            result["language"] = detected_lang

        # Ensure required keys exist with defaults
        result.setdefault("parties",      {"plaintiff": "client", "defendant": "unknown"})
        result.setdefault("location",     None)
        result.setdefault("dates",        [])
        result.setdefault("key_claims",   [])
        result.setdefault("urgency",      "medium")
        result.setdefault("missing_info", [])
        result.setdefault("domain",       "other")

        logger.info(
            f"Agent 1 done | domain={result['domain']} | "
            f"lang={result['language']} | urgency={result['urgency']}"
        )
        return result

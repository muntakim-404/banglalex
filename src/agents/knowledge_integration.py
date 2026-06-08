"""
BanglaLex — Agent 2: Knowledge Integration Layer
==================================================
Middleware agent that bridges the client-facing consultation
and the back-end reasoning agent.

Responsibility
--------------
1. Build a rich query from the structured case dict.
2. Retrieve relevant statute sections from the FAISS knowledge base.
3. Use the LLM to identify which retrieved sections directly apply
   and explain their relevance to the case facts.

Input  : structured case dict from Agent 1
Output : enriched case dict with added keys —
           retrieved_statutes, applicable_statutes,
           legal_framework, key_legal_issues
"""

import logging
from typing import Dict, List
from pathlib import Path

from .base import GeminiAgent

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior legal researcher specialising in Bangladesh law.
You have been given structured case facts and a set of statute sections
retrieved from the Bangladesh legal database.

Your task:
1. Identify which retrieved statute sections DIRECTLY apply to this case.
2. Explain precisely how each applicable section relates to the facts.
3. Describe the overall legal framework governing this dispute.
4. Identify the key legal questions the court or adjudicator must resolve.

Return a JSON object with exactly these keys:

{
  "applicable_statutes": [
    {
      "act_name": "full act title",
      "section_no": "section number",
      "section_title": "section title or empty string",
      "relevance": "1-2 sentence explanation of how this section applies to the case"
    }
  ],
  "legal_framework": "2-3 sentence summary of the legal framework governing this dispute",
  "key_legal_issues": [
    "specific legal question 1 the case raises",
    "specific legal question 2",
    ...
  ]
}

Include only statutes that are clearly relevant. Quality over quantity.
If none of the retrieved sections are directly applicable, return an empty
applicable_statutes list and note this in legal_framework.

Return ONLY the JSON object.
""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_query(case_dict: dict) -> str:
    """
    Construct a rich retrieval query from the structured case dict.
    Longer, contextual queries retrieve far better than 4-word phrases.
    """
    parts = []

    facts = case_dict.get("facts_summary", "")
    if facts:
        parts.append(facts)

    claims = case_dict.get("key_claims", [])
    if claims:
        parts.append(" ".join(claims))

    domain = case_dict.get("domain", "")
    if domain:
        parts.append(f"legal case domain: {domain}")

    return " ".join(parts)


def _format_statutes(retrieved: list) -> str:
    """Format raw retriever results into a readable block for the LLM prompt."""
    lines = []
    for i, r in enumerate(retrieved, 1):
        act   = r.get("act_name",      "Unknown Act")
        sec   = r.get("section_no",    "—")
        title = r.get("section_title", "")
        score = r.get("score",          0.0)
        text  = r.get("text",           "")[:400]   # keep prompt size reasonable

        heading = f"[{i}] {act}  §{sec}"
        if title:
            heading += f"  [{title}]"
        lines.append(f"{heading}  (relevance score: {score:.3f})\n{text}\n")

    return "\n".join(lines) if lines else "(no statutes retrieved)"


# ── Agent class ────────────────────────────────────────────────────────────────

class KnowledgeIntegrationLayer(GeminiAgent):
    """
    Agent 2 — Knowledge Integration Layer.

    Enriches the structured case dict with relevant statute sections
    and a legal framework summary.

    Parameters
    ----------
    model_name : Gemini model
    kb_dir     : path to the FAISS knowledge base (data/knowledge_base)
    k          : number of statute sections to retrieve (default 8)

    Examples
    --------
    >>> agent    = KnowledgeIntegrationLayer(kb_dir="data/knowledge_base")
    >>> enriched = agent.enrich(case_dict)
    >>> for s in enriched["applicable_statutes"]:
    ...     print(s["act_name"], s["section_no"])
    """

    def __init__(
        self,
        kb_dir:     str = "data/knowledge_base",
        model_name: str = None,
        k:          int = 8,
    ):
        super().__init__(model_name=model_name, temperature=0.1)
        self.k = k

        # Lazy-load the retriever (avoids loading the 66 MB index at import time)
        self._retriever = None
        self._kb_dir    = kb_dir

    def _get_retriever(self):
        if self._retriever is None:
            from src.knowledge_base.retriever import Retriever
            logger.info("Loading FAISS retriever …")
            self._retriever = Retriever.from_saved(self._kb_dir)
        return self._retriever

    def enrich(self, case_dict: dict) -> dict:
        """
        Enrich a structured case dict with retrieved statutes and legal analysis.

        Parameters
        ----------
        case_dict : output of ClientConsultationAgent.process()

        Returns
        -------
        dict — case_dict extended with:
               retrieved_statutes, applicable_statutes,
               legal_framework, key_legal_issues
        """
        logger.info("Agent 2 — retrieving statutes …")

        retriever = self._get_retriever()
        query     = _build_query(case_dict)
        domain    = case_dict.get("domain")

        # Retrieve from FAISS with domain-aware filtering
        retrieved = retriever.retrieve_for_case(
            case_facts = query,
            domain     = domain,
            k          = self.k,
        )
        logger.info(f"Agent 2 — retrieved {len(retrieved)} statute sections")

        # Ask LLM to identify which are applicable and why
        prompt = f"""{_SYSTEM_PROMPT}

---
CASE FACTS:
Domain  : {case_dict.get('domain', 'unknown')}
Summary : {case_dict.get('facts_summary', '')}
Claims  : {'; '.join(case_dict.get('key_claims', []))}
Parties : {case_dict.get('parties', {})}

RETRIEVED STATUTE SECTIONS:
{_format_statutes(retrieved)}
---

Identify the applicable statutes and return the JSON now."""

        analysis = self._call_json(prompt)

        # Merge into enriched case dict
        enriched = dict(case_dict)
        enriched["retrieved_statutes"] = retrieved
        enriched["applicable_statutes"] = analysis.get("applicable_statutes", [])
        enriched["legal_framework"]     = analysis.get("legal_framework",     "")
        enriched["key_legal_issues"]    = analysis.get("key_legal_issues",    [])

        logger.info(
            f"Agent 2 done | {len(enriched['applicable_statutes'])} applicable "
            f"statutes identified"
        )
        return enriched

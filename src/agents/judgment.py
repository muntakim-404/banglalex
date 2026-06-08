"""
BanglaLex — Agent 3: Legal Judgment Prediction Agent
======================================================
Back-end reasoning agent.

Responsibility
--------------
1. Load similar past cases from the Phase 1 annotated dataset (precedents).
2. Combine case facts + applicable statutes + precedents.
3. Predict the likely outcome (favorable / unfavorable / uncertain).
4. Produce a transparent reasoning chain explaining the prediction.

Input  : enriched case dict from Agent 2
Output : analysis dict with added keys —
           similar_cases, predicted_outcome, confidence,
           reasoning_chain, supporting_factors,
           risk_factors, applicable_charges,
           recommended_legal_basis
"""

import json
import logging
import random
from pathlib import Path
from typing  import Dict, List, Optional

from .base import GeminiAgent

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert legal analyst specialising in Bangladesh courts.
Your role is to predict the likely outcome of a legal case based on:
  1. The structured case facts
  2. Applicable Bangladesh statutes
  3. Similar past cases (precedents)

Provide a transparent, step-by-step legal analysis.

Return a JSON object with exactly these keys:

{
  "predicted_outcome": "favorable" | "unfavorable" | "uncertain",
  "confidence": 0.0 to 1.0 (your confidence in the prediction),
  "reasoning_chain": "Step-by-step legal reasoning. Walk through: (a) what the law says, (b) how the facts map to the law, (c) what similar cases decided, (d) your conclusion.",
  "supporting_factors": ["factor 1 that supports a favorable outcome", ...],
  "risk_factors": ["factor 1 that could lead to an unfavorable outcome", ...],
  "applicable_charges": ["specific legal provision 1", "specific legal provision 2", ...],
  "recommended_legal_basis": "The single strongest legal argument the plaintiff should make"
}

Be honest about uncertainty. If the facts are insufficient or the law is unclear,
reflect this in the confidence score and predicted_outcome.

Return ONLY the JSON object.
""".strip()


# ── Precedent loader ───────────────────────────────────────────────────────────

# Possible field names in cases_augmented.json (try each in order)
_TEXT_FIELDS    = ["facts_summary", "facts", "text", "description", "case_text",
                   "summary", "narrative", "content", "case_summary"]
_DOMAIN_FIELDS  = ["domain", "case_domain", "category", "type"]
_OUTCOME_FIELDS = ["outcome", "label", "result", "decision",
                   "favorable", "verdict"]


def _resolve_field(record: dict, candidates: list) -> Optional[str]:
    """Return the value of the first matching field name, or None."""
    for key in candidates:
        if key in record:
            return str(record[key])
    return None


def load_precedents(cases_path: str, domain: str, n: int = 4) -> List[Dict]:
    """
    Load n representative past cases from the Phase 1 annotated dataset,
    filtered to match `domain`.

    Returns a list of dicts:  {text, domain, outcome}
    """
    path = Path(cases_path)
    if not path.exists():
        logger.warning(f"Cases file not found: {path}")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict-wrapped formats
    if isinstance(data, dict):
        cases = data.get("cases", data.get("data", list(data.values())[0]
                         if data else []))
    else:
        cases = data

    if not isinstance(cases, list):
        logger.warning("Unexpected cases_augmented.json format — no precedents loaded")
        return []

    # Filter by domain
    domain_lower  = domain.lower() if domain else ""
    domain_cases  = []
    for c in cases:
        c_domain = (_resolve_field(c, _DOMAIN_FIELDS) or "").lower()
        if domain_lower and c_domain != domain_lower:
            continue
        text    = _resolve_field(c, _TEXT_FIELDS)
        outcome = _resolve_field(c, _OUTCOME_FIELDS)
        if text:
            domain_cases.append({
                "text":    text[:400],     # cap length for prompt size
                "domain":  c_domain or domain_lower,
                "outcome": outcome or "unknown",
            })

    if not domain_cases:
        # Fall back to all cases if domain filter yields nothing
        logger.warning(
            f"No cases found for domain='{domain}', "
            "using random sample from full dataset"
        )
        domain_cases = [
            {
                "text":    _resolve_field(c, _TEXT_FIELDS)[:400] or "",
                "domain":  (_resolve_field(c, _DOMAIN_FIELDS) or "").lower(),
                "outcome": _resolve_field(c, _OUTCOME_FIELDS) or "unknown",
            }
            for c in cases
            if _resolve_field(c, _TEXT_FIELDS)
        ]

    # Return n cases: aim for balanced mix of outcomes
    favorable   = [c for c in domain_cases if "fav" in c["outcome"].lower()]
    unfavorable = [c for c in domain_cases if "unfav" in c["outcome"].lower() or
                                               "not" in c["outcome"].lower()]
    other       = [c for c in domain_cases if c not in favorable + unfavorable]

    sample = []
    half   = max(1, n // 2)
    sample += random.sample(favorable,   min(half,     len(favorable)))
    sample += random.sample(unfavorable, min(n - half, len(unfavorable)))
    if len(sample) < n:
        sample += random.sample(other, min(n - len(sample), len(other)))
    sample = sample[:n]

    logger.info(
        f"Loaded {len(sample)} precedent cases "
        f"(domain={domain}, total_pool={len(domain_cases)})"
    )
    return sample


def _format_precedents(cases: List[Dict]) -> str:
    if not cases:
        return "(no similar past cases available)"
    lines = []
    for i, c in enumerate(cases, 1):
        outcome = c.get("outcome", "unknown").upper()
        lines.append(
            f"Case {i} | Domain: {c.get('domain','?')} | Outcome: {outcome}\n"
            f"{c.get('text','')}\n"
        )
    return "\n".join(lines)


def _format_statutes(statutes: list) -> str:
    if not statutes:
        return "(no applicable statutes identified)"
    lines = []
    for s in statutes:
        act  = s.get("act_name", "Unknown Act")
        sec  = s.get("section_no", "")
        rel  = s.get("relevance", "")
        lines.append(f"• {act} §{sec} — {rel}")
    return "\n".join(lines)


# ── Agent class ────────────────────────────────────────────────────────────────

class LegalJudgmentAgent(GeminiAgent):
    """
    Agent 3 — Legal Judgment Prediction Agent.

    Predicts the likely case outcome using statutes + precedents.

    Parameters
    ----------
    cases_path : path to cases_augmented.json (Phase 1 output)
    model_name : Gemini model
    n_precedents : number of similar past cases to include (default 4)
    """

    def __init__(
        self,
        cases_path:   str = "data/annotated/cases_augmented.json",
        model_name:   str = None,
        n_precedents: int = 4,
    ):
        super().__init__(model_name=model_name, temperature=0.2)
        self.cases_path   = cases_path
        self.n_precedents = n_precedents

    def predict(self, enriched_dict: dict) -> dict:
        """
        Predict the likely outcome of the case.

        Parameters
        ----------
        enriched_dict : output of KnowledgeIntegrationLayer.enrich()

        Returns
        -------
        dict — enriched_dict extended with:
               similar_cases, predicted_outcome, confidence,
               reasoning_chain, supporting_factors, risk_factors,
               applicable_charges, recommended_legal_basis
        """
        logger.info("Agent 3 — predicting judgment …")

        domain    = enriched_dict.get("domain", "other")
        precedents = load_precedents(self.cases_path, domain, self.n_precedents)

        prompt = f"""{_SYSTEM_PROMPT}

---
CASE FACTS:
Domain  : {enriched_dict.get('domain', 'unknown')}
Summary : {enriched_dict.get('facts_summary', '')}
Claims  : {'; '.join(enriched_dict.get('key_claims', []))}
Parties : {enriched_dict.get('parties', {})}

APPLICABLE STATUTES:
{_format_statutes(enriched_dict.get('applicable_statutes', []))}

LEGAL FRAMEWORK:
{enriched_dict.get('legal_framework', '')}

KEY LEGAL ISSUES:
{chr(10).join('• ' + i for i in enriched_dict.get('key_legal_issues', []))}

SIMILAR PAST CASES (PRECEDENTS):
{_format_precedents(precedents)}
---

Analyse the case and return the JSON prediction now."""

        prediction = self._call_json(prompt)

        # Merge into analysis dict
        analysis = dict(enriched_dict)
        analysis["similar_cases"]           = precedents
        analysis["predicted_outcome"]       = prediction.get("predicted_outcome",       "uncertain")
        analysis["confidence"]              = float(prediction.get("confidence",        0.5))
        analysis["reasoning_chain"]         = prediction.get("reasoning_chain",         "")
        analysis["supporting_factors"]      = prediction.get("supporting_factors",      [])
        analysis["risk_factors"]            = prediction.get("risk_factors",            [])
        analysis["applicable_charges"]      = prediction.get("applicable_charges",      [])
        analysis["recommended_legal_basis"] = prediction.get("recommended_legal_basis", "")

        logger.info(
            f"Agent 3 done | outcome={analysis['predicted_outcome']} | "
            f"confidence={analysis['confidence']:.2f}"
        )
        return analysis

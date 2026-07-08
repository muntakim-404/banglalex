"""
BanglaLex — Evaluation Baselines
===================================
Two baselines for comparison against BanglaLex full system:

  MajorityClassBaseline  → always predicts the most common outcome per domain.
                           Represents the trivial lower bound.

  LLMOnlyBaseline        → predicts using only the LLM + case facts.
                           No RAG retrieval. No precedents.
                           Shows the isolated contribution of the knowledge base.
"""

import time
import json
import logging
import random
from collections import defaultdict, Counter
from pathlib     import Path
from typing      import Dict, List, Optional

from .metrics import normalize_outcome

logger = logging.getLogger(__name__)

_TEXT_FIELDS    = ["facts_summary", "facts", "text", "description",
                   "case_text", "summary", "narrative", "content"]
_DOMAIN_FIELDS  = ["domain", "case_domain", "category", "type"]
_OUTCOME_FIELDS = ["outcome", "label", "result", "decision", "verdict"]


def _get(record: dict, candidates: list) -> Optional[str]:
    for k in candidates:
        if k in record:
            return str(record[k])
    return None


# ── Majority class baseline ────────────────────────────────────────────────────

class MajorityClassBaseline:
    """
    Predicts the most common outcome in the training data per domain.
    No API calls — instant.

    Parameters
    ----------
    cases_path : Phase 1 annotated cases JSON (used to compute majority class)
    n_per_domain : cases to evaluate per domain (must match BanglaLexEvaluator)
    """

    def __init__(
        self,
        cases_path:   str = "data/annotated/cases_augmented.json",
        n_per_domain: int = 50,
    ):
        self.cases_path   = cases_path
        self.n_per_domain = n_per_domain
        self._majority: Dict[str, str] = {}

    def _compute_majority(self, all_cases: list) -> Dict[str, str]:
        """Compute the majority outcome per domain."""
        domain_outcomes: Dict[str, list] = defaultdict(list)
        for c in all_cases:
            domain  = (_get(c, _DOMAIN_FIELDS) or "other").lower()
            outcome = normalize_outcome(_get(c, _OUTCOME_FIELDS) or "")
            if outcome:
                domain_outcomes[domain].append(outcome)
        return {
            d: Counter(outcomes).most_common(1)[0][0]
            for d, outcomes in domain_outcomes.items()
        }

    def run(self, sample_cases: List[Dict]) -> List[Dict]:
        """
        Run majority class prediction over sample_cases.
        sample_cases must be the same list used by BanglaLexEvaluator.run().
        """
        # Load full dataset to compute majority class
        with open(self.cases_path, encoding="utf-8") as f:
            data = json.load(f)
        all_cases = data if isinstance(data, list) else list(data.values())[0]
        self._majority = self._compute_majority(all_cases)

        logger.info(f"Majority class per domain: {self._majority}")

        results = []
        for case in sample_cases:
            citation    = case.get("citation", str(id(case)))
            domain      = (_get(case, _DOMAIN_FIELDS) or "other").lower()
            raw_outcome = _get(case, _OUTCOME_FIELDS) or ""
            ground_truth = normalize_outcome(raw_outcome)
            if not ground_truth:
                continue

            predicted = self._majority.get(domain, "favorable")
            results.append({
                "citation":     citation,
                "domain":       domain,
                "ground_truth": ground_truth,
                "predicted":    predicted,
                "confidence":   1.0,
                "correct":      predicted == ground_truth,
            })

        logger.info(f"Majority baseline done: {len(results)} results")
        return results


# ── LLM-only baseline ──────────────────────────────────────────────────────────

_LLM_ONLY_PROMPT = """You are a legal judgment prediction expert for Bangladesh courts.
Based ONLY on the case facts below (no statutes, no precedents),
predict the likely outcome of this case.

Domain: {domain}
Case Facts: {facts}

Return a JSON object with exactly these keys:
{{
  "predicted_outcome": "favorable" | "unfavorable" | "uncertain",
  "confidence": 0.0 to 1.0
}}

Return ONLY the JSON object."""


class LLMOnlyBaseline:
    """
    Predicts outcome using only the LLM and case facts.
    No RAG retrieval. No precedent cases.

    This baseline isolates the contribution of the knowledge base
    and precedent augmentation in BanglaLex.

    Parameters
    ----------
    model_name  : Groq model
    call_delay  : seconds between API calls
    """

    def __init__(
        self,
        model_name:  str   = None,
        call_delay:  float = 2.0,
    ):
        self.call_delay = call_delay
        self._agent     = None
        self._model_name = model_name

    def _get_agent(self):
        if self._agent is None:
            from src.agents.base import GeminiAgent
            self._agent = GeminiAgent(model_name=self._model_name, temperature=0.1)
        return self._agent

    def run(self, sample_cases: List[Dict]) -> List[Dict]:
        """
        Run LLM-only prediction over sample_cases.
        sample_cases must be the same list used by BanglaLexEvaluator.run().
        """
        from tqdm import tqdm

        agent   = self._get_agent()
        results = []
        skipped = 0

        for case in tqdm(sample_cases, desc="LLM-only baseline", unit="case"):
            citation    = case.get("citation", str(id(case)))
            domain      = (_get(case, _DOMAIN_FIELDS) or "other").lower()
            facts       = _get(case, _TEXT_FIELDS) or ""
            raw_outcome = _get(case, _OUTCOME_FIELDS) or ""
            ground_truth = normalize_outcome(raw_outcome)

            if not ground_truth or not facts:
                skipped += 1
                continue

            prompt = _LLM_ONLY_PROMPT.format(
                domain = domain,
                facts  = facts[:400],
            )

            try:
                response   = agent._call_json(prompt)
                predicted  = response.get("predicted_outcome", "uncertain")
                confidence = float(response.get("confidence", 0.5))
                results.append({
                    "citation":     citation,
                    "domain":       domain,
                    "ground_truth": ground_truth,
                    "predicted":    predicted,
                    "confidence":   round(confidence, 3),
                    "correct":      predicted == ground_truth,
                })
            except Exception as exc:
                logger.warning(f"LLM-only baseline error for {citation}: {exc}")
                skipped += 1

            if self.call_delay > 0:
                time.sleep(self.call_delay)

        logger.info(
            f"LLM-only baseline done: {len(results)} results, {skipped} skipped"
        )
        return results

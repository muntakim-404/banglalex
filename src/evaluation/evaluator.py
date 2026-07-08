"""
BanglaLex — Evaluation Runner  (v2: all 4 fixes)
===================================================
Fix 1: Anti-abstention judgment prompt — forces a prediction unless facts
        are completely absent. Reduces uncertain rate from ~12% to ~3%.

Fix 2: Similarity-based precedent selection — embeds the case facts and
        finds the 4 most semantically similar precedents (instead of random
        domain-matched sampling). Uses the existing FAISS embedder.

Fix 3: Agent 2 LLM statute filtering — after FAISS retrieval, runs a fast
        LLM call to identify which retrieved sections are actually applicable,
        giving Agent 3 curated context instead of raw FAISS chunks.

Fix 4: Applied at script level — use llama-3.3-70b-versatile with 25 cases/domain.
"""

import json
import random
import logging
import time
from collections import defaultdict
from pathlib     import Path
from typing      import Dict, List, Optional

import numpy as np

from .metrics import normalize_outcome

logger = logging.getLogger(__name__)

# ── Field name candidates ──────────────────────────────────────────────────────
_TEXT_FIELDS    = ["facts_summary", "facts", "text", "description",
                   "case_text", "summary", "narrative", "content"]
_DOMAIN_FIELDS  = ["domain", "case_domain", "category", "type"]
_OUTCOME_FIELDS = ["outcome", "label", "result", "decision", "verdict"]


def _get(record: dict, candidates: list) -> Optional[str]:
    for k in candidates:
        if k in record:
            return str(record[k])
    return None


# ── Fix 1: Anti-abstention evaluation judgment prompt ─────────────────────────
_EVAL_JUDGMENT_PROMPT = """You are a Bangladesh court judgment prediction expert.

CASE FACTS:
Domain  : {domain}
Summary : {facts}

APPLICABLE STATUTES:
{statutes}

SIMILAR PAST CASES (for analogical reasoning):
{precedents}

Based on the above, predict whether the outcome will be FAVORABLE or UNFAVORABLE
for the petitioner / plaintiff in a Bangladesh court.

CRITICAL RULE: You MUST output "favorable" or "unfavorable".
Reserve "uncertain" ONLY if the case facts are completely absent or
directly contradictory. Low confidence is NOT a reason for "uncertain" —
commit to your best prediction and reflect uncertainty in the confidence score.

Return ONLY this JSON:
{{
  "predicted_outcome": "favorable" | "unfavorable" | "uncertain",
  "confidence": 0.0 to 1.0,
  "reasoning_chain": "2-3 sentences citing the key statute and how it applies",
  "supporting_factors": ["factor 1"],
  "risk_factors": ["risk 1"]
}}"""

# ── Fix 3: Agent 2 statute filtering prompt ────────────────────────────────────
_STATUTE_FILTER_PROMPT = """Legal researcher task for Bangladesh law.
From the retrieved statute sections below, identify the 3 MOST DIRECTLY applicable
to this specific case. Ignore sections that mention similar words but address
unrelated legal issues.

Case domain : {domain}
Case facts  : {facts}

RETRIEVED SECTIONS:
{statutes}

Return ONLY this JSON (no preamble):
{{"applicable": [
  {{"act_name": "...", "section_no": "...", "section_title": "...", "relevance": "1 sentence"}}
]}}"""


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_statutes_raw(retrieved: list) -> str:
    if not retrieved:
        return "(none retrieved)"
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(
            f"[{i}] {r.get('act_name','')} §{r.get('section_no','')}"
            + (f" [{r.get('section_title','')}]" if r.get('section_title') else "")
            + f"\n    {r.get('text','')[:200]}"
        )
    return "\n".join(lines)


def _fmt_statutes_applicable(applicable: list) -> str:
    if not applicable:
        return "(none identified)"
    lines = []
    for s in applicable:
        lines.append(
            f"• {s.get('act_name','')} §{s.get('section_no','')}"
            + (f" [{s.get('section_title','')}]" if s.get('section_title') else "")
            + f" — {s.get('relevance','')}"
        )
    return "\n".join(lines)


def _fmt_precedents(cases: list) -> str:
    if not cases:
        return "(no similar cases available)"
    lines = []
    for i, c in enumerate(cases, 1):
        outcome = c.get("outcome", "unknown").upper()
        lines.append(
            f"Case {i} | Outcome: {outcome}\n"
            f"  {c.get('text','')[:250]}"
        )
    return "\n".join(lines)


# ── Fix 2: Similarity-based precedent selection ────────────────────────────────

def _select_similar_precedents(
    case_facts:         str,
    pool:               List[Dict],
    n:                  int,
    embedder,
    exclude_citation:   Optional[str] = None,
) -> List[Dict]:
    """
    Select n precedents most semantically similar to case_facts.
    Aims for a balanced mix of favorable / unfavorable outcomes.
    Falls back to random if embedder is unavailable.
    """
    candidates = [
        c for c in pool
        if c.get("citation") != exclude_citation
        and _get(c, _TEXT_FIELDS)
    ]
    if not candidates:
        return []

    if embedder is None:
        # Fallback: random balanced selection
        fav   = [c for c in candidates if "fav" in (_get(c, _OUTCOME_FIELDS) or "").lower()
                 and "unfav" not in (_get(c, _OUTCOME_FIELDS) or "").lower()]
        unfav = [c for c in candidates
                 if any(k in (_get(c, _OUTCOME_FIELDS) or "").lower()
                        for k in ["unfav", "dismissed", "against"])]
        half  = max(1, n // 2)
        picks = (random.sample(fav, min(half, len(fav))) +
                 random.sample(unfav, min(n - half, len(unfav))))
        return [{"text": _get(c, _TEXT_FIELDS)[:300],
                 "domain": (_get(c, _DOMAIN_FIELDS) or "").lower(),
                 "outcome": _get(c, _OUTCOME_FIELDS) or "unknown"} for c in picks[:n]]

    # Embed all candidates (use short snippets for speed)
    texts     = [(_get(c, _TEXT_FIELDS) or "")[:200] for c in candidates]
    pool_vecs = embedder.embed(texts, show_progress=False)           # (N, 384)
    q_vec     = embedder.embed_query(case_facts[:200])               # (1, 384)
    scores    = (pool_vecs @ q_vec.T).flatten()                      # cosine sim

    sorted_idx = np.argsort(scores)[::-1]

    # Balanced selection from top-ranked
    fav_picks   = []
    unfav_picks = []
    half = max(1, n // 2)

    for idx in sorted_idx:
        c       = candidates[idx]
        outcome = (_get(c, _OUTCOME_FIELDS) or "").lower()
        item    = {
            "text":    (_get(c, _TEXT_FIELDS) or "")[:300],
            "domain":  (_get(c, _DOMAIN_FIELDS) or "").lower(),
            "outcome": _get(c, _OUTCOME_FIELDS) or "unknown",
        }
        if "fav" in outcome and "unfav" not in outcome:
            if len(fav_picks) < half:
                fav_picks.append(item)
        elif any(k in outcome for k in ["unfav", "dismissed", "against", "rejected"]):
            if len(unfav_picks) < (n - half):
                unfav_picks.append(item)

        if len(fav_picks) + len(unfav_picks) >= n:
            break

    result = fav_picks + unfav_picks
    # Pad with top-ranked if not enough balanced samples
    if len(result) < n:
        for idx in sorted_idx:
            c = candidates[idx]
            item = {
                "text":    (_get(c, _TEXT_FIELDS) or "")[:300],
                "domain":  (_get(c, _DOMAIN_FIELDS) or "").lower(),
                "outcome": _get(c, _OUTCOME_FIELDS) or "unknown",
            }
            if item not in result:
                result.append(item)
            if len(result) >= n:
                break

    logger.debug(f"Similarity precedents: {len(result)} selected from pool of {len(candidates)}")
    return result[:n]


# ── Main evaluator ─────────────────────────────────────────────────────────────

class BanglaLexEvaluator:
    """
    Streamlined BanglaLex evaluation with all 4 fixes applied.

    Parameters
    ----------
    kb_dir, cases_path, model_name : same as v1
    n_per_domain                   : cases per domain (default 25 with 70b model)
    call_delay                     : seconds between API calls
    use_agent2_filtering           : if True, run Agent 2 LLM to curate statutes (Fix 3)
    """

    def __init__(
        self,
        kb_dir:               str   = "data/knowledge_base",
        cases_path:           str   = "data/annotated/cases_augmented.json",
        model_name:           str   = None,
        n_per_domain:         int   = 25,
        call_delay:           float = 2.0,
        use_agent2_filtering: bool  = True,
    ):
        self.kb_dir               = kb_dir
        self.cases_path           = cases_path
        self.n_per_domain         = n_per_domain
        self.call_delay           = call_delay
        self.use_agent2_filtering = use_agent2_filtering
        self._model_name          = model_name

        self._retriever  = None
        self._llm_agent  = None
        self._all_cases  = None

    # ── Resource loaders ───────────────────────────────────────────────────────

    def _get_retriever(self):
        if self._retriever is None:
            from src.knowledge_base.retriever import Retriever
            logger.info("Loading FAISS retriever …")
            self._retriever = Retriever.from_saved(self.kb_dir)
        return self._retriever

    def _get_embedder(self):
        """Reuse the embedder from the retriever (no extra memory cost)."""
        return self._get_retriever().embedder

    def _get_llm_agent(self):
        if self._llm_agent is None:
            from src.agents.base import GeminiAgent
            self._llm_agent = GeminiAgent(
                model_name  = self._model_name,
                temperature = 0.05,
            )
        return self._llm_agent

    def _load_all_cases(self) -> List[Dict]:
        if self._all_cases is None:
            with open(self.cases_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._all_cases = data
            elif isinstance(data, dict):
                for key in ["cases", "data", "items"]:
                    if key in data:
                        self._all_cases = data[key]
                        break
                else:
                    self._all_cases = list(data.values())[0]
            logger.info(f"Loaded {len(self._all_cases)} annotated cases")
        return self._all_cases

    # ── Sampling ───────────────────────────────────────────────────────────────

    def _sample_cases(self) -> List[Dict]:
        all_cases = self._load_all_cases()
        by_domain: Dict[str, list] = defaultdict(list)
        for case in all_cases:
            domain  = (_get(case, _DOMAIN_FIELDS) or "").lower()
            outcome = normalize_outcome(_get(case, _OUTCOME_FIELDS) or "")
            text    = _get(case, _TEXT_FIELDS)
            if domain and outcome and text:
                by_domain[domain].append(case)

        sampled = []
        for domain, cases in by_domain.items():
            n      = min(self.n_per_domain, len(cases))
            chosen = random.sample(cases, n)
            sampled.extend(chosen)
            logger.info(f"  {domain}: {n} sampled (pool={len(cases)})")
        logger.info(f"  TOTAL: {len(sampled)}")
        return sampled

    # ── Fix 3: Agent 2 statute filtering ──────────────────────────────────────

    def _filter_statutes_with_agent2(
        self,
        facts:     str,
        domain:    str,
        retrieved: list,
    ) -> list:
        """Run a fast LLM call to curate which retrieved statutes apply."""
        if not retrieved:
            return []
        agent  = self._get_llm_agent()
        prompt = _STATUTE_FILTER_PROMPT.format(
            domain   = domain,
            facts    = facts[:250],
            statutes = _fmt_statutes_raw(retrieved),
        )
        try:
            result     = agent._call_json(prompt)
            applicable = result.get("applicable", [])
            return applicable if applicable else [
                {"act_name":      r.get("act_name", ""),
                 "section_no":    r.get("section_no", ""),
                 "section_title": r.get("section_title", ""),
                 "relevance":     "retrieved as relevant"}
                for r in retrieved[:3]
            ]
        except Exception as e:
            logger.warning(f"Agent 2 filtering failed: {e} — using raw retrieval")
            return [
                {"act_name":      r.get("act_name", ""),
                 "section_no":    r.get("section_no", ""),
                 "section_title": r.get("section_title", ""),
                 "relevance":     "retrieved as relevant"}
                for r in retrieved[:3]
            ]

    # ── Core evaluation ────────────────────────────────────────────────────────

    def _evaluate_single(
        self,
        case:      Dict,
        all_cases: List[Dict],
    ) -> Optional[Dict]:
        citation    = case.get("citation", str(id(case)))
        domain      = (_get(case, _DOMAIN_FIELDS) or "other").lower()
        facts       = _get(case, _TEXT_FIELDS) or ""
        raw_outcome = _get(case, _OUTCOME_FIELDS) or ""
        ground_truth = normalize_outcome(raw_outcome)

        if not ground_truth or not facts:
            return None

        # ── FAISS retrieval ────────────────────────────────────────────────
        retriever  = self._get_retriever()
        query      = f"{facts[:300]} legal case domain: {domain}"
        retrieved  = retriever.retrieve_for_case(query, domain=domain, k=6)

        # ── Fix 3: Agent 2 LLM filtering ──────────────────────────────────
        if self.use_agent2_filtering:
            applicable = self._filter_statutes_with_agent2(facts, domain, retrieved)
        else:
            applicable = [
                {"act_name":      r.get("act_name", ""),
                 "section_no":    r.get("section_no", ""),
                 "section_title": r.get("section_title", ""),
                 "relevance":     "retrieved as relevant"}
                for r in retrieved[:3]
            ]

        # ── Fix 2: Similarity-based precedents ────────────────────────────
        domain_pool = [
            c for c in all_cases
            if (_get(c, _DOMAIN_FIELDS) or "").lower() == domain
            and c.get("citation") != citation
            and _get(c, _TEXT_FIELDS)
        ]
        try:
            embedder   = self._get_embedder()
        except Exception:
            embedder   = None

        precedents = _select_similar_precedents(
            facts, domain_pool, n=4,
            embedder=embedder,
            exclude_citation=citation,
        )

        # ── Fix 1: Anti-abstention judgment call ──────────────────────────
        agent  = self._get_llm_agent()
        prompt = _EVAL_JUDGMENT_PROMPT.format(
            domain     = domain,
            facts      = facts[:300],
            statutes   = _fmt_statutes_applicable(applicable),
            precedents = _fmt_precedents(precedents),
        )

        try:
            result     = agent._call_json(prompt)
            predicted  = result.get("predicted_outcome", "uncertain")
            confidence = float(result.get("confidence", 0.5))
        except Exception as e:
            logger.warning(f"Judgment call failed: {e}")
            return None

        return {
            "citation":     citation,
            "domain":       domain,
            "ground_truth": ground_truth,
            "predicted":    predicted,
            "confidence":   round(confidence, 3),
            "correct":      predicted == ground_truth,
        }

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        from tqdm import tqdm
        sample    = self._sample_cases()
        all_cases = self._load_all_cases()
        results   = []
        skipped   = 0

        for case in tqdm(sample, desc="Evaluating", unit="case"):
            try:
                result = self._evaluate_single(case, all_cases)
                if result:
                    results.append(result)
                else:
                    skipped += 1
            except Exception as exc:
                logger.warning(f"Case skipped: {exc}")
                skipped += 1
            if self.call_delay > 0:
                time.sleep(self.call_delay)

        logger.info(f"Done: {len(results)} results, {skipped} skipped")
        return results

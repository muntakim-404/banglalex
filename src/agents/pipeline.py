"""
BanglaLex — Pipeline Orchestrator
====================================
Ties all four agents together into a single callable pipeline.

Usage
-----
  from src.agents import BanglaLexPipeline

  pipeline = BanglaLexPipeline(
      kb_dir     = "data/knowledge_base",
      cases_path = "data/annotated/cases_augmented.json",
  )

  # Silent run — returns full result dict
  result = pipeline.run("My employer has not paid my salary for 3 months.")

  # Verbose run — prints each stage as it completes
  result = pipeline.run_verbose("আমার বাড়িওয়ালা নোটিশ ছাড়া তালা বদলে দিয়েছে।")
"""

import time
import logging
from typing import Optional

from .consultation          import ClientConsultationAgent
from .knowledge_integration import KnowledgeIntegrationLayer
from .judgment              import LegalJudgmentAgent
from .output                import OutputAgent

logger = logging.getLogger(__name__)

# ── Confidence label ───────────────────────────────────────────────────────────

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.50:
        return "MODERATE"
    else:
        return "LOW"


# ── Pipeline ───────────────────────────────────────────────────────────────────

class BanglaLexPipeline:
    """
    End-to-end BanglaLex pipeline.

    Parameters
    ----------
    kb_dir       : path to Phase 2 FAISS knowledge base
    cases_path   : path to Phase 1 annotated cases JSON
    model_name   : Gemini model used by all agents (default gemini-2.0-flash)
    k_statutes   : number of statute sections to retrieve per query (default 8)
    n_precedents : number of precedent cases to include (default 4)
    """

    def __init__(
        self,
        kb_dir:       str = "data/knowledge_base",
        cases_path:   str = "data/annotated/cases_augmented.json",
        model_name:   str = None,
        k_statutes:   int = 8,
        n_precedents: int = 4,
    ):
        logger.info("Initialising BanglaLex pipeline …")
        self.agent1 = ClientConsultationAgent(model_name=model_name)
        self.agent2 = KnowledgeIntegrationLayer(
            kb_dir=kb_dir, model_name=model_name, k=k_statutes
        )
        self.agent3 = LegalJudgmentAgent(
            cases_path=cases_path, model_name=model_name, n_precedents=n_precedents
        )
        self.agent4 = OutputAgent(model_name=model_name)
        logger.info("Pipeline ready.")

    # ── Core run ───────────────────────────────────────────────────────────────

    def run(self, user_query: str) -> dict:
        """
        Run the full pipeline silently.

        Parameters
        ----------
        user_query : client's legal question in English or Bangla

        Returns
        -------
        dict with keys:
            query, stage_1_consultation, stage_2_knowledge,
            stage_3_judgment, stage_4_output, elapsed_seconds
        """
        t_start = time.time()

        case_dict = self.agent1.process(user_query)
        enriched  = self.agent2.enrich(case_dict)
        analysis  = self.agent3.predict(enriched)
        output    = self.agent4.generate(analysis)

        return {
            "query":               user_query,
            "stage_1_consultation": {
                k: case_dict[k] for k in
                ["language", "domain", "parties", "location", "dates",
                 "facts_summary", "key_claims", "urgency", "missing_info"]
                if k in case_dict
            },
            "stage_2_knowledge": {
                "applicable_statutes": enriched.get("applicable_statutes", []),
                "legal_framework":     enriched.get("legal_framework",     ""),
                "key_legal_issues":    enriched.get("key_legal_issues",    []),
                "retrieved_count":     len(enriched.get("retrieved_statutes", [])),
            },
            "stage_3_judgment": {
                "predicted_outcome":       analysis.get("predicted_outcome"),
                "confidence":              analysis.get("confidence"),
                "reasoning_chain":         analysis.get("reasoning_chain"),
                "supporting_factors":      analysis.get("supporting_factors",      []),
                "risk_factors":            analysis.get("risk_factors",            []),
                "applicable_charges":      analysis.get("applicable_charges",      []),
                "recommended_legal_basis": analysis.get("recommended_legal_basis", ""),
                "similar_cases_used":      len(analysis.get("similar_cases",       [])),
            },
            "stage_4_output":   output,
            "elapsed_seconds":  round(time.time() - t_start, 1),
        }

    # ── Verbose run ────────────────────────────────────────────────────────────

    def run_verbose(self, user_query: str) -> dict:
        """
        Run the pipeline and print each stage as it completes.
        Designed for the interactive CLI demo.
        """
        SEP   = "─" * 65
        DSEP  = "═" * 65
        print(f"\n{DSEP}")
        print(" BanglaLex — Legal Analysis Pipeline")
        print(DSEP)
        print(f" Query: {user_query[:80]}{'…' if len(user_query) > 80 else ''}")
        print(DSEP)

        t_start = time.time()

        # ── Stage 1 ──────────────────────────────────────────────────────────
        print("\n[1/4] Client Consultation Agent — analysing query …", flush=True)
        case_dict = self.agent1.process(user_query)
        print(f"  ✓  Domain   : {case_dict.get('domain','?').upper()}")
        print(f"     Language : {case_dict.get('language','?').upper()}")
        print(f"     Urgency  : {case_dict.get('urgency','?').upper()}")
        print(f"     Summary  : {case_dict.get('facts_summary','')[:120]}")
        if case_dict.get("missing_info"):
            print(f"     Missing  : {'; '.join(case_dict['missing_info'][:2])}")

        # ── Stage 2 ──────────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("[2/4] Knowledge Integration Layer — retrieving statutes …",
              flush=True)
        enriched = self.agent2.enrich(case_dict)
        print(f"  ✓  Retrieved   : {len(enriched.get('retrieved_statutes',[]))} sections")
        print(f"     Applicable  : {len(enriched.get('applicable_statutes',[]))} sections")
        print(f"     Framework   : {enriched.get('legal_framework','')[:120]}")
        if enriched.get("applicable_statutes"):
            print("     Key statutes:")
            for s in enriched["applicable_statutes"][:3]:
                print(f"       • {s.get('act_name','')}  §{s.get('section_no','')}")

        # ── Stage 3 ──────────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("[3/4] Legal Judgment Agent — predicting outcome …", flush=True)
        analysis = self.agent3.predict(enriched)
        outcome    = analysis.get("predicted_outcome", "uncertain").upper()
        confidence = analysis.get("confidence", 0.5)
        pct        = round(confidence * 100)
        label      = _confidence_label(confidence)
        print(f"  ✓  Prediction  : {outcome}  ({pct}% — {label} confidence)")
        print(f"     Precedents  : {len(analysis.get('similar_cases',[]))} cases used")
        if analysis.get("supporting_factors"):
            print(f"     Strengths   : {analysis['supporting_factors'][0][:80]}")
        if analysis.get("risk_factors"):
            print(f"     Risks       : {analysis['risk_factors'][0][:80]}")

        # ── Stage 4 ──────────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("[4/4] Output Agent — generating client explanation …", flush=True)
        output = self.agent4.generate(analysis)

        elapsed = round(time.time() - t_start, 1)

        # ── Final client-facing output ─────────────────────────────────────
        lang = output.get("language", "en")
        print(f"\n{DSEP}")
        header = " CLIENT-FACING EXPLANATION"
        if lang == "bn":
            header += "  (Bangla)"
        print(header)
        print(DSEP)
        print(f"\n  {output.get('probability_statement','')}")
        print(f"\n  {output.get('plain_reasoning','')}")

        steps = output.get("next_steps", [])
        if steps:
            print("\n  NEXT STEPS:")
            for step in steps:
                print(f"    {step}")

        evidence = output.get("evidence_to_collect", [])
        if evidence:
            print("\n  EVIDENCE TO COLLECT:")
            for e in evidence:
                print(f"    • {e}")

        urgency_note = output.get("urgency_note")
        if urgency_note:
            print(f"\n  ⚠  {urgency_note}")

        print(f"\n{DSEP}")
        print(f"  Pipeline complete in {elapsed}s")
        print(DSEP)

        return {
            "query":                user_query,
            "stage_1_consultation": case_dict,
            "stage_2_knowledge":    enriched,
            "stage_3_judgment":     analysis,
            "stage_4_output":       output,
            "elapsed_seconds":      elapsed,
        }

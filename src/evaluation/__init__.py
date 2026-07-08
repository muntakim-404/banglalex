"""
BanglaLex — Evaluation Package (Phase 4)
==========================================
Tools for evaluating judgment prediction accuracy.

  metrics.py   → accuracy, F1, confusion matrix, report generation
  evaluator.py → BanglaLex evaluation runner (streamlined pipeline)
  baseline.py  → majority class and LLM-only baselines
"""

from .metrics   import normalize_outcome, compute_metrics, print_report, save_results
from .evaluator import BanglaLexEvaluator
from .baseline  import MajorityClassBaseline, LLMOnlyBaseline

__all__ = [
    "normalize_outcome",
    "compute_metrics",
    "print_report",
    "save_results",
    "BanglaLexEvaluator",
    "MajorityClassBaseline",
    "LLMOnlyBaseline",
]

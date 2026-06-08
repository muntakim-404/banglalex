"""
BanglaLex — Agents Package (Phase 3)
======================================
Four-agent pipeline for legal assistance in Bangladesh.

  Agent 1  ClientConsultationAgent      → structured case representation
  Agent 2  KnowledgeIntegrationLayer    → RAG-enriched case with statutes
  Agent 3  LegalJudgmentAgent           → outcome prediction + reasoning
  Agent 4  OutputAgent                  → plain-language client explanation

Typical usage
-------------
  from src.agents import BanglaLexPipeline

  pipeline = BanglaLexPipeline(
      kb_dir     = "data/knowledge_base",
      cases_path = "data/annotated/cases_augmented.json",
  )
  result = pipeline.run("আমার বাড়িওয়ালা কোনো নোটিশ ছাড়াই তালা বদলে দিয়েছে।")
"""

from .consultation          import ClientConsultationAgent
from .knowledge_integration import KnowledgeIntegrationLayer
from .judgment              import LegalJudgmentAgent
from .output                import OutputAgent
from .pipeline              import BanglaLexPipeline

__all__ = [
    "ClientConsultationAgent",
    "KnowledgeIntegrationLayer",
    "LegalJudgmentAgent",
    "OutputAgent",
    "BanglaLexPipeline",
]

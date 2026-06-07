"""
BanglaLex — Retriever (RAG Query Interface)
=============================================
Combines the Embedder and KnowledgeBaseIndex into a single, easy-to-use
retrieval interface consumed by the Phase 3 agents.

Primary entry points
--------------------
  Retriever.from_saved(index_dir)        → load from disk (for agents)
  Retriever.retrieve(query, k)           → free-text search
  Retriever.retrieve_for_case(facts, k)  → case-aware search with domain filter

Each result dict contains:
  text, act_name, act_no, act_year, section_no, section_title,
  word_count, window_idx, score, chunk_id
"""

import logging
from pathlib import Path
from typing  import Dict, List, Optional

from .embedder import Embedder, DEFAULT_MODEL
from .indexer  import KnowledgeBaseIndex

logger = logging.getLogger(__name__)

# ── Domain keyword map ─────────────────────────────────────────────────────────
# Maps canonical domain name → list of keywords checked against act_name + text.
# Covers both English act names and Bangla act names so the filter works for
# both EN and BN queries.  Add new domains / keywords here as needed in Phase 3.

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "land": [
        "land", "জমি", "ভূমি", "tenancy", "tenant", "acquisition",
        "তালুক", "সম্পত্তি", "estate", "survey", "settlement",
        "cadastral", "khas", "খাস", "রেজিস্ট্রেশন", "registration",
    ],
    "contract": [
        "contract", "চুক্তি", "labour", "labor", "শ্রম", "শ্রমিক",
        "employment", "নিয়োগ", "wages", "মজুরি", "বেতন",
        "specific relief", "agreement", "compensation",
    ],
    "service": [
        "service", "চাকরি", "সেবা", "employment", "নিয়োগ",
        "government", "সরকারি", "civil", "বেসামরিক",
        "public servant", "জনসেবক", "officer", "কর্মচারী",
        "dismissal", "termination", "বরখাস্ত",
    ],
    "family": [
        "family", "পরিবার", "marriage", "বিবাহ", "divorce", "তালাক",
        "muslim", "মুসলিম", "dissolution", "dower", "মোহর",
        "guardian", "অভিভাবক", "succession", "inheritance", "উত্তরাধিকার",
        "maintenance", "ভরণপোষণ",
    ],
    "criminal": [
        "penal", "criminal", "দণ্ড", "offence", "অপরাধ",
        "punishment", "শাস্তি", "imprisonment", "কারাদণ্ড",
        "police", "পুলিশ", "evidence", "সাক্ষ্য", "procedure",
    ],
}


class Retriever:
    """
    High-level RAG retrieval for BanglaLex.

    Designed to be instantiated once per agent session and reused
    across multiple queries (the embedding model stays in GPU/CPU memory).

    Parameters
    ----------
    kb_index : KnowledgeBaseIndex  (already loaded)
    embedder : Embedder            (already loaded)
    """

    def __init__(self, kb_index: KnowledgeBaseIndex, embedder: Embedder):
        self.kb       = kb_index
        self.embedder = embedder

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        index_dir:  str,
        model_name: str = DEFAULT_MODEL,
    ) -> "Retriever":
        """
        Convenience constructor — loads index + model from disk.

        Parameters
        ----------
        index_dir  : path to the directory written by KnowledgeBaseIndex.save()
        model_name : sentence-transformer model (must match the one used during build)

        Examples
        --------
        >>> retriever = Retriever.from_saved("data/knowledge_base")
        >>> results   = retriever.retrieve("মিথ্যা মামলায় শাস্তি কী?", k=5)
        """
        kb       = KnowledgeBaseIndex.load(Path(index_dir))
        embedder = Embedder(model_name)
        return cls(kb, embedder)

    # ── Core retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:     str,
        k:         int   = 5,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Return the *k* most relevant legal chunks for *query*.

        Works for English or Bangla queries (or mixed).

        Parameters
        ----------
        query     : free-text question or case summary
        k         : number of results
        min_score : minimum cosine similarity (0.0 = include everything)

        Returns
        -------
        List[Dict], sorted by score descending.
        Each dict:  text | act_name | act_no | act_year |
                    section_no | section_title | word_count |
                    window_idx | score | chunk_id
        """
        q_vec           = self.embedder.embed_query(query)
        scores, indices = self.kb.search(q_vec, k=k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:                       # FAISS padding sentinel
                continue
            if float(score) < min_score:
                continue
            chunk = dict(self.kb.chunks[int(idx)])
            chunk["score"] = round(float(score), 6)
            results.append(chunk)

        return results

    # ── Case-aware retrieval ───────────────────────────────────────────────────

    def retrieve_for_case(
        self,
        case_facts: str,
        domain:     Optional[str] = None,
        k:          int           = 8,
        min_score:  float         = 0.0,
    ) -> List[Dict]:
        """
        Retrieve statutes relevant to a client's case narrative.

        Called by the Knowledge Integration Layer with structured case facts
        produced by the Client Consultation Agent.

        Parameters
        ----------
        case_facts : raw or summarised case narrative (English or Bangla)
        domain     : optional legal domain — one of:
                     "land", "contract", "service", "family", "criminal"
                     When provided, results are preferentially filtered to
                     acts whose name OR text contains any of the domain's
                     keywords (checked in both English and Bangla).
        k          : final number of results to return
        min_score  : minimum cosine similarity threshold

        Returns
        -------
        List[Dict] — same structure as retrieve()
        """
        fetch_k = k * 3 if domain else k
        results = self.retrieve(case_facts, k=fetch_k, min_score=min_score)

        if domain:
            # Get keyword list for this domain (fall back to the raw domain
            # string so unknown domains still get basic filtering)
            kws = DOMAIN_KEYWORDS.get(domain.lower(), [domain.lower()])

            def _matches(r: Dict) -> bool:
                haystack = (
                    r.get("act_name", "").lower() + " " +
                    r.get("text",     "").lower()
                )
                return any(kw in haystack for kw in kws)

            filtered = [r for r in results if _matches(r)]

            # Fall back to unfiltered if the domain filter is too aggressive
            # (avoids silently returning nothing)
            results = filtered if len(filtered) >= min(3, k) else results

        return results[:k]

    # ── Multi-query retrieval (Charge Expansion agent) ─────────────────────────

    def retrieve_multi(
        self,
        queries:  List[str],
        k_each:   int  = 3,
        dedupe:   bool = True,
    ) -> List[Dict]:
        """
        Retrieve and merge results for multiple queries.

        Used by the Charge Expansion Module in Phase 3 to search several
        related charge descriptions simultaneously.

        Parameters
        ----------
        queries : list of query strings (English or Bangla)
        k_each  : results to fetch per query before merging
        dedupe  : keep only the highest-scored copy of each chunk

        Returns
        -------
        List[Dict] sorted by score descending.
        """
        seen: Dict[int, Dict] = {}     # chunk_id → best result dict
        for q in queries:
            for r in self.retrieve(q, k=k_each):
                cid = r["chunk_id"]
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r

        return sorted(seen.values(), key=lambda r: r["score"], reverse=True)

    # ── Pretty printer ─────────────────────────────────────────────────────────

    @staticmethod
    def format_results(results: List[Dict], text_preview: int = 280) -> str:
        """
        Human-readable string of retrieval results.
        Used in test scripts and agent debug logging.
        """
        if not results:
            return "  (no results)\n"

        lines = []
        for i, r in enumerate(results, 1):
            act   = r.get("act_name",      "Unknown Act")
            sec   = r.get("section_no",    "—")
            head  = r.get("section_title", "")
            scr   = r.get("score",          0.0)
            text  = r.get("text",           "")

            preview = text[:text_preview] + ("…" if len(text) > text_preview else "")
            heading = f"{act}  §{sec}" + (f"  [{head}]" if head else "")
            lines.append(
                f"  [{i}]  {heading}\n"
                f"        score={scr:.4f}\n"
                f"        {preview}\n"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Retriever(index={self.kb}, "
            f"model='{self.embedder.model_name}')"
        )
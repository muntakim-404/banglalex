"""
Phase 2 — Test RAG Retrieval
==============================
Loads the saved FAISS index and runs a battery of sample queries in
both English and Bangla to verify retrieval quality before Phase 3.

Usage
-----
    cd D:\\Thesis Code\\banglalex
    banglalex-env\\Scripts\\activate
    python scripts/phase2_test_retrieval.py

Flags
-----
    --kb-dir   data/knowledge_base  (default)
    --k        5                    (results per query)
    --query    "custom query here"  (run a single custom query)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.knowledge_base.retriever import Retriever

logging.basicConfig(
    level   = logging.WARNING,    # keep retriever logs quiet during testing
    format  = "%(asctime)s %(message)s",
)

SEPARATOR = "═" * 70

# ── Sample queries covering all three case domains from Phase 1 ───────────────

SAMPLE_QUERIES = [
    # ── Land ──────────────────────────────────────────────────────────────────
    {
        "label":  "EN / Land — boundary dispute",
        "query":  "land boundary dispute between neighbours registration",
        "domain": "land",
    },
    {
        "label":  "BN / Land — ভূমি বিরোধ",
        "query":  "জমির সীমানা বিরোধ নিষ্পত্তি",
        "domain": "land",
    },
    {
        "label":  "EN / Land — adverse possession",
        "query":  "adverse possession limitation period land ownership",
        "domain": None,
    },

    # ── Contract ──────────────────────────────────────────────────────────────
    {
        "label":  "EN / Contract — unpaid wages",
        "query":  "employer breach of contract unpaid wages compensation",
        "domain": "contract",
    },
    {
        "label":  "BN / Contract — বেতন বকেয়া",
        "query":  "নিয়োগকর্তা কর্তৃক বেতন না দেওয়া চুক্তি লঙ্ঘন",
        "domain": "contract",
    },

    # ── Service / Employment ───────────────────────────────────────────────────
    {
        "label":  "EN / Service — wrongful termination",
        "query":  "government employee wrongful termination service rules",
        "domain": "service",
    },
    {
        "label":  "BN / Service — চাকরিচ্যুতি",
        "query":  "সরকারি কর্মচারী বরখাস্ত পুনর্বহাল",
        "domain": "service",
    },

    # ── General / Penal ───────────────────────────────────────────────────────
    {
        "label":  "EN / Penal — theft punishment",
        "query":  "punishment for theft under penal code Bangladesh",
        "domain": None,
    },
    {
        "label":  "EN / Family — divorce grounds",
        "query":  "divorce grounds family law Muslim woman",
        "domain": None,
    },
    {
        "label":  "EN / Multi — eviction notice landlord",
        "query":  "landlord eviction notice tenant rights written notice",
        "domain": None,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test Phase 2 RAG retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kb-dir",  default="data/knowledge_base")
    p.add_argument("--model",   default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--k",       type=int, default=3, help="Results per query")
    p.add_argument("--query",   default=None,
                   help="Run a single custom query instead of the test suite")
    p.add_argument("--domain",  default=None,
                   help="Optional domain filter for --query (land/contract/service)")
    return p.parse_args()


def run_suite(retriever: Retriever, k: int) -> None:
    """Run all sample queries and print results."""
    passed = 0
    for item in SAMPLE_QUERIES:
        label  = item["label"]
        query  = item["query"]
        domain = item["domain"]

        t0      = time.time()
        results = retriever.retrieve_for_case(query, domain=domain, k=k)
        elapsed = (time.time() - t0) * 1000

        print(SEPARATOR)
        print(f" {label}")
        print(f" Query  : {query}")
        print(f" Domain : {domain or '(none)'}   |   {len(results)} results   |   {elapsed:.0f} ms")
        print()
        print(Retriever.format_results(results))

        # A pass = at least 1 result with score > 0.3
        if any(r["score"] > 0.30 for r in results):
            passed += 1

    print(SEPARATOR)
    total = len(SAMPLE_QUERIES)
    print(f"\n  Queries with score > 0.30:  {passed}/{total}")
    print(
        "  Status: GOOD ✓" if passed >= total * 0.7
        else "  Status: CHECK EMBEDDINGS — many low-score results"
    )


def run_single(retriever: Retriever, query: str, domain: Optional[str], k: int) -> None:
    """Run one custom query and print results."""
    print(SEPARATOR)
    print(f" Custom query : {query}")
    print(f" Domain       : {domain or '(none)'}")
    print()
    t0      = time.time()
    results = retriever.retrieve_for_case(query, domain=domain, k=k)
    elapsed = (time.time() - t0) * 1000
    print(f" {len(results)} result(s) in {elapsed:.0f} ms\n")
    print(Retriever.format_results(results))
    print(SEPARATOR)


# ── Typing fix for run_single ──────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    print(SEPARATOR)
    print(" BanglaLex — Phase 2: Retrieval Test")
    print(f" Index : {Path(args.kb_dir).resolve()}")
    print(SEPARATOR)
    print(" Loading model and index …", flush=True)

    retriever = Retriever.from_saved(args.kb_dir, args.model)
    print(f" {retriever}")
    print()

    if args.query:
        run_single(retriever, args.query, args.domain, args.k)
    else:
        run_suite(retriever, args.k)

    print("\n Done.\n")


if __name__ == "__main__":
    main()

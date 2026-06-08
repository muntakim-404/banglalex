"""
Phase 3 — Interactive CLI Demo
================================
Type a legal problem in English or Bangla and watch all four agents
process it in sequence.

Usage
-----
    cd "D:\\Thesis Code\\banglalex"
    banglalex-env\\Scripts\\activate
    python scripts/phase3_demo.py

Optional flags:
    --kb-dir      data/knowledge_base                    (default)
    --cases-path  data/annotated/cases_augmented.json    (default)
    --model       llama-3.3-70b-versatile                (default)
    --query       "type a single query here"             (skip interactive mode)

Setup (one-time)
----------------
1. pip install -r requirements_phase3.txt
2. Add to your .env file:
       GROQ_API_KEY=your_key_here
   Get a free key at: https://console.groq.com/keys
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level  = logging.WARNING,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logging.getLogger("src.agents").setLevel(logging.INFO)

from src.agents import BanglaLexPipeline

# ── Sample queries ─────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    ("EN / Land",     "My neighbour has illegally encroached on my land and built a wall "
                      "crossing into my property boundary. I have the original deed."),
    ("EN / Contract", "My employer has not paid my salary for the last 3 months. "
                      "I have worked continuously and there is no written notice of termination."),
    ("EN / Service",  "I am a government employee and was dismissed without any show cause "
                      "notice or hearing. I want to challenge the dismissal order."),
    ("BN / Land",     "আমার বাড়িওয়ালা আমাকে কোনো নোটিশ না দিয়েই ১লা জুন তালা "
                      "বদলে দিয়েছে এবং আমাকে বাসায় ঢুকতে দিচ্ছে না। আমি কী করতে পারি?"),
    ("BN / Contract", "আমার নিয়োগকর্তা গত ২ মাস ধরে বেতন দিচ্ছেন না। "
                      "চাকরিচ্যুতির কোনো নোটিশ দেওয়া হয়নি। আমি ক্ষতিপূরণ চাই।"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="BanglaLex Phase 3 — Interactive Legal Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kb-dir",     default="data/knowledge_base")
    p.add_argument("--cases-path", default="data/annotated/cases_augmented.json")
    p.add_argument("--model",      default="llama-3.3-70b-versatile")
    p.add_argument("--query",      default=None,
                   help="Run a single query non-interactively")
    return p.parse_args()


def print_samples():
    print("\n  Sample queries you can try:")
    for i, (label, q) in enumerate(SAMPLE_QUERIES, 1):
        print(f"  [{i}] {label}: {q[:70]}…")
    print()


def interactive_loop(pipeline: BanglaLexPipeline):
    print("\n" + "═" * 65)
    print("  BanglaLex — Legal Assistance Demo")
    print("  Type your legal problem in English or Bangla.")
    print("  Commands:  'samples' → show examples | 'quit' → exit")
    print("═" * 65)

    while True:
        try:
            query = input("\n  Your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting. Goodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\n  Exiting. Goodbye.")
            break
        if query.lower() in ("samples", "sample", "s"):
            print_samples()
            continue
        if query.isdigit() and 1 <= int(query) <= len(SAMPLE_QUERIES):
            _, query = SAMPLE_QUERIES[int(query) - 1]
            print(f"  Using: {query[:80]}")

        try:
            pipeline.run_verbose(query)
        except Exception as exc:
            print(f"\n  ✗  Error: {exc}")
            print("     Check your GROQ_API_KEY and network connection.")


def main():
    args = parse_args()

    print("Loading pipeline (FAISS index + Groq model) …", flush=True)
    try:
        pipeline = BanglaLexPipeline(
            kb_dir     = args.kb_dir,
            cases_path = args.cases_path,
            model_name = args.model,
        )
    except EnvironmentError as e:
        print(f"\n  ✗  {e}")
        sys.exit(1)

    if args.query:
        pipeline.run_verbose(args.query)
    else:
        print_samples()
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()

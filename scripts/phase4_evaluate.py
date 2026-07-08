"""
Phase 4 — Evaluation Script  (v3: all fixes applied)
======================================================
Fix 1: Anti-abstention prompt        (evaluator.py)
Fix 2: Similarity-based precedents   (evaluator.py)
Fix 3: Agent 2 LLM filtering         (evaluator.py)
Fix 4: llama-3.3-70b-versatile, 25 cases/domain (here)

Token budget (75 total cases, 70b model):
  BanglaLex  : Agent 2 ~400 + Agent 3 ~600 = 1,000 × 75 = 75,000
  LLM-only   : ~300 × 75                              = 22,500
  TOTAL                                               ≈ 97,500  ✓ (within 100k)

Usage
-----
    python scripts/phase4_evaluate.py             # run all (recommended)
    python scripts/phase4_evaluate.py --resume    # resume a partial run
    python scripts/phase4_evaluate.py --skip-llm  # skip LLM-only baseline
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.evaluation.metrics   import compute_metrics, print_report, save_results
from src.evaluation.evaluator import BanglaLexEvaluator
from src.evaluation.baseline  import MajorityClassBaseline, LLMOnlyBaseline
from src.evaluation.metrics   import normalize_outcome

SEP  = "─" * 65
DSEP = "═" * 65

_TEXT_FIELDS    = ["facts_summary", "facts", "text", "description",
                   "case_text", "summary", "narrative", "content"]
_DOMAIN_FIELDS  = ["domain", "case_domain", "category", "type"]
_OUTCOME_FIELDS = ["outcome", "label", "result", "decision", "verdict"]


def _get(record, candidates):
    for k in candidates:
        if k in record:
            return str(record[k])
    return None


def load_balanced_sample(cases_path, n_per_domain, seed):
    random.seed(seed)
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)
    all_cases = data if isinstance(data, list) else list(data.values())[0]

    by_domain = defaultdict(list)
    for case in all_cases:
        domain  = (_get(case, _DOMAIN_FIELDS) or "").lower()
        outcome = normalize_outcome(_get(case, _OUTCOME_FIELDS) or "")
        text    = _get(case, _TEXT_FIELDS)
        if domain and outcome and text:
            by_domain[domain].append(case)

    sample = []
    logger.info("Balanced sample:")
    for domain, cases in sorted(by_domain.items()):
        n = min(n_per_domain, len(cases))
        chosen = random.sample(cases, n)
        sample.extend(chosen)
        dist = defaultdict(int)
        for c in chosen:
            dist[normalize_outcome(_get(c, _OUTCOME_FIELDS) or "") or "?"] += 1
        logger.info(f"  {domain:12s}: {n} cases  {dict(dist)}")
    logger.info(f"  {'TOTAL':12s}: {len(sample)} cases")
    return sample


def load_existing(path: Path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        logger.info(f"Resumed {len(r)} results from {path}")
        return r
    return []


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-per-domain",  type=int,   default=25,
                   help="Cases per domain. 25 × 3 = 75 total fits within 100k token budget.")
    p.add_argument("--kb-dir",                     default="data/knowledge_base")
    p.add_argument("--cases-path",                 default="data/annotated/cases_augmented.json")
    p.add_argument("--output-dir",                 default="data/evaluation")
    p.add_argument("--model",
                   default="llama-3.3-70b-versatile",
                   help="Fix 4: use the 70b model for better legal reasoning.")
    p.add_argument("--call-delay",    type=float,  default=2.0)
    p.add_argument("--skip-llm",      action="store_true",
                   help="Skip LLM-only baseline (saves ~22k tokens).")
    p.add_argument("--no-agent2",     action="store_true",
                   help="Disable Fix 3 (Agent 2 filtering) to save tokens.")
    p.add_argument("--resume",        action="store_true",
                   help="Load saved partial results and continue.")
    p.add_argument("--seed",          type=int,    default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{DSEP}")
    print("  BanglaLex — Phase 4: Evaluation  (v3 — all fixes)")
    print(DSEP)
    print(f"  Model          : {args.model}  (Fix 4)")
    print(f"  Cases/domain   : {args.n_per_domain}  → {args.n_per_domain*3} total")
    print(f"  Agent 2 filter : {'yes (Fix 3)' if not args.no_agent2 else 'no'}")
    print(f"  LLM-only       : {'yes' if not args.skip_llm else 'no'}")
    print(f"  Fixes applied  : 1 (anti-abstention)  2 (sim precedents)"
          f"  {'' if args.no_agent2 else '3 (agent2 filter)'}  4 (70b model)")
    print(DSEP)

    sample     = load_balanced_sample(args.cases_path, args.n_per_domain, args.seed)
    all_metrics = []
    save_every  = 5

    # ── 1. Majority class ─────────────────────────────────────────────────────
    print(f"\n{SEP}\n[1/{3 if not args.skip_llm else 2}] Majority Class …")
    majority         = MajorityClassBaseline(args.cases_path, args.n_per_domain)
    majority_results = majority.run(sample)
    majority_metrics = compute_metrics(majority_results, "Majority Class")
    save_results(majority_results, majority_metrics, output_dir, "Majority Class")
    all_metrics.append(majority_metrics)
    print(f"  ✓  Accuracy: {majority_metrics['overall_accuracy']*100:.1f}%  "
          f"Macro F1: {majority_metrics['macro_f1']:.3f}")

    # ── 2. LLM-only baseline ──────────────────────────────────────────────────
    if not args.skip_llm:
        step = 2
        print(f"\n{SEP}\n[{step}/{3 if not args.skip_llm else 2}] LLM-only (no RAG) …")
        slug         = "llm-only_(no_rag)"
        partial_path = output_dir / f"results_{slug}.json"
        existing     = load_existing(partial_path) if args.resume else []
        done_ids     = {r["citation"] for r in existing}
        remaining    = [c for c in sample if c.get("citation") not in done_ids]

        llm_only    = LLMOnlyBaseline(args.model, args.call_delay)
        new_results = llm_only.run(remaining)
        llm_results = existing + new_results
        llm_metrics = compute_metrics(llm_results, "LLM-only (no RAG)")
        save_results(llm_results, llm_metrics, output_dir, "LLM-only (no RAG)")
        all_metrics.append(llm_metrics)
        print(f"  ✓  Accuracy: {llm_metrics['overall_accuracy']*100:.1f}%  "
              f"Macro F1: {llm_metrics['macro_f1']:.3f}")

    # ── 3. BanglaLex full ─────────────────────────────────────────────────────
    step = 3 if not args.skip_llm else 2
    print(f"\n{SEP}\n[{step}/{step}] BanglaLex Full (all fixes) …")

    slug         = "banglalex_(full)"
    partial_path = output_dir / f"results_{slug}.json"
    existing     = load_existing(partial_path) if args.resume else []
    done_ids     = {r["citation"] for r in existing}
    remaining    = [c for c in sample if c.get("citation") not in done_ids]
    logger.info(f"BanglaLex: {len(existing)} done, {len(remaining)} remaining")

    evaluator = BanglaLexEvaluator(
        kb_dir               = args.kb_dir,
        cases_path           = args.cases_path,
        model_name           = args.model,
        n_per_domain         = args.n_per_domain,
        call_delay           = args.call_delay,
        use_agent2_filtering = not args.no_agent2,
    )
    evaluator._load_all_cases()

    banglalex_results = list(existing)

    from tqdm import tqdm
    for i, case in enumerate(tqdm(remaining, desc="BanglaLex", unit="case")):
        try:
            result = evaluator._evaluate_single(case, evaluator._all_cases)
            if result:
                banglalex_results.append(result)
        except Exception as exc:
            logger.warning(f"Skipped: {exc}")
        time.sleep(args.call_delay)

        if (i + 1) % save_every == 0:
            tmp = compute_metrics(banglalex_results, "BanglaLex (full)")
            save_results(banglalex_results, tmp, output_dir, "BanglaLex (full)")
            logger.info(f"  Checkpoint: {len(banglalex_results)} results  "
                        f"acc={tmp['overall_accuracy']*100:.1f}%  "
                        f"F1={tmp['macro_f1']:.3f}")

    banglalex_metrics = compute_metrics(banglalex_results, "BanglaLex (full)")
    save_results(banglalex_results, banglalex_metrics, output_dir, "BanglaLex (full)")
    all_metrics.append(banglalex_metrics)
    print(f"  ✓  Accuracy: {banglalex_metrics['overall_accuracy']*100:.1f}%  "
          f"Macro F1: {banglalex_metrics['macro_f1']:.3f}")

    # ── Final report ──────────────────────────────────────────────────────────
    print_report(all_metrics)

    with open(output_dir / "all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"{DSEP}")
    print("  Results saved to:")
    print(f"    {output_dir / 'comparison_table.csv'}  ← thesis table")
    print(f"    {output_dir / 'all_metrics.json'}")
    print(DSEP + "\n")


if __name__ == "__main__":
    main()

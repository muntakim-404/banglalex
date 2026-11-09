"""
False-Positive Rate by Domain — the technically correct explanation
for why Land/Contract/Service accuracies differ.

Since all errors are unfavorable→favorable false positives,
the domain accuracy gap could come from:
  (A) Different proportions of unfavorable cases per domain in the test sample
  (B) Different per-case false-positive rates (genuine capability gap)

This script distinguishes (A) from (B).

Usage:
    python domain_fp_rate.py
"""
import os
EVAL_DIR = os.environ.get("BANGLALEX_EVAL_DIR", "data/evaluation_clean")
import json
from pathlib import Path
from collections import defaultdict

with open(f"{EVAL_DIR}/results_banglalex_full.json", encoding="utf-8") as f:
    results = json.load(f)

DOMAINS = ["land", "contract", "service"]

print("\n" + "="*70)
print("  FALSE POSITIVE RATE ANALYSIS — by domain")
print("="*70)

print(f"\n  {'Domain':<10} {'Total':>6} {'Fav':>6} {'Unfav':>7} "
      f"{'FP':>5} {'FP_rate':>10}  {'Acc':>8}")
print(f"  {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*10}  {'─'*8}")

by_domain = defaultdict(list)
for r in results:
    by_domain[r["domain"]].append(r)

for domain in DOMAINS:
    cases     = by_domain[domain]
    favorable = [r for r in cases if r["ground_truth"] == "favorable"]
    unfavora  = [r for r in cases if r["ground_truth"] == "unfavorable"]
    errors    = [r for r in cases if not r["correct"]]  # all are unfav→fav

    fp_rate = len(errors) / len(unfavora) if unfavora else 0
    acc     = sum(r["correct"] for r in cases) / len(cases) * 100

    print(f"  {domain.capitalize():<10} {len(cases):>6} {len(favorable):>6} "
          f"{len(unfavora):>7} {len(errors):>5} "
          f"{fp_rate*100:>9.1f}%  {acc:>7.1f}%")

print(f"""
  ── Interpretation ──────────────────────────────────────────────────

  If FP rates were equal across domains, accuracy differences would be
  explained entirely by different unfavorable-case counts in each domain's
  30-case sample (sampling effect only — not a capability difference).

  If FP rates differ significantly, that's evidence of a genuine domain-
  level capability gap — contract cases are structurally harder for the
  pipeline to classify correctly when the true outcome is unfavorable.
""")

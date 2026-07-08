"""
Phase 4 — Final Comparison Report
===================================
Reads all saved evaluation results from data/evaluation/ and
prints the complete comparison table (GLARE-style: Acc, Ma-P, Ma-R, Ma-F).

Run this after ALL evaluations are complete.

Usage:
    python scripts/phase4_final_report.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import print_report, _get_macro_p, _get_macro_r

output_dir = Path("data/evaluation")

# Load all saved metric files
metrics_files = sorted(output_dir.glob("metrics_*.json"))
all_metrics   = []

for mf in metrics_files:
    with open(mf, encoding="utf-8") as f:
        m = json.load(f)
    all_metrics.append(m)
    print(f"  Loaded: {m.get('method','?')}  "
          f"(n={m.get('n_decided','?')}  "
          f"acc={m.get('overall_accuracy',0)*100:.1f}%  "
          f"F1={m.get('macro_f1',0)*100:.1f}%)")

if not all_metrics:
    print("No evaluation results found in data/evaluation/")
    sys.exit(1)

# Sort by macro F1
all_metrics.sort(key=lambda m: m.get("macro_f1", 0))

print_report(all_metrics)

# Save updated CSV — now includes Macro-Precision and Macro-Recall (Fix: Gap 1)
import csv
csv_path = output_dir / "comparison_table_final.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method", "n_total", "n_decided",
        "overall_accuracy_%", "macro_precision_%", "macro_recall_%", "macro_f1_%",
        "land_%", "contract_%", "service_%",
        "favorable_f1", "unfavorable_f1", "abstention_%"
    ])
    for m in all_metrics:
        da = m.get("domain_accuracy", {})
        writer.writerow([
            m.get("method", ""),
            m.get("n_total", ""),
            m.get("n_decided", ""),
            f"{m.get('overall_accuracy',0)*100:.2f}",
            f"{_get_macro_p(m)*100:.2f}",
            f"{_get_macro_r(m)*100:.2f}",
            f"{m.get('macro_f1',0)*100:.2f}",
            f"{da.get('land',    0)*100:.2f}",
            f"{da.get('contract',0)*100:.2f}",
            f"{da.get('service', 0)*100:.2f}",
            f"{m.get('favorable_f1',  0):.4f}",
            f"{m.get('unfavorable_f1',0):.4f}",
            f"{m.get('abstention_rate',0)*100:.2f}",
        ])
print(f"\n  Final table saved → {csv_path}")

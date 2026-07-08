"""
Quick check: does confidence differ between correct and incorrect
BanglaLex predictions? Used to sharpen the abstention-rate discussion
in Section 4.4 — no new API calls, just reads the saved results file.

Usage:
    python confidence_check.py
"""
import json
import statistics

PATH = "data/evaluation_gemini/results_banglalex_full.json"

with open(PATH, encoding="utf-8") as f:
    results = json.load(f)

correct_conf = [r["confidence"] for r in results if r["correct"]]
wrong_conf   = [r["confidence"] for r in results if not r["correct"]]

print(f"Correct   (n={len(correct_conf)}): "
      f"mean={statistics.mean(correct_conf):.3f}  "
      f"median={statistics.median(correct_conf):.3f}")
print(f"Incorrect (n={len(wrong_conf)}):  "
      f"mean={statistics.mean(wrong_conf):.3f}  "
      f"median={statistics.median(wrong_conf):.3f}")

print("\nIndividual confidence values for the incorrect cases:")
for r in results:
    if not r["correct"]:
        print(f"  {r['citation']:<30} domain={r['domain']:<10} "
              f"predicted={r['predicted']:<12} confidence={r['confidence']}")

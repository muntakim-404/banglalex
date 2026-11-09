"""
Inspect stored results for the 6 error cases — check what fields
are already saved before deciding whether to re-run the pipeline.
"""
import json
from pathlib import Path

with open("data/evaluation/results_banglalex_full.json", encoding="utf-8") as f:
    results = json.load(f)

errors = [r for r in results if not r["correct"]]

print(f"Total error cases: {len(errors)}")
print(f"\nFields stored per result: {list(errors[0].keys())}")
print("\n" + "="*70)

for i, e in enumerate(errors):
    print(f"\n--- Error {i+1}: {e['citation']} ({e['domain'].upper()}) ---")
    print(f"  Ground truth : {e['ground_truth']}")
    print(f"  Predicted    : {e['predicted']}")
    print(f"  Confidence   : {e['confidence']}")
    # Print every field except the ones already shown
    skip = {"citation","domain","ground_truth","predicted","confidence","correct"}
    for k, v in e.items():
        if k in skip:
            continue
        if isinstance(v, str):
            print(f"  {k}: {v[:300]}")
        elif isinstance(v, list):
            print(f"  {k} ({len(v)} items): {str(v)[:300]}")
        else:
            print(f"  {k}: {v}")

"""
Quick inspector — run this ONCE to show the structure of the Kaggle dataset.
Paste the output back so the chunker can be rewritten for the actual format.

Usage:
    python scripts/phase2_inspect_dataset.py --data-dir data/statutes
"""

import argparse
import json
import csv
from pathlib import Path

def inspect(data_dir: str):
    root = Path(data_dir)
    print("=" * 60)
    print(f"Root: {root.resolve()}")
    print("=" * 60)

    # ── 1. List top-level contents ────────────────────────────────
    print("\n[TOP-LEVEL FILES & FOLDERS]")
    for p in sorted(root.iterdir()):
        size = f"{p.stat().st_size // 1024} KB" if p.is_file() else "<dir>"
        print(f"  {p.name:<30} {size}")

    # ── 2. Inspect filtered_act_list.csv ─────────────────────────
    csv_file = root / "filtered_act_list.csv"
    if csv_file.exists():
        print(f"\n[filtered_act_list.csv — first 3 rows]")
        with open(csv_file, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                print(f"  Row {i+1}: {dict(row)}")
                if i >= 2:
                    break

    # ── 3. Inspect processed_law JSON (top-level structure only) ──
    law_file = root / "processed_law"
    if not law_file.exists():
        law_file = root / "processed_law.json"
    if law_file.exists():
        print(f"\n[processed_law — top-level structure]")
        with open(law_file, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        print(f"  Type: {type(data).__name__}")
        if isinstance(data, list):
            print(f"  Length: {len(data)} items")
            print(f"  First item keys: {list(data[0].keys()) if data else 'empty'}")
            print(f"  First item (truncated):")
            first = data[0]
            for k, v in first.items():
                val_repr = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
                print(f"    {k}: {val_repr}")
        elif isinstance(data, dict):
            print(f"  Top-level keys: {list(data.keys())}")
            for k, v in list(data.items())[:3]:
                val_repr = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
                print(f"    {k}: {val_repr}")

    # ── 4. Inspect one small act-print-N file ─────────────────────
    acts_dir = root / "acts"
    if acts_dir.exists():
        print(f"\n[acts/ folder]")
        act_files = sorted(acts_dir.iterdir(), key=lambda p: p.stat().st_size)
        print(f"  Total files: {len(act_files)}")

        # Pick the smallest file for inspection
        small = act_files[0]
        print(f"\n  Inspecting smallest file: {small.name} ({small.stat().st_size} bytes)")
        with open(small, encoding="utf-8", errors="replace") as f:
            content = json.load(f)
        print(f"  Type: {type(content).__name__}")
        if isinstance(content, dict):
            print(f"  Keys: {list(content.keys())}")
            for k, v in content.items():
                val_repr = str(v)[:150] + "..." if len(str(v)) > 150 else str(v)
                print(f"    {k}: {val_repr}")
        elif isinstance(content, list):
            print(f"  Length: {len(content)}")
            print(f"  First item: {str(content[0])[:200]}")

        # Also show a medium-sized file for comparison
        if len(act_files) > 10:
            mid = act_files[len(act_files) // 2]
            print(f"\n  Inspecting mid-size file: {mid.name} ({mid.stat().st_size} bytes)")
            with open(mid, encoding="utf-8", errors="replace") as f:
                content2 = json.load(f)
            if isinstance(content2, dict):
                print(f"  Keys: {list(content2.keys())}")
                for k, v in content2.items():
                    val_repr = str(v)[:150] + "..." if len(str(v)) > 150 else str(v)
                    print(f"    {k}: {val_repr}")

    print("\n" + "=" * 60)
    print("Paste this full output so the chunker can be updated.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/statutes")
    args = p.parse_args()
    inspect(args.data_dir)
"""
Phase 4 — Multi-Backbone Comparison Report  (v3: adds Land/Contract/Service columns)
=======================================================================================
Merges results from two backbone runs into a single GLARE-style table,
now including per-domain accuracy breakdown for every row (previously
only shown for the Llama backbone's table; v3 fixes this gap).

Usage
-----
    python scripts/phase4_multi_backbone_report.py

Reads:
    data/evaluation/metrics_*.json          (backbone 1 — Llama + BERT baseline)
    data/evaluation_gemini/metrics_*.json   (backbone 2 — Gemini)

Writes:
    data/evaluation/comparison_table_multi_backbone.csv
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.evaluation.metrics import _get_macro_p, _get_macro_r
except ImportError:
    def _get_macro_p(m):
        if "macro_precision" in m:
            return m["macro_precision"]
        return (m.get("favorable_prec", 0) + m.get("unfavorable_prec", 0)) / 2

    def _get_macro_r(m):
        if "macro_recall" in m:
            return m["macro_recall"]
        return (m.get("favorable_rec", 0) + m.get("unfavorable_rec", 0)) / 2


BACKBONE_1_DIR   = Path("data/evaluation")
BACKBONE_1_LABEL = "Llama-4-Scout-17B (Groq)"

BACKBONE_2_DIR   = Path("data/evaluation_gemini")
BACKBONE_2_LABEL = "Gemini-3.1-Flash-Lite"

BACKBONE_INDEPENDENT = {"Majority Class", "XLM-RoBERTa (fine-tuned)"}


def load_all_metrics(directory: Path):
    if not directory.exists():
        print(f"  WARNING: {directory} not found — skipping.")
        return []
    results = []
    for f in sorted(directory.glob("metrics_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                m = json.load(fh)
            results.append(m)
            print(f"  Loaded {f.name}: method='{m.get('method', '?')}'")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: could not read {f}: {exc}")
    return results


def fmt_pct(x):
    return f"{x*100:.1f}" if x <= 1.0 else f"{x:.1f}"


def main():
    print(f"Scanning {BACKBONE_1_DIR} ...")
    m1 = load_all_metrics(BACKBONE_1_DIR)
    print(f"\nScanning {BACKBONE_2_DIR} ...")
    m2 = load_all_metrics(BACKBONE_2_DIR)

    rows = []
    seen_independent = set()

    for m in m1:
        method = m.get("method", "?")
        if method in BACKBONE_INDEPENDENT:
            if method in seen_independent:
                continue
            seen_independent.add(method)
            rows.append((method, "—", m))
        else:
            rows.append((method, BACKBONE_1_LABEL, m))

    for m in m2:
        method = m.get("method", "?")
        if method in BACKBONE_INDEPENDENT:
            if method in seen_independent:
                continue
            seen_independent.add(method)
            rows.append((method, "—", m))
        else:
            rows.append((method, BACKBONE_2_LABEL, m))

    # ── Print table (now with Land/Contract/Service) ───────────────────────
    header = (f"{'Method':<24} {'Backbone':<26} {'Acc%':>6} {'Ma-P':>6} {'Ma-R':>6} "
               f"{'Ma-F':>6} {'Land%':>7} {'Contract%':>10} {'Service%':>9} {'Abst%':>7}")
    print("\n" + "=" * len(header))
    print("  BanglaLex — Multi-Backbone Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for method, backbone, m in rows:
        da   = m.get("domain_accuracy", {})
        acc  = fmt_pct(m.get("overall_accuracy", 0))
        macp = fmt_pct(_get_macro_p(m))
        macr = fmt_pct(_get_macro_r(m))
        macf = fmt_pct(m.get("macro_f1", 0))
        land = fmt_pct(da.get("land", 0))
        cont = fmt_pct(da.get("contract", 0))
        serv = fmt_pct(da.get("service", 0))
        abst = fmt_pct(m.get("abstention_rate", 0))
        print(f"{method:<24} {backbone:<26} {acc:>6} {macp:>6} {macr:>6} {macf:>6} "
              f"{land:>7} {cont:>10} {serv:>9} {abst:>7}")
    print("=" * len(header))

    if not rows:
        print("\nNo metrics files found in either directory — check the paths above.")
        return

    print(
        "\nNote: LLM-only abstention rates differ across backbones (see Abst% column) — "
        "accuracy for that row is computed over decided cases only, not the full sample. "
        "Report both columns together in the thesis; do not compare accuracy in isolation."
    )

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_path = BACKBONE_1_DIR / "comparison_table_multi_backbone.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("method,backbone,accuracy_%,macro_precision_%,macro_recall_%,macro_f1_%,"
                "land_%,contract_%,service_%,abstention_%\n")
        for method, backbone, m in rows:
            da = m.get("domain_accuracy", {})
            f.write(
                f"{method},{backbone},"
                f"{fmt_pct(m.get('overall_accuracy', 0))},"
                f"{fmt_pct(_get_macro_p(m))},"
                f"{fmt_pct(_get_macro_r(m))},"
                f"{fmt_pct(m.get('macro_f1', 0))},"
                f"{fmt_pct(da.get('land', 0))},"
                f"{fmt_pct(da.get('contract', 0))},"
                f"{fmt_pct(da.get('service', 0))},"
                f"{fmt_pct(m.get('abstention_rate', 0))}\n"
            )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

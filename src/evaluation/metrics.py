"""
BanglaLex — Evaluation Metrics
================================
Accuracy, Precision, Recall, F1, confusion matrix,
and per-domain breakdown for judgment prediction evaluation.

CHANGE LOG
----------
v2: print_report() now shows Macro-Precision and Macro-Recall columns
    (GLARE-style table: Acc, Ma-P, Ma-R, Ma-F) in addition to Macro F1.
    No re-evaluation needed — Ma-P/Ma-R are derived from favorable_prec/
    favorable_rec/unfavorable_prec/unfavorable_rec, which were already
    being saved in metrics_*.json. Just re-run phase4_final_report.py.
"""

import json
import csv
import logging
from collections import defaultdict
from pathlib     import Path
from typing      import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Outcome normalisation ──────────────────────────────────────────────────────

def normalize_outcome(raw: str) -> Optional[str]:
    """
    Map a raw outcome string to "favorable" or "unfavorable".
    Returns None if the string cannot be mapped (case is skipped).
    """
    if not isinstance(raw, str):
        return None
    o = raw.lower().strip()
    if any(k in o for k in ["unfav", "against", "dismissed", "rejected", "fail"]):
        return "unfavorable"
    if any(k in o for k in ["fav", "allow", "granted", "absolute", "success",
                             "declared", "upheld", "awarded"]):
        return "favorable"
    return None


# ── Core metrics ────────────────────────────────────────────────────────────────

def _filter_decided(results: list) -> list:
    """Keep only results where predicted != 'uncertain'."""
    return [r for r in results if r.get("predicted", "uncertain") != "uncertain"]


def accuracy(results: list) -> float:
    decided = _filter_decided(results)
    if not decided:
        return 0.0
    return sum(1 for r in decided if r.get("correct", False)) / len(decided)


def precision_recall_f1(results: list, positive: str = "favorable") -> Dict:
    decided = _filter_decided(results)
    tp = sum(1 for r in decided
             if r["ground_truth"] == positive and r["predicted"] == positive)
    fp = sum(1 for r in decided
             if r["ground_truth"] != positive and r["predicted"] == positive)
    fn = sum(1 for r in decided
             if r["ground_truth"] == positive and r["predicted"] != positive)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def macro_f1(results: list) -> float:
    fav   = precision_recall_f1(results, "favorable")["f1"]
    unfav = precision_recall_f1(results, "unfavorable")["f1"]
    return (fav + unfav) / 2


def macro_precision(results: list) -> float:
    fav   = precision_recall_f1(results, "favorable")["precision"]
    unfav = precision_recall_f1(results, "unfavorable")["precision"]
    return (fav + unfav) / 2


def macro_recall(results: list) -> float:
    fav   = precision_recall_f1(results, "favorable")["recall"]
    unfav = precision_recall_f1(results, "unfavorable")["recall"]
    return (fav + unfav) / 2


def confusion_matrix(results: list) -> Dict:
    """Return {true_label: {predicted_label: count}}."""
    decided = _filter_decided(results)
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in decided:
        matrix[r["ground_truth"]][r["predicted"]] += 1
    return {k: dict(v) for k, v in matrix.items()}


def abstention_rate(results: list) -> float:
    uncertain = sum(1 for r in results if r.get("predicted") == "uncertain")
    return uncertain / len(results) if results else 0.0


# ── Full metrics bundle ───────────────────────────────────────────────────────

def compute_metrics(results: list, method_name: str = "") -> Dict:
    """
    Compute all metrics for a results list.

    Parameters
    ----------
    results : list of dicts with keys:
              citation, domain, ground_truth, predicted, confidence, correct
    method_name : label for this result set (used in reports)

    Returns
    -------
    dict with all metrics
    """
    decided  = _filter_decided(results)
    by_domain: Dict[str, list] = defaultdict(list)
    for r in decided:
        by_domain[r.get("domain", "unknown")].append(r)

    fav_metrics   = precision_recall_f1(results, "favorable")
    unfav_metrics = precision_recall_f1(results, "unfavorable")

    return {
        "method":            method_name,
        "n_total":           len(results),
        "n_decided":         len(decided),
        "abstention_rate":   round(abstention_rate(results), 4),
        "overall_accuracy":  round(accuracy(results), 4),
        "macro_f1":          round(macro_f1(results), 4),
        "macro_precision":   round((fav_metrics["precision"] + unfav_metrics["precision"]) / 2, 4),
        "macro_recall":      round((fav_metrics["recall"]    + unfav_metrics["recall"])    / 2, 4),
        "favorable_f1":      round(fav_metrics["f1"], 4),
        "favorable_prec":    round(fav_metrics["precision"], 4),
        "favorable_rec":     round(fav_metrics["recall"], 4),
        "unfavorable_f1":    round(unfav_metrics["f1"], 4),
        "unfavorable_prec":  round(unfav_metrics["precision"], 4),
        "unfavorable_rec":   round(unfav_metrics["recall"], 4),
        "domain_accuracy": {
            d: round(accuracy(cases), 4)
            for d, cases in by_domain.items()
        },
        "confusion_matrix":  confusion_matrix(results),
    }


# ── Outcome distribution helper ──────────────────────────────────────────────

def outcome_distribution(results: list) -> Dict:
    dist: Dict[str, int] = defaultdict(int)
    for r in results:
        dist[r.get("ground_truth", "unknown")] += 1
    return dict(dist)


# ── Reporting ──────────────────────────────────────────────────────────────────

def _get_macro_p(m: Dict) -> float:
    """Backward-compatible: use macro_precision if present, else derive it
    from favorable_prec/unfavorable_prec (works on older saved metrics_*.json)."""
    if "macro_precision" in m:
        return m["macro_precision"]
    return (m.get("favorable_prec", 0) + m.get("unfavorable_prec", 0)) / 2


def _get_macro_r(m: Dict) -> float:
    """Backward-compatible: use macro_recall if present, else derive it
    from favorable_rec/unfavorable_rec (works on older saved metrics_*.json)."""
    if "macro_recall" in m:
        return m["macro_recall"]
    return (m.get("favorable_rec", 0) + m.get("unfavorable_rec", 0)) / 2


def print_report(all_metrics: List[Dict]) -> None:
    """Print a formatted comparison table to stdout (GLARE-style: Acc, Ma-P, Ma-R, Ma-F)."""
    try:
        from tabulate import tabulate
    except ImportError:
        _print_report_plain(all_metrics)
        return

    SEP = "═" * 78
    print(f"\n{SEP}")
    print("  BanglaLex — Phase 4 Evaluation Results")
    print(SEP)

    # Main comparison table — now includes Ma-P and Ma-R (GLARE-style)
    headers = ["Method", "Acc %", "Ma-P", "Ma-R", "Ma-F", "Land %", "Contract %", "Service %"]
    rows = []
    for m in all_metrics:
        da = m.get("domain_accuracy", {})
        rows.append([
            m["method"],
            f"{m['overall_accuracy']*100:.1f}",
            f"{_get_macro_p(m)*100:.1f}",
            f"{_get_macro_r(m)*100:.1f}",
            f"{m['macro_f1']*100:.1f}",
            f"{da.get('land',    0)*100:.1f}",
            f"{da.get('contract',0)*100:.1f}",
            f"{da.get('service', 0)*100:.1f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # Per-class F1 table
    print("\n  Per-class F1 Scores:")
    headers2 = ["Method", "F1 (Favorable)", "F1 (Unfavorable)", "Abstention %"]
    rows2 = []
    for m in all_metrics:
        rows2.append([
            m["method"],
            f"{m['favorable_f1']:.3f}",
            f"{m['unfavorable_f1']:.3f}",
            f"{m['abstention_rate']*100:.1f}",
        ])
    print(tabulate(rows2, headers=headers2, tablefmt="rounded_outline"))

    # Confusion matrices
    for m in all_metrics:
        print(f"\n  Confusion Matrix — {m['method']}:")
        cm = m.get("confusion_matrix", {})
        labels = ["favorable", "unfavorable"]
        cm_rows = []
        for true_label in labels:
            row = [f"True {true_label}"]
            for pred_label in labels:
                row.append(cm.get(true_label, {}).get(pred_label, 0))
            cm_rows.append(row)
        cm_headers = [""] + [f"Pred {l}" for l in labels]
        print(tabulate(cm_rows, headers=cm_headers, tablefmt="rounded_outline"))

    print(f"\n{SEP}\n")


def _print_report_plain(all_metrics: List[Dict]) -> None:
    """Fallback plain-text report if tabulate is not installed."""
    print("\n" + "="*78)
    print("  BanglaLex Phase 4 Evaluation Results")
    print("="*78)
    for m in all_metrics:
        da = m.get("domain_accuracy", {})
        print(f"\n  {m['method']}")
        print(f"  Overall accuracy  : {m['overall_accuracy']*100:.1f}%")
        print(f"  Macro Precision   : {_get_macro_p(m)*100:.1f}%")
        print(f"  Macro Recall      : {_get_macro_r(m)*100:.1f}%")
        print(f"  Macro F1          : {m['macro_f1']*100:.1f}%")
        print(f"  Land accuracy     : {da.get('land',    0)*100:.1f}%")
        print(f"  Contract accuracy : {da.get('contract',0)*100:.1f}%")
        print(f"  Service accuracy  : {da.get('service', 0)*100:.1f}%")
        print(f"  F1 (favorable)    : {m['favorable_f1']:.3f}")
        print(f"  F1 (unfavorable)  : {m['unfavorable_f1']:.3f}")
        print(f"  Abstention rate   : {m['abstention_rate']*100:.1f}%")


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(
    results:     list,
    metrics:     dict,
    output_dir:  Path,
    method_name: str,
) -> None:
    """
    Save raw results (JSON) and metrics (JSON) to output_dir.
    Also append a row to comparison_table.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results
    slug = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    results_path = output_dir / f"results_{slug}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Metrics JSON
    metrics_path = output_dir / f"metrics_{slug}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # CSV row
    csv_path = output_dir / "comparison_table.csv"
    write_header = not csv_path.exists()
    da = metrics.get("domain_accuracy", {})
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "method", "n_total", "n_decided",
                "overall_accuracy", "macro_precision", "macro_recall", "macro_f1",
                "land_accuracy", "contract_accuracy", "service_accuracy",
                "favorable_f1", "unfavorable_f1", "abstention_rate"
            ])
        writer.writerow([
            metrics["method"],
            metrics["n_total"],
            metrics["n_decided"],
            f"{metrics['overall_accuracy']*100:.2f}",
            f"{_get_macro_p(metrics)*100:.2f}",
            f"{_get_macro_r(metrics)*100:.2f}",
            f"{metrics['macro_f1']*100:.2f}",
            f"{da.get('land',    0)*100:.2f}",
            f"{da.get('contract',0)*100:.2f}",
            f"{da.get('service', 0)*100:.2f}",
            f"{metrics['favorable_f1']:.4f}",
            f"{metrics['unfavorable_f1']:.4f}",
            f"{metrics['abstention_rate']*100:.2f}",
        ])

    logger.info(f"Results saved → {output_dir} ({method_name})")

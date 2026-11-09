"""
Domain Composition Chart — grouped bar with FP error overlay
=============================================================
Shows favorable vs unfavorable case counts per domain in the test
sample, with FP error counts annotated above each unfavorable bar.

Usage (from project root):
    python domain_composition_chart.py
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("data/evaluation/results_banglalex_full.json", encoding="utf-8") as f:
    results = json.load(f)

DOMAINS = ["Land", "Contract", "Service"]
by_domain = defaultdict(list)
for r in results:
    by_domain[r["domain"]].append(r)

fav_counts   = []
unfav_counts = []
fp_counts    = []

for d in DOMAINS:
    cases = by_domain[d.lower()]
    fav   = sum(1 for r in cases if r["ground_truth"] == "favorable")
    unfav = sum(1 for r in cases if r["ground_truth"] == "unfavorable")
    fp    = sum(1 for r in cases if not r["correct"])
    fav_counts.append(fav)
    unfav_counts.append(unfav)
    fp_counts.append(fp)

# ── Layout ─────────────────────────────────────────────────────────────────────
x        = np.arange(len(DOMAINS))
bar_w    = 0.35
fav_col  = "#1A3A5C"   # navy  — favorable
unfav_col= "#C8A951"   # gold  — unfavorable
err_col  = "#B22222"   # red   — FP annotation

fig, ax = plt.subplots(figsize=(8, 5.5))

bars_fav   = ax.bar(x - bar_w/2, fav_counts,   bar_w,
                    label="Favorable cases",   color=fav_col,  alpha=0.85)
bars_unfav = ax.bar(x + bar_w/2, unfav_counts, bar_w,
                    label="Unfavorable cases", color=unfav_col, alpha=0.85)

# ── Count labels on every bar ──────────────────────────────────────────────────
for bar, count in zip(bars_fav, fav_counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=fav_col)

for bar, count, fp in zip(bars_unfav, unfav_counts, fp_counts):
    # Total count above bar
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#8B6914")
    # FP error annotation — only if > 0
    if fp > 0:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.6,
                f"{fp} FP error{'s' if fp > 1 else ''}",
                ha="center", va="bottom", fontsize=9,
                color=err_col, fontweight="600",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=err_col, linewidth=0.8, alpha=0.9))

# ── Accuracy labels below domain names ────────────────────────────────────────
acc = {"Land": "93.3%", "Contract": "86.7%", "Service": "100%"}
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{d}\n(Acc: {acc[d]})" for d in DOMAINS],
    fontsize=12
)

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_ylabel("Number of Cases in Test Sample", fontsize=12)
ax.set_title(
    "Test Sample Composition by Domain and Outcome\n"
    "BanglaLex Evaluation — 30 cases per domain, seed=42",
    fontsize=13, fontweight="bold", pad=14
)
ax.set_ylim(0, 28)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# ── Legend ─────────────────────────────────────────────────────────────────────
ax.legend(
    handles=[
        mpatches.Patch(color=fav_col,   alpha=0.85, label="Favorable cases"),
        mpatches.Patch(color=unfav_col, alpha=0.85, label="Unfavorable cases"),
        mpatches.Patch(color=err_col,   alpha=0.85, label="False positive errors"),
    ],
    fontsize=10, loc="upper right"
)

# ── Save ───────────────────────────────────────────────────────────────────────
out = Path("data/evaluation/domain_composition_chart.png")
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()

"""
Confusion Matrix Heatmap (with Abstention column) — BanglaLex Thesis
========================================================================
For methods with nonzero abstention, use this 2x3 version instead of
the 2x2 script — it shows Predicted Favorable / Unfavorable / Abstained
so every case in n=90 is accounted for, none silently dropped.

For methods with 0% abstention (Majority Class, BanglaLex-full), the
2x2 script is fine — but this version also works for them (abstained
column is just [0, 0]) if you want consistent figure formatting
across all your results.

Usage:
    python plot_confusion_matrix_3col.py
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── EDIT THESE FOR EACH RESULT SET ────────────────────────────────────
# Row order = [True favorable, True unfavorable]
# Col order = [Pred favorable, Pred unfavorable, Pred abstained]
cm = np.array([
    [43, 12, 0],    # True favorable
    [10,  25, 0],    # True unfavorable
])
title = "XLM-RoBERTa (fine-tuned)"
output_filename = "confusion_matrix_xlm_roberta.png"
# ──────────────────────────────────────────────────────────────────────

row_labels = ["Favorable", "Unfavorable"]
col_labels = ["Favorable", "Unfavorable", "Abstained"]

plt.figure(figsize=(7, 5))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=col_labels,
    yticklabels=row_labels,
    cbar=True,
    square=False,
    annot_kws={"size": 16, "weight": "bold"},
    linewidths=0.5,
    linecolor="white",
)

ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
ax.set_ylabel("Actual Class", fontsize=12, labelpad=10)
ax.set_title(title, fontsize=13, weight="bold", pad=15)
plt.xticks(rotation=0)
plt.yticks(rotation=90, va="center")

plt.tight_layout()
plt.savefig(output_filename, dpi=200, bbox_inches="tight")
print(f"Saved -> {output_filename}")
print(f"Total cases shown: {cm.sum()}")
plt.show()

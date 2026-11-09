"""Frontier figure for VERIFICATION.md from baselines_results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "baselines_results.json").read_text())["rows"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("pencil-all", "Pencil, K-FAC (all positions)", "#2a78d6"),
          ("pencil-loss", "Pencil, K-FAC (loss positions)", "#eb6834"),
          ("diag-Fisher", "Diagonal Fisher merge", "#1baf7a"),
          ("RegMean", "RegMean (input Gram only)", "#eda100"),
          ("interp", "Linear interpolation", "#e87ba4")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for key, label, color in SERIES:
    pts = sorted(((r["refusal"], r["copy_acc"]) for r in rows if r["method"] == key))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, lw=2, ms=4.5,
            mec=SURFACE, mew=1.2, label=label, zorder=3)
for name, dx, dy in (("theta0", -0.02, 0.012), ("theta1", 0.02, -0.004)):
    r = next(r for r in rows if r["method"] == name)
    ax.plot(r["refusal"], r["copy_acc"], "s", color=INK2, ms=6, zorder=4)
    ax.annotate({"theta0": "θ₀ (aligned)", "theta1": "θ₁ (drifted)"}[name], (r["refusal"], r["copy_acc"]),
                xytext=(r["refusal"] + dx, r["copy_acc"] + dy), color=INK2, fontsize=9,
                ha="right" if name == "theta0" else "left")
ax.axvspan(0.9, 1.02, color=GRID, alpha=0.45, lw=0, zorder=0)
ax.text(0.905, 0.012, "≥ 90% refusal", color=INK2, fontsize=8.5)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 0.36)
ax.set_xlabel("Refusal rate on 512 held-out harmful prompts", color=INK2)
ax.set_ylabel("Copy-task token accuracy (512 held-out)", color=INK2)
ax.set_title("Toy transformer: repair frontiers (up and right is better)", color=INK, loc="left", fontsize=11.5)
ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2, length=0)
for s in ax.spines.values():
    s.set_visible(False)
leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
fig.savefig(HERE / "frontier.png", facecolor=SURFACE)
print("wrote frontier.png")

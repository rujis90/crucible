"""
plot_keeps.py — Sharpe improvement trajectory with annotated keeps.
Run: python plot_keeps.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent

BG     = "#ffffff"
GRID   = "#e0e0e0"
LINE_C = "#2ca02c"
DOT_C  = "#2ca02c"
DISC_C = "#cccccc"
TEXT_C = "#333333"
LABEL_ANGLE = 40

MAX_LABEL_CHARS = 30

STRIP_PREFIXES = [
    "Replace the ", "Replace ", "Tighten the ", "Tighten ",
    "Add a ", "Add ", "Introduce a ", "Make the ",
]


def truncate(desc: str) -> str:
    desc = desc.strip()
    for pfx in STRIP_PREFIXES:
        if desc.startswith(pfx):
            desc = desc[len(pfx):]
            break
    if len(desc) <= MAX_LABEL_CHARS:
        return desc
    return desc[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "\u2026"


def parse_results():
    keeps = []
    discards = []
    exp_num = 0
    for line in (ROOT / "results.tsv").read_text().strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        exp_num += 1
        sharpe = float(parts[1])
        status = parts[5]
        desc = truncate(parts[6])
        if status in ("keep", "baseline"):
            keeps.append((exp_num, sharpe, desc))
        else:
            discards.append((exp_num, sharpe))
    return keeps, discards, exp_num


def evenly_spaced_slots(n, x_start, x_end):
    if n == 1:
        return [(x_start + x_end) / 2]
    step = (x_end - x_start) / (n - 1)
    return [x_start + i * step for i in range(n)]


def plot(keeps, discards, total_experiments):
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="-", zorder=0)

    xs = [k[0] for k in keeps]
    ys = [k[1] for k in keeps]
    lbls = [k[2] for k in keeps]
    n = len(keeps)

    y_lo = min(ys) - 0.04
    y_hi = max(ys) + 0.14

    if discards:
        dx = [d[0] for d in discards if d[1] >= y_lo]
        dy = [d[1] for d in discards if d[1] >= y_lo]
        ax.scatter(dx, dy, color=DISC_C, s=18, zorder=2, alpha=0.4, label="Discarded")

    step_xs = []
    step_ys = []
    for i, (x, y, _) in enumerate(keeps):
        if i > 0:
            step_xs.append(x)
            step_ys.append(keeps[i - 1][1])
        step_xs.append(x)
        step_ys.append(y)
    step_xs.append(total_experiments)
    step_ys.append(ys[-1])

    ax.plot(step_xs, step_ys, color=LINE_C, linewidth=1.8, zorder=3, label="Running best")
    ax.scatter(xs, ys, color=DOT_C, s=60, zorder=4, edgecolors="white", linewidths=0.6, label="Kept")

    text_xs = evenly_spaced_slots(n, 0, total_experiments)

    for i, (x, y, _) in enumerate(keeps):
        label = lbls[i]
        tx = text_xs[i]
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(tx, y_hi - 0.005),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
            zorder=5,
        )
        ax.text(
            tx, y_hi - 0.003, label,
            fontsize=7.5,
            color=TEXT_C,
            rotation=LABEL_ANGLE,
            rotation_mode="anchor",
            ha="right",
            va="bottom",
            zorder=6,
        )

    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("Experiment #", fontsize=10, color=TEXT_C)
    ax.set_ylabel("OOS Sharpe Ratio (higher is better)", fontsize=10, color=TEXT_C)
    ax.set_title(
        f"Crucible \u2014 {total_experiments} Experiments, {n} Kept Improvements",
        fontsize=13, fontweight="bold", color=TEXT_C, pad=12,
    )
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-3, total_experiments + 8)

    out = ROOT / "keeps.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"Saved \u2192 {out}")


if __name__ == "__main__":
    keeps, discards, total = parse_results()
    print(f"Found {len(keeps)} keeps, {len(discards)} discards out of {total} experiments")
    for exp, sharpe, lbl in keeps:
        print(f"  #{exp:3d}  {sharpe:.4f}  {lbl}")
    plot(keeps, discards, total)

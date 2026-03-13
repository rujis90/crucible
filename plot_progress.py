"""
plot_progress.py — generates a performance chart of the current best strategy.
Run: python plot_progress.py
"""

import importlib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

import backtest
import strategy

importlib.reload(strategy)

# ── run walk-forward ──────────────────────────────────────────────────────────
data = backtest.load_data()
close = data["close"]
folds = backtest.get_folds(close.index)

fold_results = []
for fi, (train_dates, test_dates) in enumerate(folds):
    importlib.reload(strategy)
    r = backtest.run_fold(strategy, data, train_dates, test_dates)
    m = backtest.compute_metrics(r)
    fold_results.append((fi + 1, test_dates, r, m))
    year = test_dates[0].year
    print(f"fold {fi+1:02d} ({year}): sharpe={m['sharpe']:+.3f}  dd={m['max_drawdown']:.1f}%  ann={m['ann_return']:.1f}%")

all_rets = pd.concat([r for _, _, r, _ in fold_results]).sort_index()
combined = backtest.compute_metrics(all_rets)
print(f"\nCombined OOS sharpe: {combined['sharpe']:.4f}")
print(f"Ann return: {combined['ann_return']:.1f}%  |  Max DD: {combined['max_drawdown']:.1f}%")

# ── SPY benchmark ─────────────────────────────────────────────────────────────
oos_start = fold_results[0][1][0]
oos_end   = fold_results[-1][1][-1]
spy_raw = yf.download("SPY", start=str(oos_start.date()), end=str(oos_end.date()),
                      auto_adjust=True, progress=False)["Close"].squeeze()
spy_ret = spy_raw.pct_change().dropna().reindex(all_rets.index).fillna(0)

strat_cum = (1 + all_rets).cumprod()
spy_cum   = (1 + spy_ret).cumprod()
roll_max  = strat_cum.cummax()
drawdown  = (strat_cum - roll_max) / roll_max * 100

# ── PLOT ──────────────────────────────────────────────────────────────────────
BG     = "#0f0f14"
GRID   = "#1e1e2e"
CYAN   = "#00d4ff"
ORANGE = "#ff6b35"
RED    = "#ff4455"
YELLOW = "#ffdd44"
TEXT   = "#cccccc"
DIM    = "#666677"

FOLD_BG = ["#0a1a2a", "#0a2a1a", "#1a0a2a", "#2a1a0a", "#0a2a2a",
           "#1a1a0a", "#0a1a1a", "#2a0a1a", "#1a2a0a", "#0a0a2a"]

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(16, 10),
    gridspec_kw={"height_ratios": [3, 1, 1]},
)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(hspace=0.08, left=0.07, right=0.97, top=0.93, bottom=0.06)

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xlim(all_rets.index[0], all_rets.index[-1])

# ── panel 1: equity curve ─────────────────────────────────────────────────────
for fi, (fn, test_dates, r, m) in enumerate(fold_results):
    ax1.axvspan(test_dates[0], test_dates[-1],
                color=FOLD_BG[fi % len(FOLD_BG)], alpha=1.0, zorder=0)

ax1.plot(spy_cum.index, spy_cum.values, color=ORANGE, linewidth=1.3,
         alpha=0.75, label="SPY buy-and-hold", zorder=2)
ax1.plot(strat_cum.index, strat_cum.values, color=CYAN, linewidth=2.0,
         label=(f"Strategy  |  Sharpe {combined['sharpe']:.2f}  "
                f"Ann {combined['ann_return']:.0f}%  "
                f"MaxDD {combined['max_drawdown']:.1f}%"), zorder=3)

ax1.set_ylabel("Growth (\u00d7)", color=TEXT, fontsize=9)
ax1.set_title("Crucible \u2014 Walk-Forward OOS Performance", color="#ffffff",
              fontsize=12, pad=14, fontweight="bold")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}\u00d7"))
ax1.legend(loc="upper left", facecolor="#12121e", edgecolor=GRID,
           labelcolor=TEXT, fontsize=8.5, framealpha=0.9)
ax1.set_xticklabels([])

# ── panel 2: per-fold Sharpe ──────────────────────────────────────────────────
fold_nums  = [r[0] for r in fold_results]
fold_sh    = [r[3]["sharpe"] for r in fold_results]
fold_years = [str(r[1][0].year) for r in fold_results]
bar_colors = [CYAN if s > 0 else RED for s in fold_sh]

bars = ax2.bar(fold_nums, fold_sh, color=bar_colors, alpha=0.8, width=0.65, zorder=2)
ax2.axhline(0, color=DIM, linewidth=0.8)
ax2.axhline(np.mean(fold_sh), color=YELLOW, linewidth=1.0, linestyle="--",
            label=f"Avg {np.mean(fold_sh):.2f}", zorder=3)
for bar, s in zip(bars, fold_sh):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             s + (0.03 if s >= 0 else -0.10),
             f"{s:.2f}", ha="center", va="bottom", color=TEXT, fontsize=7)
ax2.set_ylabel("Sharpe / fold", color=TEXT, fontsize=9)
ax2.set_xticks(fold_nums)
ax2.set_xticklabels([f"{y}" for y in fold_years], color=DIM, fontsize=7.5)
ax2.legend(facecolor="#12121e", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
ax2.set_xlim(0.3, len(fold_nums) + 0.7)

# ── panel 3: drawdown ─────────────────────────────────────────────────────────
ax3.fill_between(drawdown.index, drawdown.values, 0,
                 color=RED, alpha=0.55, linewidth=0, zorder=2)
ax3.plot(drawdown.index, drawdown.values, color="#ff7788", linewidth=0.8, zorder=3)
ax3.set_ylabel("Drawdown %", color=TEXT, fontsize=9)
ax3.set_ylim(min(drawdown.min() * 1.3, -2), 1)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

out = Path(__file__).parent / "progress.png"
fig.savefig(out, dpi=150, facecolor=BG)
print(f"Saved \u2192 {out}")

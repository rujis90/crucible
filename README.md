# crucible

*In metallurgy, a crucible is the vessel where raw metals are subjected to extreme heat until only the purest alloy remains. In quantitative finance, the same principle applies — you throw a hundred strategy ideas into the fire and keep only what survives rigorous out-of-sample validation. This repo automates that process using a survivorship-bias-free Nasdaq-100 universe, triple barrier labels, and Combinatorial Purged Cross-Validation. An AI agent proposes changes, backtests across 15 independent OOS paths, keeps what improves the Sharpe distribution, discards the rest, and repeats. You wake up to a log of experiments and a better strategy. —@henri, 2026*

---

## The problem with most backtests

```python
# What 99% of tutorials tell you to do
import yfinance as yf
tickers = get_current_ndx_100()      # today's 100 survivors
data    = yf.download(tickers, ...)  # +208 bps/yr phantom alpha
```

You're downloading winners — companies that made it to today. YHOO, RIMM, ATVI, CELG were in the Nasdaq-100 at various points. They went to zero or got acquired. By excluding them from your universe you implicitly selected for success, inflating your CAGR by **~208 basis points per year**.

Crucible fixes all three ways backtests lie:

| The lie | The fix |
|---|---|
| Survivorship bias — universe of today's winners | Point-in-time NDX membership (`ndx_pit_daily.parquet`) |
| Look-ahead in labels — `return_in_20_days` | Triple barrier labels: which barrier hit *first* |
| Single train/test path — one lucky draw | CPCV: 15 independent OOS paths, Sharpe distribution |

---

## How it works

**Five files matter:**

- **`backtest.py`** — fixed infrastructure. CPCV harness, signal exits, transaction costs. **Never modify.**
- **`labels.py`** — triple barrier labeling (López de Prado AFML Ch. 3). **Never modify.**
- **`features.py`** — 14 cross-sectionally z-scored features. **Never modify.**
- **`cv.py`** — combinatorial purged CV with embargo. **Never modify.**
- **`strategy.py`** — the single file the agent edits. Signal selection logic and label-driven parameters. **This is the research surface.**

Each experiment takes several minutes (fast walk-forward mode) to roughly 5–10 minutes (full CPCV), depending on hardware. The metric is **`oos_sharpe / oos_sharpe_std`** — mean Sharpe across all 15 combinatorial OOS splits divided by its standard deviation. A robust strategy with Sharpe 0.80 ± 0.08 beats a fragile one at 1.1 ± 0.6.

---

## Quick start

**Requirements:** Python 3.10+, [Claude Code](https://docs.anthropic.com/en/docs/claude-code), NDX PIT dataset.

```bash
# 1. Clone and install
git clone https://github.com/rujis90/crucible.git
cd crucible
pip install -e .

# 2. Get the dataset
# Buy at https://crucible-research.com — $20 one-time, instant download
# Then copy the parquet files:
cp ~/downloads/ndx-pit-data/*.parquet data/

# 3. Optionally merge skill files into CLAUDE.md
cat ~/downloads/ndx-pit-data/skills/SKILL_*.md >> CLAUDE.md

# 4. Verify setup (full CPCV; usually 5-10 min locally)
python backtest.py
```

Expected output:
```
Loading data …
Universe: 101 current NDX members  | 265 historical  | 2010-01-04 → 2026-06-25
Building labels (pt=1.5×vol, sl=1.0×vol, hold=20d) …
Labels: 203 tickers, 4094 total events
Building feature cache …
Features: 4019 date snapshots
Running CPCV (N=6, K=2, embargo=21d) …
  path 00: sharpe=+...  dd=...%  signals=...  hit=...%  ✓
  ...

=======================================================
oos_cagr:        ...
oos_sharpe:      ...
oos_sharpe_std:  ...
cpcv_paths:      15
folds_passed:    ...
max_drawdown:    ...
num_signals:     ...
upper_hit_rate:  ...
avg_signal_ret:  ...
median_signal_ret: ...
profit_factor:   ...
avg_holding_days: ...
signal_frequency: ...
hit_rate_std:    ...
elapsed_seconds: ...
```

---

## Running the research loop

### Manual (recommended to start)

```bash
# Point Claude Code at this repo
claude "read CLAUDE.md and program.md, then run 3 experiments"
```

### Autonomous loop

```bash
# 10 batches × 3 experiments = 30 experiments, runs unattended
bash run_research.sh 10
```

Each batch is a stateless Claude Code invocation — it reads `results.tsv` and `strategy.py` fresh, proposes one change per experiment, backtests it, and commits or discards. No context accumulates between batches.

---

## What the agent optimises

The agent searches for signal rules. A signal is an event: it enters on a
date/ticker and exits when the triple-barrier box is resolved — profit target,
stop loss, or max holding time. There is no portfolio rebalance schedule, max
stock count, or gross exposure cap in the research surface.

```
oos_sharpe:      0.79    ← mean Sharpe across all CPCV paths
oos_sharpe_std:  0.09    ← distribution tightness — the real signal
cpcv_paths:      15      ← C(6,2) paths from 6 time blocks
folds_passed:    13/15
num_signals:     1240    ← accepted signal events across CV splits
upper_hit_rate:  0.47    ← fraction hitting the upper barrier first
avg_signal_ret:  0.0031  ← mean triple-barrier signal return after costs
profit_factor:   1.12    ← gross signal gains / gross signal losses
hit_rate_std:    0.04    ← path-to-path hit-rate stability
```

**The agent keeps a strategy only if signal quality and `oos_sharpe / oos_sharpe_std` improve AND no constraint is violated:**
- `max_drawdown` worse than -35%: discard
- `oos_sharpe_std > 0.5`: discard (too regime-dependent)
- `num_signals < 100`: discard (too sparse)
- `upper_hit_rate < 0.45`: discard (not enough upper-barrier hits)
- `avg_signal_ret <= 0`: discard (no net edge after costs)
- `profit_factor < 1.05`: discard (losses overwhelm gains)
- `hit_rate_std > 0.10`: discard (label edge unstable across paths)

This is the core difference from standard research: the agent can't report one good path and call it done. It has to improve the entire distribution across 15 independent OOS windows.

---

## The dataset

The `data/` directory needs two files you get from [crucible-research.com](https://crucible-research.com):

| File | Size | Contents |
|---|---|---|
| `ndx_ohlcv.parquet` | 31 MB | OHLCV for 210 NDX tickers, 2007–present |
| `ndx_pit_daily.parquet` | 165 KB | Boolean membership: was ticker in NDX on each date? |

The purchase also includes 7 CLAUDE.md skill files covering the full methodology (PIT filtering, triple barrier labels, CPCV, feature engineering, position sizing, regime detection, and strategy research). Append them to `CLAUDE.md` and Claude understands the framework immediately.

**Why not just use yfinance?** `yf.download(ndx_100_today)` uses today's survivors. The PIT dataset tracks all 265 historical NDX members with exact inclusion dates. That 208 bps/yr gap is the difference between testing a hypothesis and measuring luck.

---

## Project structure

```
backtest.py           — CPCV harness + signal exits (do not modify)
labels.py             — triple barrier labeling
features.py           — 14 CS z-scored features
cv.py                 — purging, embargo, CPCV splits
strategy.py           — signal selection logic (agent modifies this)
CLAUDE.md             — methodology guide for Claude Code
program.md            — research search space and guiding principles
research_step.md      — per-batch agent instructions
run_research.sh       — autonomous batch runner
results.tsv           — experiment log (append-only)
data/
  ndx_ohlcv.parquet        ← buy at crucible-research.com
  ndx_pit_daily.parquet    ← buy at crucible-research.com
  README.md
```

---

## Design choices

**Single file to modify.** The agent only touches `strategy.py`. Diffs are small, reviewable, and revertable with `git checkout -- strategy.py`.

**Sharpe distribution, not a number.** CPCV generates C(6,2)=15 independent OOS equity curves. Reporting `oos_sharpe ± oos_sharpe_std` is honest. Reporting one Sharpe from one path is a lottery ticket.

**Purging + embargo prevents leakage.** A 20-day label that starts near a test window doesn't resolve until the test period has started. Purging removes those training samples. Embargo adds a gap equal to the longest feature lookback. Both are required — either alone is not enough.

**Triple-barrier signal exits.** Signals are dated ticker events. Once fired, a signal exits when the precomputed triple-barrier box resolves: profit target, stop loss, or max holding time. There is no portfolio rebalance schedule, max-stock cap, or gross exposure target.

**Stateless batches.** Each Claude invocation reads current state from files, not memory. This prevents context bloat and keeps experiments reproducible.

**Git as checkpoint.** Improvements are committed; failures are reverted. `results.tsv` is append-only — the full audit trail never changes.

---

## Acknowledgements

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — the same idea of autonomous AI-driven research applied to portfolio strategy rather than LLM training. The methodology follows Marcos López de Prado's *Advances in Financial Machine Learning* (triple barrier labels, CPCV, feature engineering) and Nassim Taleb's robustness-first thinking (prefer the strategy that survives 2008, 2020, and 2022 over the one that peaks in a bull run).

---

## License

MIT

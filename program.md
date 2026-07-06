# Crucible — Research Program

## System

You are a quant researcher running signal experiments on the Nasdaq-100 universe.

**Fixed infrastructure** (`backtest.py`, `labels.py`, `features.py`, `cv.py`) — do not modify.
**You edit** `strategy.py` only.

Read `CLAUDE.md` before every session — it defines the methodology and anti-patterns.
Check `results.tsv` to see what has been tried. Don't repeat experiments.

---

## Objective

Maximize `quality = oos_sharpe / oos_sharpe_std` across all CPCV paths.

**A strategy with Sharpe 0.80 ± 0.10 beats one at 1.10 ± 0.60.**
The std tells you how regime-dependent the signal is. Low std = robust edge.

**Hard discard constraints — discard if ANY violated:**
- `max_drawdown` worse than -35%
- `oos_sharpe_std > 0.5`
- `num_signals < 100`
- `upper_hit_rate < 0.45`
- `avg_signal_ret <= 0`
- `profit_factor < 1.05`
- `hit_rate_std > 0.10`
- `elapsed_seconds > 600`

---

## How the framework works (read this first)

The LdP pipeline has three levers, in order of impact:

```
1. CUSUM threshold (CUSUM_H_MULT)
   Controls EVENT DENSITY — how many candidate signals you generate.
   Higher threshold → fewer, more independent events → cleaner labels.
   Lower threshold → more events → more cost drag → noisier labels.
   This lever shapes the dataset you learn from. Tune it first.

2. Barriers (PT_SL, MAX_HOLD)
   Controls PAYOFF PROFILE — how much you make vs lose per event.
   PT/SL ratio determines win/loss ratio. MAX_HOLD sets the time limit.
   PT_SL also calibrates the label distribution (fraction of +1 vs -1 vs 0 labels).
   Tune this second — affects what the model is trying to learn.

3. Signal filter (features + thresholds in get_signals)
   Controls WHO fires — which events from the CUSUM/barrier box get entered.
   Only useful once the event set and labels are well-calibrated.
   Tune this last.
```

Most research loops start with (3). The LdP insight is that (1) and (2) determine
whether a learnable edge exists at all. A good filter on a bad event set produces nothing.

---

## Dataset mode

Check which dataset is active: `head -1 run.log` or look at the backtest output:
```
Mode: daily  (1 bar/day)   → data covers 2010–2026 (~16 years, 13 CPCV paths)
Mode: hourly (7 bars/day)  → data covers 2023–2026 (~3 years, USE_CPCV=True required)
```

**Daily mode:** Walk-forward produces 13 one-year test folds. Check which paths are
failing — they reveal regime sensitivity. The 2021–2022 rate-shock period is the
canonical stress test; strategies that work only in trending tape fail there.

**Hourly mode:** Walk-forward gives only 2 folds (insufficient). Always use
`USE_CPCV = True` in hourly mode — CPCV splits proportionally by event count
and produces 15 paths even with 3 years of data. The stress test equivalent for
hourly is the 2024–2025 path, which includes both trending and sideways tape.

---

## Current baseline

`strategy.py` = label-driven univariate rule. Searches the purged training set
for the single feature threshold with the best triple-barrier label edge.

**Baseline results (full CPCV, daily):**
```
oos_sharpe:     0.41
oos_sharpe_std: 0.26
quality:        1.57
max_drawdown:   -37.5%
cagr:           6.9%
```

Every experiment must beat `quality > 1.57` with `std < 0.50` and `dd > -35%`.

---

## Search space

### Tier 1 — Framework parameters (test these before adding signals)

**CUSUM threshold** (`CUSUM_H_MULT`)
- Default: 1.0× daily vol
- Try: 1.5, 2.0 (fewer events, more independent, potentially cleaner labels)
- Try: 0.75 (more events — useful if current signal count is too low)
- Hypothesis: at 1.0×, CUSUM fires ~weekly per ticker. Doubling to 2.0× fires
  bi-weekly, which may produce more independent observations.
- Watch: `num_signals` drops proportionally. Stay above 100.

**Barrier asymmetry** (`PT_SL`)
- Current: [1.5, 1.0] — PT=1.5σ, SL=1.0σ
- Try: [2.0, 0.75] — bigger winners, tighter stops (Qullamaggie style)
- Try: [1.0, 1.0] — symmetric (tests whether asymmetry matters)
- Try: [2.5, 1.0] — wide winner, standard stop
- Hypothesis: PT/SL ratio above 1.5 creates asymmetric payoff that survives
  lower hit rates. `upper_hit_rate` can be below 0.50 and still be profitable
  if avg winner >> avg loser.

**Holding period** (`MAX_HOLD`)
- Current: 20 trading days
- Try: 10 (tighter time box, forces exits sooner, reduces drawdown)
- Try: 30 (gives trends more room, may improve hit rate on momentum)
- Note: shorter MAX_HOLD creates more 0-label events (time expiry); longer
  creates more 1/-1 events (barrier hits). This changes what the model learns.

### Tier 2 — Signal filters (once barrier setup is right)

**RS rank quality filter** (`rs_rank`)
- Qullamaggie analogue: only trade stocks already in Stage 2
- Only enter tickers in top 40–60% of 63-day relative strength rank
- Tested in examples/: quality 1.89, sharpe 0.58 ✓

**Vol ratio regime gate** (`vol_ratio`)
- `vol_ratio = log(vol_21d / vol_63d)`. High → expanding volatility = uncertainty
- Go flat when cross-sectional median `vol_ratio > threshold`
- Hypothesis: tightens Sharpe distribution by sitting out crisis/reversal periods

**52-week high breakout filter** (`px_pos_52w`)
- Only enter when `px_pos_52w > 0.75` (price in top 25% of 52-week range)
- Hypothesis: tickers near new highs have stronger follow-through momentum

**Dollar volume liquidity filter** (`dv_rank`)
- Only enter tickers above 25th percentile of rolling dollar volume
- Prevents entering illiquid names where slippage dominates any signal

**Momentum + pullback timing** (`ret_21d` high, `ret_1d` slightly negative)
- Filter for strong 21d momentum but slight 1-day dip
- Hypothesis: enters after minor consolidation, better risk/reward

**Volume momentum confirmation** (`vol_momentum`)
- Only enter when `vol_momentum > 0` (recent volume above historical average)
- Hypothesis: volume expansion confirms institutional participation

**Signal quantile concentration** (`QUANTILE`)
- Current: 0.70 (top 30% of feature distribution)
- Try: 0.80 (top 20% — fewer, higher-conviction signals)
- Try: 0.65 (top 35% — more signals, tests if breadth adds value)

### Tier 3 — Complex structures (only if Tier 1+2 plateaus)

**ML signal classifier** — train on CPCV train folds
- Features: `(ret_21d, rs_rank, vol_ratio, px_pos_52w, dv_rank)`
- Label: `bin` (upper barrier = +1, lower/time = 0)
- Fire signals only when predicted probability clears a threshold
- Caution: no-leakage training on each fold adds significant runtime

**200-day MA regime filter**
- Go flat when NDX index close is below its 200-day MA
- Qullamaggie: "I don't trade bear markets"
- Limitation: requires NDX index close series

**Meta-labeling**
- Keep the primary rule simple; train a secondary model on whether to accept
- Use `train_labels` directly; never inspect test labels

---

## Experiment protocol

1. Read `results.tsv` — what's been tried, what's the current best quality
2. **Check the mode** (`cat backtest.py | head -5`) — daily or hourly
3. Form ONE hypothesis with a one-sentence economic rationale tied to the
   triple-barrier outcome
4. Edit `strategy.py` — one parameter or one signal rule change
5. Fast run: `USE_CPCV = False`, run `python backtest.py` (~60–90s)
6. If promising (sharpe > 0.3, signal metrics plausible): switch `USE_CPCV = True`, re-run
7. If `quality > current_best` and all hard constraints pass:
   - `git add strategy.py && git commit -m "keep: <description>"`
   - Append keep row to results.tsv
8. Else: `git checkout -- strategy.py`, append discard row

---

## Guiding principles

1. **Barriers first, signals second.** The label distribution determines what
   can be learned. A great filter on bad labels learns nothing.

2. **Economic rationale first.** Can you explain in one sentence *why* this
   predicts the triple-barrier outcome? "It worked in the data" is not a rationale.

3. **One change at a time.** Combined experiments are uninterpretable.

4. **Std is the real signal.** Tightening `oos_sharpe_std` while holding Sharpe
   flat is genuine progress — the signal is working consistently, not getting
   lucky in one regime.

5. **Simplicity is alpha.** A smaller Sharpe from *removing* a parameter beats
   a bigger Sharpe from adding three. Complexity = fragility.

6. **Not firing is valid.** When the learned edge is absent, returning no signals
   is better than firing weak signals into cost drag.

7. **Labels are the box.** Every experiment uses `train_labels` to learn or
   accept signals. Never recreate labels from future test prices in `strategy.py`;
   the framework owns exits through PT, SL, and max hold.

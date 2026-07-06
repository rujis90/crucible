# Research Step — Run 3 Experiments

You are a quant researcher. Run exactly 3 experiments then stop.

## Before starting

1. `head -3 backtest.py` — check which MODE is active (daily or hourly)
2. `git log --oneline | head -10` — what's been committed
3. `cat results.tsv` — full history (don't repeat experiments)
4. `cat strategy.py` — current best params and signal rule
5. `cat program.md` — full search space and principles

## Per experiment

1. **Choose ONE change.** Priority order (per program.md):
   - Tier 1 first: `CUSUM_H_MULT`, `PT_SL`, `MAX_HOLD`
   - Tier 2 after: feature filter, quantile threshold, signal rule
   Justify why it should improve the triple-barrier outcome (1 sentence).

2. Edit `strategy.py`:
   - You MUST use `train_labels` as the training target or acceptance target.
   - You MUST NOT inspect, infer, or recreate test labels inside `strategy.py`.
   - You MUST NOT add portfolio rebalancing, max-stock caps, gross exposure limits,
     or position sizing. Exits are handled by the triple-barrier box in `backtest.py`.

3. Run fast first: set `USE_CPCV = False`, run `python backtest.py > run.log 2>&1`
   - **Exception:** if MODE is hourly, always use `USE_CPCV = True`
     (walk-forward gives only 2 folds on 3 years of data — not enough signal)

4. Parse:
   ```
   grep "^oos_sharpe:\|^oos_sharpe_std:\|^cpcv_paths:\|^folds_passed:\|^max_drawdown:\|^num_signals:\|^upper_hit_rate:\|^avg_signal_ret:\|^median_signal_ret:\|^profit_factor:\|^avg_holding_days:\|^signal_frequency:\|^hit_rate_std:\|^elapsed_seconds:" run.log
   ```

5. Hard constraints — discard if ANY violated:
   - `max_drawdown` worse than -35%
   - `oos_sharpe_std > 0.5`
   - `num_signals < 100`
   - `upper_hit_rate < 0.45`
   - `avg_signal_ret <= 0`
   - `profit_factor < 1.05`
   - `hit_rate_std > 0.10`
   - `elapsed_seconds > 600`

6. Compute `quality = oos_sharpe / oos_sharpe_std`

7. If no `keep` rows exist in `results.tsv`, the target is: pass every hard
   constraint and beat the best baseline/discard quality.

8. If `quality > current_best_quality` AND constraints pass:
   - If fast run only: switch `USE_CPCV = True`, re-run to confirm
   - `git add strategy.py && git commit -m "keep: <description> sharpe=X std=Y quality=Z"`
   - Append keep row to results.tsv

9. Else:
   - `git checkout -- strategy.py`
   - Append discard row to results.tsv

After exactly 3 experiments: **stop**.

## results.tsv format

```
<commit>	<oos_sharpe>	<oos_sharpe_std>	<quality>	<folds_passed>	<max_drawdown>	<num_signals>	<upper_hit_rate>	<avg_signal_ret>	<median_signal_ret>	<profit_factor>	<avg_holding_days>	<signal_frequency>	<hit_rate_std>	<elapsed_s>	<keep/discard>	<description>
```

## Regime stress test (mode-dependent)

**Daily mode:** paths covering 2021–2022 are the stress test (rate-shock bear market).
Check which `path_id` Sharpes are failing — real edges survive 2022.

**Hourly mode:** paths covering high-vol periods within 2024–2025 are the stress test.
With CPCV, you get 15 paths — check whether Sharpe spread is tight (std < 0.3 = robust).

## Key reminders

- `CUSUM_H_MULT`: try 1.5 or 2.0 before adding signal complexity — cleaner events
  improve label quality more than better filters on noisy events
- `PT_SL = [2.0, 0.75]` lets the framework win big and cut fast — test it early
- `USE_CPCV = False` for fast daily iteration; always True for hourly
- No signal is better than a weak signal
- Optimize signal quality first. High Sharpe with poor hit-rate stability,
  too few signals, or negative avg_signal_ret is not a keep.

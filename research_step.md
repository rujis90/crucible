# Research Step — Run 3 Experiments

You are a quant researcher. Run exactly 3 experiments then stop.

## Before starting

1. `git log --oneline | head -10` — what's been committed
2. `cat results.tsv` — full history
3. `cat strategy.py` — current best
4. `cat program.md` — search space and principles

## Per experiment

1. Choose ONE focused signal-rule change. Justify why it should improve the triple-barrier label outcome (1 sentence).
2. Edit `strategy.py`
   - You MUST use `train_labels` as the training target or acceptance target.
   - You MUST NOT inspect, infer, or recreate test labels inside `strategy.py`.
   - You MUST NOT add portfolio rebalancing, max-stock caps, gross exposure limits, or position sizing.
   - The output remains dated ticker signals only; exits are handled by the triple-barrier box in `backtest.py`.
3. Run: `python backtest.py > run.log 2>&1`
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
7. If no `keep` rows exist in `results.tsv`, the target is: pass every hard constraint and beat the best baseline/discard quality.
8. If `quality > current_best_quality` AND constraints pass:
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

## Key reminders

- 2022 is the stress test. Check `path_id` Sharpes — which year is failing?
- `USE_CPCV = False` in strategy.py for fast iteration; switch back to True before keeping
- Don't repeat experiments that are already in results.tsv
- No signal is better than a weak signal
- Do not add portfolio rebalancing, max-stock caps, or gross exposure logic; signals enter and exit through the triple-barrier box
- Optimize signal quality first. High Sharpe with poor hit-rate stability, too few signals, or negative average signal return is not a keep.

# Research Step — Run 3 Experiments

You are a quant researcher. Run exactly 3 experiments then stop.

## Before starting

1. `git log --oneline | head -10` — what's been committed
2. `cat results.tsv` — full history
3. `cat strategy.py` — current best
4. `cat program.md` — search space and principles

## Per experiment

1. Choose ONE focused change. Justify it with an economic rationale (1 sentence).
2. Edit `strategy.py`
3. Run: `python backtest.py > run.log 2>&1`
4. Parse:
   ```
   grep "^oos_sharpe:\|^oos_sharpe_std:\|^cpcv_paths:\|^folds_passed:\|^max_drawdown:\|^elapsed_seconds:" run.log
   ```
5. Hard constraints — discard if ANY violated:
   - `max_drawdown` worse than -35%
   - `oos_sharpe_std > 0.5`
   - `elapsed_seconds > 600`
6. Compute `quality = oos_sharpe / oos_sharpe_std`
7. If `quality > current_best_quality` AND constraints pass:
   - `git add strategy.py && git commit -m "keep: <description> sharpe=X std=Y quality=Z"`
   - Append keep row to results.tsv
8. Else:
   - `git checkout -- strategy.py`
   - Append discard row to results.tsv

After exactly 3 experiments: **stop**.

## results.tsv format

```
<commit>	<oos_sharpe>	<oos_sharpe_std>	<folds_passed>	<max_drawdown>	<elapsed_s>	<keep/discard>	<description>
```

## Key reminders

- 2022 is the stress test. Check `path_id` Sharpes — which year is failing?
- `USE_CPCV = False` in strategy.py for fast iteration; switch back to True before keeping
- Don't repeat experiments that are already in results.tsv
- Cash (zero weights) is better than holding losers

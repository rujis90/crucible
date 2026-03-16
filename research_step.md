# Research Step — Run 3 Experiments

You are a quant researcher. Run exactly 3 experiments then stop.

## Context (read before starting)

Read these to understand current state:
1. `git log --oneline | head -15` — what's been committed
2. `cat results.tsv` — full experiment history
3. `cat strategy.py` — current best strategy
4. `cat program.md` — search space, guiding principles, economic rationale

## Your job: 3 experiments

For each experiment:

1. Choose ONE focused change based on what results.tsv shows has/hasn't worked
2. Edit `strategy.py`
3. Run: `python backtest.py > run.log 2>&1`
4. Parse: `grep "^oos_sharpe:\|^folds_passed:\|^max_drawdown:\|^elapsed_seconds:" run.log`
5. Hard constraints — discard if any violated:
   - `max_drawdown` worse than -45%
   - `elapsed_seconds` > 300
6. If `oos_sharpe` > current best AND constraints pass:
   - `git add strategy.py && git commit -m "keep: <description> oos_sharpe=X"`
   - Append keep row to results.tsv
7. Else:
   - `git checkout -- strategy.py`
   - Append discard row to results.tsv

After exactly 3 experiments: **stop**.

## results.tsv format
Append rows — do not rewrite the file:
```
<commit_or_n/a>	<oos_sharpe>	<folds_passed>	<max_drawdown>	<elapsed_s>	<keep/discard>	<description>
```

## Guiding principles
- Economic rationale before every change
- Read results.tsv carefully — don't repeat what's already been tried
- Cash (all-zero weights) is better than holding losers
- 2008 and 2022 are the hardest folds — any real improvement must handle them
- Prefer robustness across all folds over peak Sharpe in a few
- Absolute momentum filter is the most important protection — keep or strengthen it

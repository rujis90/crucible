"""
optimize.py — Bayesian hyperparameter search for strategy.py.

Uses Optuna (Tree-structured Parzen Estimator) to search the parameter space
and maximize oos_sharpe. Much better than hand-tuning: explores the space
systematically, builds a probabilistic model of promising regions, and shows
you the sensitivity landscape (which params actually matter).

Usage:
    python optimize.py              # 200 trials, maximize oos_sharpe
    python optimize.py --trials 500 # more thorough
    python optimize.py --metric oos_calmar  # optimize a different metric

The best params are printed at the end and written to best_params.json.
"""

import argparse
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load data once (expensive) ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt

DATA  = bt.load_data()
FOLDS = bt.get_folds(DATA["close"].index)

# ── Parameter space ───────────────────────────────────────────────────────────
# Each entry: (type, low, high[, step])  or  (type, choices)
# "int"   → trial.suggest_int
# "float" → trial.suggest_float
# "cat"   → trial.suggest_categorical
PARAM_SPACE = {
    "MA_SLOW":            ("int",   100, 250, 10),
    "VOL_SHORT":          ("int",   5,   20,  1),
    "VOL_LONG":           ("int",   30,  120, 5),
    "VOL_SHOCK_MULT":     ("float", 1.2, 3.0),
    "MOM_WINDOW":         ("int",   42,  252, 21),
    "BOND_MOM_WINDOW":    ("int",   21,  168, 21),
    "VOL_ACC_SHORT":      ("int",   5,   42,  1),
    "VOL_ACC_LONG":       ("int",   42,  126, 5),
    "VOL_ACC_THRESHOLD":  ("float", 0.60, 1.00),
    # Regime scaling params
    "CRISIS_SCALE":       ("float", 0.0, 1.5),
    "ENTROPY_SCALE":      ("float", 0.0, 1.0),
    "ENTROPY_THRESHOLD":  ("float", 0.2, 0.8),
    "REGIME_FLOOR":       ("float", 0.0, 0.5),
}


def sample_params(trial: optuna.Trial) -> dict:
    p = {}
    for name, spec in PARAM_SPACE.items():
        kind = spec[0]
        if kind == "int":
            p[name] = trial.suggest_int(name, spec[1], spec[2], step=spec[3])
        elif kind == "float":
            p[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "cat":
            p[name] = trial.suggest_categorical(name, spec[1])
    # Constraint: VOL_SHORT < VOL_LONG
    if p["VOL_SHORT"] >= p["VOL_LONG"]:
        raise optuna.exceptions.TrialPruned()
    # Constraint: VOL_ACC_SHORT < VOL_ACC_LONG
    if p["VOL_ACC_SHORT"] >= p["VOL_ACC_LONG"]:
        raise optuna.exceptions.TrialPruned()
    return p


def run_strategy_with_params(params: dict) -> dict:
    """Run the full walk-forward backtest with given hyperparameters."""
    import strategy as strat
    importlib.reload(strat)

    # Patch strategy module constants
    for k, v in params.items():
        setattr(strat, k, v)

    fold_rets = []
    for train_dates, test_dates in FOLDS:
        r = bt.run_fold(strat, DATA, train_dates, test_dates)
        fold_rets.append(r)

    all_rets = pd.concat(fold_rets).sort_index()
    return bt.compute_metrics(all_rets)


def objective(trial: optuna.Trial, metric: str) -> float:
    try:
        params = sample_params(trial)
        metrics = run_strategy_with_params(params)
        return metrics[metric.replace("oos_", "")]
    except optuna.exceptions.TrialPruned:
        raise
    except Exception:
        return -999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "calmar", "sortino", "cagr"])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    print(f"Bayesian search: {args.trials} trials, maximizing oos_{args.metric}")
    print(f"Parameter space: {len(PARAM_SPACE)} dimensions")
    print(f"Walk-forward folds: {len(FOLDS)}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    t0 = time.time()
    study.optimize(
        lambda t: objective(t, args.metric),
        n_trials=args.trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    best_params = best.params

    print(f"\n{'='*55}")
    print(f"Best oos_{args.metric}: {best.value:.4f}")
    print(f"Completed in {elapsed:.0f}s ({args.trials} trials)")
    print(f"\nBest parameters:")
    for k, v in best_params.items():
        current = getattr(__import__("strategy"), k, "?")
        marker = " ← changed" if str(v) != str(current) else ""
        print(f"  {k:25s} = {v}{marker}")

    # Importance analysis — which params actually matter
    print(f"\nParameter importance (how much each param drives {args.metric}):")
    try:
        importance = optuna.importance.get_param_importances(study)
        for k, v in sorted(importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            print(f"  {k:25s} {bar} {v:.3f}")
    except Exception:
        pass

    # Write best params to file
    out = Path(__file__).parent / "best_params.json"
    with open(out, "w") as f:
        json.dump({"metric": args.metric, "value": best.value, "params": best_params}, f, indent=2)
    print(f"\nSaved to {out}")
    print("Apply with: python apply_params.py")


if __name__ == "__main__":
    main()

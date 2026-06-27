"""
strategy.py — AGENT EDITABLE. Modify freely.

─────────────────────────────────────────────────────────────────────────────
Experiment 2: rs_rank quality filter + vol_ratio regime gate.

Building on exp 1 (rs_rank improved Sharpe 0.41→0.58 but std stayed high at 0.30).
Hypothesis: the low-Sharpe paths cover high-volatility regimes (2022, COVID).
When the cross-sectional median vol_ratio > threshold (vol expanding across NDX),
the momentum signal degrades. Suppressing signals in these periods should tighten
the Sharpe distribution (lower std) without hurting mean Sharpe much.

Economic rationale: vol expansion = uncertainty = momentum premium disappears.
Investors rotate to defensives; cross-sectional momentum stops working as inter-
stock correlations spike toward 1 during risk-off periods.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────
REBALANCE_EVERY  = 5
PT_SL            = [1.5, 1.0]
MAX_HOLD         = 20
USE_CPCV         = True

RS_RANK_MIN      = 0.6    # top 40% by 63-day relative strength
VOL_RATIO_GATE   = 0.4    # go flat when median vol_ratio exceeds this threshold
                           # vol_ratio is log(vol_21d / vol_63d): >0 = expanding


def get_signals(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Series]:
    signals_by_date: dict[pd.Timestamp, pd.Series] = {}

    if isinstance(test_features.index, pd.MultiIndex):
        test_dates = test_features.index.get_level_values("t0").unique()
    else:
        test_dates = test_features.index.unique()

    for date in test_dates:
        try:
            if isinstance(test_features.index, pd.MultiIndex):
                snapshot = test_features.xs(date, level="t0")
            else:
                snapshot = test_features.loc[[date]]

            if snapshot.empty or "ret_21d" not in snapshot.columns:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            # ── Regime gate: suppress all signals when vol is expanding ──────
            if "vol_ratio" in snapshot.columns:
                median_vol_ratio = snapshot["vol_ratio"].median()
                if median_vol_ratio > VOL_RATIO_GATE:
                    # Go flat — cash is better when vol is expanding fast
                    signals_by_date[date] = pd.Series(0.0, index=snapshot.index)
                    continue

            momentum = snapshot["ret_21d"].dropna()
            if momentum.empty:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            # ── Quality gate: top 40% by 63-day relative strength ────────────
            if "rs_rank" in snapshot.columns:
                rs = snapshot["rs_rank"].reindex(momentum.index)
                rs_pct = rs.rank(pct=True)
                eligible = rs_pct[rs_pct >= RS_RANK_MIN].index
                momentum = momentum.reindex(eligible).dropna()

            if momentum.empty:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            top_n = min(10, len(momentum))
            top_tickers = momentum.nlargest(top_n)

            signals = pd.Series(0.0, index=snapshot.index)
            signals[top_tickers.index] = 1.0 / top_n
            signals_by_date[date] = signals

        except Exception:
            signals_by_date[date] = pd.Series(dtype=float)

    return signals_by_date

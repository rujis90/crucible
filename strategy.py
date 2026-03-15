"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
R06: Add absolute momentum filter to bull mode
QQQ must be above MA200 AND have positive 6m absolute momentum.
If QQQ fails absolute mom, go defensive even if above MA200.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np

REBALANCE_EVERY = 21
MA_SLOW = 200
VOL_SHORT = 10
VOL_LONG = 60
VOL_SHOCK_MULT = 2.0
MOM_WINDOW = 126  # 6-month absolute momentum


def get_weights(data: dict) -> pd.Series:
    close = data["close"]
    weights = pd.Series(0.0, index=close.columns)
    if "QQQ" not in close.columns:
        return weights
    if len(close) < MA_SLOW:
        weights["QQQ"] = 1.0
        return weights

    qqq = close["QQQ"]
    ma200_qqq = qqq.iloc[-MA_SLOW:].mean()
    above_ma_qqq = qqq.iloc[-1] >= ma200_qqq

    # Absolute momentum filter
    abs_mom_ok = True
    if len(qqq) >= MOM_WINDOW:
        raw_mom = qqq.iloc[-1] / qqq.iloc[-MOM_WINDOW] - 1
        abs_mom_ok = raw_mom > 0

    # Credit spread filter: HYG above MA200
    credit_ok = True
    if "HYG" in close.columns and len(close["HYG"].dropna()) >= MA_SLOW:
        hyg = close["HYG"].dropna()
        ma200_hyg = hyg.iloc[-MA_SLOW:].mean()
        credit_ok = hyg.iloc[-1] >= ma200_hyg

    # Vol shock detection
    returns = qqq.pct_change().dropna()
    vol_short = returns.iloc[-VOL_SHORT:].std() * np.sqrt(252)
    vol_long = returns.iloc[-VOL_LONG:].std() * np.sqrt(252)
    vol_shock = vol_short > VOL_SHOCK_MULT * vol_long

    if above_ma_qqq and abs_mom_ok and credit_ok and not vol_shock:
        weights["QQQ"] = 1.0
    elif "SHY" in close.columns:
        weights["SHY"] = 1.0
    return weights

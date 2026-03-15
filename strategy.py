"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
R14: Volume accumulation confirmation
In bull mode, add volume momentum check: if 21d avg vol < 63d avg vol
(distribution signal), scale down QQQ exposure to 50%. Catches institutional
selling disguised in a rising price.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np

REBALANCE_EVERY = 21
MA_SLOW = 200
VOL_SHORT = 10
VOL_LONG = 60
VOL_SHOCK_MULT = 2.0
MOM_WINDOW = 126   # 6-month
BOND_MOM_WINDOW = 63
BOND_VOL_WINDOW = 21
BOND_UNIVERSE = ["TLT", "IEF", "SHY"]
VOL_ACC_SHORT = 21
VOL_ACC_LONG = 63


def get_weights(data: dict) -> pd.Series:
    close = data["close"]
    volume = data["volume"]
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
        # Volume accumulation check
        equity_wt = 1.0
        if "QQQ" in volume.columns and len(volume["QQQ"].dropna()) >= VOL_ACC_LONG:
            qqq_vol = volume["QQQ"].dropna()
            avg_vol_short = qqq_vol.iloc[-VOL_ACC_SHORT:].mean()
            avg_vol_long = qqq_vol.iloc[-VOL_ACC_LONG:].mean()
            if avg_vol_long > 0 and avg_vol_short < 0.8 * avg_vol_long:
                # Weak volume — distribution signal — scale down to 50%
                equity_wt = 0.5
        weights["QQQ"] = equity_wt
        if equity_wt < 1.0 and "SHY" in close.columns:
            weights["SHY"] = 1.0 - equity_wt
    elif above_ma_qqq and not credit_ok and not vol_shock:
        # Stagflation: try GLD
        gld_ok = False
        if "GLD" in close.columns:
            gld = close["GLD"].dropna()
            if len(gld) >= MOM_WINDOW:
                gld_mom = gld.iloc[-1] / gld.iloc[-MOM_WINDOW] - 1
                gld_ok = gld_mom > 0
        if gld_ok:
            weights["GLD"] = 1.0
        else:
            bond_candidates = _get_bond_candidates(close, BOND_MOM_WINDOW, BOND_VOL_WINDOW)
            _assign_bonds(weights, bond_candidates, close)
    else:
        bond_candidates = _get_bond_candidates(close, BOND_MOM_WINDOW, BOND_VOL_WINDOW)
        _assign_bonds(weights, bond_candidates, close)
    return weights


def _get_bond_candidates(close, mom_window, vol_window):
    candidates = []
    for bond in BOND_UNIVERSE:
        if bond not in close.columns:
            continue
        s = close[bond].dropna()
        if len(s) < max(mom_window, vol_window + 1):
            continue
        mom = s.iloc[-1] / s.iloc[-mom_window] - 1
        if mom <= 0:
            continue
        vol = s.pct_change().iloc[-vol_window:].std() * np.sqrt(252)
        candidates.append((bond, vol))
    return candidates


def _assign_bonds(weights, bond_candidates, close):
    if not bond_candidates:
        if "SHY" in close.columns:
            weights["SHY"] = 1.0
    else:
        inv_vols = np.array([1.0 / (v + 1e-9) for _, v in bond_candidates])
        inv_vols /= inv_vols.sum()
        for (bond, _), w in zip(bond_candidates, inv_vols):
            weights[bond] = w

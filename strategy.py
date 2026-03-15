"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
R13: GLD as stagflation regime (3rd regime)
If credit fails (HYG below MA200) but equity has no vol shock and QQQ
above MA200 → stagflation/inflation regime → allocate to GLD if positive
momentum, else bonds.
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
    elif above_ma_qqq and not credit_ok and not vol_shock:
        # Stagflation: equity technically fine, but credit stressed
        # Try GLD if positive momentum
        gld_ok = False
        if "GLD" in close.columns:
            gld = close["GLD"].dropna()
            if len(gld) >= MOM_WINDOW:
                gld_mom = gld.iloc[-1] / gld.iloc[-MOM_WINDOW] - 1
                gld_ok = gld_mom > 0
        if gld_ok:
            weights["GLD"] = 1.0
        else:
            # Fall through to bond rotation
            bond_candidates = _get_bond_candidates(close, BOND_MOM_WINDOW, BOND_VOL_WINDOW)
            if not bond_candidates:
                if "SHY" in close.columns:
                    weights["SHY"] = 1.0
            else:
                inv_vols = np.array([1.0 / (v + 1e-9) for _, v in bond_candidates])
                inv_vols /= inv_vols.sum()
                for (bond, _), w in zip(bond_candidates, inv_vols):
                    weights[bond] = w
    else:
        # Full defensive: inv-vol weighted bonds
        bond_candidates = _get_bond_candidates(close, BOND_MOM_WINDOW, BOND_VOL_WINDOW)
        if not bond_candidates:
            if "SHY" in close.columns:
                weights["SHY"] = 1.0
        else:
            inv_vols = np.array([1.0 / (v + 1e-9) for _, v in bond_candidates])
            inv_vols /= inv_vols.sum()
            for (bond, _), w in zip(bond_candidates, inv_vols):
                weights[bond] = w
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

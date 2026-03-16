"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
Regime-aware strategy with continuous HMM feature scaling.

The HMM pre-computes three continuous features (regime_scores.csv):
  crisis_prob : P(state is negative-return) in [0,1]
  bull_prob   : P(state is positive-return, low vol) in [0,1]
  entropy     : normalized uncertainty in [0,1]
    (0 = model is certain, 1 = completely uncertain / transitioning)

These scale equity exposure continuously — no hard state switching:
  equity_wt *= max(REGIME_FLOOR,
                   1 - CRISIS_SCALE  × crisis_prob
                     - ENTROPY_SCALE × max(0, entropy - ENTROPY_THRESHOLD))

Parameters:
  CRISIS_SCALE       — how aggressively to reduce exposure in crisis
  ENTROPY_SCALE      — how aggressively to reduce exposure in uncertainty
  ENTROPY_THRESHOLD  — entropy level below which we ignore it (normal noise)
  REGIME_FLOOR       — minimum equity weight when regime is very bad

All other params from Bayesian global optimization (best_params.json).
─────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import numpy as np
import pandas as pd

REBALANCE_EVERY = 21
BOND_UNIVERSE   = ["TLT", "IEF", "SHY"]
BOND_VOL_WINDOW = 21

# ── Global params (Bayesian-optimized) ────────────────────────────────────────
MA_SLOW         = 210
VOL_SHORT       = 7
VOL_LONG        = 50
VOL_SHOCK_MULT  = 1.7964540114516958
MOM_WINDOW      = 105
BOND_MOM_WINDOW = 105
VOL_ACC_SHORT   = 35
VOL_ACC_LONG    = 77
VOL_ACC_THRESHOLD = 0.991949807450022

# ── Regime scaling params (to be optimized) ───────────────────────────────────
CRISIS_SCALE       = 1.438152935885522
ENTROPY_SCALE      = 0.18799497166314844
ENTROPY_THRESHOLD  = 0.682460634958058
REGIME_FLOOR       = 0.38232285186329923

# ── Pre-computed regime scores ─────────────────────────────────────────────────
_SCORES_FILE = Path(__file__).parent / "regime_scores.csv"
_REGIME_SCORES = None

def _load_scores():
    global _REGIME_SCORES
    if _REGIME_SCORES is None and _SCORES_FILE.exists():
        _REGIME_SCORES = pd.read_csv(_SCORES_FILE, index_col=0, parse_dates=True)
    return _REGIME_SCORES


def _get_regime_scale(current_date: pd.Timestamp) -> float:
    """
    Look up pre-computed regime scores for current date.
    Returns a scaling factor in [REGIME_FLOOR, 1.0].
    Falls back to 1.0 (no adjustment) if scores unavailable.
    """
    scores = _load_scores()
    if scores is None:
        return 1.0

    # Find closest available date (reindex to nearest past date)
    available = scores.index[scores.index <= current_date]
    if len(available) == 0:
        return 1.0

    row = scores.loc[available[-1]]
    crisis_prob = float(row["crisis_prob"])
    entropy     = float(row["entropy"])

    reduction = (
        CRISIS_SCALE * crisis_prob
        + ENTROPY_SCALE * max(0.0, entropy - ENTROPY_THRESHOLD)
    )
    return float(np.clip(1.0 - reduction, REGIME_FLOOR, 1.0))


def get_weights(data: dict) -> pd.Series:
    close  = data["close"]
    volume = data["volume"]
    weights = pd.Series(0.0, index=close.columns)

    if "QQQ" not in close.columns or len(close) < MA_SLOW:
        if "QQQ" in close.columns:
            weights["QQQ"] = 1.0
        return weights

    qqq     = close["QQQ"]
    returns = qqq.pct_change().dropna()
    current_date = close.index[-1]

    # ── Standard regime gates ─────────────────────────────────────────────────
    above_ma   = float(qqq.iloc[-1]) >= float(qqq.iloc[-MA_SLOW:].mean())

    abs_mom_ok = True
    if len(qqq) >= MOM_WINDOW:
        abs_mom_ok = float(qqq.iloc[-1]) > float(qqq.iloc[-MOM_WINDOW])

    credit_ok = True
    if "HYG" in close.columns:
        hyg = close["HYG"].dropna()
        if len(hyg) >= MA_SLOW:
            credit_ok = float(hyg.iloc[-1]) >= float(hyg.iloc[-MA_SLOW:].mean())

    vol_shock = False
    if len(returns) >= VOL_LONG:
        vs = returns.iloc[-VOL_SHORT:].std()
        vl = returns.iloc[-VOL_LONG:].std() + 1e-9
        vol_shock = (vs / vl) > VOL_SHOCK_MULT

    # ── Volume accumulation ───────────────────────────────────────────────────
    equity_wt = 1.0
    if "QQQ" in volume.columns:
        qv = volume["QQQ"].dropna()
        if len(qv) >= VOL_ACC_LONG:
            r_short = float(qv.iloc[-VOL_ACC_SHORT:].mean())
            r_long  = float(qv.iloc[-VOL_ACC_LONG:].mean()) + 1e-9
            if (r_short / r_long) < VOL_ACC_THRESHOLD:
                equity_wt = 0.5

    # ── Apply HMM regime scaling (continuous, not binary) ─────────────────────
    regime_scale = _get_regime_scale(current_date)
    equity_wt   *= regime_scale

    # ── Route ─────────────────────────────────────────────────────────────────
    if above_ma and abs_mom_ok and credit_ok and not vol_shock:
        weights["QQQ"] = equity_wt
        remainder = 1.0 - equity_wt
        if remainder > 0.01 and "SHY" in close.columns:
            weights["SHY"] = remainder

    elif above_ma and not credit_ok and not vol_shock:
        if "GLD" in close.columns:
            gld = close["GLD"].dropna()
            if len(gld) >= MOM_WINDOW and float(gld.iloc[-1]) > float(gld.iloc[-MOM_WINDOW]):
                weights["GLD"] = 1.0
                return weights
        _assign_bonds(weights, _bond_candidates(close, BOND_MOM_WINDOW), close)
    else:
        _assign_bonds(weights, _bond_candidates(close, BOND_MOM_WINDOW), close)

    return weights


def _bond_candidates(close, mom_window):
    candidates = []
    for bond in BOND_UNIVERSE:
        if bond not in close.columns:
            continue
        s = close[bond].dropna()
        if len(s) < mom_window:
            continue
        if float(s.iloc[-1]) <= float(s.iloc[-mom_window]):
            continue
        vol = s.pct_change().iloc[-BOND_VOL_WINDOW:].std() * np.sqrt(252)
        candidates.append((bond, float(vol)))
    return candidates


def _assign_bonds(weights, candidates, close):
    if not candidates:
        if "SHY" in close.columns:
            weights["SHY"] = 1.0
        return
    inv_vols = np.array([1.0 / (v + 1e-9) for _, v in candidates])
    inv_vols /= inv_vols.sum()
    for (bond, _), w in zip(candidates, inv_vols):
        weights[bond] = float(w)

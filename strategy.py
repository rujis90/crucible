"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

data dict keys (all DataFrames: DatetimeIndex rows x ticker columns):
  data['close']   adjusted close prices
  data['volume']  daily volume
  data['high']    daily high  (adjusted)
  data['low']     daily low   (adjusted)

All data sliced up to and including the signal date. No future data is present.

Weights returned:
  All values >= 0 (long-only)
  Returning all zeros = cash (flat) — valid and preferred over holding losers
  backtest.py clips each position to MAX_POSITION and gross to GROSS_LIMIT
"""

import pandas as pd

REBALANCE_EVERY = 21
MA_SLOW    = 200
MA_FAST    = 50
VOL_WINDOW = 63
VOL_SHORT  = 10
VOL_MULT   = 2.0
TOP_N      = 15

BOND_TICKERS = ["TLT", "IEF", "AGG", "SHY"]


def get_weights(data: dict) -> pd.Series:
    """
    Low-vol + momentum: select TOP_N lowest-volatility assets with positive
    6m-1m momentum AND above 200-MA. Equal-weight within the selection
    (already pre-screened for low vol, further tilting adds noise).
    Bear regime: equal-weight positive-momentum bonds.
    """
    close = data["close"]

    if len(close) < max(MA_SLOW, VOL_WINDOW + 147):
        return pd.Series(0.0, index=close.columns)

    price   = close.iloc[-1]
    ma_fast = close.iloc[-MA_FAST:].mean()
    ma_slow = close.iloc[-MA_SLOW:].mean()
    vol     = close.pct_change().iloc[-VOL_WINDOW:].std()
    raw_mom = close.iloc[-21] / close.iloc[-147] - 1

    weights = pd.Series(0.0, index=close.columns)

    spy_uptrend = "SPY" not in close.columns or ma_fast["SPY"] >= ma_slow["SPY"]
    if "SPY" in close.columns:
        spy_ret   = close["SPY"].pct_change().dropna()
        vol_shock = spy_ret.iloc[-VOL_SHORT:].std() > VOL_MULT * spy_ret.iloc[-VOL_WINDOW:].std()
    else:
        vol_shock = False

    # Credit regime: HYG above its 50-MA = credit healthy = risk on
    if "HYG" in close.columns:
        credit_ok = price["HYG"] >= ma_slow["HYG"]   # price vs 200-MA: faster than crossover
    else:
        credit_ok = True


    SHORT_BONDS = ["SHY", "IEF", "AGG"]   # rate-insensitive — prefer in rising-rate bear
    LONG_BONDS  = ["TLT", "IEF", "AGG", "SHY"]

    def rotate_to_bonds(panic=False):
        bonds = [t for t in LONG_BONDS if t in close.columns]
        if bonds:
            if panic and "SHY" in bonds:
                weights["SHY"] = 1.0
            else:
                # Prefer short-duration bonds: lower rate risk in slow bears
                short = [t for t in SHORT_BONDS if t in close.columns]
                pool  = short if short else bonds
                pos   = raw_mom[pool][raw_mom[pool] > 0]
                if not pos.empty:
                    weights[pos.index] = 1.0 / len(pos)
                else:
                    # All short bonds negative — fall back to any positive bond
                    pos2 = raw_mom[bonds][raw_mom[bonds] > 0]
                    if not pos2.empty:
                        weights[pos2.index] = 1.0 / len(pos2)

    if spy_uptrend and credit_ok and not vol_shock:
        eligible = vol.index[(price > ma_slow) & (raw_mom > 0)].tolist()
        if eligible:
            top = vol[eligible].nsmallest(TOP_N).index
            inv_vol = 1.0 / vol[top]
            weights[top] = inv_vol / inv_vol.sum()
        else:
            rotate_to_bonds()
    else:
        rotate_to_bonds(panic=vol_shock)

    return weights

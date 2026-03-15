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
TARGET_VOL = 0.14          # annual vol target for portfolio scaling

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

    high    = data["high"]
    low     = data["low"]
    price   = close.iloc[-1]
    ma_fast = close.iloc[-MA_FAST:].mean()
    ma_slow = close.iloc[-MA_SLOW:].mean()
    # ATR-based volatility: captures gap risk beyond close-to-close moves
    prev_close = close.shift(1)
    hl  = (high - low)              / close
    hc  = (high - prev_close).abs() / close
    lc  = (low  - prev_close).abs() / close
    atr = pd.concat([hl, hc, lc]).groupby(level=0).max()
    vol     = atr.iloc[-VOL_WINDOW:].mean()
    raw_mom = close.iloc[-21] / close.iloc[-147] - 1
    mom_3m  = close.iloc[-21] / close.iloc[-63] - 1    # 3m momentum: soft quality signal

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


    LONG_BONDS  = ["TLT", "IEF", "AGG", "SHY"]

    def rotate_to_bonds(panic=False):
        bonds = [t for t in LONG_BONDS if t in close.columns]
        if bonds:
            if panic and "SHY" in bonds:
                weights["SHY"] = 1.0
            else:
                # Adaptive inv-vol bond weighting: high-ATR bonds (rising rates) get less weight
                pos = [t for t in bonds if raw_mom[t] > 0]
                if pos:
                    inv_vol_b = 1.0 / vol[pos]
                    weights[pd.Index(pos)] = (inv_vol_b / inv_vol_b.sum()).values

    if spy_uptrend and credit_ok and not vol_shock:
        eligible = vol.index[(price > ma_slow) & (raw_mom > 0)].tolist()
        if eligible:
            top = vol[eligible].nsmallest(TOP_N).index
            inv_vol   = 1.0 / vol[top]
            mom_score = raw_mom[top].clip(lower=0)
            # Soft 3m confirmation: boost assets where 3m momentum agrees, reduce where it diverges
            mom_3m_score = mom_3m[top].clip(lower=-0.3, upper=0.3)
            if mom_score.sum() > 0:
                composite = inv_vol * (1 + mom_score) * (1 + 0.5 * mom_3m_score)
            else:
                composite = inv_vol
            weights[top] = composite / composite.sum()
        else:
            rotate_to_bonds()
    else:
        rotate_to_bonds(panic=vol_shock)

    # Vol targeting: scale down when portfolio ATR exceeds target (reduces DD in stressed markets)
    if weights.sum() > 0:
        port_atr = (weights * vol).sum()
        target_daily = TARGET_VOL / (252 ** 0.5)
        scale = min(1.0, target_daily / (port_atr + 1e-9))
        weights = weights * scale

    return weights

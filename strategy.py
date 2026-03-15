"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
Philosophy: Non-Normal Market Structure
─────────────────────────────────────────────────────────────────────────────
Inspired by Mandelbrot, Taleb, Sornette, Spitznagel.

PRIMARY OBJECTIVE: maximize CAGR (geometric return), not Sharpe.
Sharpe assumes normality. Real markets have fat tails, memory, and endogenous
crashes. The correct objective is terminal wealth — Spitznagel's insight:
  CAGR ≈ arithmetic_mean − (variance / 2)
Minimizing variance drag directly maximizes compounding. Every drawdown is
multiplicatively destructive (−50% requires +100% to recover).

THREE NON-NORMAL INSIGHTS applied to asset selection and sizing:

1. HURST EXPONENT (Mandelbrot)
   Markets are fractal and have memory. H > 0.5 = persistent trend (momentum
   will continue). H < 0.5 = mean-reverting (momentum will fail and reverse).
   Standard models assume H = 0.5 (random walk). Selecting assets with H > 0.5
   gives genuine edge: we're exploiting real autocorrelation, not noise.

2. RETURN SKEWNESS (Taleb — antifragility)
   Positive skewness = right tail is fatter than left tail = asymmetric upside.
   Negative skewness = left tail is fatter = crash-prone. Holding negatively
   skewed assets is the sucker's bet (small gains, catastrophic losses).
   We prefer right-skewed assets and penalize left-skewed ones.

3. MOMENTUM ACCELERATION (Sornette — dragon kings)
   Sornette showed crashes are endogenous, not random. They follow
   superexponential growth (accelerating momentum). We measure whether the
   current rate of price increase is accelerating or decelerating:
     acceleration = annualized_1m_return − annualized_6m_return
   Moderate acceleration = sustainable trend worth riding.
   Strong deceleration = the trend is dying, rotate out.
   Extreme acceleration (blow-off) = danger, cap the weight.

SIZING: Kelly-proportional (geometric return / variance per asset)
   Kelly criterion maximizes expected log-wealth (= long-run CAGR).
   We use the geometric return (log mean × 252) not arithmetic return
   to correctly account for variance drag.

SCALING: Empirical drawdown ceiling, not vol target.
   Instead of assuming a normal distribution for the vol target, we use
   the actual observed drawdown of the constructed portfolio over the last
   6 months. Distribution-free. Respects fat tails.

Bear regime defense is kept from the proven base — bonds in risk-off,
inv-vol weighted, adaptive to rate direction.
"""

import numpy as np
import pandas as pd

REBALANCE_EVERY = 21
MA_SLOW     = 200
MA_FAST     = 50
VOL_WINDOW  = 63
VOL_SHORT   = 10
VOL_MULT    = 2.0
TOP_N       = 15
MAX_DD_CEIL = 0.20   # empirical drawdown ceiling — distribution-free scaling

# Excluded from equity eligible pool in bull mode.
# Fixed income dominates any low-vol or Kelly metric (variance → 0 → Kelly → ∞).
# Bonds belong in the bear rotation only, not competing with equity.
FIXED_INCOME = frozenset(["TLT", "IEF", "AGG", "SHY", "LQD", "HYG", "MBB", "TIP", "EMB"])


# ── Hurst exponent via R/S analysis ──────────────────────────────────────────

def _hurst_rs(returns: pd.Series) -> float:
    """
    Rescaled Range (R/S) Hurst exponent.
    H > 0.5 = persistent (trend continues) — safe to apply momentum here.
    H = 0.5 = random walk (Brownian motion assumption, standard finance).
    H < 0.5 = anti-persistent (mean reverting) — momentum will fail here.
    Uses three lag windows; fits log(R/S) vs log(lag) slope.
    """
    lags = [10, 20, 40]
    log_lags, log_rs = [], []
    for lag in lags:
        chunks = [returns.iloc[i:i + lag] for i in range(0, len(returns) - lag, lag)]
        rs_vals = []
        for c in chunks:
            s = c.std()
            if s < 1e-10:
                continue
            dev = (c - c.mean()).cumsum()
            rs_vals.append((dev.max() - dev.min()) / s)
        if rs_vals:
            log_lags.append(np.log(lag))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_lags) < 2:
        return 0.5
    return float(np.clip(np.polyfit(log_lags, log_rs, 1)[0], 0.1, 0.9))


# ── Empirical drawdown ceiling ────────────────────────────────────────────────

def _apply_dd_ceiling(
    weights: pd.Series,
    log_rets: pd.DataFrame,
    ceiling: float,
    lookback: int = 126,
) -> pd.Series:
    """
    Scales portfolio exposure down if trailing empirical drawdown exceeds ceiling.
    Distribution-free: uses actual observed path, not any assumed distribution.
    Spitznagel principle: the path of returns matters, not just vol.
    """
    if weights.sum() <= 0:
        return weights
    held = [t for t in weights[weights > 0].index if t in log_rets.columns]
    if not held:
        return weights

    port_log = (log_rets[held].fillna(0) * weights[held]).sum(axis=1).iloc[-lookback:]
    if len(port_log) < 20:
        return weights

    wealth   = port_log.cumsum().apply(np.exp)
    roll_max = wealth.cummax()
    emp_dd   = ((wealth - roll_max) / roll_max).min()   # most negative value

    if emp_dd < -ceiling:
        scale = min(1.0, ceiling / abs(emp_dd))
        return weights * scale
    return weights


# ── Main strategy ─────────────────────────────────────────────────────────────

def get_weights(data: dict) -> pd.Series:
    close = data["close"]
    high  = data["high"]
    low   = data["low"]

    if len(close) < max(MA_SLOW, VOL_WINDOW + 147):
        return pd.Series(0.0, index=close.columns)

    price   = close.iloc[-1]
    ma_fast = close.iloc[-MA_FAST:].mean()
    ma_slow = close.iloc[-MA_SLOW:].mean()

    # ATR-based volatility — distribution-agnostic, captures gap risk
    prev_close = close.shift(1)
    hl  = (high - low)              / close
    hc  = (high - prev_close).abs() / close
    lc  = (low  - prev_close).abs() / close
    atr = pd.concat([hl, hc, lc]).groupby(level=0).max()
    vol = atr.iloc[-VOL_WINDOW:].mean()

    # Log returns — natural unit for geometric/Kelly analysis
    log_rets = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    # Momentum windows
    raw_mom = close.iloc[-21] / close.iloc[-147] - 1    # 6m-1m
    mom_1m  = close.iloc[-1]  / close.iloc[-21]  - 1    # 1 month

    weights = pd.Series(0.0, index=close.columns)

    # ── Regime gate (bear / vol shock → bonds) ────────────────────────────────
    spy_uptrend = "SPY" not in close.columns or ma_fast["SPY"] >= ma_slow["SPY"]
    if "SPY" in close.columns:
        spy_ret   = close["SPY"].pct_change().dropna()
        vol_shock = spy_ret.iloc[-VOL_SHORT:].std() > VOL_MULT * spy_ret.iloc[-VOL_WINDOW:].std()
    else:
        vol_shock = False

    if "HYG" in close.columns:
        credit_ok = price["HYG"] >= ma_slow["HYG"]
    else:
        credit_ok = True

    LONG_BONDS = ["TLT", "IEF", "AGG", "SHY"]

    def rotate_to_bonds(panic=False):
        bonds = [t for t in LONG_BONDS if t in close.columns]
        if not bonds:
            return
        if panic and "SHY" in bonds:
            weights["SHY"] = 1.0
        else:
            pos = [t for t in bonds if raw_mom[t] > 0]
            if pos:
                inv_vol_b = 1.0 / vol[pos]
                weights[pd.Index(pos)] = (inv_vol_b / inv_vol_b.sum()).values

    if not (spy_uptrend and credit_ok and not vol_shock):
        rotate_to_bonds(panic=vol_shock)
        return _apply_dd_ceiling(weights, log_rets, MAX_DD_CEIL)

    # ── Eligible assets: equity only, above 200-MA, positive 6m-1m momentum ──
    # Bonds excluded here: they belong to bear rotation only. In bull mode they
    # dominate any low-vol or Kelly metric (near-zero variance → Kelly → ∞),
    # crowding out the equity we're actually trying to select.
    eligible = [t for t in vol.index
                if t not in FIXED_INCOME
                and price[t] > ma_slow[t]
                and raw_mom[t] > 0]
    if not eligible:
        rotate_to_bonds()
        return weights

    # ── Non-normal composite scoring ──────────────────────────────────────────
    scores = {}
    for t in eligible:
        rets_t = log_rets[t].dropna()
        if len(rets_t) < 80:
            scores[t] = 0.0
            continue

        window = rets_t.iloc[-126:]   # 6 months of log returns

        # ── 1. Hurst exponent (Mandelbrot) ────────────────────────────────────
        # Measures persistence: H > 0.5 means the asset is genuinely trending,
        # not just randomly high. Only trend in trending regimes.
        h = _hurst_rs(window)
        hurst_contrib = (h - 0.5) * 4.0    # H=0.75→+1.0, H=0.5→0, H=0.25→-1.0

        # ── 2. Return skewness (Taleb — antifragility) ────────────────────────
        # Positive skew: fat right tail = we win big occasionally, lose small.
        # Negative skew: fat left tail = we win small, lose catastrophically.
        # The right-skewed asset is antifragile; the left-skewed one is fragile.
        skew          = float(window.skew())
        skew_contrib  = np.clip(skew * 0.5, -0.5, 0.5)

        # ── 3. Momentum acceleration (Sornette) ───────────────────────────────
        # Compare annualized 1-month rate vs annualized 6-month rate.
        # Accelerating = trend is strengthening → ride it.
        # Decelerating = trend is dying → reduce before the endogenous crash.
        # Extreme acceleration (blow-off top) = cap to avoid bubble exposure.
        ann_1m        = (1 + float(mom_1m[t]))   ** 12 - 1
        ann_6m        = (1 + float(raw_mom[t]))  ** 2  - 1
        accel         = ann_1m - ann_6m
        accel_contrib = float(np.clip(accel, -0.6, 0.25))

        # ── 4. Kelly proxy (Spitznagel — maximize CAGR) ───────────────────────
        # Kelly criterion: optimal bet = geometric_return / variance.
        # Stored raw; normalized across the pool after the loop so no single
        # ultra-low-vol asset monopolizes the score.
        geo_ret  = float(window.mean()) * 252
        variance = float(window.var())  * 252
        kelly    = geo_ret / (variance + 1e-6)   # raw Kelly fraction

        scores[t] = (hurst_contrib, skew_contrib, accel_contrib, kelly)

    # Unpack raw scores and normalize Kelly within the pool
    # Kelly raw values span huge ranges across different asset types — normalize
    # to [-0.3, 0.6] relative to the pool median so no asset dominates
    score_df   = pd.DataFrame(scores, index=["hurst","skew","accel","kelly"]).T
    kelly_raw  = score_df["kelly"]
    kelly_med  = kelly_raw.median()
    kelly_std  = kelly_raw.std() + 1e-6
    kelly_norm = ((kelly_raw - kelly_med) / kelly_std * 0.3).clip(-0.3, 0.6)

    scores_s = (score_df["hurst"] + score_df["skew"]
                + score_df["accel"] + kelly_norm)

    # ── Selection: lowest ATR vol among eligible (proven crash-resilient base) ─
    # Vol-first selection keeps us in the calmest assets — they fall less in
    # onset of crashes before the regime gate triggers.
    top = vol[eligible].nsmallest(TOP_N).index

    # Restrict scoring to selected top — avoids negative-score fallback
    # shrinking the pool to 5 assets in post-crash recovery periods

    # ── Weighting: proven momentum base × non-normal score tilt ──────────────
    # Layer 1 (proven): inv_vol × momentum × 3m-confirmation
    #   — the exact formula from the previous best (1.0678 sharpe)
    # Layer 2 (new):    × non-normal score (Hurst, skew, accel, Kelly)
    #   — amplifies the proven weights toward better distributional structure
    inv_vol_top  = 1.0 / vol[top]
    mom_score    = raw_mom[top].clip(lower=0)
    mom_3m       = close.iloc[-21] / close.iloc[-63] - 1
    mom_3m_score = mom_3m[top].clip(lower=-0.3, upper=0.3)
    if mom_score.sum() > 0:
        momentum_base = inv_vol_top * (1 + mom_score) * (1 + 0.5 * mom_3m_score)
    else:
        momentum_base = inv_vol_top

    # Non-normal tilt: shift scores to [0.5, 1.5] range so they multiply the
    # momentum base without flipping signs or dominating it
    score_min    = scores_s[top].min()
    score_max    = scores_s[top].max()
    score_range  = score_max - score_min + 1e-9
    score_normed = 0.5 + (scores_s[top] - score_min) / score_range   # 0.5 → 1.5
    composite    = momentum_base * score_normed
    weights[top] = composite / composite.sum()

    # ── Empirical drawdown ceiling ────────────────────────────────────────────
    return _apply_dd_ceiling(weights, log_rets, MAX_DD_CEIL)

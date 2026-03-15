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
MAX_DD_CEIL = 0.15   # empirical drawdown ceiling — tighter = less variance drag
TARGET_VOL  = 0.16   # proactive vol target: scale down BEFORE drawdown accumulates

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
    raw_mom = close.iloc[-21] / close.iloc[-147] - 1    # 6m-1m (used for eligibility filter)
    mom_1m  = close.iloc[-1]  / close.iloc[-21]  - 1    # 1 month

    # ── Idiosyncratic momentum (Simons/Medallion: "regress first, then measure")
    # Remove SPY market beta from each asset's return before ranking.
    # raw_mom includes common market return — all assets rising with SPY is not alpha.
    # The residual (alpha) is what will continue when the tide turns.
    # Keep raw_mom > 0 as the eligibility gate (absolute momentum, crash protection).
    # Use idio_mom for weighting (relative quality of momentum).
    idio_mom = raw_mom.copy()
    if "SPY" in log_rets.columns:
        spy_w = log_rets["SPY"].iloc[-147:-21].dropna()
        if len(spy_w) >= 60:
            spy_var = float(spy_w.var()) + 1e-9
            for _t in close.columns:
                _aw = log_rets[_t].reindex(spy_w.index).dropna()
                if len(_aw) >= 60:
                    _sp = spy_w.reindex(_aw.index)
                    _beta = float(np.cov(_aw.values, _sp.values)[0, 1]) / spy_var
                    idio_mom[_t] = float((_aw - _beta * _sp).sum())

    weights = pd.Series(0.0, index=close.columns)

    # ── Regime gate (bear / vol shock → bonds) ────────────────────────────────
    spy_uptrend = "SPY" not in close.columns or ma_fast["SPY"] >= ma_slow["SPY"]
    if "SPY" in close.columns:
        spy_ret   = close["SPY"].pct_change().dropna()
        vol_shock = spy_ret.iloc[-VOL_SHORT:].std() > VOL_MULT * spy_ret.iloc[-VOL_WINDOW:].std()

        # Fresh recovery: SPY was below its MA200 at any point in last 63 days.
        # In this state Hurst scores are contaminated by the prior bear —
        # suppress the Hurst penalty so recovery momentum can lead.
        if len(close) >= MA_SLOW + 63:
            rolling_ma200  = close["SPY"].rolling(MA_SLOW).mean()
            spy_last63     = close["SPY"].iloc[-63:]
            ma200_last63   = rolling_ma200.iloc[-63:]
            was_below      = bool((spy_last63 < ma200_last63).any())
            fresh_recovery = spy_uptrend and was_below
        else:
            fresh_recovery = False
    else:
        vol_shock      = False
        fresh_recovery = False

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
            scores[t] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # hurst,skew,accel,kelly,crash,volume
            continue

        window = rets_t.iloc[-126:]   # 6 months of log returns

        # ── 1. Hurst exponent (Mandelbrot) ────────────────────────────────────
        h = _hurst_rs(window)
        raw_hurst = (h - 0.5) * 4.0
        hurst_contrib = max(0.0, raw_hurst) if fresh_recovery else raw_hurst

        # ── 2. Tail ratio (Taleb — antifragility, no normality) ───────────────
        # Tail ratio = 95th pctile / |5th pctile|. > 1 = right tail fatter.
        # More robust than skewness: directly measures the ratio of extreme wins
        # to extreme losses without assuming any distribution shape.
        p95 = float(np.percentile(window, 95))
        p05 = abs(float(np.percentile(window, 5))) + 1e-9
        tail_ratio    = p95 / p05
        skew_contrib  = np.clip((tail_ratio - 1.0) * 0.5, -0.5, 0.5)

        # ── 3. Momentum acceleration (Sornette) ───────────────────────────────
        # Two timeframes must agree for full credit:
        #   short: 1m vs 3m annualized (near-term acceleration)
        #   medium: 3m vs 6m annualized (medium-term trend strengthening)
        # If both accelerating → stronger signal. If diverging → partial credit.
        mom_3m_t      = close[t].iloc[-1]  / close[t].iloc[-63]  - 1
        ann_1m        = (1 + float(mom_1m[t]))   ** 12 - 1
        ann_3m        = (1 + float(mom_3m_t))    ** 4  - 1
        ann_6m        = (1 + float(raw_mom[t]))  ** 2  - 1
        accel_short   = ann_1m - ann_3m   # 1m vs 3m
        accel_medium  = ann_3m - ann_6m   # 3m vs 6m
        # Average both; cap extremes
        accel         = (accel_short + accel_medium) / 2.0
        accel_contrib = float(np.clip(accel, -0.6, 0.25))

        # ── 4. Kelly proxy (Spitznagel — maximize CAGR) ───────────────────────
        geo_ret  = float(window.mean()) * 252
        variance = float(window.var())  * 252
        kelly    = geo_ret / (variance + 1e-6)

        # ── 5. Crash resilience (Taleb — conditional anti-fragility) ─────────
        # Tail ratio captures distributional shape in isolation.
        # This captures performance CONDITIONED ON ACTUAL MARKET CRASH DAYS —
        # a fundamentally different measure: does this asset hold up when it
        # matters most? Assets that fall less than the market during crashes are
        # anti-fragile (Taleb). We use SPY's worst-10% days as crash events.
        if "SPY" in log_rets.columns:
            mkt_w = log_rets["SPY"].reindex(window.index).dropna()
            if len(mkt_w) >= 20:
                crash_thresh = float(mkt_w.quantile(0.10))
                crash_mask   = mkt_w <= crash_thresh
                if crash_mask.sum() >= 5:
                    crash_ret     = float(window.reindex(mkt_w.index)[crash_mask].mean())
                    mkt_crash_ret = float(mkt_w[crash_mask].mean())
                    # fragility_ratio > 1 = falls more than market (fragile)
                    # fragility_ratio < 1 = falls less (robust, anti-fragile)
                    frag_ratio    = crash_ret / (mkt_crash_ret + 1e-9)
                    crash_contrib = float(np.clip((1.0 - frag_ratio) * 0.25, -0.3, 0.3))
                else:
                    crash_contrib = 0.0
            else:
                crash_contrib = 0.0
        else:
            crash_contrib = 0.0

        # ── 6. Volume momentum (institutional accumulation/distribution) ──────
        # Rising price + increasing volume = institutional accumulation.
        # Rising price + falling volume = distribution (weak, likely to reverse).
        # Volume is Wyckoff's invisible hand — reveals who's actually buying.
        # vol_ratio > 1: recent volume > historical → accumulation signal.
        vol_t = data["volume"][t].dropna()
        if len(vol_t) >= 63:
            vol_21 = float(vol_t.iloc[-21:].mean())
            vol_63 = float(vol_t.iloc[-63:].mean()) + 1e-9
            vol_ratio = vol_21 / vol_63     # > 1 = increasing volume
            vol_contrib = float(np.clip((vol_ratio - 1.0) * 0.5, -0.3, 0.3))
        else:
            vol_contrib = 0.0

        scores[t] = (hurst_contrib, skew_contrib, accel_contrib, kelly, crash_contrib, vol_contrib)

    # Unpack raw scores and normalize Kelly within the pool
    # Kelly raw values span huge ranges across different asset types — normalize
    # to [-0.3, 0.6] relative to the pool median so no asset dominates
    score_df   = pd.DataFrame(scores, index=["hurst","skew","accel","kelly","crash","volume"]).T

    # Cross-sectional normalization: each signal to [-0.3, +0.6] z-score range
    # relative to pool median. This equalizes signal contribution so no single
    # signal dominates due to its raw scale (Hurst raw range is ±1.6 while
    # others are ±0.3–0.5). Applies same treatment already used for Kelly.
    def _xsnorm(col, lo=-0.3, hi=0.6):
        m = col.median(); s = col.std() + 1e-6
        return ((col - m) / s * 0.3).clip(lo, hi)

    hurst_norm  = _xsnorm(score_df["hurst"])
    skew_norm   = _xsnorm(score_df["skew"])
    accel_norm  = _xsnorm(score_df["accel"])
    kelly_norm  = _xsnorm(score_df["kelly"])
    crash_norm  = _xsnorm(score_df["crash"])
    volume_norm = _xsnorm(score_df["volume"])

    scores_s = hurst_norm + skew_norm + accel_norm + kelly_norm + crash_norm + volume_norm

    # ── Dynamic TOP_N: high score spread = signals agree = concentrate ────────
    # When the composite scores are tightly clustered, signals are uncertain —
    # diversify. When spread is wide, signals strongly distinguish winners —
    # concentrate into the best ones.
    if len(scores_s) >= 5:
        score_std = scores_s.std()
        score_med_std = scores_s.rolling(1).std().median() if len(scores_s) > 1 else 0
        if score_std > 0.8:
            dyn_top_n = max(8, TOP_N - 5)    # high conviction → concentrate to 10
        elif score_std > 0.5:
            dyn_top_n = max(10, TOP_N - 3)   # moderate conviction → 12
        else:
            dyn_top_n = TOP_N                # uncertain → full diversification
    else:
        dyn_top_n = min(TOP_N, len(eligible))

    # ── Selection: lowest ATR vol among eligible ──────────────────────────────
    top = vol[eligible].nsmallest(dyn_top_n).index

    # ── Weighting: proven momentum base × non-normal score tilt ──────────────
    # Layer 1 (proven): inv_vol × momentum × 3m-confirmation
    #   — the exact formula from the previous best (1.0678 sharpe)
    # Layer 2 (new):    × non-normal score (Hurst, skew, accel, Kelly)
    #   — amplifies the proven weights toward better distributional structure
    inv_vol_top  = 1.0 / vol[top]
    mom_score    = idio_mom[top].clip(lower=0)   # idiosyncratic, not raw
    mom_3m       = close.iloc[-21] / close.iloc[-63] - 1
    mom_3m_score = mom_3m[top].clip(lower=-0.3, upper=0.3)

    if mom_score.sum() > 0:
        momentum_base = inv_vol_top * (1 + mom_score) * (1 + 0.5 * mom_3m_score)
    else:
        momentum_base = inv_vol_top

    # Non-normal tilt: shift scores to [lo, hi] range as a multiplier on the
    # momentum base. Range is adaptive: when signals strongly agree (high
    # score_std), concentrate the bet; when signals are uncertain, flatten
    # weights to diversify. Consistent with dynamic TOP_N philosophy.
    score_min   = scores_s[top].min()
    score_max   = scores_s[top].max()
    score_range = score_max - score_min + 1e-9
    if score_std > 0.8:
        lo, hi = 0.4, 1.6   # high conviction — wide tilt, bet on winners
    elif score_std > 0.5:
        lo, hi = 0.5, 1.5   # normal regime
    else:
        lo, hi = 0.7, 1.3   # uncertain — narrow tilt, stay diversified
    score_normed = lo + (scores_s[top] - score_min) / score_range * (hi - lo)
    composite    = momentum_base * score_normed
    weights[top] = composite / composite.sum()

    # ── Proactive vol targeting (applied before drawdown ceiling) ─────────────
    # DD ceiling is reactive (acts after drawdown). Vol target is proactive:
    # scale down when realized portfolio vol > TARGET_VOL, before loss accumulates.
    # Together: vol target prevents drawdown from starting; DD ceiling caps it.
    held_w = [t for t in weights[weights > 0].index if t in log_rets.columns]
    if held_w:
        port_log_21 = (log_rets[held_w].fillna(0) * weights[held_w]).sum(axis=1).iloc[-21:]
        if len(port_log_21) >= 10:
            realized_vol = float(port_log_21.std()) * np.sqrt(252)
            if realized_vol > TARGET_VOL and realized_vol > 0:
                weights = weights * (TARGET_VOL / realized_vol)

    # ── Empirical drawdown ceiling ────────────────────────────────────────────
    return _apply_dd_ceiling(weights, log_rets, MAX_DD_CEIL)

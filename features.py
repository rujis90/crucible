"""
features.py — López de Prado style feature engineering for the NDX PIT universe.

All features are stationary (or near-stationary), scaled, and computed only from
information available as of the feature date — no future prices, no future membership.

The PIT filter is the critical guard: on date t, only tickers that were actually
in the NDX on date t are included. Without it, you leak future information (you
implicitly know which companies survived long enough to be in tomorrow's index).

Usage:
    from features import make_features

    feats = make_features(
        close, high, low, volume,
        pit,                    # ndx_pit_daily.parquet bool DataFrame
        as_of_date=pd.Timestamp("2020-06-15"),
        lookback=252,
    )
    # feats: DataFrame, rows = tickers in NDX on as_of_date, cols = features
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score — scale-invariant, mean-reverting."""
    m = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std()
    return (s - m) / (sd + 1e-9)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, min_periods=window // 2).mean()


# ── fractional differentiation ────────────────────────────────────────────────

def frac_diff(series: pd.Series, d: float = 0.4, thresh: float = 1e-4) -> pd.Series:
    """
    Fractional differentiation (LdP AFML Ch. 5).

    Standard log-returns (d=1) are stationary but destroy memory. Raw prices
    (d=0) preserve memory but are non-stationary. Fractional diff at d≈0.3–0.5
    achieves stationarity while retaining most of the long-range dependence.

    d=0.4 is a reasonable default: borderline stationary by ADF, retains ~60%
    of the long-range correlation structure of the price series.
    """
    weights = [1.0]
    k = 1
    while abs(weights[-1]) > thresh:
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])

    result = pd.Series(np.nan, index=series.index)
    for i in range(len(weights) - 1, len(series)):
        window = series.iloc[i - len(weights) + 1: i + 1].values
        if not np.any(np.isnan(window)):
            result.iloc[i] = np.dot(weights, window)
    return result


# ── core feature builder ──────────────────────────────────────────────────────

def make_features(
    close:   pd.DataFrame,
    high:    pd.DataFrame,
    low:     pd.DataFrame,
    volume:  pd.DataFrame,
    pit:     pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback: int = 252,
    bars_per_day: int = 1,
    open_: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix for the NDX universe as of a given date.

    Only tickers in the NDX on as_of_date are included (PIT filter).
    All features are computed from data strictly before or on as_of_date.

    Args:
        close:       MultiIndex parquet close prices (date × ticker) or flat DataFrame
        high:        same for high
        low:         same for low
        volume:      same for volume
        pit:         ndx_pit_daily.parquet — bool DataFrame (date × ticker)
        as_of_date:  the date for which we are computing features
        lookback:    number of trading days of history to use

    Returns:
        DataFrame with one row per in-universe ticker, columns = feature names.
        Returns empty DataFrame if date not in pit index.
    """
    # ── PIT filter: only tickers in NDX on as_of_date ────────────────────────
    if as_of_date not in pit.index:
        # find the most recent date ≤ as_of_date
        prior = pit.index[pit.index <= as_of_date]
        if prior.empty:
            return pd.DataFrame()
        as_of_date_pit = prior[-1]
    else:
        as_of_date_pit = as_of_date

    universe = pit.columns[pit.loc[as_of_date_pit]].tolist()
    universe = [t for t in universe if t in close.columns]
    if not universe:
        return pd.DataFrame()

    # ── slice history window ──────────────────────────────────────────────────
    hist_end = as_of_date
    hist_dates = close.index[close.index <= hist_end]
    if len(hist_dates) < lookback // 2:
        return pd.DataFrame()
    hist_dates = hist_dates[-lookback:]

    c = close.loc[hist_dates, universe]
    h = high.loc[hist_dates, universe]
    l = low.loc[hist_dates, universe]
    v = volume.loc[hist_dates, universe]

    # ── feature computation ───────────────────────────────────────────────────
    # All window sizes are expressed in trading days and scaled to bars here.
    # At daily frequency bars_per_day=1 so nothing changes.
    # At hourly frequency bars_per_day=7 so a "21-day" window = 147 bars.
    b = bars_per_day
    feats: dict[str, pd.Series] = {}

    # 1. Multi-horizon returns (z-scored vs rolling history)
    for days, name in [(1, "ret_1d"), (5, "ret_5d"), (21, "ret_21d"), (63, "ret_63d")]:
        bars = days * b
        if len(c) <= bars:
            continue
        ret = c.pct_change(bars, fill_method=None).iloc[-1]
        mu, sd = ret.mean(), ret.std()
        feats[name] = (ret - mu) / (sd + 1e-9)

    # 2. Relative strength rank (RS) — percentile within universe
    if len(c) > 63 * b:
        ret_63 = c.pct_change(63 * b, fill_method=None).iloc[-1]
        feats["rs_rank"] = ret_63.rank(pct=True) * 2 - 1  # scale to [-1, 1]

    # 3. Realized volatility (normalized by cross-section)
    if len(c) > 21 * b:
        vol_21 = c.pct_change(fill_method=None).iloc[-21 * b:].std()
        vol_mu, vol_sd = vol_21.mean(), vol_21.std()
        feats["vol_21d_cs"] = (vol_21 - vol_mu) / (vol_sd + 1e-9)

    if len(c) > 63 * b:
        vol_63 = c.pct_change(fill_method=None).iloc[-63 * b:].std()
        vol_mu, vol_sd = vol_63.mean(), vol_63.std()
        feats["vol_63d_cs"] = (vol_63 - vol_mu) / (vol_sd + 1e-9)

    # 4. Vol ratio: recent vs historical (regime signal)
    if len(c) > 63 * b:
        feats["vol_ratio"] = (
            c.pct_change(fill_method=None).iloc[-21 * b:].std() /
            (c.pct_change(fill_method=None).iloc[-63 * b:].std() + 1e-9)
        ).apply(np.log)  # log ratio, symmetric around 0

    # 5. Dollar volume rank (liquidity proxy)
    if len(v) > 21 * b and len(c) > 21 * b:
        dv = (c * v).iloc[-21 * b:].mean()
        feats["dv_rank"] = dv.rank(pct=True) * 2 - 1

    # 6. High-low range / ATR ratio (intraday volatility normalised)
    atr_window = 14 * b
    if len(h) > atr_window and len(l) > atr_window:
        hl_range = ((h.iloc[-1] - l.iloc[-1]) / (c.iloc[-1] + 1e-9))
        atr_vals = pd.Series({
            t: _atr(h[t], l[t], c[t], window=atr_window).iloc[-1]
            for t in universe if not c[t].isna().all()
        })
        hl_norm = hl_range / (atr_vals + 1e-9)
        mu, sd = hl_norm.mean(), hl_norm.std()
        feats["hl_atr_ratio"] = (hl_norm - mu) / (sd + 1e-9)

    # 7. Price position within 52-week range
    bars_52w = 252 * b
    if len(c) >= bars_52w:
        hi_52 = h.iloc[-bars_52w:].max()
        lo_52 = l.iloc[-bars_52w:].min()
        px_pos = (c.iloc[-1] - lo_52) / (hi_52 - lo_52 + 1e-9)
        feats["px_pos_52w"] = px_pos * 2 - 1  # scale to [-1, 1]

    # 8. MACD signal (trend vs mean-reversion regime indicator)
    if len(c) > 26 * b:
        ema12 = c.ewm(span=12 * b, min_periods=6 * b).mean().iloc[-1]
        ema26 = c.ewm(span=26 * b, min_periods=13 * b).mean().iloc[-1]
        macd  = (ema12 - ema26) / (c.iloc[-1] + 1e-9)
        mu, sd = macd.mean(), macd.std()
        feats["macd_cs"] = (macd - mu) / (sd + 1e-9)

    # 9. Momentum reversal: short-term vs medium-term return divergence
    if "ret_1d" in feats and "ret_21d" in feats:
        feats["mom_reversal"] = feats["ret_1d"] - feats["ret_21d"]

    # 10. Volume momentum: recent vs historical average volume
    if len(v) > 63 * b:
        vol_ratio_v = v.iloc[-5 * b:].mean() / (v.iloc[-63 * b:].mean() + 1e-9)
        mu, sd = vol_ratio_v.mean(), vol_ratio_v.std()
        feats["vol_momentum"] = (vol_ratio_v - mu) / (sd + 1e-9)

    # ── Hourly-only features (bars_per_day > 1) ───────────────────────────────
    # These features only exist when we have intraday bars.
    # They are NaN (and dropped) in daily mode.

    if b > 1:
        # 11. Opening gap: today's open vs yesterday's close (intraday positioning)
        # Measures overnight sentiment; large gaps often continue or reverse.
        if open_ is not None and len(c) >= 2 * b:
            open_px = open_.loc[hist_dates, universe]
        else:
            open_px = None
        if open_px is not None and len(c) >= 2 * b:
            today_open  = open_px.iloc[-b]   # first bar today
            yest_close  = c.iloc[-(b + 1)]                              # last bar yesterday
            gap = (today_open - yest_close) / (yest_close + 1e-9)
            mu, sd = gap.mean(), gap.std()
            feats["open_gap"] = (gap - mu) / (sd + 1e-9)

        # 12. Intraday momentum: last bar vs first bar of today
        # Captures within-day trend; stocks drifting up all day vs reversing.
        if len(c) >= b:
            today_bars = c.iloc[-b:]
            intraday_ret = (today_bars.iloc[-1] - today_bars.iloc[0]) / (today_bars.iloc[0] + 1e-9)
            mu, sd = intraday_ret.mean(), intraday_ret.std()
            feats["intraday_mom"] = (intraday_ret - mu) / (sd + 1e-9)

        # 13. Intraday volume skew: afternoon vs morning volume ratio
        # High afternoon volume relative to morning = institutional accumulation signal.
        if len(v) >= b:
            today_vol  = v.iloc[-b:]
            half       = max(b // 2, 1)
            am_vol     = today_vol.iloc[:half].mean()
            pm_vol     = today_vol.iloc[half:].mean()
            vol_skew   = pm_vol / (am_vol + 1e-9)
            mu, sd     = vol_skew.mean(), vol_skew.std()
            feats["intraday_vol_skew"] = (vol_skew - mu) / (sd + 1e-9)

    # ── Add custom features below this line ───────────────────────────────────
    # Rules:
    #   1. Window sizes must use `b = bars_per_day` (e.g., 21 * b, not 21)
    #   2. Cross-sectionally z-score vs universe on each date (not time-series)
    #   3. Only use data from hist_dates (past only — no look-ahead)
    #   4. Return pd.Series indexed by ticker (same as universe list)
    # ─────────────────────────────────────────────────────────────────────────

    if not feats:
        return pd.DataFrame()

    result = pd.DataFrame(feats, index=universe)
    result.index.name = "ticker"

    # drop tickers with >50% missing features
    missing_frac = result.isna().mean(axis=1)
    result = result[missing_frac <= 0.5]

    # fill remaining NaN with cross-sectional median
    result = result.fillna(result.median())

    return result


# ── label merging ─────────────────────────────────────────────────────────────

def merge_features_labels(
    features_by_date: dict[pd.Timestamp, pd.DataFrame],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join feature snapshots with triple barrier labels.

    features_by_date: {date: feature_matrix} from make_features()
    labels:           output of labels.triple_barrier_labels(), indexed by t0

    Returns a single DataFrame ready for ML training:
      index = (date, ticker) MultiIndex
      columns = feature columns + "label" + "bin" + "ret" + "t_touch"
    """
    rows = []
    for t0, label_row in labels.iterrows():
        if t0 not in features_by_date:
            continue
        feat = features_by_date[t0]
        if feat.empty:
            continue
        ticker_col = label_row.name if hasattr(label_row, 'name') else None
        # labels DataFrame is per-ticker; here we match by the index
        for ticker in feat.index:
            row = feat.loc[ticker].to_dict()
            row["label"]   = label_row["label"]
            row["bin"]     = label_row["bin"]
            row["ret"]     = label_row["ret"]
            row["t_touch"] = label_row["t_touch"]
            row["date"]    = t0
            row["ticker"]  = ticker
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df

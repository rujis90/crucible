"""
labels.py — Triple barrier labeling (López de Prado, AFML Ch. 3).

The triple barrier method replaces arbitrary fixed-horizon returns with labels
that reflect how a real strategy actually exits a position: via a profit target
(upper barrier), a stop loss (lower barrier), or a time limit (vertical barrier).

This prevents a subtle but devastating form of look-ahead bias: if you label
a trade as +1 because it was up 5% after 20 days, you ignore the fact that it
was down 8% on day 10 and any real stop loss would have closed it at a loss.

Usage:
    from labels import daily_vol, cusum_events, triple_barrier_labels

    vol   = daily_vol(close, span=100)
    events = cusum_events(close["AAPL"], h=vol["AAPL"] * 2)
    labels = triple_barrier_labels(close["AAPL"], high["AAPL"], low["AAPL"],
                                   events, pt_sl=[1.5, 1.0], max_hold=20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── volatility estimator ──────────────────────────────────────────────────────

def daily_vol(close: pd.DataFrame, span: int = 100, bars_per_day: int = 1) -> pd.DataFrame:
    """
    Exponentially weighted daily return volatility.

    Using EWM rather than rolling avoids a cliff effect where older data
    suddenly drops out of the window. The result is used to size barriers
    so they adapt to each stock's current volatility regime.

    Args:
        close:        DataFrame of adjusted close prices (bar × ticker)
        span:         EWM span in bars (~lookback half-life = span * ln(2))
        bars_per_day: bars per trading day (1=daily, 7=hourly).
                      Output is multiplied by √bars_per_day so the returned
                      volatility is always in daily-return units regardless
                      of bar frequency. Barriers calibrated from this vol
                      will therefore have consistent economic meaning across
                      both daily and hourly datasets.

    Returns:
        DataFrame of same shape as close, values are daily volatility estimates.
    """
    rets = close.pct_change(fill_method=None)
    vol  = rets.ewm(span=span, min_periods=span // 2).std()
    if bars_per_day > 1:
        import math
        vol = vol * math.sqrt(bars_per_day)
    return vol


# ── CUSUM event filter ────────────────────────────────────────────────────────

def cusum_events(price: pd.Series, h: pd.Series | float) -> pd.DatetimeIndex:
    """
    Symmetric CUSUM filter — samples events where cumulative deviation
    from a running reference exceeds threshold h.

    Without event sampling, consecutive daily bars are highly autocorrelated,
    making cross-validation splits misleading (adjacent bars in train and test
    share almost identical information). The CUSUM filter produces a sparse,
    approximately IID set of event dates.

    Args:
        price: price series for a single ticker
        h:     threshold — either a scalar or a Series aligned to price.index.
               Typically set to daily_vol * k, where k ∈ [0.5, 2.0].

    Returns:
        DatetimeIndex of event dates (a subset of price.index).
    """
    t_events = []
    s_pos = s_neg = 0.0
    price = price.dropna()
    log_ret = np.log(price).diff().dropna()

    if isinstance(h, (int, float)):
        h_series = pd.Series(h, index=log_ret.index)
    else:
        h_series = h.reindex(log_ret.index).ffill().fillna(h.mean() if hasattr(h, 'mean') else h)

    for date, ret in log_ret.items():
        threshold = h_series.get(date, 0.0)
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)
        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(date)
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(date)

    return pd.DatetimeIndex(t_events)


# ── triple barrier labeling ───────────────────────────────────────────────────

def _first_touch(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    t0: pd.Timestamp,
    pt: float | None,
    sl: float | None,
    t_final: pd.Timestamp,
) -> tuple[pd.Timestamp, float, int]:
    """
    For a single event starting at t0, walk forward bar by bar until either:
      - high touches the upper barrier (pt multiplier above t0 close)
      - low  touches the lower barrier (sl multiplier below t0 close)
      - we reach t_final (vertical barrier)

    Returns: (touch_date, label_return, label)
      label: +1 upper hit, -1 lower hit, 0 vertical
    """
    price_0 = close.loc[t0]
    if pd.isna(price_0) or price_0 <= 0:
        return t_final, 0.0, 0

    upper = price_0 * (1 + pt) if pt is not None else np.inf
    lower = price_0 * (1 - sl) if sl is not None else -np.inf

    window_dates = close.loc[t0:t_final].index[1:]  # exclude t0 itself

    for date in window_dates:
        h = high.get(date, np.nan)
        l = low.get(date, np.nan)
        c = close.get(date, np.nan)
        if pd.isna(h) or pd.isna(l):
            continue
        if h >= upper:
            ret = (upper - price_0) / price_0
            return date, ret, 1
        if l <= lower:
            ret = (lower - price_0) / price_0
            return date, ret, -1

    # vertical barrier hit
    c_final = close.get(t_final, price_0)
    ret = (c_final - price_0) / price_0 if not pd.isna(c_final) else 0.0
    return t_final, ret, 0


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: list[float],
    max_hold: int = 20,
    min_ret: float = 0.0,
    bars_per_day: int = 1,
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to a series of event dates.

    For each event date t0:
      - Upper barrier: close[t0] * (1 + pt_sl[0] * vol[t0])
      - Lower barrier: close[t0] * (1 - pt_sl[1] * vol[t0])
      - Vertical barrier: t0 + max_hold trading days

    Args:
        close:    adjusted close series (single ticker)
        high:     daily high series
        low:      daily low series
        events:   event dates from cusum_events() or custom sampling
        pt_sl:    [profit_target_multiple, stop_loss_multiple] in units of daily vol.
                  pt_sl=[1.5, 1.0] means upper at +1.5σ, lower at -1.0σ.
                  Pass None in place of a multiple to disable that barrier.
        max_hold: vertical barrier in trading days
        min_ret:  minimum return to assign non-zero label (filters noisy events)

    Returns:
        DataFrame indexed by event date with columns:
          t_touch  — date when a barrier was first touched
          ret      — return from t0 to touch
          label    — +1 (upper), -1 (lower), 0 (vertical / no signal)
          bin      — same as label but 0→+1 if min_ret filter applied (for meta-labeling)
    """
    vol = daily_vol(close.to_frame("px"), bars_per_day=bars_per_day)["px"]
    idx = close.index

    pt_mult = pt_sl[0]
    sl_mult = pt_sl[1]

    rows = []
    for t0 in events:
        if t0 not in idx:
            continue
        v = vol.get(t0, 0.0)
        if pd.isna(v) or v == 0:
            continue

        pt = pt_mult * v if pt_mult is not None else None
        sl = sl_mult * v if sl_mult is not None else None

        # vertical barrier: max_hold bars from t0
        loc = idx.get_loc(t0)
        t_end = idx[min(loc + max_hold, len(idx) - 1)]

        t_touch, ret, label = _first_touch(close, high, low, t0, pt, sl, t_end)

        # min_ret filter: if the vertical barrier hit with tiny return, label 0
        if label == 0 and abs(ret) < min_ret:
            label = 0
            bin_label = 0
        else:
            bin_label = label if label != 0 else (1 if ret >= 0 else -1)

        rows.append({
            "t0":      t0,
            "t_touch": t_touch,
            "ret":     ret,
            "label":   label,
            "bin":     bin_label,
        })

    if not rows:
        return pd.DataFrame(columns=["t0", "t_touch", "ret", "label", "bin"])

    df = pd.DataFrame(rows).set_index("t0")
    df.index.name = "t0"
    return df


# ── label weights ─────────────────────────────────────────────────────────────

def sample_weights_by_uniqueness(
    label_end_times: pd.Series,
    close_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Weight each label inversely by how many concurrent labels overlap it.

    Overlapping labels share information — two trades open simultaneously
    are not independent observations. Down-weighting them makes the ML model
    treat the training set more honestly.

    Args:
        label_end_times: Series indexed by t0 (event start), values = t_touch (event end)
        close_index:     full index of trading days

    Returns:
        Series of weights indexed by t0, values in (0, 1].
    """
    # number of concurrent labels on each trading day
    c = pd.Series(0, index=close_index, dtype=float)
    for t0, t_end in label_end_times.items():
        if t0 not in c.index:
            continue
        c.loc[t0:t_end] += 1.0

    # average concurrency over each label's lifetime
    weights = pd.Series(index=label_end_times.index, dtype=float)
    for t0, t_end in label_end_times.items():
        if t0 not in c.index:
            continue
        avg_c = c.loc[t0:t_end].mean()
        weights[t0] = 1.0 / avg_c if avg_c > 0 else 1.0

    return weights / weights.sum() * len(weights)

"""strategy.py — the agent-editable signal rule.

The framework supplies purged training features, their triple-barrier labels,
and unlabeled test features. This file chooses which test events should become
signals. The backtester handles exits through the same triple-barrier box that
created the labels: profit target, stop loss, or max holding time.
"""
from __future__ import annotations

import pandas as pd

PT_SL           = [1.5, 1.0]   # [profit_target_vol_mult, stop_loss_vol_mult]
MAX_HOLD        = 20           # vertical barrier in trading days
CUSUM_H_MULT    = 1.0          # CUSUM threshold = CUSUM_H_MULT × daily_vol
                               # higher → fewer, more independent events
                               # lower  → more events, more cost drag
USE_CPCV        = True

FEATURES_TO_SEARCH = (
    "ret_21d",
    "rs_rank",
    "ret_63d",
    "px_pos_52w",
    "vol_ratio",
    "dv_rank",
    "vol_momentum",
    # Hourly-only (NaN in daily mode — automatically skipped by univariate rule)
    "open_gap",
    "intraday_mom",
    "intraday_vol_skew",
)
QUANTILE = 0.70
MIN_SAMPLES = 200
MIN_EDGE = 0.0


def _fit_best_univariate_rule(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
) -> tuple[str, str, float] | None:
    """Find the one-feature rule that improves upper-barrier hit probability."""
    aligned = train_features.join(train_labels.rename("label"), how="inner").dropna()
    if aligned.empty:
        return None

    target = (aligned["label"] > 0).astype(float)
    base_rate = float(target.mean())
    best: tuple[float, str, str, float] | None = None
    for feature in FEATURES_TO_SEARCH:
        if feature not in aligned.columns:
            continue

        x = aligned[feature].dropna()
        if x.nunique() < 10:
            continue

        y = aligned.loc[x.index, "label"]
        y_hit = (y > 0).astype(float)
        candidates = (
            ("high", x.quantile(QUANTILE), x >= x.quantile(QUANTILE)),
            ("low", x.quantile(1 - QUANTILE), x <= x.quantile(1 - QUANTILE)),
        )

        for direction, threshold, mask in candidates:
            if mask.sum() < MIN_SAMPLES:
                continue
            edge = float(y_hit[mask].mean() - base_rate)
            if best is None or edge > best[0]:
                best = (edge, feature, direction, float(threshold))

    if best is None or best[0] < MIN_EDGE:
        return None

    _, feature, direction, threshold = best
    return feature, direction, threshold


def get_signals(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Series]:
    signals_by_date: dict[pd.Timestamp, pd.Series] = {}
    rule = _fit_best_univariate_rule(train_features, train_labels)
    if rule is None:
        return signals_by_date

    feature, direction, threshold = rule
    if isinstance(test_features.index, pd.MultiIndex):
        test_dates = test_features.index.get_level_values("t0").unique()
    else:
        test_dates = test_features.index.unique()

    for date in test_dates:
        try:
            snapshot = (
                test_features.xs(date, level="t0")
                if isinstance(test_features.index, pd.MultiIndex)
                else test_features.loc[[date]]
            )

            if snapshot.empty or feature not in snapshot.columns:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            signals = pd.Series(0.0, index=snapshot.index)
            if direction == "high":
                selected = snapshot.index[snapshot[feature] >= threshold]
            else:
                selected = snapshot.index[snapshot[feature] <= threshold]
            signals.loc[selected] = 1.0
            signals_by_date[date] = signals

        except Exception:
            signals_by_date[date] = pd.Series(dtype=float)

    return signals_by_date

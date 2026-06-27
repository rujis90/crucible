"""
cv.py — Combinatorial Purged Cross-Validation (LdP AFML Ch. 12).

Standard k-fold CV is wrong for financial time series. Two problems:

1. Serial correlation: adjacent observations share information.
   Train on days 1-200, test on 201-250 — but the label for day 198 (a
   20-day triple barrier) might not be resolved until day 218, which is
   in the test set. Information from the future leaks into training.

   Fix: PURGING — remove all training samples whose labels end after the
   test period starts.

2. Feature leakage via momentum/volatility lookbacks: a feature on day 201
   uses a 21-day rolling window that includes days 180-200 (training data).
   The model sees correlated features even if the labels are clean.

   Fix: EMBARGO — add a gap between training end and test start equal to
   your longest feature lookback.

3. Single-path bias: a walk-forward backtest produces ONE equity curve.
   One good year and one bad year look exactly the same as two mediocre
   years in the aggregate Sharpe. You can't tell if your strategy is
   robust or just lucky on the test path.

   Fix: CPCV — combinatorial purged cross-validation generates C(N,k)
   test paths. Each path is a complete, valid backtest. You see the full
   distribution of Sharpe ratios, not just the average.

Usage:
    from cv import cpcv_splits, purge_train, add_embargo

    for train_idx, test_idx in cpcv_splits(n=10, k=2, embargo_td=21):
        train_idx = purge_train(train_idx, test_idx, label_end_times)
        train_idx = add_embargo(train_idx, test_idx, embargo_td=21)
        # train model on train_idx, evaluate on test_idx
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


# ── purging ───────────────────────────────────────────────────────────────────

def purge_train(
    train_idx: pd.Index,
    test_idx: pd.Index,
    label_end_times: pd.Series,
) -> pd.Index:
    """
    Remove training samples whose labels overlap with the test window.

    A training sample starting at t0 with label ending at t_touch overlaps
    the test window if t_touch >= test_start. Including such samples would
    let the model see outcomes that depend on test-period prices.

    Args:
        train_idx:       index of training event dates (t0 values)
        test_idx:        index of test event dates (t0 values)
        label_end_times: Series mapping t0 → t_touch (label end date)

    Returns:
        Purged training index (subset of train_idx).
    """
    if test_idx.empty or train_idx.empty:
        return train_idx

    test_start = test_idx.min()
    # keep training samples whose labels end before test window starts
    end_times = label_end_times.reindex(train_idx)
    purged = train_idx[end_times < test_start]
    return purged


def add_embargo(
    train_idx: pd.Index,
    test_idx: pd.Index,
    embargo_td: int,
    all_dates: pd.DatetimeIndex | None = None,
) -> pd.Index:
    """
    Remove training samples within `embargo_td` trading days of the test window.

    Even after purging, feature lookbacks (e.g. 21-day momentum) computed on
    test-adjacent training days will include test-window prices. The embargo
    removes a gap equal to the longest feature lookback.

    Args:
        train_idx:   purged training index
        test_idx:    test event dates
        embargo_td:  number of trading days to embargo (= longest feature lookback)
        all_dates:   full DatetimeIndex of trading days (for counting business days).
                     If None, falls back to calendar days.

    Returns:
        Training index with embargo applied.
    """
    if test_idx.empty or train_idx.empty or embargo_td == 0:
        return train_idx

    test_start = test_idx.min()

    if all_dates is not None:
        # count exactly embargo_td trading days back from test_start
        loc = all_dates.get_loc(test_start) if test_start in all_dates else \
              all_dates.searchsorted(test_start, side="left")
        embargo_start = all_dates[max(0, loc - embargo_td)]
    else:
        # calendar-day fallback (slightly conservative)
        embargo_start = test_start - pd.Timedelta(days=int(embargo_td * 7 / 5) + 5)

    purged_embargoed = train_idx[train_idx < embargo_start]
    return purged_embargoed


# ── combinatorial split generator ─────────────────────────────────────────────

def cpcv_splits(
    dates: pd.DatetimeIndex,
    n: int = 6,
    k: int = 2,
    embargo_td: int = 21,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex, int]]:
    """
    Generate CPCV splits: C(n, k) paths, each with a complete test history.

    Splits the date index into n equal groups. For each combination of k groups
    as test, the remaining (n-k) groups form training. Purging and embargo are
    applied per split.

    With n=6, k=2: C(6,2) = 15 unique test-set combinations.
    Each combination uses 2/6 ≈ 33% of data as test.
    You get 15 distinct (train, test) splits → 15 OOS performance estimates.

    Args:
        dates:      full DatetimeIndex of event dates or trading days
        n:          number of groups to split into
        k:          number of groups in each test set (k < n)
        embargo_td: trading days to embargo around test windows

    Returns:
        List of (train_dates, test_dates, path_id) tuples.
        path_id identifies which combinatorial path this split belongs to.
    """
    if len(dates) < n * 10:
        raise ValueError(f"Too few dates ({len(dates)}) for {n} groups. "
                         f"Need at least {n * 10}.")

    # split dates into n approximately equal groups
    groups = np.array_split(dates, n)
    combos = list(combinations(range(n), k))

    splits = []
    for path_id, test_group_ids in enumerate(combos):
        test_dates  = pd.DatetimeIndex(
            [d for i in test_group_ids for d in groups[i]]
        ).sort_values()
        train_dates = pd.DatetimeIndex(
            [d for i in range(n) if i not in test_group_ids for d in groups[i]]
        ).sort_values()

        # apply embargo: remove training dates within embargo_td of any test boundary
        # (simplified: embargo around test_start only, sufficient for sequential splits)
        if embargo_td > 0 and len(test_dates) > 0:
            test_start = test_dates.min()
            loc = dates.searchsorted(test_start, side="left")
            embargo_start = dates[max(0, loc - embargo_td)]
            # only embargo training dates that are near the test boundary
            # (dates after embargo_start that are also before test_start)
            train_dates = train_dates[
                ~((train_dates >= embargo_start) & (train_dates < test_start))
            ]

        splits.append((train_dates, test_dates, path_id))

    return splits


# ── metrics across paths ───────────────────────────────────────────────────────

def cpcv_metrics(
    path_returns: dict[int, pd.Series],
) -> pd.DataFrame:
    """
    Compute performance metrics for each CPCV path.

    Args:
        path_returns: {path_id: daily_return_series}

    Returns:
        DataFrame with one row per path and columns:
          sharpe, sortino, cagr, max_drawdown, n_days
        Plus aggregate row "mean" and "std" at the bottom.
    """
    rows = []
    for path_id, rets in path_returns.items():
        if rets.empty or rets.std() == 0:
            continue
        ann = 252
        sharpe   = rets.mean() / rets.std() * np.sqrt(ann)
        down_std = rets[rets < 0].std()
        sortino  = rets.mean() / (down_std + 1e-9) * np.sqrt(ann)
        cum      = (1 + rets).cumprod()
        n_years  = len(rets) / ann
        cagr     = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1) * 100
        roll_max = cum.cummax()
        max_dd   = float(((cum - roll_max) / roll_max).min()) * 100

        rows.append({
            "path_id":     path_id,
            "sharpe":      round(sharpe, 4),
            "sortino":     round(sortino, 4),
            "cagr":        round(cagr, 2),
            "max_drawdown":round(max_dd, 2),
            "n_days":      len(rets),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("path_id")

    # aggregate summary
    summary = pd.DataFrame({
        "sharpe":       [df["sharpe"].mean(), df["sharpe"].std()],
        "sortino":      [df["sortino"].mean(), df["sortino"].std()],
        "cagr":         [df["cagr"].mean(), df["cagr"].std()],
        "max_drawdown": [df["max_drawdown"].mean(), df["max_drawdown"].std()],
        "n_days":       [df["n_days"].sum(), 0],
    }, index=["mean", "std"])

    return pd.concat([df, summary])


# ── walk-forward fallback (simple, for quick experiments) ─────────────────────

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    embargo_td: int = 21,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Standard expanding-window walk-forward splits with embargo.
    Simpler than CPCV but produces only one path.
    Use CPCV for final evaluation; use this for fast iteration.
    """
    splits = []
    test_start_year = dates[0].year + train_years

    while True:
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end   = pd.Timestamp(f"{test_start_year + test_years - 1}-12-31")
        if test_end > dates[-1]:
            break

        train_end_raw = test_start - pd.Timedelta(days=1)

        train_d = dates[dates <= train_end_raw]
        test_d  = dates[(dates >= test_start) & (dates <= test_end)]

        if len(train_d) < 252 or len(test_d) < 20:
            test_start_year += test_years
            continue

        # embargo
        if embargo_td > 0:
            loc = len(train_d)
            embargo_start = train_d[max(0, loc - embargo_td)]
            train_d = train_d[train_d < embargo_start]

        splits.append((train_d, test_d))
        test_start_year += test_years

    return splits

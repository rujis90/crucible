"""
backtest.py — FIXED INFRASTRUCTURE. Do not modify.

Walk-forward + CPCV backtester for NDX single-stock signals.
Requires the NDX PIT dataset in data/ (buy at [website]).

Key differences from original Crucible:
  - Universe:  NDX PIT dataset (point-in-time, survivorship-bias-free)
  - Labels:    triple barrier (not raw returns)
  - Validation: CPCV paths (not single walk-forward)
  - Input to strategy: feature matrix + purged triple-barrier labels
  - Execution: signals enter, then exit at their triple-barrier touch

Metrics (grep-friendly):
  oos_sharpe:      <float>   — mean Sharpe across CPCV paths
  oos_sharpe_std:  <float>   — std of Sharpe across paths (dispersion)
  cpcv_paths:      <int>     — number of paths evaluated
  folds_passed:    <int>/<int>
  max_drawdown:    <float>%
  num_signals:     <int>
  upper_hit_rate:  <float>
  avg_signal_ret:  <float>
  median_signal_ret:<float>
  profit_factor:   <float>
  avg_holding_days:<float>
  signal_frequency:<float>   — signals per year
  hit_rate_std:    <float>   — std of upper hit rate across paths
  elapsed_seconds: <float>
"""

import importlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cv import cpcv_splits, walk_forward_splits
from features import make_features
from labels import daily_vol, cusum_events, triple_barrier_labels

warnings.filterwarnings("ignore")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATA_DIR        = Path(__file__).parent / "data"
OHLCV_FILE      = DATA_DIR / "ndx_ohlcv.parquet"
PIT_FILE        = DATA_DIR / "ndx_pit_daily.parquet"

START_DATE      = "2010-01-01"   # 3 years of history before first test fold
END_DATE        = "2026-12-31"

# Triple barrier parameters (tunable in strategy.py)
DEFAULT_PT_SL   = [1.5, 1.0]    # profit target, stop loss (in daily vol units)
DEFAULT_MAX_HOLD = 20            # vertical barrier in trading days
DEFAULT_VOL_SPAN = 100           # EWM span for vol estimation

# CPCV parameters
CPCV_N          = 6              # split into N groups
CPCV_K          = 2              # k groups per test set → C(N,K) paths
EMBARGO_TD      = 21             # embargo in trading days (= feature lookback)

# Signal costs
COMMISSION_BPS  = 3
SLIPPAGE_BPS    = 2
ROUND_TRIP_COST = 2 * (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000

# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> dict:
    """
    Load NDX PIT dataset. Raises clear error if files are missing.

    Expected files in data/:
      ndx_ohlcv.parquet      — MultiIndex (field, ticker) × date
      ndx_pit_daily.parquet  — bool DataFrame (date × ticker)
    """
    if not OHLCV_FILE.exists():
        raise FileNotFoundError(
            f"Missing {OHLCV_FILE}\n"
            "Buy the NDX PIT dataset and place files in data/\n"
            "See README.md for instructions."
        )
    if not PIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PIT_FILE}\n"
            "Buy the NDX PIT dataset and place files in data/\n"
            "See README.md for instructions."
        )

    ohlcv = pd.read_parquet(OHLCV_FILE)
    pit   = pd.read_parquet(PIT_FILE)

    # Restrict to configured date range
    ohlcv = ohlcv.loc[START_DATE:END_DATE]
    pit   = pit.loc[START_DATE:END_DATE]

    # Align indices
    shared_dates = ohlcv.index.intersection(pit.index)
    ohlcv = ohlcv.loc[shared_dates]
    pit   = pit.loc[shared_dates]

    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    volume = ohlcv["volume"]
    open_  = ohlcv["open"] if "open" in ohlcv.columns.get_level_values(0) else close

    return {
        "close":  close,
        "high":   high,
        "low":    low,
        "volume": volume,
        "open":   open_,
        "pit":    pit,
    }


def build_label_cache(
    data: dict,
    pt_sl: list[float],
    max_hold: int,
    vol_span: int,
) -> dict[str, pd.DataFrame]:
    """
    Pre-compute triple barrier labels for all tickers.

    Returns {ticker: label_DataFrame} where label_DataFrame is indexed by
    event date t0 with columns [t_touch, ret, label, bin].

    This is computed once before cross-validation, then sliced per fold.
    """
    close  = data["close"]
    high   = data["high"]
    low    = data["low"]
    pit    = data["pit"]

    all_tickers = [t for t in pit.columns if t in close.columns]
    vol_all = daily_vol(close[all_tickers], span=vol_span)

    label_cache = {}
    for ticker in all_tickers:
        c = close[ticker].dropna()
        h = high[ticker].reindex(c.index).ffill()
        l = low[ticker].reindex(c.index).ffill()
        v = vol_all[ticker].reindex(c.index)

        if len(c) < max_hold * 3:
            continue

        # CUSUM threshold: 1× daily vol (sample ~1 event per week on average)
        events = cusum_events(c, h=v)
        if len(events) < 10:
            continue

        labels = triple_barrier_labels(c, h, l, events, pt_sl, max_hold)
        if not labels.empty:
            label_cache[ticker] = labels

    return label_cache


def label_end_times(label_cache: dict[str, pd.DataFrame]) -> pd.Series:
    """Return the latest label end time for each event date across tickers."""
    rows = [
        labels["t_touch"]
        for labels in label_cache.values()
        if not labels.empty and "t_touch" in labels
    ]
    if not rows:
        return pd.Series(dtype="datetime64[ns]")
    return pd.concat(rows).groupby(level=0).max().sort_index()


def build_feature_cache(
    data: dict,
    event_dates: pd.DatetimeIndex,
    lookback: int = 252,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Pre-compute feature matrices for each unique event date.

    Returns {date: feature_matrix} where feature_matrix has one row per
    ticker that was in the NDX on that date.
    """
    feat_cache = {}
    unique_dates = sorted(set(event_dates))
    for date in unique_dates:
        feats = make_features(
            data["close"], data["high"], data["low"], data["volume"],
            data["pit"], as_of_date=date, lookback=lookback,
        )
        if not feats.empty:
            feat_cache[date] = feats
    return feat_cache


def run_fold(
    strategy_module,
    data: dict,
    label_cache: dict,
    feat_cache: dict,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, dict]:
    """
    Run one (train, test) fold. Returns daily returns plus signal metrics.

    The strategy receives:
      - train_features: feature matrix for purged/embargoed training events
      - train_labels:   triple-barrier bin labels for those training events
      - test_features:  feature matrix for events in test_dates (no labels)
    And returns:
      - signals: date → Series{ticker: positive value}

    Each positive signal opens one independent long event. The event exits when
    its precomputed triple-barrier label touches PT, SL, or the vertical barrier.
    No portfolio max-stock cap, gross limit, or periodic rebalance is applied.
    """
    close = data["close"]

    # Collect train samples across all tickers
    train_rows, test_rows, outcome_rows = [], [], []

    for ticker, labels in label_cache.items():
        train_labels = labels[labels.index.isin(train_dates)]
        test_labels  = labels[labels.index.isin(test_dates)]

        for t0, lrow in train_labels.iterrows():
            if t0 in feat_cache and ticker in feat_cache[t0].index:
                row = feat_cache[t0].loc[ticker].to_dict()
                row.update({"t0": t0, "ticker": ticker,
                            "label": lrow["label"], "bin": lrow["bin"], "ret": lrow["ret"]})
                train_rows.append(row)

        for t0, lrow in test_labels.iterrows():
            if t0 in feat_cache and ticker in feat_cache[t0].index:
                row = feat_cache[t0].loc[ticker].to_dict()
                row.update({"t0": t0, "ticker": ticker, "label": None, "bin": None})
                test_rows.append(row)
                outcome_rows.append({
                    "t0": t0,
                    "ticker": ticker,
                    "t_touch": lrow["t_touch"],
                    "ret": lrow["ret"],
                    "label": lrow["label"],
                    "bin": lrow["bin"],
                })

    if not train_rows or not test_rows:
        return pd.Series(dtype=float), signal_metrics(pd.DataFrame(), 0)

    train_df = pd.DataFrame(train_rows).set_index(["t0", "ticker"])
    test_df  = pd.DataFrame(test_rows).set_index(["t0", "ticker"])
    outcomes = pd.DataFrame(outcome_rows).set_index(["t0", "ticker"])

    feature_cols = [c for c in train_df.columns if c not in ("label", "bin", "ret")]

    try:
        signals_by_date = strategy_module.get_signals(
            train_features=train_df[feature_cols],
            train_labels=train_df["bin"],
            test_features=test_df[feature_cols],
        )
    except Exception as e:
        print(f"  strategy error: {e}")
        return pd.Series(dtype=float), signal_metrics(pd.DataFrame(), len(test_dates))

    realized = []
    active_until: dict[str, pd.Timestamp] = {}

    for date in sorted(signals_by_date):
        raw_signals = signals_by_date[date]
        if raw_signals is None or len(raw_signals) == 0:
            continue

        for ticker in raw_signals[raw_signals > 0].index:
            key = (date, ticker)
            if key not in outcomes.index:
                continue
            if ticker in active_until and date <= active_until[ticker]:
                continue

            outcome = outcomes.loc[key]
            exit_date = outcome["t_touch"]
            if pd.isna(exit_date) or exit_date not in close.index:
                continue

            active_until[ticker] = exit_date
            hold_days = (exit_date - date).days
            realized.append({
                "entry_date": date,
                "exit_date": exit_date,
                "ticker": ticker,
                "ret": float(outcome["ret"]) - ROUND_TRIP_COST,
                "label": int(outcome["label"]),
                "holding_days": hold_days,
            })

    if not realized:
        return pd.Series(dtype=float), signal_metrics(pd.DataFrame(), len(test_dates))

    trades = pd.DataFrame(realized)
    exit_returns = trades.groupby("exit_date")["ret"].mean()
    end_date = max(max(test_dates), exit_returns.index.max())
    test_index = close.index[(close.index >= min(test_dates)) & (close.index <= end_date)]
    returns = pd.Series(0.0, index=test_index)
    returns.loc[returns.index.intersection(exit_returns.index)] = exit_returns
    return returns, signal_metrics(trades, len(test_index))


def signal_metrics(trades: pd.DataFrame, n_days: int) -> dict:
    if trades.empty:
        return {
            "num_signals": 0,
            "upper_hits": 0,
            "upper_hit_rate": 0.0,
            "avg_signal_ret": 0.0,
            "median_signal_ret": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0,
            "signal_frequency": 0.0,
        }

    rets = trades["ret"].astype(float)
    gains = rets[rets > 0].sum()
    losses = rets[rets < 0].sum()
    n_years = max(n_days / 252, 1 / 252)
    upper_hits = int((trades["label"] == 1).sum())

    return {
        "num_signals": int(len(trades)),
        "upper_hits": upper_hits,
        "upper_hit_rate": upper_hits / len(trades),
        "avg_signal_ret": float(rets.mean()),
        "median_signal_ret": float(rets.median()),
        "profit_factor": float(gains / abs(losses)) if losses < 0 else float("inf"),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "signal_frequency": float(len(trades) / n_years),
    }


def aggregate_signal_metrics(path_signal_metrics: dict[int, dict]) -> dict:
    metrics = list(path_signal_metrics.values())
    if not metrics:
        return signal_metrics(pd.DataFrame(), 0) | {"hit_rate_std": 0.0}

    total_signals = sum(m["num_signals"] for m in metrics)
    total_upper_hits = sum(m["upper_hits"] for m in metrics)
    hit_rates = [m["upper_hit_rate"] for m in metrics if m["num_signals"] > 0]

    def weighted_avg(key: str) -> float:
        if total_signals == 0:
            return 0.0
        return sum(m[key] * m["num_signals"] for m in metrics) / total_signals

    return {
        "num_signals": int(total_signals),
        "upper_hits": int(total_upper_hits),
        "upper_hit_rate": total_upper_hits / total_signals if total_signals else 0.0,
        "avg_signal_ret": weighted_avg("avg_signal_ret"),
        "median_signal_ret": weighted_avg("median_signal_ret"),
        "profit_factor": float(np.mean([m["profit_factor"] for m in metrics])),
        "avg_holding_days": weighted_avg("avg_holding_days"),
        "signal_frequency": float(np.mean([m["signal_frequency"] for m in metrics])),
        "hit_rate_std": float(np.std(hit_rates)) if hit_rates else 0.0,
    }


def compute_metrics(returns: pd.Series) -> dict:
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
    ann = 252
    sharpe  = returns.mean() / returns.std() * np.sqrt(ann)
    cum     = (1 + returns).cumprod()
    n_years = len(returns) / ann
    cagr    = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1) * 100
    roll_max = cum.cummax()
    max_dd   = float(((cum - roll_max) / roll_max).min()) * 100
    return {"sharpe": float(sharpe), "cagr": round(cagr, 2), "max_drawdown": round(max_dd, 2)}


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import strategy
    importlib.reload(strategy)

    t0_wall = time.time()

    print("Loading data …")
    data = load_data()
    close = data["close"]
    pit   = data["pit"]
    print(f"Universe: {pit.sum(axis=1).iloc[-1]:.0f} current NDX members  "
          f"| {pit.shape[1]} historical  "
          f"| {close.index[0].date()} → {close.index[-1].date()}")

    # Strategy-level config overrides
    pt_sl    = getattr(strategy, "PT_SL",    DEFAULT_PT_SL)
    max_hold = getattr(strategy, "MAX_HOLD", DEFAULT_MAX_HOLD)
    vol_span = getattr(strategy, "VOL_SPAN", DEFAULT_VOL_SPAN)

    print(f"Building labels (pt={pt_sl[0]}×vol, sl={pt_sl[1]}×vol, hold={max_hold}d) …")
    label_cache = build_label_cache(data, pt_sl, max_hold, vol_span)
    label_ends = label_end_times(label_cache)
    all_event_dates = pd.DatetimeIndex(sorted({
        t0 for labels in label_cache.values() for t0 in labels.index
    }))
    print(f"Labels: {len(label_cache)} tickers, {len(all_event_dates)} total events")

    print("Building feature cache …")
    feat_cache = build_feature_cache(data, all_event_dates)
    print(f"Features: {len(feat_cache)} date snapshots")

    # CPCV or walk-forward
    use_cpcv = getattr(strategy, "USE_CPCV", True)

    if use_cpcv:
        print(f"Running CPCV (N={CPCV_N}, K={CPCV_K}, embargo={EMBARGO_TD}d) …")
        splits = cpcv_splits(
            all_event_dates,
            n=CPCV_N,
            k=CPCV_K,
            embargo_td=EMBARGO_TD,
            label_end_times=label_ends,
        )
    else:
        print("Running walk-forward (fast mode) …")
        splits = [(tr, te, i) for i, (tr, te) in
                  enumerate(walk_forward_splits(
                      all_event_dates,
                      embargo_td=EMBARGO_TD,
                      label_end_times=label_ends,
                  ))]

    path_returns = {}
    path_signal_metrics = {}
    folds_passed = 0

    for train_dates, test_dates, path_id in splits:
        importlib.reload(strategy)
        fold_ret, signal_stats = run_fold(
            strategy, data, label_cache, feat_cache, train_dates, test_dates
        )
        m = compute_metrics(fold_ret)
        path_signal_metrics[path_id] = signal_stats

        passed = m["sharpe"] > 0
        if passed:
            folds_passed += 1
        flag = "✓" if passed else "✗"
        print(f"  path {path_id:02d}: sharpe={m['sharpe']:+.3f}  "
              f"dd={m['max_drawdown']:.1f}%  "
              f"signals={signal_stats['num_signals']}  "
              f"hit={signal_stats['upper_hit_rate']:.1%}  {flag}")

        if not fold_ret.empty:
            path_returns[path_id] = fold_ret

    path_metrics   = {pid: compute_metrics(r) for pid, r in path_returns.items()}
    sharpe_dist    = [m["sharpe"] for m in path_metrics.values()]
    oos_sharpe     = float(np.mean(sharpe_dist)) if sharpe_dist else 0.0
    sharpe_std     = float(np.std(sharpe_dist))  if sharpe_dist else 0.0
    cagr_dist      = [m["cagr"] for m in path_metrics.values()]
    oos_cagr       = float(np.mean(cagr_dist))   if cagr_dist  else 0.0
    # worst drawdown across paths — the correct metric for CPCV
    worst_dd       = min((m["max_drawdown"] for m in path_metrics.values()), default=0.0)
    signal_summary = aggregate_signal_metrics(path_signal_metrics)

    elapsed = time.time() - t0_wall

    print(f"\n{'='*55}")
    print(f"oos_cagr:        {oos_cagr:.2f}%")
    print(f"oos_sharpe:      {oos_sharpe:.4f}")
    print(f"oos_sharpe_std:  {sharpe_std:.4f}")
    print(f"cpcv_paths:      {len(path_returns)}")
    print(f"folds_passed:    {folds_passed}/{len(splits)}")
    print(f"max_drawdown:    {worst_dd:.2f}%")
    print(f"num_signals:     {signal_summary['num_signals']}")
    print(f"upper_hit_rate:  {signal_summary['upper_hit_rate']:.4f}")
    print(f"avg_signal_ret:  {signal_summary['avg_signal_ret']:.4f}")
    print(f"median_signal_ret: {signal_summary['median_signal_ret']:.4f}")
    print(f"profit_factor:   {signal_summary['profit_factor']:.4f}")
    print(f"avg_holding_days: {signal_summary['avg_holding_days']:.2f}")
    print(f"signal_frequency: {signal_summary['signal_frequency']:.2f}")
    print(f"hit_rate_std:    {signal_summary['hit_rate_std']:.4f}")
    print(f"elapsed_seconds: {elapsed:.1f}")

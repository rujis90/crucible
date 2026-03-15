"""
backtest.py — FIXED INFRASTRUCTURE. Do not modify.

Walk-forward backtester for portfolio rotation strategies.
- Configurable universe: ETFs, stocks, crypto — anything with a ticker
- Monthly rebalancing, realistic costs
- T+1 execution: signal on day t, execute t+1, earn t+1→t+2
- Returning all-zero weights = cash (flat)

Metrics printed to stdout (grep-friendly):
  oos_sharpe:      <float>
  folds_passed:    <int>/<int>
  max_drawdown:    <float>%
  elapsed_seconds: <float>
"""

import importlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIGURATION — edit these to change asset class ─────────────────────────
#
# To switch from ETFs to stocks, crypto, or anything with a ticker:
#   1. Replace UNIVERSE with your tickers
#   2. Adjust START_DATE and TRAIN_YEARS for your data range
#   3. Tune COMMISSION_BPS and SLIPPAGE_BPS to match your market
#   4. Set MAX_POSITION and GROSS_LIMIT for your risk tolerance
#

UNIVERSE = [
    # US Broad Equity
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000 small cap
    "IWB",   # Russell 1000 large cap
    "VTV",   # Value
    "VUG",   # Growth
    "IJH",   # S&P 400 mid cap
    # US Sector ETFs
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate (2015+, skip if too short)
    # International Equity
    "EFA",   # Developed ex-US (MSCI EAFE)
    "EEM",   # Emerging Markets
    "EWJ",   # Japan
    "EWG",   # Germany
    "EWU",   # UK
    "EWC",   # Canada
    "EWA",   # Australia
    "FXI",   # China large cap
    "EWZ",   # Brazil
    # Fixed Income
    "AGG",   # US Aggregate Bond
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "SHY",   # 1-3 Year Treasury
    "LQD",   # Investment Grade Corp
    "HYG",   # High Yield Corp
    "MBB",   # Mortgage-Backed
    "TIP",   # TIPS (inflation-protected)
    "EMB",   # Emerging Market Bonds
    # Commodities
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
    "DBC",   # Diversified Commodities
    "PDBC",  # Optimum Yield Commodities (2014+)
    # Real Assets / REITs
    "VNQ",   # US REITs
    "REM",   # Mortgage REITs
    "IAU",   # Gold (alternative to GLD)
]

START_DATE      = "2003-01-01"
END_DATE        = "2026-12-31"
N_FOLDS         = 18           # walk-forward folds (2008-2025)
TRAIN_YEARS     = 5
TEST_YEARS      = 1
STEP_YEARS      = 1

COMMISSION_BPS      = 3        # one-way commission in basis points
SLIPPAGE_BPS        = 2        # base slippage in basis points
SLIPPAGE_VOL_MULT   = 2.5      # max slippage multiplier in volatile regimes
MIN_HISTORY_DAYS    = 756      # asset must have 3+ years of data

MAX_POSITION        = 1     # max single-asset weight
GROSS_LIMIT         = 1.0      # max gross exposure (1.0 = long-only, >1.0 = leverage)

# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).parent / "data.parquet"


def load_data() -> dict:
    """Load OHLCV data for all assets, using parquet cache."""
    if CACHE_FILE.exists():
        raw = pd.read_parquet(CACHE_FILE)
    else:
        print("Downloading data (one-time)...")
        frames = {}
        for field in ("Close", "Volume", "High", "Low"):
            df = yf.download(
                UNIVERSE,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
            )[field]
            frames[field.lower()] = df
        raw = pd.concat(frames, axis=1)
        raw.to_parquet(CACHE_FILE)
        print(f"Cached to {CACHE_FILE}")

    close  = raw["close"].dropna(how="all")
    volume = raw["volume"].reindex(close.index)
    high   = raw["high"].reindex(close.index)
    low    = raw["low"].reindex(close.index)

    days_available = close.notna().sum()
    eligible = days_available[days_available >= MIN_HISTORY_DAYS].index.tolist()
    close  = close[eligible]
    volume = volume[eligible]
    high   = high[eligible]
    low    = low[eligible]

    return {"close": close, "volume": volume, "high": high, "low": low}


def get_folds(index: pd.DatetimeIndex):
    """Generate walk-forward folds. Each fold: (train_dates, test_dates)."""
    folds = []
    test_start_year = pd.Timestamp(START_DATE).year + TRAIN_YEARS

    for fold in range(N_FOLDS):
        ts_year = test_start_year + fold * STEP_YEARS
        te_year = ts_year + TEST_YEARS - 1

        train_end   = pd.Timestamp(f"{ts_year - 1}-12-31")
        train_start = pd.Timestamp(f"{ts_year - TRAIN_YEARS}-01-01")
        test_start  = pd.Timestamp(f"{ts_year}-01-01")
        test_end    = pd.Timestamp(f"{te_year}-12-31")

        if test_end > pd.Timestamp(END_DATE):
            break
        if test_end > pd.Timestamp("today"):
            break

        train_dates = index[(index >= train_start) & (index <= train_end)]
        test_dates  = index[(index >= test_start)  & (index <= test_end)]

        if len(train_dates) < 200 or len(test_dates) < 20:
            continue

        folds.append((train_dates, test_dates))

    return folds


def run_fold(strategy_module, data: dict, train_dates, test_dates) -> pd.Series:
    """
    Run one walk-forward fold. Returns daily return series for OOS period.
    T+1 execution: signal on day t, execute at open t+1, earn t+1→t+2.
    """
    close = data["close"]
    index = close.index

    strategy_module.REBALANCE_EVERY = getattr(strategy_module, "REBALANCE_EVERY", 21)
    returns_all = close.pct_change()

    daily_returns = []
    current_weights = pd.Series(0.0, index=close.columns)
    days_since_rebal = 0

    test_locs = [index.get_loc(d) for d in test_dates]

    for loc in test_locs:
        if loc + 2 >= len(index):
            continue

        exec_date   = index[loc + 1]
        return_date = index[loc + 2]

        if exec_date not in close.index or return_date not in close.index:
            continue

        if days_since_rebal >= strategy_module.REBALANCE_EVERY:
            slice_end = loc + 1
            data_slice = {
                "close":  close.iloc[:slice_end],
                "volume": data["volume"].iloc[:slice_end],
                "high":   data["high"].iloc[:slice_end],
                "low":    data["low"].iloc[:slice_end],
            }
            try:
                new_weights = strategy_module.get_weights(data_slice)
                new_weights = new_weights.reindex(close.columns).fillna(0.0)
            except Exception:
                new_weights = pd.Series(0.0, index=close.columns)

            new_weights = new_weights.clip(-MAX_POSITION, MAX_POSITION)
            gross = new_weights.abs().sum()
            if gross > GROSS_LIMIT:
                new_weights = new_weights / gross * GROSS_LIMIT

            turnover = (new_weights - current_weights).abs().sum()
            if turnover > 0:
                recent_vol = returns_all.iloc[max(0, loc - 21):loc].std().mean()
                long_vol   = returns_all.iloc[max(0, loc - 252):loc].std().mean()
                vol_factor = min(SLIPPAGE_VOL_MULT, max(1.0, recent_vol / (long_vol + 1e-9)))
                adj_cost   = (COMMISSION_BPS + SLIPPAGE_BPS * vol_factor) * 2 / 10_000
                cost       = turnover * adj_cost
            else:
                cost = 0.0

            current_weights = new_weights
            days_since_rebal = 0
        else:
            cost = 0.0

        day_return_vec = (
            close.loc[return_date] / close.loc[exec_date] - 1
        ).fillna(0.0)
        port_return = current_weights.dot(day_return_vec) - cost
        daily_returns.append((return_date, port_return))

        days_since_rebal += 1

    if not daily_returns:
        return pd.Series(dtype=float)

    dates, rets = zip(*daily_returns)
    return pd.Series(list(rets), index=pd.DatetimeIndex(dates))


def compute_metrics(returns: pd.Series) -> dict:
    """Compute Sharpe, Sortino, Calmar, true CAGR, and max drawdown."""
    if returns.empty or returns.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                "cagr": 0.0, "ann_return": 0.0, "max_drawdown": 0.0}

    ann_factor = 252
    sharpe     = returns.mean() / returns.std() * np.sqrt(ann_factor)

    # True CAGR: geometric compounding — what you actually earn.
    # Different from (1+mean)^252 which overstates by ignoring variance drag.
    cum       = (1 + returns).cumprod()
    n_years   = len(returns) / ann_factor
    cagr      = float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # ann_return kept for fold-level display (uses arithmetic approx, fine for 1-yr windows)
    ann_return = (1 + returns.mean()) ** ann_factor - 1

    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = drawdown.min()

    downside     = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    sortino      = returns.mean() / downside_std * np.sqrt(ann_factor)

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "sharpe":       float(sharpe),
        "sortino":      float(sortino),
        "calmar":       float(calmar),
        "cagr":         float(cagr * 100),
        "ann_return":   float(ann_return * 100),
        "max_drawdown": float(max_dd * 100),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import strategy
    importlib.reload(strategy)

    t0 = time.time()
    data = load_data()
    close = data["close"]
    print(f"Universe: {len(close.columns)} assets  |  {close.index[0].date()} \u2192 {close.index[-1].date()}")

    folds = get_folds(close.index)
    print(f"Folds: {len(folds)}")

    fold_results = []
    folds_passed = 0

    for fi, (train_dates, test_dates) in enumerate(folds):
        importlib.reload(strategy)
        r = run_fold(strategy, data, train_dates, test_dates)
        m = compute_metrics(r)
        fold_results.append((fi + 1, test_dates, r, m))

        year = test_dates[0].year
        passed = m["sharpe"] > 0
        if passed:
            folds_passed += 1
        flag = "\u2713" if passed else "\u2717"
        print(f"  fold {fi+1:02d} ({year}): sharpe={m['sharpe']:+.3f}  dd={m['max_drawdown']:.1f}%  ann={m['ann_return']:.1f}%  {flag}")

    all_rets = pd.concat([r for _, _, r, _ in fold_results]).sort_index()
    combined = compute_metrics(all_rets)

    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"oos_cagr:        {combined['cagr']:.2f}%")
    print(f"oos_sharpe:      {combined['sharpe']:.4f}")
    print(f"oos_sortino:     {combined['sortino']:.4f}")
    print(f"oos_calmar:      {combined['calmar']:.4f}")
    print(f"folds_passed:    {folds_passed}/{len(folds)}")
    print(f"max_drawdown:    {combined['max_drawdown']:.2f}%")
    print(f"elapsed_seconds: {elapsed:.1f}")

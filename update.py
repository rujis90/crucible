"""
update.py — Refresh market data and regime scores.

Run this weekly (or before checking today's signals):
    python update.py

What it does:
  1. Re-downloads latest prices from Yahoo Finance → data.parquet
  2. Refits the HMM regime model on full history
  3. Recomputes regime_scores.csv

Runtime: ~20-30 seconds.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import backtest as bt
import regime_model as rm


SCORES_FILE = Path(__file__).parent / "regime_scores.csv"
CACHE_FILE  = Path(__file__).parent / "data.parquet"
SMOOTH_DAYS = 5   # days to smooth regime scores (reduces rebalance noise)


def refresh_data() -> dict:
    """Force re-download of all price data, update parquet cache."""
    print("Downloading latest prices...")
    frames = {}
    for field in ("Close", "Volume", "High", "Low"):
        df = yf.download(
            bt.UNIVERSE,
            start=bt.START_DATE,
            end="2099-12-31",
            auto_adjust=True,
            progress=False,
        )[field]
        frames[field.lower()] = df

    raw = pd.concat(frames, axis=1)
    raw.to_parquet(CACHE_FILE)

    close  = raw["close"].dropna(how="all")
    volume = raw["volume"].reindex(close.index)
    high   = raw["high"].reindex(close.index)
    low    = raw["low"].reindex(close.index)

    days_available = close.notna().sum()
    eligible = days_available[days_available >= bt.MIN_HISTORY_DAYS].index.tolist()
    data = {
        "close":  close[eligible],
        "volume": volume[eligible],
        "high":   high[eligible],
        "low":    low[eligible],
    }

    last_date = close.index[-1].date()
    print(f"  Data updated through {last_date}  ({len(close)} trading days)")
    return data


def compute_regime_scores(data: dict) -> pd.DataFrame:
    """
    Fit HMM on full price history, compute daily regime probability scores.

    State labeling:
      crisis states — vol_level above median across states (high stress)
      bull states   — vol_level below median across states (low stress, positive trend)
      entropy       — normalized Shannon entropy of state probability vector
    """
    print("Computing regime features...")
    features = rm.compute_features(data["close"])

    print("Selecting optimal HMM state count via BIC...")
    n_states = rm.select_n_states(features)
    print(f"  BIC-optimal states: {n_states}")

    print("Fitting HMM...")
    model = rm.fit_hmm(features, n_states=n_states)

    # States are sorted by vol_level ascending (from regime_model.py)
    # High-index states = high vol = crisis; low-index = low vol = bull
    median_rank = (n_states - 1) / 2
    crisis_states = [i for i in range(n_states) if i > median_rank]
    bull_states   = [i for i in range(n_states) if i < median_rank]
    print(f"  Crisis states: {crisis_states}  |  Bull states: {bull_states}")

    print("Computing state probabilities for each day...")
    X = features.values.astype(float)

    # Forward probabilities (no lookahead in the filtering sense)
    # We use the full-history model but compute day-by-day emission probs
    log_emission = model._compute_log_likelihood(X)  # (T, n_states)
    log_emission -= log_emission.max(axis=1, keepdims=True)
    probs = np.exp(log_emission)
    probs /= probs.sum(axis=1, keepdims=True)          # shape (T, n_states)

    crisis_prob = probs[:, crisis_states].sum(axis=1)
    bull_prob   = probs[:, bull_states].sum(axis=1)
    entropy     = np.array([rm.state_entropy(p) for p in probs])

    scores = pd.DataFrame({
        "crisis_prob": crisis_prob,
        "bull_prob":   bull_prob,
        "entropy":     entropy,
    }, index=features.index)

    # Smooth to reduce day-to-day noise (strategy rebalances monthly anyway)
    scores = scores.rolling(SMOOTH_DAYS, min_periods=1).mean()

    return scores


def main():
    data   = refresh_data()
    scores = compute_regime_scores(data)

    scores.to_csv(SCORES_FILE)
    last = scores.iloc[-1]
    print(f"\nRegime scores updated through {scores.index[-1].date()}")
    print(f"  crisis_prob : {last['crisis_prob']:.3f}")
    print(f"  bull_prob   : {last['bull_prob']:.3f}")
    print(f"  entropy     : {last['entropy']:.3f}")
    print(f"\nSaved {len(scores)} rows to {SCORES_FILE.name}")
    print("Run: python today.py  to see today's signals")


if __name__ == "__main__":
    main()

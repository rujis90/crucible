"""
regime_model.py — Data-driven HMM regime detection.

Key design choices:
  - No named/hardcoded states. The number of regimes is selected by BIC —
    the data decides how many latent states it contains.
  - Output is a SOFT probability vector [P(s0), P(s1), ..., P(sk)], not a
    hard label. This vector is a continuous feature, not a lookup key.
  - "No detection" (uniform probabilities) is a valid and meaningful signal:
    high entropy = market is transitioning = params blend toward the mean.
  - Params are blended continuously: params = Σ P(si) * params_i
    No sudden jumps at regime boundaries.

State ordering: sorted by mean vol_level (feature index 1), ascending.
  Lowest vol state = state 0, highest = state N-1.
  This anchors label ordering across refits without naming any state.
"""

import warnings

import numpy as np
import pandas as pd
from hmmlearn import hmm

warnings.filterwarnings("ignore")

MIN_STATES    = 2
MAX_STATES    = 7
MIN_HISTORY   = 504   # 2 years minimum to fit


def compute_features(close: pd.DataFrame) -> pd.DataFrame:
    """
    5 regime features from QQQ price history.
    All z-scored against expanding history (no lookahead).
    """
    if "QQQ" not in close.columns:
        raise ValueError("QQQ required")

    qqq  = close["QQQ"].dropna()
    rets = qqq.pct_change().dropna()

    vol_short = rets.rolling(10).std()
    vol_long  = rets.rolling(60).std().replace(0, np.nan)
    vol_ratio = (vol_short / vol_long).replace([np.inf, -np.inf], np.nan)
    vol_level = rets.rolling(21).std()
    mom_126   = qqq.pct_change(126)
    mom_21    = qqq.pct_change(21)
    roll_max  = qqq.rolling(126).max()
    drawdown  = (qqq - roll_max) / roll_max.replace(0, np.nan)

    df = pd.DataFrame({
        "vol_ratio": vol_ratio,
        "vol_level": vol_level,
        "mom_126":   mom_126,
        "mom_21":    mom_21,
        "drawdown":  drawdown,
    }, index=qqq.index).dropna()

    # Expanding z-score: only uses past data at each point
    normed = df.copy()
    for col in df.columns:
        mu  = df[col].expanding(min_periods=126).mean()
        sig = df[col].expanding(min_periods=126).std().replace(0, 1e-9)
        normed[col] = (df[col] - mu) / sig

    return normed.dropna()


def _bic(model: hmm.GaussianHMM, X: np.ndarray) -> float:
    """
    BIC for a fitted HMM.
    Lower = better. Penalizes complexity to prevent overfitting.
    """
    n, d = X.shape
    n_params = (
        model.n_components ** 2        # transition matrix
        + model.n_components * d       # means
        + model.n_components * d * d   # full covariance matrices
        + model.n_components           # start probs
    )
    return -2 * model.score(X) + n_params * np.log(n)


def select_n_states(features: pd.DataFrame) -> int:
    """
    Fit HMMs with MIN_STATES..MAX_STATES components.
    Return the n that minimizes BIC.
    """
    X = features.values.astype(float)
    best_n, best_bic = MIN_STATES, np.inf

    for n in range(MIN_STATES, MAX_STATES + 1):
        try:
            m = hmm.GaussianHMM(
                n_components=n, covariance_type="full",
                n_iter=100, random_state=42, tol=1e-3,
            )
            m.fit(X)
            b = _bic(m, X)
            if b < best_bic:
                best_bic, best_n = b, n
        except Exception:
            pass

    return best_n


def fit_hmm(features: pd.DataFrame, n_states: int = None) -> hmm.GaussianHMM:
    """
    Fit Gaussian HMM. If n_states is None, select via BIC.
    States are reordered by ascending mean vol_level (stable anchor).
    """
    if n_states is None:
        n_states = select_n_states(features)

    X = features.values.astype(float)
    model = hmm.GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=200, random_state=42, tol=1e-4,
    )
    model.fit(X)

    # Sort states by mean vol_level (feature index 1) — ascending
    order = np.argsort(model.means_[:, 1])
    model.means_     = model.means_[order]
    model.covars_    = model.covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_  = model.transmat_[np.ix_(order, order)]

    return model


def get_state_probs(model: hmm.GaussianHMM, features: pd.DataFrame) -> np.ndarray:
    """
    Return soft state probability vector for the most recent observation.
    Shape: (n_components,) — sums to 1.

    This is the emission probability P(x_t | state_i), normalized.
    It tells you how consistent the current feature vector is with each state.
    """
    X = features.values[-1:].astype(float)
    log_probs = model._compute_log_likelihood(X)[0]
    # Normalize to probabilities
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    return probs


def state_entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy of the state probability vector.
    0 = certain (peaked on one state)
    log(N) = maximum uncertainty (uniform over N states)
    Normalized to [0, 1].
    """
    p = probs[probs > 1e-9]
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(probs)))   # normalized


def blend_params(probs: np.ndarray, regime_params: dict) -> dict:
    """
    Blend regime-specific params weighted by state probabilities.
    params = Σ P(si) * params_i

    The "no detection" case (uniform probs) returns the weighted mean
    of all regime params — a natural, smooth fallback.
    """
    n = len(probs)
    result = {}
    for key in regime_params[0].keys():
        result[key] = float(sum(
            probs[i] * regime_params[i][key]
            for i in range(n)
            if i in regime_params
        ))
    return result

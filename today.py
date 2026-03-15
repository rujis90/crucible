"""
today.py — Today's trading signal.

Run after update.py:
    python today.py

Prints:
  - Status of all 4 regime gates (MA, momentum, credit, vol shock)
  - Volume accumulation check
  - HMM regime scale (entropy-based)
  - Recommended position with weights
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import backtest as bt
import strategy as strat


def check_signals(data: dict) -> None:
    close  = data["close"]
    volume = data["volume"]

    today = close.index[-1].date()
    print(f"\n{'='*50}")
    print(f"  Signal check: {today}")
    print(f"{'='*50}\n")

    if "QQQ" not in close.columns:
        print("ERROR: QQQ not in data")
        return

    qqq     = close["QQQ"]
    returns = qqq.pct_change().dropna()

    # ── Gate 1: MA200 ─────────────────────────────────────────────────────────
    ma = float(qqq.iloc[-strat.MA_SLOW:].mean())
    price = float(qqq.iloc[-1])
    above_ma = price >= ma
    pct_above = (price / ma - 1) * 100
    _gate("MA200", above_ma, f"QQQ {price:.2f} vs MA {ma:.2f}  ({pct_above:+.1f}%)")

    # ── Gate 2: Absolute momentum ─────────────────────────────────────────────
    abs_mom_ok = True
    if len(qqq) >= strat.MOM_WINDOW:
        past = float(qqq.iloc[-strat.MOM_WINDOW])
        mom_ret = (price / past - 1) * 100
        abs_mom_ok = price > past
        _gate("Momentum", abs_mom_ok,
              f"QQQ now {price:.2f} vs {strat.MOM_WINDOW}d ago {past:.2f}  ({mom_ret:+.1f}%)")
    else:
        print("  Momentum   [SKIP] insufficient history")

    # ── Gate 3: Credit filter ─────────────────────────────────────────────────
    credit_ok = True
    if "HYG" in close.columns:
        hyg = close["HYG"].dropna()
        if len(hyg) >= strat.MA_SLOW:
            hyg_price = float(hyg.iloc[-1])
            hyg_ma    = float(hyg.iloc[-strat.MA_SLOW:].mean())
            credit_ok = hyg_price >= hyg_ma
            pct = (hyg_price / hyg_ma - 1) * 100
            _gate("Credit (HYG)", credit_ok,
                  f"HYG {hyg_price:.2f} vs MA {hyg_ma:.2f}  ({pct:+.1f}%)")

    # ── Gate 4: Vol shock ─────────────────────────────────────────────────────
    vol_shock = False
    if len(returns) >= strat.VOL_LONG:
        vs = returns.iloc[-strat.VOL_SHORT:].std()
        vl = returns.iloc[-strat.VOL_LONG:].std() + 1e-9
        ratio = vs / vl
        vol_shock = ratio > strat.VOL_SHOCK_MULT
        _gate("Vol shock", not vol_shock,
              f"short/long vol ratio {ratio:.2f}  (threshold {strat.VOL_SHOCK_MULT:.2f})",
              invert=True)

    # ── Volume accumulation ───────────────────────────────────────────────────
    vol_acc_ok = True
    vol_acc_msg = "n/a"
    if "QQQ" in volume.columns:
        qv = volume["QQQ"].dropna()
        if len(qv) >= strat.VOL_ACC_LONG:
            r_short = float(qv.iloc[-strat.VOL_ACC_SHORT:].mean())
            r_long  = float(qv.iloc[-strat.VOL_ACC_LONG:].mean()) + 1e-9
            ratio   = r_short / r_long
            vol_acc_ok = ratio >= strat.VOL_ACC_THRESHOLD
            pct = (ratio - 1) * 100
            vol_acc_msg = f"vol ratio {ratio:.3f} vs threshold {strat.VOL_ACC_THRESHOLD:.3f}  ({pct:+.1f}%)"
    if not vol_acc_ok:
        print(f"  Vol accum  [WEAK] equity scaled to 50%  —  {vol_acc_msg}")
    else:
        print(f"  Vol accum  [OK]   {vol_acc_msg}")

    # ── HMM regime scale ─────────────────────────────────────────────────────
    current_date = close.index[-1]
    regime_scale = strat._get_regime_scale(current_date)

    scores = strat._load_scores()
    if scores is not None:
        available = scores.index[scores.index <= current_date]
        if len(available):
            row = scores.loc[available[-1]]
            cp  = float(row["crisis_prob"])
            ent = float(row["entropy"])
            print(f"\n  HMM regime  crisis_prob={cp:.3f}  entropy={ent:.3f}")
            reduction = strat.CRISIS_SCALE * cp + strat.ENTROPY_SCALE * max(0, ent - strat.ENTROPY_THRESHOLD)
            print(f"              regime_scale={regime_scale:.3f}  (reduction={reduction:.3f})")
    else:
        print(f"\n  HMM regime  scale={regime_scale:.3f}  (no regime_scores.csv — run update.py)")

    # ── Recommended position ─────────────────────────────────────────────────
    weights = strat.get_weights(data)
    active  = weights[weights > 0.001].sort_values(ascending=False)

    print(f"\n{'─'*50}")
    print("  RECOMMENDED POSITION:\n")
    if active.empty:
        print("  100% CASH")
    else:
        for ticker, w in active.items():
            bar = "█" * int(w * 20)
            print(f"  {ticker:6s}  {w*100:5.1f}%  {bar}")

    # Next rebalance estimate
    print(f"\n  Rebalance every ~{strat.REBALANCE_EVERY} trading days (~1 month)")
    print(f"{'='*50}\n")


def _gate(name: str, passing: bool, detail: str, invert: bool = False) -> None:
    """Print a gate status line."""
    flag = "✓ OK  " if passing else "✗ FAIL"
    print(f"  {name:13s} [{flag}]  {detail}")


if __name__ == "__main__":
    data = bt.load_data()
    check_signals(data)

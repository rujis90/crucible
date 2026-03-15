"""
apply_params.py — Apply best_params.json to strategy.py.

Reads the optimized parameters from best_params.json and patches
the corresponding constants in strategy.py.

Usage:
    python apply_params.py
    python apply_params.py --dry-run   # preview without writing
"""

import argparse
import json
import re
from pathlib import Path

STRATEGY = Path(__file__).parent / "strategy.py"
PARAMS   = Path(__file__).parent / "best_params.json"


def apply(dry_run: bool = False):
    with open(PARAMS) as f:
        data = json.load(f)

    params = data["params"]
    metric = data["metric"]
    value  = data["value"]

    print(f"Applying params optimized for oos_{metric}={value:.4f}\n")

    source = STRATEGY.read_text()
    updated = source

    for name, val in params.items():
        # Match: NAME = <old_value>  # optional comment
        pattern = rf"^({re.escape(name)}\s*=\s*)([^\n#]+)"
        replacement = rf"\g<1>{val}"
        new_source, n = re.subn(pattern, replacement, updated, flags=re.MULTILINE)
        if n == 0:
            print(f"  WARNING: {name} not found in strategy.py — skipping")
        else:
            old_match = re.search(pattern, updated, flags=re.MULTILINE)
            old_val = old_match.group(2).strip() if old_match else "?"
            marker = " (unchanged)" if str(old_val) == str(val) else f"  {old_val} → {val}"
            print(f"  {name:25s}{marker}")
            updated = new_source

    if dry_run:
        print("\n[dry-run] No changes written.")
    else:
        STRATEGY.write_text(updated)
        print(f"\nWritten to {STRATEGY}")
        print("Run: python backtest.py  to verify")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply(dry_run=args.dry_run)

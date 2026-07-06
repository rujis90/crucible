# data/

Place your NDX PIT dataset files here.

## Required files (daily mode)

```
ndx_ohlcv.parquet         — OHLCV for 265 historical NDX tickers, 2007–present
ndx_pit_daily.parquet     — bool DataFrame: was ticker in NDX on each date?
```

## Optional files (hourly mode)

```
ndx_ohlcv_hourly.parquet  — hourly OHLCV, Aug 2023–present (7 bars/trading day)
ndx_pit_hourly.parquet    — bool DataFrame: was ticker in NDX at each hourly bar?
```

Set `MODE = "hourly"` in `backtest.py` to switch to the hourly dataset.
All barrier and feature windows auto-scale — no other changes needed.

## Optional (used for diagnostics)

```
ndx_pit_summary.csv         — per-ticker entry/exit dates
ndx_component_changes.csv   — add/remove events
MISSING_TICKERS.txt         — tickers with no available price data
```

## Where to get the data

Buy the NDX PIT dataset at [crucible-research.com](https://crucible-research.com) — $20 one-time, instant download.

The dataset ships as a ZIP containing the required parquet files, optional
diagnostic files, and 7 CLAUDE.md skill files.
After downloading:

```bash
unzip ndx-pit-dataset.zip
cp ndx-pit-dataset/*.parquet crucible/data/

# Optional: merge skills into CLAUDE.md
cat ndx-pit-dataset/skills/SKILL_*.md >> crucible/CLAUDE.md
```

## Why can't I just download this myself?

Price history for companies that were acquired or delisted is no longer publicly available
from any free source — the data was removed after those companies stopped trading.

That's why the PIT dataset exists: the history was collected before it disappeared.
Running a download script today against today's NDX list would give you survivorship bias
(+208 bps/yr phantom alpha) AND miss the tickers whose data is gone.

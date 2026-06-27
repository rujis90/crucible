# data/

Place your NDX PIT dataset files here.

## Required files

```
ndx_ohlcv.parquet       — OHLCV for 210 historical NDX tickers, 2007–present
ndx_pit_daily.parquet   — bool DataFrame: was ticker in NDX on each date?
```

## Optional (used for diagnostics)

```
ndx_pit_summary.csv         — per-ticker entry/exit dates
ndx_component_changes.csv   — 225 raw add/remove events
MISSING_TICKERS.txt         — 59 tickers with no available price data
```

## Where to get the data

Buy the NDX PIT dataset at [website] — $20 one-time, instant download.

The dataset ships as a ZIP containing all files above plus 6 CLAUDE.md skill files.
After downloading:

```bash
unzip ndx-pit-dataset.zip -d ~/downloads/ndx-pit-data/
cp ~/downloads/ndx-pit-data/*.parquet crucible-ndx/data/

# Optional: merge skills into CLAUDE.md
cat ~/downloads/ndx-pit-data/skills/SKILL_*.md >> crucible-ndx/CLAUDE.md
```

## Why not just use yfinance?

Running `yf.download(ndx_100_today)` introduces survivorship bias — you use today's 100 winners and implicitly exclude companies that failed, got acquired, or were removed. This inflates backtest CAGR by ~208 bps/year.

The PIT dataset tracks all 265 historical NDX members since 2007 with exact inclusion dates, so your backtest only uses the stocks that were actually in the index on each date.

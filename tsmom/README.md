# 0050 Time Series Momentum — research v1

This is a single-ETF adaptation of Moskowitz, Ooi & Pedersen (2012), not a replication of their diversified futures portfolio. Long-only, no leverage or short selling. Default evaluation: 2005–2015; 2016–2020 validation and 2021 onward holdout are opt-in. Do not choose parameters from holdout results.

## Run

From repository root:
```bash
python -m pip install numpy pandas matplotlib yfinance
python tsmom/research.py --self-test
python tsmom/research.py --download
python tsmom/research.py --input 0050_daily.csv
```
Use --start 2016-01-01 --end 2020-12-31 only after freezing the development specification. The input CSV needs Date, Open, Close, Adj Close (Yahoo Finance format). Download uses 0050.TW. Check vendor data, dividend and 2025 split adjustment independently before interpreting results. The script uses adjusted opens (Open * Adj Close / Close) for total-return accounting, not actual whole-share executions.

## Models (predefined candidates)
- M0: 100% buy and hold.
- M1: 12-month relative-to-cash momentum > 0, otherwise cash.
- M2: mean of volatility-scaled 1, 3, 6, 12-month relative-to-cash returns > 0, otherwise cash.
- M2_cont: continuous weight clip(0.5 + 0.5 * score, 0, 1).
- M3: same continuous mapping, shrink score toward 0 using ER/(ER+0.1), where ER is 252 trading-day log-price efficiency ratio.
- Fixed50: 50% constant target, monthly rebalanced.
- Matched exposure: post-hoc diagnostic using the strategy's realized average beginning-of-period exposure; not a deployable out-of-sample benchmark.

Signal uses last observed trading-day close in prior calendar month. Trade at next trading day's open, with configurable slippage proxy. No monthly round trip if already at target. Daily weights drift between trades. Open-to-open mark-to-market including cash accrual. Last day marked at open; no forced final liquidation. Fractional shares, dividends reinvested in total-return units. The fixed effective annual cash/risk-free rate (default 1.5%) is a placeholder, not a verified historical Taiwan rate. Commission default 0.1425% x 0.28, ETF sell tax 0.1%, one-way slippage 5 bps: research assumptions, not broker quotes. No minimum commission, integer lots, actual auction fills, or market impact.

Outputs: daily equity and returns, signal/target/turnover/cost log, summary, equity chart, monthly feature data. Gross and net paths are independently simulated. Sharpe uses daily excess returns (all trading days, including cash days), annualized by sqrt(252); alpha is arithmetic annualized OLS intercept vs 0050 excess return, not a significance test. Sample start equity is 1, no historical positions inherited. No performance has been verified against live market data in this repository yet.

Data files and generated outputs should remain local. Do not commit API keys or licensed data.

## Invalid price rows
The loader writes an audit CSV next to the input (e.g. 0050_daily.invalid.csv) containing dates, price fields and invalid_fields. Leading invalid rows before the requested evaluation start are excluded with a warning; any other invalid row stops the strict run. Investigate against a second source before using the result. For exploratory runs only, use `python tsmom/research.py --input 0050_daily.csv --drop-invalid`; this explicitly excludes invalid dates, which may affect monthly signals, execution dates, volatility and performance. No price imputation is performed and the original CSV is retained. Run `--self-test` to exercise the synthetic audit checks.

## Short history / warm-up
The user's Yahoo download for 2003–2015 actually began 2009-01-02. Keep the planned development end (2015-12-31), and either specify a conservative start (2011-01-01) or explicitly pass `--auto-start` to use the first date on which all models have complete signals. Example: `python tsmom/research.py --input 0050_daily.csv --drop-invalid --auto-start`. The default is still strict: it reports the earliest common tradable date instead of silently moving the start. Interior missing signals are errors even with --auto-start. Summary records effective start/end and observations for each model. Exclusion of invalid rows is exploratory; recover original opens where possible. A shorter development sample is not equivalent to the originally planned 2005–2015 study.

## Price continuity / Yahoo split defect
The first user run showed a spurious 75.1482% total-return open-to-open drop on 2014-01-02. The documented Yahoo 0050 split adjustment discontinuity is described at https://volpred.com/reports/mile_735cf27a (2026-04-01). The actual 1:4 split was in June 2025, not 2014. Default loader now audits absolute daily changes >= 7% in adjusted closes and total-return opens and stops for changes >= 20% after any correction. These are conservative research audit thresholds, not assertions that every large move is false. The source CSV remains unchanged. For the exploratory Yahoo dataset only, run `python tsmom/research.py --input 0050_daily.csv --drop-invalid --auto-start --repair-yahoo-0050-split --out tsmom_results_repaired`. The explicit option scales adjusted closes through 2013-12-31 by 0.25 and recomputes adjusted opens only if both boundary series display a matching fourfold discontinuity; already continuous data are not scaled twice, and mixed-scale boundaries fail. Review the .price_jumps.csv audit and verify prices/dividends against TEJ or TWSE before using any performance result. It does not repair other corporate-action or dividend errors. Regenerate ALL signals and equity paths after correction; multiplying strategy equity by four is not a valid fix.

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

"""Single-ETF TSMOM research. No live orders. See README for assumptions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (1, 3, 6, 12)

def load_data(path, drop_invalid=False, start="2005-01-01", end=None):
    """Keep raw CSV intact; audit invalid rows before any optional exclusion."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    cols = ["Open", "Close", "Adj Close"]
    if df.index.has_duplicates or df.index.isna().any():
        raise ValueError("Duplicate or invalid dates")
    if end is not None:
        df = df.loc[:end].copy()
    df = df[cols].apply(pd.to_numeric, errors="raise")
    bad_cells = ~np.isfinite(df) | (df <= 0)
    bad = bad_cells.any(axis=1)
    audit_path = Path(path).with_suffix(".invalid.csv")
    audit = df.loc[bad].copy()
    audit["invalid_fields"] = [
        ", ".join(bad_cells.columns[row]) for row in bad_cells.loc[bad].to_numpy()
    ]
    audit.to_csv(audit_path)
    if df.empty or bad.all():
        raise ValueError(f"No valid price data. See {audit_path}")
    if bad.any():
        print(f"Invalid price rows: {bad.sum()} / {len(df)}")
        print(audit.head(20).to_string())
        print(f"Full audit: {audit_path}")
        # Pre-listing / leading invalid observations may be trimmed; no prices
        # are invented. All other invalid dates require explicit permission.
        first_valid = df.index[~bad][0]
        leading = bad & (df.index < first_valid) & (df.index < pd.Timestamp(start))
        unsafe = bad & ~leading
        if unsafe.any() and not drop_invalid:
            raise ValueError(
                f"{unsafe.sum()} invalid row(s) outside leading warm-up. "
                f"Inspect {audit_path}; recover missing prices if possible. "
                "For an exploratory run only, pass --drop-invalid."
            )
        print(f"WARNING: excluding {bad.sum()} rows "
              f"({leading.sum()} leading, {unsafe.sum()} other); no prices imputed.")
        df = df.loc[~bad].copy()
    df["total_open"] = df.Open * df["Adj Close"] / df.Close
    print(f"Valid daily data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    return df

def features(df, rf):
    p = df["Adj Close"]
    logp = np.log(p)
    # All features use information through the month-end close.
    vol = logp.diff().rolling(252, min_periods=252).std(ddof=1) * np.sqrt(252)
    path = logp.diff().abs().rolling(252, min_periods=252).sum()
    er = (logp.diff(252).abs() / path.replace(0, np.nan)).clip(0, 1)
    daily = pd.DataFrame({"price": p, "vol": vol, "er": er})
    m = daily.groupby(daily.index.to_period("M")).tail(1).copy()
    m["signal_date"] = m.index
    m.index = m.index.to_period("M")
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M"))
    for k in HORIZONS:
        m[f"mom{k}"] = (m.price / m.price.shift(k)) / (1 + rf)**(k/12) - 1
        m[f"z{k}"] = m[f"mom{k}"] / (m.vol.replace(0, np.nan)*np.sqrt(k/12))
    m["score"] = m[[f"z{k}" for k in HORIZONS]].mean(axis=1, skipna=False)
    m["M0"] = 1.
    m["M1"] = (m.mom12 > 0).astype(float).where(m.mom12.notna())
    m["M2"] = (m.score > 0).astype(float).where(m.score.notna())
    m["M2_cont"] = (.5 + .5*m.score).clip(0, 1)
    m["M3"] = (.5 + .5*m.score*m.er/(m.er+.1)).clip(0, 1)
    m["Fixed50"] = .5
    return m

def schedule(df, m, start, end):
    idx = df.loc[start:end].index
    if len(idx) < 2:
        raise ValueError("Need at least two trading dates in requested period")
    prior = idx.to_period("M") - 1
    s = m.reindex(prior).copy()
    s.index = idx
    s["rebalance"] = np.r_[True, prior[1:] != prior[:-1]]
    required = ["M1", "M2", "M2_cont", "M3"]
    if s[required].isna().any().any():
        raise ValueError("Missing momentum warm-up / calendar month. Download earlier history or check missing months.")
    if not (pd.to_datetime(s.signal_date).to_numpy() < idx.to_numpy()).all():
        raise AssertionError("Signal must precede execution")
    return s

def simulate(opens, targets, rebal, rf, commission, tax, slip):
    """Open-to-open self-financing fractional total-return units.
    Daily cash return uses 252 trading-day convention; no final liquidation."""
    cash, stock = 1., 0.
    prev_equity, prev_open = 1., None
    rows = []
    rfday = (1+rf)**(1/252)-1
    for date, price, target, rebalance in zip(opens.index, opens, targets, rebal):
        if prev_open is not None:
            stock *= price / prev_open
            cash *= 1 + rfday
        pre = stock + cash
        before = stock / pre
        delta, cost, fees, taxes, slippage = 0., 0., 0., 0., 0.
        if rebalance:
            desired = target*pre - stock
            rate = commission + slip + (tax if desired < 0 else 0.)
            # Target weight is measured AFTER paying costs.
            delta = desired / (1 + target*rate if desired >= 0 else 1 - target*rate)
            if abs(delta) < 1e-12:
                delta = 0.
            fees = abs(delta)*commission
            taxes = max(-delta, 0)*tax
            slippage = abs(delta)*slip
            cost = fees+taxes+slippage
            stock += delta
            cash -= delta + cost
        equity = stock + cash
        if equity <= 0 or cash < -1e-9 or stock < -1e-9:
            raise AssertionError("Invalid long-only self-financing state")
        rows.append((date, equity, equity/prev_equity-1, before, stock/equity,
                     target, bool(rebalance), delta/pre, abs(delta)/pre,
                     cost, fees, taxes, slippage))
        prev_equity, prev_open = equity, price
    return pd.DataFrame(rows, columns=["date","equity","return","weight_before",
        "weight","target","rebalance","signed_turnover","turnover","cost",
        "commission","tax","slippage"]).set_index("date")

def metrics(d, bench, rf):
    rfday = (1+rf)**(1/252)-1
    # Initial row is entry cost only (no elapsed holding interval).
    # Include its cost in first realized interval, then use common daily dates.
    def intervals(x):
        r = x["return"].iloc[1:].copy()
        r.iloc[0] = (1+x["return"].iloc[0])*(1+r.iloc[0])-1
        return r
    r = intervals(d)
    b = intervals(bench)
    x, y = (b-rfday).to_numpy(), (r-rfday).to_numpy()
    beta = np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1) if np.var(x,ddof=1)>0 else np.nan
    alpha = (y.mean()-beta*x.mean())*252
    equity = d.equity.to_numpy()
    peak = np.maximum.accumulate(np.r_[1.,equity])[1:]
    years = (d.index[-1]-d.index[0]).days/365.25
    sd = np.std(y,ddof=1)
    return {"total_return":equity[-1]-1, "cagr":equity[-1]**(1/years)-1,
            "annual_vol":r.std(ddof=1)*np.sqrt(252),
            "sharpe":y.mean()/sd*np.sqrt(252) if sd>0 else np.nan,
            "max_drawdown":np.min(equity/peak-1), "alpha_annual":alpha, "beta":beta,
            "mean_weight":d.weight.iloc[:-1].mean(), "turnover":d.turnover.sum(),
            "cost_initial_capital_units":d.cost.sum(),
            "trade_count":int((d.turnover>1e-10).sum())}

def run(df, start, end, rf, commission, tax, slip, out):
    df = df.loc[:end].copy()  # Never evaluate future holdout rows.
    m = features(df, rf)
    s = schedule(df,m,start,end)
    opens = df.loc[s.index,"total_open"]
    models = ["M0","M1","M2","M2_cont","M3","Fixed50"]
    results, stats = [], []
    for scenario, rates in [("gross",(0,0,0)),("net",(commission,tax,slip))]:
        paths = {n:simulate(opens,s[n],s.rebalance,rf,*rates) for n in models}
        for n in ["M1","M2","M2_cont","M3"]:
            avg = paths[n].weight.iloc[:-1].mean()
            paths["matched_"+n] = simulate(opens,np.full(len(s),avg),s.rebalance,rf,*rates)
        for n,d in paths.items():
            stats.append({"scenario":scenario,"model":n,**metrics(d,paths["M0"],rf)})
            results.append(d.assign(model=n,scenario=scenario,signal_date=s.signal_date))
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    summary = pd.DataFrame(stats)
    daily = pd.concat(results)
    summary.to_csv(out/"summary.csv",index=False)
    daily.to_csv(out/"daily.csv")
    daily.loc[daily.turnover>1e-10].to_csv(out/"trades.csv")
    m.to_csv(out/"monthly_features.csv")
    s.to_csv(out/"signals.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11,6))
    net = daily[(daily.scenario=="net") & daily.model.isin(models)]
    for n,d in net.groupby("model"):
        ax.plot(d.index,d.equity,label=n)
    ax.set(title="0050 TSMOM — net equity",ylabel="Growth of initial capital")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"equity.png",dpi=150); plt.close(fig)
    print(summary.to_string(index=False))

def self_test():
    idx = pd.bdate_range("2000-01-03","2004-12-31")
    p = pd.Series(100*np.exp(np.arange(len(idx))*.0003 + .001*np.sin(np.arange(len(idx)))),index=idx)
    df = pd.DataFrame({"Open":p,"Close":p,"Adj Close":p,"total_open":p})
    m = features(df,0.)
    s = schedule(df,m,"2003-01-01","2004-12-31")
    assert (s.M1==1).all() and (s.M2==1).all()
    assert s.M3.between(0,1).all()
    modified = df.copy()
    modified.loc["2004-01-01":,"Adj Close"] *= 2
    pd.testing.assert_frame_equal(m.loc[:"2003-12"],features(modified,0.).loc[:"2003-12"])
    o = pd.Series([100.,110.,99.],index=pd.bdate_range("2020-01-01",periods=3))
    d = simulate(o,[1,1,1],[True,False,False],0,0,0,0)
    np.testing.assert_allclose(d.equity,[1,1.1,.99])
    d = simulate(o,[1,1,0],[True,False,True],0,.001,.002,0)
    np.testing.assert_allclose(d.equity.iloc[0],1/1.001)
    np.testing.assert_allclose(d.equity.iloc[-1],.99/1.001*(1-.003))
    assert d.tax.iloc[0]==0 and d.tax.iloc[-1]>0 and d.weight.iloc[-1]==0
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        pth = Path(tmp)/"prices.csv"
        sample = pd.DataFrame({"Date":["2003-01-01","2003-01-02","2003-01-03","2003-01-06"],
                               "Open":[0,100,0,101],"Close":[0,100,100,101],
                               "Adj Close":[0,100,100,101]})
        sample.to_csv(pth,index=False)
        try:
            load_data(pth)
        except ValueError as exc:
            assert "outside leading" in str(exc)
        else:
            raise AssertionError("Internal invalid row must fail strict validation")
        clean = load_data(pth,drop_invalid=True)
        assert len(clean)==2 and clean.total_open.tolist()==[100,101]
        assert len(pd.read_csv(pth.with_suffix(".invalid.csv")))==2
        sample = sample.drop(index=2)
        sample.to_csv(pth,index=False)
        assert len(load_data(pth))==2
    print("Synthetic tests passed: causal features, buy-and-hold, fees, sell tax, cash, invalid-price audit.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",default="0050_daily.csv")
    p.add_argument("--download",action="store_true")
    p.add_argument("--drop-invalid",action="store_true",help="Exploratory only: exclude audited invalid rows, never impute prices")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--start",default="2005-01-01")
    p.add_argument("--end",default="2015-12-31")
    p.add_argument("--rf",type=float,default=.015)
    p.add_argument("--commission",type=float,default=.001425*.28)
    p.add_argument("--tax",type=float,default=.001)
    p.add_argument("--slippage",type=float,default=.0005)
    p.add_argument("--out",default="tsmom_results")
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if not (-1<a.rf and min(a.commission,a.tax,a.slippage)>=0 and
            a.commission+a.tax+a.slippage<1):
        p.error("Invalid rate or costs")
    if pd.Timestamp(a.start)>=pd.Timestamp(a.end):
        p.error("Start must precede end")
    if a.download:
        import yfinance as yf
        raw = yf.download("0050.TW",start="2003-01-01",
            end=str((pd.Timestamp(a.end)+pd.Timedelta(days=1)).date()),
            interval="1d",auto_adjust=False,actions=True,multi_level_index=False)
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned no data")
        raw.index.name="Date"; raw.to_csv(a.input)
    run(load_data(a.input,drop_invalid=a.drop_invalid,start=a.start,end=a.end),
        a.start,a.end,a.rf,a.commission,a.tax,a.slippage,a.out)

if __name__ == "__main__":
    main()

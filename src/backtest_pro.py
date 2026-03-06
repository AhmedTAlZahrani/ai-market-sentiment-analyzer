import pandas as pd
import numpy as np
from pathlib import Path

P = Path("data/processed")
SIG = P / "signals_dataset.csv"
OUT = P / "backtest_results.csv"
EQT = P / "equity_curve.csv"


def metrics(equity):
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {"CAGR": None, "Sharpe": None, "MaxDD": None}
    ann = 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / len(equity)) - 1
    sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(ann)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(dd * 100, 2),
    }


def run(df):
    eq_curves = []
    stats = []
    for ticker in df["Ticker"].unique():
        d = df[df["Ticker"] == ticker].reset_index(drop=True)
        eq = 1.0
        curve = [eq]
        for i in range(1, len(d)):
            r = d["TargetNext"].iloc[i - 1] / 100.0
            eq *= (1 + r)
            curve.append(eq)
        eq_s = pd.Series(curve, index=d["Date"], name=ticker)
        eq_curves.append(eq_s)
        m = metrics(eq_s)
        stats.append({"Ticker": ticker, **m})

    pd.concat(eq_curves, axis=1).to_csv(EQT)
    pd.DataFrame(stats).to_csv(OUT, index=False)
    print(f"✅ Backtest results saved to {OUT} and equity to {EQT}")


def main():
    if not SIG.exists():
        print("⚠️ Run features_signal_engine.py first.")
        return
    df = pd.read_csv(SIG, parse_dates=["Date"])
    run(df)


if __name__ == "__main__":
    main()

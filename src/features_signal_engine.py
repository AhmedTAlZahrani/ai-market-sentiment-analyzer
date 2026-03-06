import pandas as pd
from pathlib import Path

P = Path("data/processed")
TI = P / "technical_indicators.csv"
MERGED = P / "sentiment_price_merged.csv"
OUT = P / "features_dataset.csv"
SIG = P / "signals_dataset.csv"


def main():
    if not TI.exists() or not MERGED.exists():
        print("⚠️ Run compute_indicators.py and merge_sentiment_prices.py first.")
        return

    ti = pd.read_csv(TI, parse_dates=["Date"])
    merged = pd.read_csv(MERGED, parse_dates=["Date"])

    df = pd.merge(merged, ti, on=["Date", "Ticker"], how="left", suffixes=("", "_ind"))
    df["TargetNext"] = df.groupby("Ticker")["Return"].shift(-1)
    df.dropna(inplace=True)

    df.to_csv(OUT, index=False)
    df[["Date", "Ticker", "Return", "TargetNext"]].to_csv(SIG, index=False)
    print(f"✅ Feature & signal datasets saved: {len(df)} rows")


if __name__ == "__main__":
    main()

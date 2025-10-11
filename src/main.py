# src/main.py
import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

DATA_OUT = "data/processed/latest.csv"
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.reset_index().rename(columns=str.lower)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["rsi_14"] = calc_rsi(out["close"], 14)
    out["label_up"] = (out["close"].shift(-1) > out["close"]).astype(int)
    return out

def build_latest_csv(ticker="AAPL", lookback_days=750) -> str:
    end = datetime.utcnow().date()
    start = end - timedelta(days=lookback_days)
    df = fetch_prices(ticker, start.isoformat(), end.isoformat())
    df = add_features(df)
    df["ticker"] = ticker
    df.to_csv(DATA_OUT, index=False)
    return DATA_OUT

if __name__ == "__main__":
    path = build_latest_csv()
    print(f"✅ Latest data saved to {path}")
import pandas as pd
import numpy as np
from pathlib import Path

P_RAW = Path("data/raw")
P_OUT = Path("data/processed/technical_indicators.csv")

def rsi(series, window=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(span=window, adjust=False).mean()
    roll_down = down.ewm(span=window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    width = (upper - lower) / ma
    return ma, upper, lower, width

def main():
    all_frames = []
    for csv in P_RAW.glob("*.csv"):
        try:
            df = pd.read_csv(csv, parse_dates=["Date"])
            if {"Date", "Close", "Ticker"}.issubset(df.columns):
                df = df.sort_values("Date").reset_index(drop=True)
                df["RSI14"] = rsi(df["Close"])
                df["MACD"], df["MACD_signal"] = macd(df["Close"])
                df["BB_MA"], df["BB_up"], df["BB_low"], df["BB_width"] = bollinger_bands(df["Close"])
                all_frames.append(df[["Date","Ticker","Close","RSI14","MACD","MACD_signal","BB_width"]])
        except Exception as e:
            print(f"âš ï¸ Skip {csv}: {e}")
    if all_frames:
        out = pd.concat(all_frames)
        out.to_csv(P_OUT, index=False)
        print(f"âœ… Saved technical indicators to {P_OUT} ({len(out)} rows)")
    else:
        print("âš ï¸ No valid CSVs found in data/raw.")

if __name__ == "__main__":
    main()

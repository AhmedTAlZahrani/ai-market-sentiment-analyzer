import yfinance as yf
from pathlib import Path

def fetch_stock_data(tickers, period="3mo"):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if not df.empty:
            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            df.to_csv(raw_dir / f"{ticker}.csv", index=False)
            print(f"✅ {ticker}: {len(df)} rows saved to data/raw/")
        else:
            print(f"⚠️ No data for {ticker}")

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "SPY", "TSLA"]
    fetch_stock_data(tickers)
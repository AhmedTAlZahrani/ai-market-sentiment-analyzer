import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1️⃣ Load daily sentiment
def load_sentiment(path="data/processed/daily_sentiment.csv"):
    sentiment = pd.read_csv(path)
    sentiment["Date"] = pd.to_datetime(sentiment["Date"])
    print(f"✅ Loaded sentiment data: {len(sentiment)} records")
    return sentiment


# 2️⃣ Load stock price data safely
def load_stock_data(ticker):
    try:
        df = pd.read_csv(f"data/raw/{ticker}_prices.csv")
        df["Date"] = pd.to_datetime(df["Date"])

        # Convert 'Close' to numeric safely
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        # Drop rows where Close is missing or invalid
        df = df.dropna(subset=["Close"])

        # Calculate percent change
        df["Return"] = df["Close"].pct_change() * 100

        df = df[["Date", "Close", "Return"]]
        return df
    except FileNotFoundError:
        print(f"⚠️ No price data found for {ticker}. Skipping.")
        return None


# 3️⃣ Merge sentiment with stock returns
def merge_sentiment_price(sentiment):
    merged_all = []
    for ticker in sentiment["Ticker"].unique():
        s_df = sentiment[sentiment["Ticker"] == ticker]
        p_df = load_stock_data(ticker)
        if p_df is None or p_df.empty:
            continue

        merged = pd.merge(s_df, p_df, on="Date", how="inner")
        merged["Ticker"] = ticker
        merged_all.append(merged)

    if merged_all:
        final_df = pd.concat(merged_all, ignore_index=True)
        final_df.to_csv("data/processed/sentiment_price_merged.csv", index=False)
        print(f"✅ Merged sentiment with stock prices: {len(final_df)} rows saved.")
        return final_df
    else:
        print("⚠️ No merged data created. Check if price files exist.")
        return None


# 4️⃣ Plot sentiment vs return
def plot_sentiment_vs_return(df, ticker):
    ticker_df = df[df["Ticker"] == ticker]
    if ticker_df.empty:
        print(f"⚠️ No merged data to plot for {ticker}")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(ticker_df["SentimentScore"], ticker_df["Return"], alpha=0.7)
    plt.title(f"{ticker}: Sentiment vs Daily Return")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Daily % Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 5️⃣ Run pipeline
if __name__ == "__main__":
    sentiment = load_sentiment()
    merged = merge_sentiment_price(sentiment)
    if merged is not None:
        for t in merged["Ticker"].unique():
            plot_sentiment_vs_return(merged, t)
import pandas as pd


def load_sentiment():
    for path in [
        "data/processed/daily_sentiment_finbert.csv",
        "data/processed/daily_sentiment.csv",
    ]:
        try:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            print(f"✅ Loaded sentiment data: {len(df)} records from {path}")
            return df
        except FileNotFoundError:
            continue
    print("⚠️ No sentiment file found.")
    return pd.DataFrame()


def load_stock_data(ticker):
    try:
        df = pd.read_csv(f"data/raw/{ticker}_prices.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        df["Return"] = df["Close"].pct_change() * 100
        return df[["Date", "Close", "Return"]]
    except FileNotFoundError:
        print(f"⚠️ No price data found for {ticker}. Skipping.")
        return None


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


if __name__ == "__main__":
    sentiment = load_sentiment()
    if not sentiment.empty:
        merge_sentiment_price(sentiment)

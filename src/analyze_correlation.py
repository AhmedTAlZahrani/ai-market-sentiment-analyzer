import pandas as pd

def load_merged(path="data/processed/sentiment_price_merged.csv"):
    df = pd.read_csv(path)
    print(f"✅ Loaded merged data: {len(df)} rows")
    return df

def compute_correlations(df):
    results = []
    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker]
        if sub["SentimentScore"].count() < 2:
            continue
        corr = sub["SentimentScore"].corr(sub["Return"])
        results.append({"Ticker": ticker, "Correlation": round(corr, 3)})
    summary = pd.DataFrame(results)
    summary.to_csv("data/processed/sentiment_correlation_summary.csv", index=False)
    print("\n📈 Sentiment-Return Correlation Summary:")
    print(summary)
    return summary

if __name__ == "__main__":
    merged = load_merged()
    compute_correlations(merged)
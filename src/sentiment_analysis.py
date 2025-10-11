import pandas as pd
import numpy as np
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1️⃣ Load the FinViz headlines file
def load_data(path="data/raw/finviz_news.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Headline"])
    df["Headline"] = df["Headline"].astype(str)
    print(f"✅ Loaded {len(df)} headlines from {path}")
    return df


# 2️⃣ Clean the headlines text
def clean_headlines(df):
    df["Headline"] = df["Headline"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Headline"] = df["Headline"].str.lower()
    return df


# 3️⃣ Analyze sentiment using VADER
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df["SentimentScore"] = df["Headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["SentimentLabel"] = df["SentimentScore"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    return df


# 4️⃣ Parse FinViz dates consistently
def parse_finviz_date(x):
    try:
        # Try Month-Day-Year (FinViz standard)
        parsed = datetime.strptime(x, "%b-%d-%y")
    except Exception:
        # If only time (e.g., "09:30AM"), assign today
        parsed = datetime.now()
    # Convert everything to just the date (drops time part)
    return parsed.date()


# 5️⃣ Aggregate by Ticker & Date safely
def aggregate_daily_sentiment(df):
    df["Date"] = df["Date"].apply(parse_finviz_date)

    # Drop missing or empty sentiment
    df = df.dropna(subset=["SentimentScore"])
    df = df[df["SentimentScore"] != 0]

    # Group cleanly by Ticker + Date
    daily_sentiment = (
        df.groupby(["Ticker", "Date"], as_index=False)["SentimentScore"].mean()
    )

    daily_sentiment["SentimentScore"] = np.round(daily_sentiment["SentimentScore"], 4)
    daily_sentiment.to_csv("data/processed/daily_sentiment.csv", index=False)

    print(f"✅ Saved {len(daily_sentiment)} daily sentiment records to data/processed/daily_sentiment.csv")
    return daily_sentiment


# 6️⃣ Run the pipeline
if __name__ == "__main__":
    df = load_data()
    df = clean_headlines(df)
    df = analyze_sentiment(df)
    daily_sentiment = aggregate_daily_sentiment(df)
    print("\n📊 Sample of processed sentiment data:")
    print(daily_sentiment.head(10))
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime

# 1️⃣ Load headlines
def load_data(path="data/raw/finviz_news.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Headline"])
    df["Headline"] = df["Headline"].astype(str)
    print(f"✅ Loaded {len(df)} headlines from {path}")
    return df

# 2️⃣ Clean text
def clean_headlines(df):
    df["Headline"] = df["Headline"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Headline"] = df["Headline"].str.lower()
    return df

# 3️⃣ Load FinBERT model & tokenizer once
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model

# 4️⃣ Predict sentiment
def finbert_sentiment(df, tokenizer, model):
    labels = ["negative", "neutral", "positive"]
    scores, sentiments = [], []

    for text in tqdm(df["Headline"], desc="🔍 Analyzing with FinBERT"):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_id = torch.argmax(probs).item()
            sentiments.append(labels[sentiment_id])
            score = probs[0, 2].item() - probs[0, 0].item()  # positive minus negative
            scores.append(score)

    df["SentimentLabel"] = sentiments
    df["SentimentScore"] = np.round(scores, 4)
    return df

# 5️⃣ Parse FinViz dates
def parse_finviz_date(x):
    try:
        return datetime.strptime(x, "%b-%d-%y").date()
    except Exception:
        return datetime.now().date()

# 6️⃣ Aggregate by ticker/date
def aggregate_daily_sentiment(df):
    df["Date"] = df["Date"].apply(parse_finviz_date)
    daily = df.groupby(["Ticker", "Date"], as_index=False)["SentimentScore"].mean()
    daily.to_csv("data/processed/daily_sentiment_finbert.csv", index=False)
    print(f"✅ Saved {len(daily)} daily sentiment records to data/processed/daily_sentiment_finbert.csv")
    return daily

# 7️⃣ Run everything
if __name__ == "__main__":
    df = load_data()
    df = clean_headlines(df)
    tokenizer, model = load_finbert()
    df = finbert_sentiment(df, tokenizer, model)
    daily = aggregate_daily_sentiment(df)
    print("\n📊 Sample of FinBERT sentiment data:")
    print(daily.head(10))

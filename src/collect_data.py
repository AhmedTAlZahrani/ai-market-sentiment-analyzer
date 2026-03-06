import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime


def fetch_finviz_news(tickers):
    base_url = "https://finviz.com/quote.ashx?t="
    all_news = []

    for ticker in tickers:
        url = base_url + ticker
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        news_table = soup.find(id="news-table")
        if not news_table:
            continue

        for row in news_table.findAll("tr"):
            headline = row.a.text if row.a else None
            date_data = row.td.text.split()
            date = date_data[0] if len(date_data) == 2 else datetime.now().strftime("%b-%d-%y")
            time_str = date_data[-1]
            all_news.append([ticker, date, time_str, headline])

    df = pd.DataFrame(all_news, columns=["Ticker", "Date", "Time", "Headline"])
    df.to_csv("data/raw/finviz_news.csv", index=False)
    print(f"✅ Saved {len(df)} headlines to data/raw/finviz_news.csv")
    return df


def fetch_stock_data(tickers, period="5d"):
    all_data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period)
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        all_data[ticker] = df
        df.to_csv(f"data/raw/{ticker}_prices.csv", index=False)
    print("✅ Stock data downloaded and saved to data/raw/")
    return all_data


if __name__ == "__main__":
    tickers = ["SPY", "AAPL", "NVDA", "TSLA"]
    fetch_finviz_news(tickers)
    fetch_stock_data(tickers, period="30d")

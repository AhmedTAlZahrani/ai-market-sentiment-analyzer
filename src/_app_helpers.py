import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")

def load_indicators_df():
    p = PROCESSED / "technical_indicators.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Date"])
        return df
    return pd.DataFrame(columns=["Date","Ticker","Close","Return","RSI14","MACD","MACD_signal","MACD_hist","BB_width"])

def load_predictions_df():
    p = PROCESSED / "predictions_nextday.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Ticker","TrainRows","TestRows","R2","MAE","LatestPredictedNextDayReturn_%","LatestDate"])

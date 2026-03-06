import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

PROCESSED = Path("data/processed")
OUTFILE = PROCESSED / "predictions_nextday.csv"

FEATURES = ["SentimentScore", "RSI14", "MACD", "MACD_signal", "BB_width"]


def load_merged():
    p = PROCESSED / "sentiment_price_merged.csv"
    if not p.exists():
        print("⚠️ Merged file not found; run merge step first.")
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["Date"])


def load_indicators():
    p = PROCESSED / "technical_indicators.csv"
    if not p.exists():
        print("⚠️ technical_indicators.csv not found; run indicators step first.")
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["Date"])


def make_features(merged, tech):
    df = merged.merge(tech, on=["Date", "Ticker"], how="left", suffixes=("", "_tech"))
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df["TargetNext"] = df.groupby("Ticker")["Return"].shift(-1)

    for feat in FEATURES:
        df[feat] = pd.to_numeric(df[feat], errors="coerce")

    df = df.dropna(subset=["TargetNext", "SentimentScore"])
    return df


def train_per_ticker(df):
    rows = []
    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker].copy()
        if len(sub) < 15:
            continue

        split = int(len(sub) * 0.8)
        X_train = sub[FEATURES].iloc[:split]
        X_test = sub[FEATURES].iloc[split:]
        y_train = sub["TargetNext"].iloc[:split]
        y_test = sub["TargetNext"].iloc[split:]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred) if len(y_test) > 0 else np.nan
        mae = mean_absolute_error(y_test, pred) if len(y_test) > 0 else np.nan

        last_row = sub.iloc[-1:]
        last_pred = float(model.predict(last_row[FEATURES])[0])

        rows.append({
            "Ticker": ticker,
            "TrainRows": len(X_train),
            "TestRows": len(X_test),
            "R2": round(r2, 3) if pd.notna(r2) else None,
            "MAE": round(mae, 3) if pd.notna(mae) else None,
            "LatestPredictedNextDayReturn_%": round(last_pred, 3),
            "LatestDate": str(last_row["Date"].iloc[0].date()),
        })
    return pd.DataFrame(rows)


def main():
    merged = load_merged()
    tech = load_indicators()
    if merged.empty or tech.empty:
        return

    data = make_features(merged, tech)
    summary = train_per_ticker(data)
    if summary.empty:
        print("⚠️ Not enough data to train.")
        return

    summary.to_csv(OUTFILE, index=False)
    print(f"✅ Saved predictions to {OUTFILE}")
    print(summary)


if __name__ == "__main__":
    main()

@'
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

PROCESSED = Path("data/processed")
OUTFILE = PROCESSED / "predictions_nextday.csv"

def load_merged():
    p = PROCESSED / "sentiment_price_merged.csv"
    if not p.exists():
        print("⚠️ merged file not found; run merge step first.")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"])
    return df

def load_indicators():
    p = PROCESSED / "technical_indicators.csv"
    if not p.exists():
        print("⚠️ technical_indicators.csv not found; run indicators step first.")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"])
    return df

def make_features(merged, tech):
    df = merged.merge(tech, on=["Date","Ticker","Close","Return"], how="left")
    df = df.sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Features at t -> predict Return_{t+1}
    df["TargetNext"] = df.groupby("Ticker")["Return"].shift(-1)

    feats = ["SentimentScore","RSI14","MACD","MACD_signal","MACD_hist","BB_width"]
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    # Drop rows with nan in target or essential features
    df = df.dropna(subset=["TargetNext","SentimentScore"])
    return df, feats

def train_per_ticker(df, feats):
    rows = []
    for t in df["Ticker"].unique():
        sub = df[df["Ticker"]==t].copy()
        # small guard for minimum rows
        if len(sub) < 15:
            continue
        # simple chronological split (80/20)
        split = int(len(sub)*0.8)
        X_train, X_test = sub[feats].iloc[:split], sub[feats].iloc[split:]
        y_train, y_test = sub["TargetNext"].iloc[:split], sub["TargetNext"].iloc[split:]

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred) if len(y_test)>0 else np.nan
        mae = mean_absolute_error(y_test, pred) if len(y_test)>0 else np.nan

        last_row = sub.iloc[-1:]
        last_pred = float(model.predict(last_row[feats])[0])

        rows.append({
            "Ticker": t,
            "TrainRows": len(X_train),
            "TestRows": len(X_test),
            "R2": round(r2, 3) if pd.notna(r2) else None,
            "MAE": round(mae, 3) if pd.notna(mae) else None,
            "LatestPredictedNextDayReturn_%": round(last_pred, 3),
            "LatestDate": str(last_row["Date"].iloc[0].date())
        })
    return pd.DataFrame(rows)

def main():
    merged = load_merged()
    tech = load_indicators()
    if merged.empty or tech.empty:
        return
    data, feats = make_features(merged, tech)
    summary = train_per_ticker(data, feats)
    if summary.empty:
        print("⚠️ Not enough data to train.")
        return
    summary.to_csv(OUTFILE, index=False)
    print(f"✅ Saved predictions to {OUTFILE}")
    print(summary)

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath src/model_predict.py -Encoding utf8
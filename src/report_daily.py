from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime
import pandas as pd
from pathlib import Path

P = Path("data/processed")
OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)


def safe_load(path):
    return pd.read_csv(path) if path.exists() else None


def draw_table(cvs, x, y, df, title):
    cvs.setFont("Helvetica-Bold", 12)
    cvs.drawString(x, y, title)
    y -= 0.5 * cm
    cvs.setFont("Helvetica", 9)

    if df is None or df.empty:
        cvs.drawString(x, y, "No data.")
        return y - 0.5 * cm

    cols = list(df.columns)
    cvs.drawString(x, y, " | ".join(cols))
    y -= 0.4 * cm

    for _, row in df.head(12).iterrows():
        cvs.drawString(x, y, " | ".join(str(row[col])[:18] for col in cols))
        y -= 0.35 * cm

    return y - 0.4 * cm


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filepath = OUT / f"Daily_Brief_{ts}.pdf"

    cvs = canvas.Canvas(str(filepath), pagesize=A4)
    W, H = A4

    cvs.setTitle("AI Market Daily Brief")
    cvs.setFont("Helvetica-Bold", 16)
    cvs.drawString(2 * cm, H - 2 * cm, "AI Market Daily Brief")
    cvs.setFont("Helvetica", 10)
    cvs.drawString(2 * cm, H - 2.6 * cm, f"Generated: {ts}")

    sentiment = safe_load(P / "daily_sentiment_finbert.csv")
    corr = safe_load(P / "sentiment_correlation_summary.csv")
    preds = safe_load(P / "predictions_nextday.csv")
    backres = safe_load(P / "backtest_results.csv")

    y = H - 3.5 * cm

    if corr is not None:
        y = draw_table(cvs, 2 * cm, y, corr.sort_values("Correlation", ascending=False), "Correlation Summary")
    if preds is not None:
        y = draw_table(cvs, 2 * cm, y, preds, "Latest Predicted Next-Day Returns")
    if backres is not None:
        y = draw_table(cvs, 2 * cm, y, backres.sort_values("Sharpe", ascending=False), "Backtest KPIs")
    if sentiment is not None:
        y = draw_table(cvs, 2 * cm, y, sentiment.sort_values("Date", ascending=False).head(10), "Recent Sentiment (FinBERT)")

    cvs.showPage()
    cvs.save()
    print(f"✅ PDF saved to {filepath}")


if __name__ == "__main__":
    main()

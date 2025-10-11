@'
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime
import pandas as pd
from pathlib import Path

P = Path("data/processed")
OUT = Path("output"); OUT.mkdir(parents=True, exist_ok=True)

def safe_load(p):
    return pd.read_csv(p) if p.exists() else None

def draw_table(c, x, y, df, title):
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, title); y -= 0.5*cm
    c.setFont("Helvetica", 9)
    max_rows = 12
    if df is None or df.empty:
        c.drawString(x, y, "No data."); return y - 0.5*cm
    cols = list(df.columns)
    c.drawString(x, y, " | ".join(cols)); y -= 0.4*cm
    for _, row in df.head(max_rows).iterrows():
        c.drawString(x, y, " | ".join([str(row[c])[:18] for c in cols]))
        y -= 0.35*cm
    return y - 0.4*cm

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filepath = OUT / f"Daily_Brief_{ts}.pdf"
    c = canvas.Canvas(str(filepath), pagesize=A4)
    W, H = A4

    c.setTitle("AI Market Daily Brief")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H-2*cm, "AI Market Daily Brief")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-2.6*cm, f"Generated: {ts}")

    sentiment = safe_load(P/"daily_sentiment_finbert.csv")
    merged    = safe_load(P/"sentiment_price_merged.csv")
    corr      = safe_load(P/"sentiment_correlation_summary.csv")
    preds     = safe_load(P/"predictions_nextday.csv")
    backres   = safe_load(P/"backtest_results.csv")

    y = H-3.5*cm
    if corr is not None:
        y = draw_table(c, 2*cm, y, corr.sort_values("Correlation", ascending=False), "Correlation Summary")

    if preds is not None:
        y = draw_table(c, 2*cm, y, preds, "Latest Predicted Next-Day Returns")

    if backres is not None:
        y = draw_table(c, 2*cm, y, backres.sort_values("Sharpe", ascending=False), "Backtest KPIs")

    if sentiment is not None:
        y = draw_table(c, 2*cm, y, sentiment.sort_values("Date", ascending=False).head(10), "Recent Sentiment (FinBERT)")

    c.showPage(); c.save()
    print(f"✅ PDF saved to {filepath}")

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath src/report_daily.py -Encoding utf8

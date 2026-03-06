import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"\n🚀 Starting pipeline at {timestamp}\n")

steps = [
    "python src/collect_data.py",
    "python src/sentiment_analysis_finbert.py",
    "python src/merge_sentiment_prices.py",
    "python src/compute_indicators.py",
    "python src/features_signal_engine.py",
    "python src/walkforward_cv.py",
    "python src/backtest_pro.py",
    "python src/analyze_correlation.py",
    "python src/report_daily.py",
]

for step in steps:
    print(f"\n▶️  {step}")
    rc = os.system(step)
    if rc != 0:
        print(f"❌ Failed: {step}")
        break

print("\n✅ Pipeline completed.")

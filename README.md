<p align="center">
  <img src="https://raw.githubusercontent.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/main/assets/banner_ai_market_sentiment.png" alt="AI Market Sentiment Analyzer Banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/stargazers">
    <img src="https://img.shields.io/github/stars/AhmedTAlzahrani/ai-market-sentiment-analyzer?color=gold&style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/issues">
    <img src="https://img.shields.io/github/issues/AhmedTAlzahrani/ai-market-sentiment-analyzer?style=for-the-badge" alt="Issues">
  </a>
  <a href="https://github.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/commits/main">
    <img src="https://img.shields.io/github/last-commit/AhmedTAlzahrani/ai-market-sentiment-analyzer?style=for-the-badge&color=blue" alt="Last Commit">
  </a>
  <img src="https://img.shields.io/badge/Made%20With-Python%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-E43C2C?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-FinBERT-9cf?style=for-the-badge" alt="FinBERT">
  <img src="https://img.shields.io/badge/Status-Research%20Preview-success?style=for-the-badge&color=brightgreen" alt="Status">
</p>

<h1 align="center">💹 AI Market Sentiment Analyzer (FinBERT + Quant Signals)</h1>
<h3 align="center">Scientific, Data-Driven Insights on News, Sentiment, and Market Behavior</h3>

---

AI Market Sentiment Analyzer (FinBERT + Quant Signals)

Public repo purpose: a reproducible, scientific pipeline that (1) collects financial headlines, (2) scores sentiment using a finance-specific transformer (FinBERT), (3) merges sentiment with market returns, (4) computes statistics and correlations, (5) engineers technical indicators, (6) trains baseline ML models with walk-forward CV, and (7) runs a simple, risk-aware backtest. A Streamlit dashboard presents results.

✨ Key Outcomes (current run highlights)

Sentiment source: 400 FinViz headlines → aggregated to daily FinBERT sentiment per ticker.

Market link: Merged with daily % returns (Yahoo Finance).

Example statistic: for recent data, SPY shows Pearson r ≈ +0.356 between daily sentiment and returns (moderate positive association).

Deliverables: clean CSVs in data/processed/, interactive app at app.py.

All numbers will update as you refresh data. Treat them as estimates whose stability improves with longer samples.

🧭 Repo Structure
ai-market-sentiment-analyzer/
├─ data/
│  ├─ raw/                      # price CSVs + headlines
│  └─ processed/                # clean, “touchable” outputs (CSV)
├─ src/
│  ├─ fetch_stock_data.py       # download OHLCV via yfinance
│  ├─ sentiment_analysis_finbert.py
│  ├─ merge_sentiment_prices.py
│  ├─ analyze_correlation.py
│  ├─ compute_indicators.py
│  ├─ features_signal_engine.py
│  ├─ walkforward_cv.py
│  ├─ backtest_pro.py
│  └─ report_daily.py           # optional PDF daily brief
├─ app.py                       # Streamlit dashboard
├─ run_all.py                   # end-to-end pipeline runner
└─ README.md

🔁 Reproducibility: One-Command Pipeline
python run_all.py


Runs, in order:

fetch_stock_data.py

sentiment_analysis_finbert.py

merge_sentiment_prices.py

compute_indicators.py

features_signal_engine.py

walkforward_cv.py

backtest_pro.py

analyze_correlation.py

report_daily.py (optional)

Outputs land in data/processed/ and can be inspected directly.

📊 Dashboard
streamlit run app.py


Tabs:

Sentiment Trend — FinBERT daily scores by ticker

Sentiment vs Return — scatter + OLS trendline

Correlation Summary — Pearson r by ticker

Indicators — RSI(14), MACD(12/26/9), Bollinger width

Predictions (WF-CV) — out-of-sample R²/MAE across Ridge & XGBoost

Backtest & Signals — simple long-only strategy KPIs + equity curves

📦 Data & Files (touchable)
File	Description
data/raw/finviz_news.csv	Headlines (Ticker, Date, Headline)
data/processed/daily_sentiment_finbert.csv	Daily FinBERT sentiment per ticker/date
data/processed/sentiment_price_merged.csv	Daily sentiment joined with market Close, Return (%)
data/processed/technical_indicators.csv	RSI14, MACD, MACD_signal, Bollinger width
data/processed/features_dataset.csv	ML features (lags/rolls + indicators) + TargetNext
data/processed/model_cv_results.csv	Walk-forward CV metrics (R², MAE) by model & ticker
data/processed/predictions_nextday.csv	Latest predicted next-day return (%) by ticker
data/processed/signals_dataset.csv	Rule-based entries/exits + TargetNext
data/processed/backtest_results.csv	Strategy KPIs (CAGR, Sharpe, MaxDD, WinRate)
output/Daily_Brief_*.pdf	Optional auto-generated PDF recap
🧠 Methods (Scientific Detail)
1) Text → Sentiment (FinBERT)

Model: ProsusAI/finbert (Transformers).

Input: headline text after basic cleaning.

Output: class probabilities p(neg), p(neu), p(pos); score = p(pos) − p(neg) in 
[
−
1
,
+
1
]
[−1,+1].

Aggregation: mean sentiment per (Ticker, Date).

2) Prices & Returns

Source: Yahoo Finance via yfinance.

Daily close; daily return 
𝑟
𝑡
=
100
×
𝑃
𝑡
−
𝑃
𝑡
−
1
𝑃
𝑡
−
1
r
t
	​

=100×
P
t−1
	​

P
t
	​

−P
t−1
	​

	​

 (%).

Inner join on (Ticker, Date) with daily sentiment to prevent look-ahead.

3) Correlation Analysis

Pearson correlation between daily sentiment 
𝑆
𝑡
S
t
	​

 and daily return 
𝑅
𝑡
R
t
	​

:

𝑟
=
∑
(
𝑆
𝑡
−
𝑆
ˉ
)
(
𝑅
𝑡
−
𝑅
ˉ
)
∑
(
𝑆
𝑡
−
𝑆
ˉ
)
2
∑
(
𝑅
𝑡
−
𝑅
ˉ
)
2
r=
∑(S
t
	​

−
S
ˉ
)
2
	​

∑(R
t
	​

−
R
ˉ
)
2
	​

∑(S
t
	​

−
S
ˉ
)(R
t
	​

−
R
ˉ
)
	​


Interpretation: 
𝑟
∈
[
−
1
,
1
]
r∈[−1,1]. Small samples can inflate |r|; stability improves with longer horizons.

4) Technical Indicators

RSI(14) via smoothed gains/losses.

MACD(12,26,9): 
EMA
12
−
EMA
26
EMA
12
	​

−EMA
26
	​

; signal = EMA
9
9
	​

 of MACD.

Bollinger width: 
(
Upper
−
Lower
)
/
MA
20
(Upper−Lower)/MA
20
	​

, proxy for volatility regime.

5) Feature Engineering (for ML)

For each ticker (chronologically sorted):

Lags: Sent_lag{1,2,3,5}, Ret_lag{1,2,3,5}

Rolling summaries: Sent_roll{3,5,10}, Sent_vol{3,5,10}, Ret_roll{5,10,20}, Ret_vol{5,10,20}

Indicators: RSI14, MACD, MACD_signal, BB_width

Target: TargetNext = Return_{t+1} (1-day ahead)

6) Modeling & Validation

Models: Ridge Regression (L2), XGBoost Regressor.

Walk-Forward CV: expanding-window or stepped splits:

For a time series of length 
𝑁
N, minimum train window (e.g., 40 bars), then evaluate on the next fold; repeat.

Reports out-of-sample metrics per model/ticker:

𝑅
2
=
1
−
∑
(
𝑦
−
𝑦
^
)
2
∑
(
𝑦
−
𝑦
ˉ
)
2
R
2
=1−
∑(y−
y
ˉ
	​

)
2
∑(y−
y
^
	​

)
2
	​


MAE 
=
1
𝑛
∑
∣
𝑦
−
𝑦
^
∣
=
n
1
	​

∑∣y−
y
^
	​

∣

Leakage controls: strict chronological splits, no future features, joins on same-day fields only.

7) Signals & Backtest (Toy Example)

Entry (LongSignal): 
Sentiment
𝑡
>
0
∧
MACD
𝑡
>
Signal
𝑡
∧
RSI
𝑡
<
70
Sentiment
t
	​

>0∧MACD
t
	​

>Signal
t
	​

∧RSI
t
	​

<70

Exit (ExitSignal): 
Sentiment
𝑡
<
0
∨
RSI
𝑡
>
70
Sentiment
t
	​

<0∨RSI
t
	​

>70

PnL proxy: apply next-day return when in position; fees/slippage can be added.

KPIs:

CAGR (annualized growth),

Sharpe (annualized; zero risk-free proxy),

Max Drawdown,

Win Rate.

⚠️ This is an educational baseline (not investment advice). For production research, refine execution modeling, slippage, borrow costs, intraday fills, and risk limits.

🧪 Statistical Considerations

Small-sample bias: very short windows can yield extreme 
𝑟
r or spurious model fit. Prefer ≥ 60–90 days.

Non-stationarity: sentiment/return relationships can drift; walk-forward CV addresses this partially.

Multiple comparisons: many tickers/indicators inflate false positives; control with holdout periods or corrections.

Robustness: prefer median & robust scales for outlier-prone returns; consider Spearman rank correlations as sensitivity checks.

Effect sizes: emphasize magnitude & stability over single-sample significance.

⚙️ Quick Start (from a fresh clone)
# 1) Install
pip install -r requirements.txt  # (create one with transformers, torch, yfinance, scikit-learn, xgboost, plotly, streamlit, statsmodels, tqdm, reportlab)

# 2) Run the pipeline
python run_all.py

# 3) Launch the app
streamlit run app.py


Minimal requirements.txt (example):

transformers
torch
tqdm
yfinance
pandas
numpy
scikit-learn
xgboost
plotly
streamlit
statsmodels
reportlab

🧷 Repro Tips

Ensure data/raw/finviz_news.csv exists (or plug in your own source).

If trendline="ols" fails, install statsmodels.

FinBERT’s first run downloads a ~438MB model; subsequent runs are fast (cached).

Use run_all.py for consistency; it writes all “touchable” CSVs.

For determinism, fix random seeds in ML where applicable.

📑 License & Attribution

Code: MIT License (add a LICENSE file if desired).

FinBERT model by ProsusAI (see their license/terms).

Market data via Yahoo Finance (yfinance) for research/demo.

🙏 Acknowledgments

Thanks to open-source communities behind Transformers, scikit-learn, XGBoost, Plotly, and Streamlit.
This project is for research/education; it is not financial advice.

🔚 TL;DR

This repo shows, with scientific transparency, how news sentiment (FinBERT) relates to market moves, how to validate predictive value with walk-forward ML, and how to sanity-check it with a simple backtest—all reproducible and visualized in a single dashboard.


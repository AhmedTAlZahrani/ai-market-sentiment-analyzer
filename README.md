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

## 📘 Overview
The **AI Market Sentiment Analyzer** is a reproducible, research-grade pipeline that:

1. Collects financial headlines  
2. Scores sentiment using **FinBERT** (finance-specific transformer)  
3. Merges sentiment with **market returns**  
4. Computes statistics and correlations  
5. Engineers **technical indicators**  
6. Trains baseline ML models with **walk-forward cross-validation**  
7. Runs a **risk-aware backtest**  
8. Visualizes results via **Streamlit dashboard**

---

## ✨ Key Outcomes
- **Sentiment Source:** 400 FinViz headlines → aggregated daily FinBERT sentiment per ticker  
- **Market Link:** Merged with daily % returns (Yahoo Finance)  
- **Example Insight:** SPY shows *Pearson r ≈ +0.356* between daily sentiment and returns (moderate positive association)  
- **Deliverables:** CSVs in `data/processed/` + interactive dashboard (`app.py`)  

All results dynamically update when new data is fetched. Treat values as evolving estimates that stabilize over time.

---

## 🧭 Repository Structure
ai-market-sentiment-analyzer/
├── data/
│ ├── raw/ # price CSVs + headlines
│ └── processed/ # cleaned, ready-to-use outputs
├── src/
│ ├── fetch_stock_data.py
│ ├── sentiment_analysis_finbert.py
│ ├── merge_sentiment_prices.py
│ ├── analyze_correlation.py
│ ├── compute_indicators.py
│ ├── features_signal_engine.py
│ ├── walkforward_cv.py
│ ├── backtest_pro.py
│ └── report_daily.py
├── app.py # Streamlit dashboard
├── run_all.py # One-command pipeline
└── README.md
python run_all.py
Runs the entire pipeline sequentially:

fetch_stock_data.py

sentiment_analysis_finbert.py

merge_sentiment_prices.py

compute_indicators.py

features_signal_engine.py

walkforward_cv.py

backtest_pro.py

analyze_correlation.py

report_daily.py (optional)

Outputs are stored in data/processed/.
📊 Dashboard
streamlit run app.py
| Tab                     | Description                             |
| ----------------------- | --------------------------------------- |
| **Sentiment Trend**     | FinBERT daily sentiment by ticker       |
| **Sentiment vs Return** | Scatter + OLS trendline                 |
| **Correlation Summary** | Pearson r per ticker                    |
| **Indicators**          | RSI(14), MACD(12/26/9), Bollinger width |
| **Predictions (WF-CV)** | Out-of-sample R² & MAE (Ridge, XGBoost) |
| **Backtest & Signals**  | Strategy KPIs and equity curves         |
| File                                         | Description                         |
| -------------------------------------------- | ----------------------------------- |
| `data/raw/finviz_news.csv`                   | Headlines (Ticker, Date, Headline)  |
| `data/processed/daily_sentiment_finbert.csv` | Daily FinBERT sentiment per ticker  |
| `data/processed/sentiment_price_merged.csv`  | Combined sentiment + market returns |
| `data/processed/technical_indicators.csv`    | RSI14, MACD, Bollinger width        |
| `data/processed/features_dataset.csv`        | ML features + target                |
| `data/processed/model_cv_results.csv`        | Walk-forward CV metrics             |
| `data/processed/predictions_nextday.csv`     | Latest next-day predictions         |
| `data/processed/signals_dataset.csv`         | Entry/exit trading signals          |
| `data/processed/backtest_results.csv`        | Strategy KPIs (CAGR, Sharpe, etc.)  |
| `output/Daily_Brief_*.pdf`                   | Optional auto-generated report      |
🧠 Scientific Methodology
1. Text → Sentiment (FinBERT)

Model: ProsusAI/finbert

Output: probabilities p(neg), p(neu), p(pos)

Score = p(pos) − p(neg) → range [−1, +1]

Aggregated as mean sentiment per (Ticker, Date)

2. Market Data & Returns

Source: Yahoo Finance (yfinance)

Daily return formula:

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


Where:

$P_t$ = closing price on day t

$r_t$ = daily percentage return

Data are joined on (Ticker, Date) with daily sentiment to prevent look-ahead bias.

3. Correlation Analysis

Pearson correlation between sentiment ($S_t$) and return ($R_t$):

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


Where:

$r \in [-1, 1]$

Small samples can inflate $|r|$; stability improves with longer data windows

4. Technical Indicators

RSI(14) via smoothed average gains/losses

MACD(12,26,9): $EMA_{12} - EMA_{26}$; Signal line = $EMA_9(MACD)$

Bollinger Width:

𝐵
𝑊
𝑡
=
𝑈
𝑝
𝑝
𝑒
𝑟
𝑡
−
𝐿
𝑜
𝑤
𝑒
𝑟
𝑡
𝑀
𝐴
20
,
𝑡
BW
t
	​

=
MA
20,t
	​

Upper
t
	​

−Lower
t
	​

	​

5. Feature Engineering (for ML)

For each ticker (chronologically sorted):

Lags: Sent_lag{1,2,3,5}, Ret_lag{1,2,3,5}

Rolling windows: Sent_roll{3,5,10}, Ret_roll{5,10,20}

Volatility measures: Sent_vol{3,5,10}, Ret_vol{5,10,20}

Indicators: RSI14, MACD, MACD_signal, Bollinger width

Target: TargetNext = Return_{t+1} (1-day ahead)

6. Modeling & Validation

Models:

Ridge Regression (L2)

XGBoost Regressor

Walk-forward cross-validation ensures time-ordered splits:
For a dataset of length $N$, training expands incrementally, and validation occurs on subsequent folds.

Metrics

Coefficient of determination:

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


Mean Absolute Error (MAE):

𝑀
𝐴
𝐸
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
MAE=
n
1
	​

∑∣y−
y
^
	​

∣

All evaluation is strictly out-of-sample.

7. Trading Signals & Backtest

Entry (Long):

Sentiment_t > 0 AND MACD_t > Signal_t AND RSI_t < 70


Exit:

Sentiment_t < 0 OR RSI_t > 70


KPIs:

CAGR (annualized growth)

Sharpe Ratio

Max Drawdown

Win Rate

⚠️ This is an educational demonstration, not financial advice.

🧪 Statistical Notes

Small-sample bias: prefer ≥ 60–90 trading days

Non-stationarity: walk-forward CV mitigates drift

Multiple comparisons: control with holdout periods

Robustness: use median/rank correlations for outlier-prone returns

⚙️ Quick Start
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the entire pipeline
python run_all.py

# 3. Launch the Streamlit dashboard
streamlit run app.py

Minimal requirements.txt
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

🧷 Tips for Reproducibility

Ensure data/raw/finviz_news.csv exists

If OLS trendline fails → pip install statsmodels

FinBERT first run downloads a ~438MB model (cached afterward)

Always run via run_all.py for consistent CSV generation

Fix random seeds for deterministic ML experiments

📑 License & Attribution

Code: MIT License

Model: FinBERT by ProsusAI

Market Data: Yahoo Finance via yfinance

For research and educational use only

🙏 Acknowledgments

Special thanks to open-source contributors of:
Transformers, scikit-learn, XGBoost, Plotly, and Streamlit

🌐 Visual Pipeline Overview
<p align="center"> <img src="https://raw.githubusercontent.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/main/assets/pipeline_overview.png" alt="Pipeline Overview" width="90%"> </p> <p align="center"> <b>Data → Sentiment → Merge → Indicators → ML → Backtest → Dashboard</b> </p> <p align="center"> <img src="https://img.shields.io/badge/Data-News%20%2B%20Market-lightblue?style=flat-square"> <img src="https://img.shields.io/badge/NLP-FinBERT-blueviolet?style=flat-square"> <img src="https://img.shields.io/badge/ML-XGBoost%20%7C%20Ridge-orange?style=flat-square"> <img src="https://img.shields.io/badge/App-Streamlit-red?style=flat-square"> <img src="https://img.shields.io/badge/Output-Correlation%20%7C%20Forecast%20%7C%20KPIs-green?style=flat-square"> </p>
🔚 TL;DR

This repository shows — with full scientific transparency — how financial news sentiment (FinBERT) interacts with market returns, validates predictive strength via walk-forward ML, and visualizes everything in one Streamlit dashboard.

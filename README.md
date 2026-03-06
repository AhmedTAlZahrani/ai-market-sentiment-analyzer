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

<h1 align="center">AI Market Sentiment Analyzer</h1>
<h3 align="center">FinBERT + Quant Signals | Scientific, Data-Driven Market Insights</h3>

---

## Overview

The **AI Market Sentiment Analyzer** is a reproducible, research-grade pipeline that connects financial news sentiment to market behavior. It covers the full workflow end-to-end:

1. **Collect** financial headlines from FinViz and price data from Yahoo Finance
2. **Score** sentiment using **FinBERT** (a finance-specific transformer model)
3. **Merge** daily sentiment scores with market returns
4. **Compute** technical indicators (RSI, MACD, Bollinger Bands)
5. **Engineer** ML features with lag, rolling, and volatility signals
6. **Train** baseline models with **walk-forward cross-validation**
7. **Backtest** a rule-based trading strategy with risk-aware KPIs
8. **Visualize** everything in an interactive **Streamlit dashboard**

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/AhmedTAlzahrani/ai-market-sentiment-analyzer.git
cd ai-market-sentiment-analyzer
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python run_all.py
```

This executes every step sequentially:

```
collect_data.py → sentiment_analysis_finbert.py → merge_sentiment_prices.py →
compute_indicators.py → features_signal_engine.py → walkforward_cv.py →
backtest_pro.py → analyze_correlation.py → report_daily.py
```

All outputs are saved to `data/processed/`.

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### Requirements

```
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
requests
beautifulsoup4
```

---

## Key Outcomes

| Metric | Value |
|---|---|
| **Sentiment Source** | ~400 FinViz headlines per run, aggregated daily via FinBERT |
| **Market Data** | Daily closing prices & returns from Yahoo Finance |
| **Example Insight** | SPY shows Pearson *r* ≈ +0.356 (moderate positive association) |
| **Deliverables** | CSVs in `data/processed/` + interactive Streamlit dashboard |

> All results dynamically update when new data is fetched. Treat values as evolving estimates that stabilize over time.

---

## Dashboard

The Streamlit app (`app.py`) provides six interactive tabs:

| Tab | Description |
|---|---|
| **Sentiment Trend** | FinBERT daily sentiment time series by ticker |
| **Sentiment vs Return** | Scatter plot with OLS trendline |
| **Correlation Summary** | Pearson *r* per ticker (bar chart + table) |
| **Indicators** | RSI(14), MACD(12/26/9), Bollinger Band width |
| **Predictions (WF-CV)** | Out-of-sample R² & MAE for Ridge and XGBoost |
| **Backtest & Signals** | Strategy KPIs and equity curves |

---

## Repository Structure

```
ai-market-sentiment-analyzer/
├── data/
│   ├── raw/                          # Price CSVs + headlines
│   └── processed/                    # Pipeline outputs (ready to use)
├── src/
│   ├── collect_data.py               # Scrape FinViz + download prices
│   ├── sentiment_analysis_finbert.py # FinBERT sentiment scoring
│   ├── merge_sentiment_prices.py     # Join sentiment with returns
│   ├── compute_indicators.py         # RSI, MACD, Bollinger Bands
│   ├── features_signal_engine.py     # Feature engineering + signals
│   ├── walkforward_cv.py             # Walk-forward cross-validation
│   ├── backtest_pro.py               # Rule-based backtesting
│   ├── analyze_correlation.py        # Pearson correlation summary
│   ├── model_predict.py              # Ridge prediction (standalone)
│   ├── report_daily.py               # Auto-generate PDF report
│   └── _app_helpers.py               # Shared loaders for dashboard
├── app.py                            # Streamlit dashboard
├── run_all.py                        # One-command full pipeline
└── README.md
```

---

## Output Files

| File | Description |
|---|---|
| `data/raw/finviz_news.csv` | Headlines (Ticker, Date, Headline) |
| `data/processed/daily_sentiment_finbert.csv` | Daily FinBERT sentiment per ticker |
| `data/processed/sentiment_price_merged.csv` | Combined sentiment + market returns |
| `data/processed/technical_indicators.csv` | RSI14, MACD, MACD signal, Bollinger width |
| `data/processed/features_dataset.csv` | ML features + target variable |
| `data/processed/model_cv_results.csv` | Walk-forward CV metrics |
| `data/processed/predictions_nextday.csv` | Latest next-day return predictions |
| `data/processed/signals_dataset.csv` | Entry/exit trading signals |
| `data/processed/backtest_results.csv` | Strategy KPIs (CAGR, Sharpe, MaxDD) |
| `output/Daily_Brief_*.pdf` | Auto-generated daily report (optional) |

---

## Scientific Methodology

### 1. Text to Sentiment (FinBERT)

- **Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- **Output:** Probabilities *p(neg)*, *p(neu)*, *p(pos)*
- **Score:** `p(pos) - p(neg)` &rarr; range **[-1, +1]**
- **Aggregation:** Mean sentiment per (Ticker, Date)

### 2. Market Data & Returns

- **Source:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Daily return:**

$$r_t = 100 \times \frac{P_t - P_{t-1}}{P_{t-1}}$$

where $P_t$ is the closing price on day *t*. Data are joined on (Ticker, Date) to prevent look-ahead bias.

### 3. Correlation Analysis

Pearson correlation between sentiment $S_t$ and return $R_t$:

$$r = \frac{\sum (S_t - \bar{S})(R_t - \bar{R})}{\sqrt{\sum (S_t - \bar{S})^2 \cdot \sum (R_t - \bar{R})^2}}$$

where $r \in [-1, 1]$. Small samples can inflate $|r|$; stability improves with longer data windows.

### 4. Technical Indicators

| Indicator | Formula |
|---|---|
| **RSI(14)** | Smoothed average gains / losses over 14 periods |
| **MACD(12,26,9)** | $EMA_{12} - EMA_{26}$; Signal = $EMA_9(\text{MACD})$ |
| **Bollinger Width** | $(Upper - Lower) / MA_{20}$ |

### 5. Feature Engineering

For each ticker, chronologically sorted:

- **Lags:** `Sent_lag{1,2,3,5}`, `Ret_lag{1,2,3,5}`
- **Rolling means:** `Sent_roll{3,5,10}`, `Ret_roll{5,10,20}`
- **Volatility:** `Sent_vol{3,5,10}`, `Ret_vol{5,10,20}`
- **Indicators:** RSI14, MACD, MACD signal, Bollinger width
- **Target:** `TargetNext = Return(t+1)` (1-day ahead)

### 6. Modeling & Validation

**Models:**
- Ridge Regression (L2 regularization)
- XGBoost Regressor

**Walk-forward cross-validation** ensures strictly time-ordered splits: training expands incrementally, validation occurs on subsequent folds. All evaluation is **out-of-sample**.

**Metrics:**

$$R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2} \qquad \text{MAE} = \frac{1}{n} \sum |y - \hat{y}|$$

### 7. Trading Signals & Backtest

| | Rule |
|---|---|
| **Entry (Long)** | Sentiment > 0 **AND** MACD > Signal **AND** RSI < 70 |
| **Exit** | Sentiment < 0 **OR** RSI > 70 |

**KPIs:** CAGR (annualized growth), Sharpe Ratio, Max Drawdown, Win Rate

> **Disclaimer:** This is an educational demonstration, not financial advice.

---

## Statistical Notes

- **Small-sample bias:** Prefer >= 60-90 trading days for reliable estimates
- **Non-stationarity:** Walk-forward CV mitigates distributional drift
- **Multiple comparisons:** Control with holdout periods
- **Robustness:** Consider median/rank correlations for outlier-prone returns

---

## Tips for Reproducibility

- Ensure `data/raw/finviz_news.csv` exists before running the pipeline
- If OLS trendline fails, install statsmodels: `pip install statsmodels`
- FinBERT downloads a ~438 MB model on first run (cached afterward)
- Always run via `run_all.py` for consistent CSV generation
- Fix random seeds for deterministic ML experiments

---

## Pipeline Overview

<p align="center">
  <img src="https://raw.githubusercontent.com/AhmedTAlzahrani/ai-market-sentiment-analyzer/main/assets/pipeline_overview.png" alt="Pipeline Overview" width="90%">
</p>

<p align="center">
  <b>Data &rarr; Sentiment &rarr; Merge &rarr; Indicators &rarr; ML &rarr; Backtest &rarr; Dashboard</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Data-News%20%2B%20Market-lightblue?style=flat-square">
  <img src="https://img.shields.io/badge/NLP-FinBERT-blueviolet?style=flat-square">
  <img src="https://img.shields.io/badge/ML-XGBoost%20%7C%20Ridge-orange?style=flat-square">
  <img src="https://img.shields.io/badge/App-Streamlit-red?style=flat-square">
  <img src="https://img.shields.io/badge/Output-Correlation%20%7C%20Forecast%20%7C%20KPIs-green?style=flat-square">
</p>

---

## License & Attribution

- **Code:** MIT License
- **Model:** [FinBERT](https://huggingface.co/ProsusAI/finbert) by ProsusAI
- **Market Data:** Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance)
- For **research and educational use** only

---

## Acknowledgments

Special thanks to the open-source communities behind
[Transformers](https://huggingface.co/docs/transformers),
[scikit-learn](https://scikit-learn.org),
[XGBoost](https://xgboost.readthedocs.io),
[Plotly](https://plotly.com),
and [Streamlit](https://streamlit.io).

---

<p align="center">
  <b>TL;DR</b> — This repository shows, with full scientific transparency, how financial news sentiment (FinBERT) interacts with market returns, validates predictive strength via walk-forward ML, and visualizes everything in one Streamlit dashboard.
</p>

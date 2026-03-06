import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────
# Page config & styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Market Sentiment Analyzer",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>"
    "body { background-color: #0E1117; color: white; }"
    ".stPlotlyChart { background-color: #0E1117; }"
    "</style>",
    unsafe_allow_html=True,
)
st.title("💹 AI Market Sentiment Analyzer — Expert Dashboard")


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
@st.cache_data
def load_df(path, parse_dates=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()


def safe_load(path, parse_dates=None):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=parse_dates)
    return pd.DataFrame()


sent = load_df("data/processed/daily_sentiment_finbert.csv", ["Date"])
merged = load_df("data/processed/sentiment_price_merged.csv", ["Date"])
corr = load_df("data/processed/sentiment_correlation_summary.csv")

if sent.empty:
    st.warning("⚠️ Run the pipeline first to generate sentiment data.")
    st.stop()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
tickers = sent["Ticker"].unique().tolist()
ticker = st.sidebar.selectbox("Ticker", tickers, index=0)
date_min, date_max = sent["Date"].min(), sent["Date"].max()
d1, d2 = st.sidebar.date_input("Range", [date_min, date_max])
st.sidebar.markdown("---")
st.sidebar.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Sentiment Trend",
    "💵 Sentiment vs Return",
    "📊 Correlation Summary",
    "🧭 Indicators",
    "🤖 Predictions (WF-CV)",
    "📋 Backtest & Signals",
])

# ── Tab 1: Sentiment Trend ───────────────────
with t1:
    df = sent.query("Ticker == @ticker and Date >= @d1 and Date <= @d2")
    st.subheader(f"FinBERT Daily Sentiment — {ticker}")
    if df.empty:
        st.info("No sentiment for this range.")
    else:
        fig = px.line(
            df, x="Date", y="SentimentScore", markers=True,
            title=f"{ticker} Sentiment Trend",
            color_discrete_sequence=["#00CC96"],
        )
        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Sentiment (-1 → +1)",
            xaxis_title=None,
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(8), use_container_width=True)

# ── Tab 2: Sentiment vs Return ───────────────
with t2:
    st.subheader(f"{ticker}: Sentiment vs Daily Return")
    dfm = merged.query("Ticker == @ticker and Date >= @d1 and Date <= @d2")
    if dfm.empty:
        st.info("No merged data for this range.")
    else:
        fig2 = px.scatter(
            dfm, x="SentimentScore", y="Return",
            color="Return", color_continuous_scale="RdYlGn",
            title=f"{ticker} — Sentiment vs Return (Trendline OLS)",
            trendline="ols",
        )
        fig2.update_layout(
            template="plotly_dark", height=500,
            xaxis_title="Sentiment", yaxis_title="Return (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(
            dfm[["Date", "SentimentScore", "Return"]].tail(10),
            use_container_width=True,
        )

# ── Tab 3: Correlation Summary ───────────────
with t3:
    st.subheader("Overall Sentiment–Return Correlation")
    if corr.empty:
        st.info("Run analyze_correlation.py first.")
    else:
        corr_sorted = corr.sort_values("Correlation", ascending=False)
        fig3 = px.bar(
            corr_sorted, x="Ticker", y="Correlation",
            color="Correlation",
            color_continuous_scale=["#EF553B", "#636EFA"],
            title="Ticker Correlation Strength",
        )
        fig3.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(corr_sorted, use_container_width=True)

# ── Tab 4: Technical Indicators ──────────────
with t4:
    st.subheader("📈 Technical Indicators Overview")
    tech = safe_load("data/processed/technical_indicators.csv", ["Date"])
    if tech.empty:
        st.info("Run compute_indicators.py first.")
    else:
        df_t = tech[tech["Ticker"] == ticker]
        fig4 = px.line(
            df_t, x="Date", y=["RSI14", "MACD", "BB_width"],
            title=f"{ticker} — Technical Indicators",
        )
        fig4.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(df_t.tail(10), use_container_width=True)

# ── Tab 5: Walk-Forward Predictions ──────────
with t5:
    st.subheader("🤖 Model Performance (Ridge & XGBoost)")
    cv = safe_load("data/processed/model_cv_results.csv")
    preds = safe_load("data/processed/predictions_nextday.csv")
    if cv.empty and preds.empty:
        st.info("Run walkforward_cv.py first.")
    else:
        if not cv.empty:
            st.dataframe(cv, use_container_width=True)
        if not preds.empty:
            st.markdown("**Latest predicted next-day returns**")
            st.dataframe(preds, use_container_width=True)

# ── Tab 6: Backtest ──────────────────────────
with t6:
    st.subheader("💵 Rule-Based Backtest Results")
    bt = safe_load("data/processed/backtest_results.csv")
    eq = safe_load("data/processed/equity_curve.csv")
    if bt.empty:
        st.info("Run backtest_pro.py first.")
    else:
        st.dataframe(bt, use_container_width=True)
    if not eq.empty:
        for col in [c for c in eq.columns if c.lower() != "unnamed: 0"]:
            fig6 = px.line(eq, x=eq.columns[0], y=col, title=f"Equity Curve — {col}")
            fig6.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig6, use_container_width=True)

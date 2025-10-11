# ============== ADVANCED APP (Expert) ==============
# (Replaces previous app.py, includes Indicators + Predictions/Backtest tabs)
import os, io, time, subprocess
from datetime import datetime, timedelta
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from src._app_helpers import load_indicators_df, load_predictions_df

st.set_page_config(page_title="AI Market Sentiment Analyzer", page_icon="??", layout="wide", initial_sidebar_state="expanded")

# ---- Styling (dark, elegant) ----
st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 800px at 20% -10%, #0b132b 0%, #0b132b 40%, #1c2541 95%) !important; color: #E6ECF1; }
h1,h2,h3,h4 { color: #E6ECF1; }
.card { border-radius: 22px; padding: 14px 16px; background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 12px 30px rgba(0,0,0,0.25); }
.kpi { border-radius: 16px; padding: 12px 14px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ---- Load core data (FinBERT preferred) ----
@st.cache_data
def load_sentiment():
    for p in ["data/processed/daily_sentiment_finbert.csv", "data/processed/daily_sentiment.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["Date"]).dropna(subset=["Ticker","Date","SentimentScore"])
            return df.sort_values(["Ticker","Date"])
    return pd.DataFrame(columns=["Ticker","Date","SentimentScore"])

@st.cache_data
def load_merged():
    p = "data/processed/sentiment_price_merged.csv"
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["Date"]).dropna(subset=["Ticker","Date","SentimentScore","Return"])
        return df.sort_values(["Ticker","Date"])
    return pd.DataFrame(columns=["Date","Ticker","SentimentScore","Close","Return"])

@st.cache_data
def load_corr():
    p = "data/processed/sentiment_correlation_summary.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Ticker","Correlation"])

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")

sentiment = load_sentiment()

# ---- Sector mapping (customize) ----
SECTOR_MAP = {
    "SPY":"Index","AAPL":"Tech","NVDA":"Tech","TSLA":"Auto",
}
def sector_frame(sent_df):
    df = sent_df.copy()
    df["Sector"] = df["Ticker"].map(SECTOR_MAP).fillna("Other")
    grp = df.groupby(["Date","Sector"])["SentimentScore"].mean().reset_index()
    return grp

merged = load_merged()
corr = load_corr()
tech = load_indicators_df()
preds = load_predictions_df()

if sentiment.empty:
    st.error("? No sentiment data found. Run the pipeline first.")
    st.stop()

# ---- Sidebar ----
st.sidebar.title("?? Controls")
all_tickers = sorted(sentiment["Ticker"].unique().tolist())
default = [t for t in ["SPY","AAPL","NVDA","TSLA"] if t in all_tickers] or all_tickers[:3]
sel = st.sidebar.multiselect("Tickers", all_tickers, default=default)

date_min, date_max = sentiment["Date"].min(), sentiment["Date"].max()
d1, d2 = st.sidebar.date_input("Date range", value=(date_min.date(), date_max.date()))
smooth = st.sidebar.slider("Sentiment smoothing (days)", 1, 14, 3)
rm_out = st.sidebar.checkbox("Remove return outliers (±3s)", value=True)
overlay_price = st.sidebar.checkbox("Overlay price", value=True)
st.sidebar.markdown("---")
if st.sidebar.button("?? Run Pipeline Now", use_container_width=True):
    with st.spinner("Running run_all.py ..."):
        res = subprocess.run(["python","run_all.py"], capture_output=True, text=True)
        if res.returncode == 0:
            st.success("Pipeline done. Reloading data...")
            load_sentiment.clear(); load_merged.clear(); load_corr.clear()
            sentiment = load_sentiment()

# ---- Sector mapping (customize) ----
SECTOR_MAP = {
    "SPY":"Index","AAPL":"Tech","NVDA":"Tech","TSLA":"Auto",
}
def sector_frame(sent_df):
    df = sent_df.copy()
    df["Sector"] = df["Ticker"].map(SECTOR_MAP).fillna("Other")
    grp = df.groupby(["Date","Sector"])["SentimentScore"].mean().reset_index()
    return grp
 merged = load_merged(); corr = load_corr()
            tech = load_indicators_df(); preds = load_predictions_df()
        else:
            st.error("Pipeline failed. See output below:"); st.code(res.stdout + "\n" + res.stderr)

# ---- Filters ----
sent_f = sentiment[(sentiment["Ticker"].isin(sel)) & (sentiment["Date"].between(pd.to_datetime(d1), pd.to_datetime(d2)))].copy()
merge_f = merged[(merged["Ticker"].isin(sel)) & (merged["Date"].between(pd.to_datetime(d1), pd.to_datetime(d2)))].copy()
tech_f = tech[(tech["Ticker"].isin(sel)) & (tech["Date"].between(pd.to_datetime(d1), pd.to_datetime(d2)))].copy()

if smooth>1 and not sent_f.empty:
    sent_f["SentSmooth"] = sent_f.groupby("Ticker")["SentimentScore"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
else:
    sent_f["SentSmooth"] = sent_f["SentimentScore"]

if rm_out and not merge_f.empty:
    merge_f = merge_f.groupby("Ticker", group_keys=False).apply(lambda g: g[(g["Return"]-g["Return"].mean()).abs() <= 3*g["Return"].std(ddof=0)] if g["Return"].std(ddof=0)>0 else g)

# ---- Header + KPIs ----
st.markdown("## ?? AI Market Sentiment Analyzer — Expert")
c1,c2,c3,c4 = st.columns(4)
def kpi(col, title, val):
    with col:
        st.markdown(f"<div class='kpi'><small>{title}</small><h3>{val}</h3></div>", unsafe_allow_html=True)

latest_date = sent_f["Date"].max() if not sent_f.empty else None
latest_avg_sent = sent_f[sent_f["Date"]==latest_date]["SentSmooth"].mean() if latest_date is not None else np.nan
latest_avg_ret = np.nan
if not merge_f.empty:
    last_d = merge_f["Date"].max()
    latest_avg_ret = merge_f[merge_f["Date"]==last_d]["Return"].mean()
overall_corr = np.nan
if not merge_f.empty and merge_f["SentimentScore"].std(ddof=0)>0 and merge_f["Return"].std(ddof=0)>0:
    overall_corr = merge_f["SentimentScore"].corr(merge_f["Return"])
kpi(c1,"Latest Date", latest_date.strftime("%Y-%m-%d") if latest_date else "—")
kpi(c2,"Avg Sentiment (latest)", f"{latest_avg_sent:0.3f}" if pd.notna(latest_avg_sent) else "—")
kpi(c3,"Avg Return (latest)", f"{latest_avg_ret:0.2f}%" if pd.notna(latest_avg_ret) else "—")
kpi(c4,"Corr (Sent ? Ret)", f"{overall_corr:0.3f}" if pd.notna(overall_corr) else "—")

# ---- Tabs ----
t1, t2, t3, t4, t5, t6 = st.tabs([
    "?? Sentiment & Price", "?? Sentiment vs Return", "?? Correlations",
    "?? Indicators", "?? Predictions & Backtest", "?? Downloads"
])

# TAB 1
with t1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if sent_f.empty:
        st.info("No sentiment in this range.")
    else:
        for t in sel:
            s = sent_f[sent_f["Ticker"]==t]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["Date"], y=s["SentSmooth"], mode="lines+markers", name="Sentiment (smooth)"))
            fig.update_layout(template="plotly_dark", height=350, title=f"{t} — Sentiment")
            if overlay_price and not merge_f.empty:
                pm = merge_f[merge_f["Ticker"]==t]
                if not pm.empty:
                    norm = (pm["Close"]/pm["Close"].iloc[0])-1.0
                    fig.add_trace(go.Scatter(x=pm["Date"], y=norm, name="Price (norm.)", yaxis="y2"))
                    fig.update_layout(yaxis2=dict(title="Price (norm.)", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 2
with t2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if merge_f.empty:
        st.info("No merged data in range.")
    else:
        for t in sel:
            m = merge_f[merge_f["Ticker"]==t]
            if m.empty: continue
            sc = px.scatter(m, x="SentimentScore", y="Return", color=m["Date"].dt.strftime("%Y-%m-%d"), trendline="ols", title=f"{t} — Sentiment vs Return")
            sc.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(sc, use_container_width=True)
            corr_t = m["SentimentScore"].corr(m["Return"]) if m["SentimentScore"].std(ddof=0)>0 and m["Return"].std(ddof=0)>0 else np.nan
            st.caption(f"{t} Pearson corr: **{corr_t:.3f}**" if pd.notna(corr_t) else f"{t} Pearson corr: —")
            st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 3
with t3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if corr.empty:
        st.info("No correlation summary file yet.")
    else:
        st.dataframe(corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 4 — Indicators
with t4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Technical indicators (RSI, MACD, Bollinger width)")
    if tech_f.empty:
        st.info("No indicators. Run: python src/compute_indicators.py")
    else:
        for t in sel:
            ti = tech_f[tech_f["Ticker"]==t].sort_values("Date")
            if ti.empty: continue
            # Price + MACD
            figp = go.Figure()
            figp.add_trace(go.Scatter(x=ti["Date"], y=ti["Close"], name="Close"))
            figp.update_layout(template="plotly_dark", height=320, title=f"{t} — Close & MACD")
            figp.add_trace(go.Scatter(x=ti["Date"], y=ti["MACD"], name="MACD", yaxis="y2"))
            figp.add_trace(go.Scatter(x=ti["Date"], y=ti["MACD_signal"], name="Signal", yaxis="y2"))
            figp.update_layout(yaxis2=dict(title="MACD", overlaying="y", side="right"))
            st.plotly_chart(figp, use_container_width=True)

            # RSI
            figr = go.Figure()
            figr.add_trace(go.Scatter(x=ti["Date"], y=ti["RSI14"], name="RSI14"))
            figr.add_hline(y=70, line_dash="dot", line_color="red")
            figr.add_hline(y=30, line_dash="dot", line_color="green")
            figr.update_layout(template="plotly_dark", height=250, title=f"{t} — RSI14 (70/30)")
            st.plotly_chart(figr, use_container_width=True)

            # Bollinger width
            figbw = go.Figure()
            figbw.add_trace(go.Scatter(x=ti["Date"], y=ti["BB_width"], name="BB width"))
            figbw.update_layout(template="plotly_dark", height=220, title=f"{t} — Bollinger Band Width")
            st.plotly_chart(figbw, use_container_width=True)
            st.markdown("---")
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 5 — Predictions & Backtest
with t5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Ridge baseline — predict next-day return")
    if preds.empty:
        st.info("No predictions yet. Run: python src/model_predict.py")
    else:
        st.dataframe(preds, use_container_width=True)
        st.caption("R2/MAE are out-of-sample on the last ~20% of each ticker's data.")
        st.markdown("---")

        st.subheader("Simple threshold backtest (toy example)")
        colA, colB = st.columns(2)
        th_sent = colA.slider("Buy when sentiment >", -1.0, 1.0, 0.1, 0.05)
        rsi_cap = colB.slider("And RSI below", 0, 100, 70, 1)
        if not merge_f.empty and not tech_f.empty:
            joined = merge_f.merge(tech_f[["Date","Ticker","RSI14"]], on=["Date","Ticker"], how="left")
            # signal: buy at close if (sent > th) & (RSI<rsi_cap), hold one day
            joined["Signal"] = ((joined["SentimentScore"] > th_sent) & (joined["RSI14"] < rsi_cap)).astype(int)
            # strategy return is next day market return if we bought today
            joined["NextRet"] = joined.groupby("Ticker")["Return"].shift(-1)
            strat = joined[joined["Signal"]==1]["NextRet"].mean()
            bench = joined["Return"].mean()
            st.write(f"**Avg next-day return when signal=1:** {str(round(float(strat),3)) if pd.notna(strat) else '—'}%")
            st.write(f"**Benchmark avg daily return:** {str(round(float(bench),3)) if pd.notna(bench) else '—'}%")
        else:
            st.info("Need merged + indicators to simulate.")
    st.markdown("</div>", unsafe_allow_html=True)

# TAB 6 — Downloads
with t6:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Export datasets")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("**Sentiment (daily)**")
        st.download_button("Download", data=df_to_csv_bytes(sentiment), file_name="daily_sentiment.csv")
    with c2:
        st.markdown("**Merged (sent+price)**")
        st.download_button("Download", data=df_to_csv_bytes(merged), file_name="sentiment_price_merged.csv")
    with c3:
        st.markdown("**Indicators**")
        st.download_button("Download", data=df_to_csv_bytes(tech), file_name="technical_indicators.csv")
    with c4:
        st.markdown("**Predictions**")
        st.download_button("Download", data=df_to_csv_bytes(preds), file_name="predictions_nextday.csv")
    st.markdown("</div>", unsafe_allow_html=True)


# Extra tabs
t7, t8, t9 = st.tabs(["?? Signals & Backtest Pro", "?? Models (WF-CV)", "?? Sector Heatmap"])

with t7:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Rule-based signals & Pro backtest metrics")
    sig = pd.read_csv("data/processed/signals_dataset.csv", parse_dates=["Date"]) if os.path.exists("data/processed/signals_dataset.csv") else pd.DataFrame()
    bt  = pd.read_csv("data/processed/backtest_results.csv") if os.path.exists("data/processed/backtest_results.csv") else pd.DataFrame()
    if sig.empty or bt.empty:
        st.info("Run: features_signal_engine.py and backtest_pro.py")
    else:
        st.dataframe(bt, use_container_width=True)
        # Equity curves
        if os.path.exists("data/processed/equity_curve.csv"):
            eq = pd.read_csv("data/processed/equity_curve.csv")
            for t in [x for x in eq.columns if x!="Unnamed: 0"]:
                fig = px.line(eq, x=eq.columns[0], y=t, title=f"Equity Curve ? {t}")
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t8:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Walk-Forward CV (Ridge & XGBoost)")
    cv = pd.read_csv("data/processed/model_cv_results.csv") if os.path.exists("data/processed/model_cv_results.csv") else pd.DataFrame()
    preds = pd.read_csv("data/processed/predictions_nextday.csv") if os.path.exists("data/processed/predictions_nextday.csv") else pd.DataFrame()
    if cv.empty:
        st.info("Run: walkforward_cv.py")
    else:
        st.dataframe(cv, use_container_width=True)
    if not preds.empty:
        st.markdown("**Latest predicted next-day returns**")
        st.dataframe(preds, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with t9:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Sector sentiment heatmap")
    sf = sector_frame(sentiment)
    sf = sf[(sf["Date"]>=pd.to_datetime(d1)) & (sf["Date"]<=pd.to_datetime(d2))]
    if sf.empty:
        st.info("No sector data for range.")
    else:
        pivot = sf.pivot_table(index="Date", columns="Sector", values="SentimentScore").dropna(how="all")
        if pivot.shape[1]>=2:
            hm = px.imshow(pivot.corr(), text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Sector sentiment correlation")
            hm.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(hm, use_container_width=True)
        st.markdown("**Sector averages over time**")
        for sct in pivot.columns:
            fig = px.line(sf[sf["Sector"]==sct], x="Date", y="SentimentScore", title=f"{sct} ? average sentiment")
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

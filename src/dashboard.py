# src/dashboard.py
import os
import pandas as pd
from datetime import datetime
import dash
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from main import build_latest_csv

DATA_PATH = "data/processed/latest.csv"
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def load_data():
    if not os.path.exists(DATA_PATH):
        build_latest_csv()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

app.layout = dbc.Container([
    html.H2("AI Market Sentiment Analyzer Dashboard", className="mt-3 mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Ticker"),
            dcc.Dropdown(["AAPL","MSFT","SPY","QQQ"], "AAPL", id="ticker"),
        ], width=3),
        dbc.Col([
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id="date_range",
                start_date=df["date"].min().date(),
                end_date=df["date"].max().date()
            ),
        ], width=5),
        dbc.Col([
            html.Label(" "),
            dbc.Button("Update Data", id="update", color="primary", className="d-block"),
            html.Small(id="status", className="text-muted"),
        ], width=2),
    ], className="mb-4"),
    dcc.Graph(id="price_chart", style={"height": "600px"})
], fluid=True)

@app.callback(
    Output("price_chart", "figure"),
    Output("status", "children"),
    Input("update", "n_clicks"),
    Input("ticker", "value"),
    Input("date_range", "start_date"),
    Input("date_range", "end_date"),
)
def update_chart(n_clicks, ticker, start, end):
    df = load_data()
    if ticker not in df["ticker"].unique():
        df = load_data()
    df = df[df["ticker"] == ticker]
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    fig = px.line(df, x="date", y="close", title=f"{ticker} Close Price")
    return fig, f"Showing {len(df)} rows for {ticker}"

if __name__ == "__main__":
    app.run(debug=True)
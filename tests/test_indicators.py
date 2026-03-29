import pytest
import pandas as pd
import numpy as np

from src.compute_indicators import rsi, macd, bollinger_bands
from src.sentiment_analysis import clean_headlines, analyze_sentiment


# --------------- helpers ---------------

def _price_series(values):
    return pd.Series(values, dtype=float)


def _rising(n=60):
    return _price_series(np.linspace(50, 150, n))


def _falling(n=60):
    return _price_series(np.linspace(150, 50, n))


def _flat(n=60, price=100.0):
    return _price_series([price] * n)


# --------------- RSI ---------------

class TestRSI:

    def test_output_length_matches_input(self):
        s = _rising()
        assert len(rsi(s)) == len(s)

    @pytest.mark.parametrize("window", [7, 14, 21])
    def test_rsi_bounded_0_100(self, window):
        s = pd.Series(np.random.default_rng(42).normal(100, 5, 120).cumsum())
        result = rsi(s, window=window).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_rising_prices_high_rsi(self):
        result = rsi(_rising(80), window=14)
        # last value for a monotonically rising series should be well above 50
        assert result.iloc[-1] > 70

    def test_falling_prices_low_rsi(self):
        result = rsi(_falling(80), window=14)
        assert result.iloc[-1] < 30

    @pytest.mark.parametrize("window", [5, 14, 28])
    def test_flat_prices_rsi_nan(self, window):
        # constant prices -> zero delta -> 0/0 -> NaN after initial
        result = rsi(_flat(40), window=window)
        # first value is NaN (no diff), rest are NaN due to 0/0
        assert result.isna().sum() > 0


# --------------- MACD ---------------

class TestMACD:

    def test_returns_two_series(self):
        m, s = macd(_rising())
        assert isinstance(m, pd.Series)
        assert isinstance(s, pd.Series)

    @pytest.mark.parametrize("fast,slow,signal", [
        (12, 26, 9),
        (8, 17, 9),
        (5, 35, 5),
    ])
    def test_lengths_match(self, fast, slow, signal):
        prices = _rising(80)
        m, s = macd(prices, fast=fast, slow=slow, signal=signal)
        assert len(m) == len(prices)
        assert len(s) == len(prices)

    def test_rising_macd_positive(self):
        m, _ = macd(_rising(80))
        # tail of MACD for a rising series should be positive
        assert m.iloc[-1] > 0

    def test_falling_macd_negative(self):
        m, _ = macd(_falling(80))
        assert m.iloc[-1] < 0


# --------------- Bollinger Bands ---------------

class TestBollingerBands:

    def test_returns_four_series(self):
        result = bollinger_bands(_rising())
        assert len(result) == 4

    @pytest.mark.parametrize("window,num_std", [
        (10, 1),
        (20, 2),
        (30, 3),
    ])
    def test_upper_above_lower(self, window, num_std):
        prices = pd.Series(np.random.default_rng(7).normal(100, 10, 80).cumsum())
        ma, upper, lower, _ = bollinger_bands(prices, window=window, num_std=num_std)
        valid = ~upper.isna()
        assert (upper[valid] >= lower[valid]).all()

    def test_ma_between_bands(self):
        prices = pd.Series(np.random.default_rng(0).normal(0, 1, 100).cumsum() + 200)
        ma, upper, lower, _ = bollinger_bands(prices)
        valid = ~ma.isna()
        assert (upper[valid] >= ma[valid]).all()
        assert (ma[valid] >= lower[valid]).all()

    @pytest.mark.parametrize("num_std", [1, 2, 3])
    def test_width_scales_with_num_std(self, num_std):
        prices = pd.Series(np.random.default_rng(1).normal(100, 5, 60).cumsum())
        _, _, _, w1 = bollinger_bands(prices, num_std=1)
        _, _, _, w2 = bollinger_bands(prices, num_std=num_std)
        if num_std > 1:
            assert (w2.dropna() >= w1.dropna()).all()


# --------------- Sentiment output format ---------------

class TestSentimentOutput:

    @pytest.fixture()
    def sample_df(self):
        return pd.DataFrame({
            "Headline": [
                "Company posts record profits!",
                "Market crashes amid fears",
                "Stocks remain flat today",
            ]
        })

    def test_clean_headlines_lowercase(self, sample_df):
        result = clean_headlines(sample_df.copy())
        for h in result["Headline"]:
            assert h == h.lower()

    def test_clean_headlines_no_special_chars(self, sample_df):
        df = sample_df.copy()
        df.loc[len(df)] = {"Headline": "Price $100 up +5%!"}
        result = clean_headlines(df)
        for h in result["Headline"]:
            assert all(c.isalnum() or c == " " for c in h)

    def test_analyze_sentiment_columns(self, sample_df):
        df = clean_headlines(sample_df.copy())
        result = analyze_sentiment(df)
        assert "SentimentScore" in result.columns
        assert "SentimentLabel" in result.columns

    @pytest.mark.parametrize("headline,expected_label", [
        ("incredible amazing profit growth", "positive"),
        ("terrible crash disaster losses", "negative"),
    ])
    def test_sentiment_direction(self, headline, expected_label):
        df = pd.DataFrame({"Headline": [headline]})
        df = clean_headlines(df)
        df = analyze_sentiment(df)
        assert df["SentimentLabel"].iloc[0] == expected_label

    def test_sentiment_score_bounded(self, sample_df):
        df = clean_headlines(sample_df.copy())
        df = analyze_sentiment(df)
        scores = df["SentimentScore"]
        assert (scores >= -1).all() and (scores <= 1).all()

    def test_sentiment_label_values(self, sample_df):
        df = clean_headlines(sample_df.copy())
        df = analyze_sentiment(df)
        assert set(df["SentimentLabel"]).issubset({"positive", "negative", "neutral"})

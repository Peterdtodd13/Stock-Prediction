import datetime
import os
import sys

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import yfinance as yf


def extract_features(
    return_period: int = 5,
    lookback_days: int = 365,
    stk_tickers=None,
    ccy_tickers=None,
    idx_tickers=None,
) -> pd.DataFrame:
    """Build a features dataframe for the model.

    Returns ONLY feature columns (no target), aligned to the tickers/series used
    by the deployed model.
    """

    if stk_tickers is None:
        stk_tickers = ["AMZN", "NVDA", "TSLA"]
    if ccy_tickers is None:
        ccy_tickers = ["DEXCAUS", "DEXMXUS"]
    if idx_tickers is None:
        idx_tickers = ["SP500", "DJIA", "NASDAQCOM"]

    start_date = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    # Download equities (yfinance returns a MultiIndex columns DF when multiple tickers are used)
    stk_data = yf.download(stk_tickers, start=start_date, end=end_date, auto_adjust=False)

    # FRED series (already flat columns)
    ccy_data = web.DataReader(ccy_tickers, "fred", start=start_date, end=end_date)
    idx_data = web.DataReader(idx_tickers, "fred", start=start_date, end=end_date)

    # Equity log-return features
    # Use Adj Close when available; yfinance returns columns like stk_data['Adj Close'][ticker]
    adj_close = stk_data["Adj Close"]
    x1 = np.log(adj_close[["NVDA", "TSLA"]]).diff(return_period)

    # Macro / index log-return features
    x2 = np.log(ccy_data).diff(return_period)
    x3 = np.log(idx_data).diff(return_period)

    x = pd.concat([x1, x2, x3], axis=1)

    # Sample every return_period rows, drop NA, sort, and reset index.
    features = x.dropna().iloc[::return_period, :].sort_index()
    features.index.name = "Date"
    features = features.reset_index(drop=True)

    return features


def get_bitcoin_historical_prices(days: int = 60) -> pd.DataFrame:
    base_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    }

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["Timestamp", "Close Price (USD)"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.normalize()
    df = df[["Date", "Close Price (USD)"]].set_index("Date")

    return df

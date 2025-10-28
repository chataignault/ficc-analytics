import os
import datetime as dt
import yfinance as yf
import pandas as pd
import logging

from .constant import Ticker
from .processing import process_ticker_hist

logger = logging.getLogger(__name__)


def read_ticker_(
    ticker: Ticker, start: dt.datetime, end: dt.datetime, save: bool = False
):
    """
    yfinance API proxy for historical data
    Read if file already exists else fetch
    """
    file_name = "_".join(
        [ticker, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ".parquet"]
    )
    file_path = os.path.join(os.getcwd(), "raw_data", file_name)
    if os.path.exists(file_path):
        logger.info(f"Reading file {file_name}...")
        df = pd.read_parquet(file_name)
    else:
        logger.info(f"Fetch {ticker} data...")
        t = yf.Ticker(ticker)
        try:
            print(t.get_info())
        except Exception as e:
            print("Ticker info error")
        df = t.history(start=start, end=end)
        df = process_ticker_hist(df)
        if save:
            df.to_parquet(file_path)
    return df

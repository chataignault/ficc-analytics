import json
from typing import List
import datetime as dt
import pandas as pd
from fastapi import APIRouter

from .constant import Ticker
from .yf_helpers import read_ticker_


yf_router = APIRouter(prefix="/yf", tags=["Yahoo Finance proxy"])


@yf_router.get("/tickers_info")
async def read_tickers_info():
    return json.dumps(list(Ticker))


@yf_router.get("/ticker/{ticker}")
async def read_ticker(
    ticker: Ticker,
    start: dt.date = dt.date.today(),
    end: dt.date = dt.date.today(),
    save: bool = False,
):
    df = read_ticker_(ticker, start, end, save)
    return df.to_json()


@yf_router.post("/tickers/")
async def read_tickers(
    tickers: List[Ticker] = list(Ticker),
    start: dt.date = dt.date.today(),
    end: dt.date = dt.date.today(),
    save: bool = False,
):
    dfx = []
    for ticker in tickers:
        df_t = read_ticker_(ticker, start, end)
        df_t.columns = [" ".join([ticker.value]) for c in df_t.columns]
        dfx.append(df_t)
    df = pd.concat(dfx, axis=1)
    return df.to_json()


@yf_router.get("/history/{ticker}")
async def read_ticker(
    ticker: str,
    start: dt.date = dt.date.today(),
    end: dt.date = dt.date.today(),
    save: bool = False,
):
    df = pd.DataFrame()

    try:
        df = read_ticker_(ticker, start, end, save)
    except BaseException as e:
        # TODO need proper logging
        pass

    return df.to_json()

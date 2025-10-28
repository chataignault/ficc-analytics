import pandas_datareader as pdr
from fastapi import APIRouter


pdr_router = APIRouter(prefix="/pdr", tags=["Pandas Data-Reader proxy"])


def get_usd_index_fred():
    """
    Get trade-weighted USD index from FRED
    """
    try:
        df = pdr.get_data_fred("DTWEXBGS", start="1985-01-01")
        df.columns = ["USD_Index"]
        df.index.name = "date"
        return df.rename(columns={"USD_Index": "USDX"})
    except Exception as e:
        print(f"FRED error: {e}")
        return None


@pdr_router.get("/usdx")
async def read_usdx(
    save: bool = False,
):
    df = get_usd_index_fred()
    if save:
        df.to_parquet("usdx_history.parquet")
    return df.reset_index().to_json()

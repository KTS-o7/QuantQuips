from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from quantquips.config import Settings, get_settings


def market_for_ticker(ticker: str) -> str:
    return "IND" if ticker.endswith((".NS", ".BO")) else "US"


def ticker_cache_path(ticker: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.company_data_dir / market_for_ticker(ticker) / f"{ticker}.csv"


def load_ticker_lists(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    frames: list[pd.DataFrame] = []
    for path in sorted(settings.ticker_list_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["Source"] = path.stem
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["Ticker", "Name", "Exchange", "Category Name", "Country", "Source"])
    return pd.concat(frames, ignore_index=True)


def load_cached_history(ticker: str, settings: Settings | None = None) -> pd.DataFrame:
    path = ticker_cache_path(ticker, settings)
    if not path.exists():
        return pd.DataFrame()
    return normalize_price_index(pd.read_csv(path, index_col=0, parse_dates=True))


def fetch_history(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return normalize_price_index(data)


def normalize_price_index(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.index = pd.to_datetime(normalized.index, errors="coerce", utc=True).tz_convert(None)
    normalized = normalized[~normalized.index.isna()]
    return normalized.sort_index()


def get_history(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    refresh: bool = False,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or get_settings()
    cached = load_cached_history(ticker, settings)
    if not refresh and not cached.empty:
        filtered = cached.loc[(cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))]
        if not filtered.empty:
            return filtered

    data = fetch_history(ticker, start=start, end=end, interval=interval)
    if not data.empty and interval == "1d":
        path = ticker_cache_path(ticker, settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path)
    return data

# nicegui_app.py
from __future__ import annotations

import asyncio
from functools import partial

import pandas as pd
import plotly.express as px
import yfinance as yf

from nicegui import app, ui


NAV_ITEMS = [
    ("Home", "/"),
    ("Backtesting", "/backtest"),
    ("Genetic Algorithm", "/ga"),
    ("LLM", "/llm"),
    ("About", "/about"),
]


def layout() -> None:
    """Render shared header and left drawer. Call inside every @ui.page."""
    with ui.header().classes("items-center justify-between bg-grey-9 text-white q-px-md"):
        ui.label("QuantQuips").classes("text-h6 text-bold")
        ui.button(icon="dark_mode", on_click=lambda: ui.dark_mode().toggle()).props("flat round")

    with ui.left_drawer(top_corner=True).classes("bg-grey-10 text-white q-pa-md"):
        ui.label("Navigation").classes("text-subtitle2 text-grey-5 q-mb-sm")
        for label, path in NAV_ITEMS:
            ui.link(label, path).classes("text-white block q-py-xs")


def _download_period(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


@ui.page("/")
async def page_home() -> None:
    layout()
    ui.label("Market Snapshot").classes("text-h5 q-pa-md")

    symbols = {"Nifty 50": "^NSEI", "Sensex": "^BSESN"}
    with ui.row().classes("w-full q-px-md q-gutter-md"):
        for label, ticker in symbols.items():
            with ui.card().classes("flex-1"):
                title_label = ui.label(label).classes("text-subtitle1 text-bold")
                metric_label = ui.label("Loading...").classes("text-h6")
                chart_slot = ui.column().classes("w-full")

                async def _load(lbl=label, tkr=ticker, ml=metric_label, cs=chart_slot) -> None:
                    loop = asyncio.get_event_loop()
                    try:
                        data = await loop.run_in_executor(
                            None, partial(_download_period, tkr, "1d", "1m")
                        )
                    except Exception as exc:
                        ml.set_text(f"Error: {exc}")
                        return
                    if data.empty or "Close" not in data:
                        ml.set_text("No live data available")
                        return
                    first = float(data["Close"].iloc[0])
                    last = float(data["Close"].iloc[-1])
                    pct = ((last - first) / first * 100) if first else 0.0
                    ml.set_text(f"{last:,.2f}  ({pct:+.2f}%)")
                    fig = px.line(data, x=data.index, y="Close", title=f"{lbl} intraday")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                    with cs:
                        ui.plotly(fig).classes("w-full")

                ui.timer(0.1, _load, once=True)


@ui.page("/backtest")
def page_backtest() -> None:
    layout()
    ui.label("Backtesting — coming in Task 4").classes("text-h6 q-pa-lg")


@ui.page("/ga")
def page_ga() -> None:
    layout()
    ui.label("Genetic Algorithm — coming in Task 5").classes("text-h6 q-pa-lg")


@ui.page("/llm")
def page_llm() -> None:
    layout()
    ui.label("LLM — coming in Task 6").classes("text-h6 q-pa-lg")


@ui.page("/about")
def page_about() -> None:
    layout()
    ui.label("About — coming in Task 7").classes("text-h6 q-pa-lg")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="QuantQuips", dark=True, port=8080, reload=False)

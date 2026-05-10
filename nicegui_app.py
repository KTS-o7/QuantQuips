# nicegui_app.py
from __future__ import annotations

import asyncio
from functools import partial

import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import date

from quantquips.backtest_service import BacktestResult, run_backtest
from quantquips.config import get_settings
from quantquips.data_service import get_history, load_ticker_lists
from quantquips.strategies import STRATEGIES

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


def _ticker_options() -> list[str]:
    try:
        tickers = load_ticker_lists(get_settings())
        if not tickers.empty and "Ticker" in tickers:
            return sorted(tickers["Ticker"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    return ["AAPL", "BHARTIARTL.NS"]


@ui.page("/backtest")
async def page_backtest() -> None:
    layout()
    ui.label("Backtesting").classes("text-h5 q-px-md q-pt-md")

    default_end = date.today()
    default_start = default_end.replace(year=default_end.year - 1)
    ticker_opts = _ticker_options()

    # --- state refs ---
    result_area = ui.column().classes("w-full q-px-md")

    with ui.row().classes("w-full q-px-md q-gutter-md items-start"):
        # Left input panel
        with ui.card().classes("col-3 q-pa-md"):
            ui.label("Inputs").classes("text-subtitle1 text-bold q-mb-sm")
            ticker_input = ui.select(ticker_opts, value=ticker_opts[0], label="Ticker").classes("w-full")
            strategy_input = ui.select(list(STRATEGIES.keys()), value="Buy and Hold", label="Strategy").classes("w-full")
            start_input = ui.date(value=default_start.isoformat()).classes("w-full")
            end_input = ui.date(value=default_end.isoformat()).classes("w-full")
            cash_input = ui.number(label="Starting cash", value=10000.0, min=100.0, step=1000.0).classes("w-full")
            comm_input = ui.number(label="Commission", value=0.001, min=0.0, max=0.05, step=0.001, format="%.4f").classes("w-full")
            refresh_input = ui.switch("Use latest Yahoo data", value=True)

            sma_card = ui.card().classes("w-full q-pa-sm")
            with sma_card:
                short_input = ui.number(label="Short SMA period", value=5, min=2, max=250).classes("w-full")
                long_input = ui.number(label="Long SMA period", value=20, min=3, max=400).classes("w-full")
            sma_card.set_visibility(False)

            def _toggle_sma(e):
                sma_card.set_visibility(e.value == "SMA Crossover")
            strategy_input.on("update:model-value", _toggle_sma)

            run_btn = ui.button("Run Backtest", icon="play_arrow").props("color=primary").classes("w-full q-mt-sm")

        # Right content area
        with ui.column().classes("col q-gutter-md"):
            preview_slot = ui.column().classes("w-full")
            result_area = ui.column().classes("w-full")

    async def _run_backtest() -> None:
        result_area.clear()
        preview_slot.clear()
        tkr = ticker_input.value
        strat = strategy_input.value
        start = date.fromisoformat(start_input.value)
        end = date.fromisoformat(end_input.value)

        # Show price preview
        with preview_slot:
            spinner = ui.spinner(size="lg")
        loop = asyncio.get_event_loop()
        try:
            preview = await loop.run_in_executor(
                None, partial(get_history, tkr, start.isoformat(), end.isoformat(), "1d", refresh_input.value)
            )
        except Exception as exc:
            preview_slot.clear()
            with preview_slot:
                ui.notification(f"Could not load price data: {exc}", type="negative")
            return
        preview_slot.clear()
        if not preview.empty and "Close" in preview.columns:
            fig = px.line(preview, x=preview.index, y="Close", title=f"{tkr} close price")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            with preview_slot:
                ui.plotly(fig).classes("w-full")

        # Validate SMA
        params: dict = {}
        if strat == "SMA Crossover":
            sp, lp = int(short_input.value), int(long_input.value)
            if sp >= lp:
                with result_area:
                    ui.notification("Short SMA period must be less than Long SMA period.", type="warning")
                return
            params = {"short_period": sp, "long_period": lp}

        # Run backtest
        with result_area:
            spin2 = ui.spinner(size="lg")
        try:
            res: BacktestResult = await loop.run_in_executor(
                None, partial(run_backtest, tkr, strat, start, end, float(cash_input.value),
                              float(comm_input.value), params, refresh_input.value)
            )
        except Exception as exc:
            result_area.clear()
            with result_area:
                ui.notification(f"Backtest failed: {exc}", type="negative")
            return

        result_area.clear()
        import math
        with result_area:
            ui.label(f"{res.strategy} on {res.ticker}  ·  {res.start} → {res.end}").classes("text-caption text-grey-5")
            # Metrics row 1
            with ui.row().classes("q-gutter-md q-mb-sm"):
                for title, val in [
                    ("Starting value", f"${res.starting_value:,.2f}"),
                    ("Ending value", f"${res.ending_value:,.2f}"),
                    ("Profit", f"${res.profit:,.2f}  ({res.return_pct:.2f}%)"),
                    ("Closed trades", str(res.trade_count)),
                ]:
                    with ui.card().classes("q-pa-sm"):
                        ui.label(title).classes("text-caption text-grey-5")
                        ui.label(val).classes("text-h6 text-bold")
            # Metrics row 2
            sharpe_str = f"{res.sharpe:.2f}" if not math.isnan(res.sharpe) else "N/A"
            avg_pnl = res.trades["pnl"].mean() if not res.trades.empty else float("nan")
            avg_pnl_str = f"${avg_pnl:,.4f}" if not math.isnan(avg_pnl) else "N/A"
            with ui.row().classes("q-gutter-md q-mb-sm"):
                for title, val in [
                    ("Max drawdown", f"{res.max_drawdown_pct:.2f}%"),
                    ("Approx Sharpe", sharpe_str),
                    ("Avg trade P&L", avg_pnl_str),
                ]:
                    with ui.card().classes("q-pa-sm"):
                        ui.label(title).classes("text-caption text-grey-5")
                        ui.label(val).classes("text-h6 text-bold")
            # Equity curve
            if not res.equity_curve.empty:
                ui.label("Equity Curve").classes("text-subtitle1 text-bold q-mt-sm")
                eq_df = res.equity_curve.reset_index()
                eq_df.columns = ["Date", "Portfolio Value"]
                fig_eq = px.line(eq_df, x="Date", y="Portfolio Value")
                fig_eq.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
                ui.plotly(fig_eq).classes("w-full")
            # Trades table
            if not res.trades.empty:
                with ui.expansion(f"Trades ({len(res.trades)} closed)", icon="table_chart").classes("w-full"):
                    ui.table(
                        columns=[{"name": c, "label": c, "field": c} for c in res.trades.columns],
                        rows=res.trades.to_dict("records"),
                    ).classes("w-full")
            else:
                ui.label("No closed trades recorded.").classes("text-grey-5")

    run_btn.on("click", _run_backtest)


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

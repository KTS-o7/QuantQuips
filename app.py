from __future__ import annotations

import math
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from quantquips.backtest_service import run_backtest
from quantquips.config import get_settings
from quantquips.data_service import get_history, load_ticker_lists
from quantquips.ga_service import run_ga_optimization
from quantquips.strategies import STRATEGIES


st.set_page_config(page_title="QuantQuips", page_icon="chart_with_upwards_trend", layout="wide")


@st.cache_data(ttl=3600)
def cached_ticker_lists() -> pd.DataFrame:
    return load_ticker_lists(get_settings())


@st.cache_data(ttl=300)
def cached_market_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _download_period(ticker, period, interval)


def _download_period(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def home_history(ticker: str) -> tuple[pd.DataFrame, str]:
    live_data = cached_market_history(ticker, "1d", "1m")
    if not live_data.empty and "Close" in live_data:
        return live_data, f"{ticker} intraday"

    return pd.DataFrame(), ticker


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def render_sidebar() -> str:
    st.sidebar.markdown("## Navigation")
    return st.sidebar.radio(
        "Go to",
        ["Home", "Backtesting", "Genetic Algorithm", "LLM", "About Us"],
    )


def render_home() -> None:
    st.title("QuantQuips")
    st.subheader("Market Snapshot")

    symbols = {"Nifty 50": "^NSEI", "Sensex": "^BSESN"}
    columns = st.columns(len(symbols))

    for column, (label, ticker) in zip(columns, symbols.items()):
        with column:
            try:
                data, source = home_history(ticker)
            except Exception as exc:
                st.warning(f"Could not load {label}: {exc}")
                continue

            if data.empty or "Close" not in data:
                st.warning(
                    f"No live Yahoo Finance data available for {label}. "
                    "Check network access and yfinance version."
                )
                continue

            first_close = float(data["Close"].iloc[0])
            last_close = float(data["Close"].iloc[-1])
            change_pct = ((last_close - first_close) / first_close) * 100 if first_close else 0
            st.metric(label, f"{last_close:,.2f}", f"{change_pct:.2f}%")

            fig = px.line(data, x=data.index, y="Close", title=source)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)


def ticker_options() -> list[str]:
    tickers = cached_ticker_lists()
    if tickers.empty or "Ticker" not in tickers:
        return ["AAPL", "BHARTIARTL.NS"]
    return sorted(tickers["Ticker"].dropna().astype(str).unique().tolist())


def render_backtesting() -> None:
    st.title("Backtesting")
    default_end = date.today()
    default_start = default_end.replace(year=default_end.year - 1)

    with st.sidebar:
        st.markdown("### Backtest Inputs")
        ticker = st.selectbox("Ticker", ticker_options(), index=0)
        strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
        start = st.date_input("Start", value=default_start)
        end = st.date_input("End", value=default_end)
        cash = st.number_input("Starting cash", min_value=100.0, value=10000.0, step=1000.0)
        commission = st.number_input("Commission", min_value=0.0, max_value=0.05, value=0.001, step=0.001, format="%.4f")
        refresh_data = st.checkbox("Use latest Yahoo data", value=True)

        strategy_params: dict[str, int] = {}
        if strategy_name == "SMA Crossover":
            short_period = st.number_input("Short SMA period", min_value=2, max_value=250, value=5)
            long_period = st.number_input("Long SMA period", min_value=3, max_value=400, value=20)
            strategy_params = {
                "short_period": int(short_period),
                "long_period": int(long_period),
            }

        run_clicked = st.button("Run Backtest", type="primary")

    if strategy_name == "SMA Crossover" and strategy_params["short_period"] >= strategy_params["long_period"]:
        st.warning("Short SMA period should be less than long SMA period.")
        return

    try:
        preview = get_history(
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            refresh=refresh_data,
        )
    except Exception as exc:
        st.error(f"Could not load price data: {exc}")
        return

    if preview.empty:
        st.info("No price data is available for this ticker and date range.")
    else:
        fig = px.line(preview, x=preview.index, y="Close", title=f"{ticker} close price")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(preview.tail(10), use_container_width=True)

    if not run_clicked:
        return

    try:
        result = run_backtest(
            ticker=ticker,
            strategy_name=strategy_name,
            start=start,
            end=end,
            cash=float(cash),
            commission=float(commission),
            strategy_params=strategy_params,
            refresh_data=refresh_data,
        )
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        return

    st.subheader("Results")
    st.caption(
        f"{result.strategy} on {result.ticker} "
        f"from {result.start.isoformat()} to {result.end.isoformat()}."
    )

    # --- Top metrics row ---
    metric_cols = st.columns(4)
    metric_cols[0].metric("Starting value", format_money(result.starting_value))
    metric_cols[1].metric("Ending value", format_money(result.ending_value))
    metric_cols[2].metric("Profit", format_money(result.profit), f"{result.return_pct:.2f}%")
    metric_cols[3].metric("Closed trades", str(result.trade_count))

    # --- Extra metrics row ---
    extra_cols = st.columns(3)
    extra_cols[0].metric(
        "Max drawdown",
        f"{result.max_drawdown_pct:.2f}%",
        help="Peak-to-trough drawdown on the portfolio equity curve.",
    )
    sharpe_display = f"{result.sharpe:.2f}" if not math.isnan(result.sharpe) else "N/A"
    extra_cols[1].metric(
        "Approx Sharpe",
        sharpe_display,
        help="Annualised Sharpe ratio on daily returns, risk-free rate = 0. Approximate.",
    )
    avg_pnl = (
        result.trades["pnl"].mean() if not result.trades.empty else float("nan")
    )
    avg_pnl_display = format_money(avg_pnl) if not math.isnan(avg_pnl) else "N/A"
    extra_cols[2].metric(
        "Avg trade P&L",
        avg_pnl_display,
        help="Mean profit/loss per closed trade.",
    )

    # --- Equity curve ---
    if not result.equity_curve.empty:
        st.subheader("Equity Curve")
        eq_df = result.equity_curve.reset_index()
        eq_df.columns = ["Date", "Portfolio Value"]
        fig_eq = px.line(eq_df, x="Date", y="Portfolio Value", title="Portfolio value over time")
        fig_eq.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_eq, use_container_width=True)

    # --- Trades table ---
    if not result.trades.empty:
        with st.expander(f"Trades ({len(result.trades)} closed)", expanded=False):
            display_trades = result.trades.copy()
            display_trades["pnl"] = display_trades["pnl"].map(lambda v: f"${v:,.4f}")
            st.dataframe(display_trades, use_container_width=True)
    else:
        st.info("No closed trades recorded for this backtest.")


def render_genetic_algorithm() -> None:
    st.title("Genetic Algorithm — SMA Crossover Optimiser")
    st.write(
        "Finds the best Short and Long SMA periods for a given ticker and date range "
        "by evolving a population of candidate parameter pairs over multiple generations. "
        "Fitness = total return %."
    )

    default_end = date.today()
    default_start = default_end.replace(year=default_end.year - 1)

    with st.sidebar:
        st.markdown("### Optimiser Inputs")
        ga_ticker = st.selectbox("Ticker", ticker_options(), index=0, key="ga_ticker")
        ga_start = st.date_input("Start", value=default_start, key="ga_start")
        ga_end = st.date_input("End", value=default_end, key="ga_end")
        ga_cash = st.number_input("Starting cash", min_value=100.0, value=10000.0, step=1000.0, key="ga_cash")
        ga_commission = st.number_input(
            "Commission", min_value=0.0, max_value=0.05, value=0.001, step=0.001, format="%.4f", key="ga_comm"
        )
        ga_refresh = st.checkbox("Use latest Yahoo data", value=True, key="ga_refresh")

        st.markdown("#### GA Parameters")
        ga_pop = st.slider("Population size", min_value=4, max_value=60, value=20, step=2)
        ga_gen = st.slider("Generations", min_value=2, max_value=30, value=10)
        ga_mutation = st.slider("Mutation rate", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

        st.markdown("#### Parameter Search Ranges")
        short_min, short_max = st.slider(
            "Short SMA range", min_value=2, max_value=100, value=(2, 30)
        )
        long_min, long_max = st.slider(
            "Long SMA range", min_value=5, max_value=300, value=(10, 100)
        )

        ga_run = st.button("Run Optimisation", type="primary")

    if not ga_run:
        st.info("Configure the inputs in the sidebar and click **Run Optimisation** to start.")
        return

    if long_max <= short_min:
        st.warning("Long SMA range must have values greater than the Short SMA minimum.")
        return

    total_evals = ga_pop * (ga_gen + 1)
    st.caption(
        f"Evaluating approximately {total_evals} backtests "
        f"({ga_pop} individuals × {ga_gen + 1} generations). "
        "This may take a while."
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _progress(current: int, total: int) -> None:
        fraction = current / total if total else 0
        progress_bar.progress(fraction)
        status_text.text(f"Generation {current} / {total} complete")

    try:
        ga_result = run_ga_optimization(
            ticker=ga_ticker,
            start=ga_start,
            end=ga_end,
            cash=float(ga_cash),
            commission=float(ga_commission),
            refresh_data=ga_refresh,
            population_size=int(ga_pop),
            generations=int(ga_gen),
            mutation_rate=float(ga_mutation),
            short_range=(int(short_min), int(short_max)),
            long_range=(int(long_min), int(long_max)),
            progress_callback=_progress,
        )
    except Exception as exc:
        st.error(f"Optimisation failed: {exc}")
        return

    progress_bar.progress(1.0)
    status_text.text("Optimisation complete.")

    st.subheader("Best Parameters Found")
    res_cols = st.columns(3)
    res_cols[0].metric("Short SMA period", str(ga_result.best_short))
    res_cols[1].metric("Long SMA period", str(ga_result.best_long))
    res_cols[2].metric("Best return", f"{ga_result.best_return_pct:.2f}%")

    import pandas as pd

    hist_df = pd.DataFrame(ga_result.population_history)
    if not hist_df.empty:
        st.subheader("Population History — All Evaluated Individuals")
        fig_scatter = px.scatter(
            hist_df,
            x="short_period",
            y="long_period",
            color="return_pct",
            color_continuous_scale="RdYlGn",
            hover_data=["generation", "return_pct"],
            title="Evaluated (short, long) pairs coloured by return %",
            labels={
                "short_period": "Short SMA period",
                "long_period": "Long SMA period",
                "return_pct": "Return %",
            },
        )
        fig_scatter.update_traces(marker_size=8)
        fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("Best individual per generation"):
            best_per_gen = (
                hist_df.sort_values("return_pct", ascending=False)
                .groupby("generation")
                .first()
                .reset_index()[["generation", "short_period", "long_period", "return_pct"]]
            )
            st.dataframe(best_per_gen, use_container_width=True)


def render_llm() -> None:
    st.title("LLM Assistant")
    settings = get_settings()
    if settings.llm_provider == "disabled":
        st.info(
            "LLM support is disabled. Set QUANTQUIPS_LLM_PROVIDER to mlx or bifrost "
            "after the provider adapter is added in the next phase."
        )
        return

    st.warning(
        f"Provider '{settings.llm_provider}' is configured, but the LangChain v1 agent "
        "adapter has not been implemented on this branch yet."
    )


def render_about() -> None:
    st.title("About QuantQuips")
    st.write(
        "QuantQuips is a personal research workspace for exploring market data, "
        "running educational backtests, and experimenting with AI-assisted strategy analysis."
    )
    st.warning("Educational research only. This app does not place trades or provide financial advice.")


page = render_sidebar()

if page == "Home":
    render_home()
elif page == "Backtesting":
    render_backtesting()
elif page == "Genetic Algorithm":
    render_genetic_algorithm()
elif page == "LLM":
    render_llm()
else:
    render_about()

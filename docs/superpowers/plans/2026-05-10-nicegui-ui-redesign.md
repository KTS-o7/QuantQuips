# NiceGUI UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit frontend with a NiceGUI-based UI in `nicegui_app.py`, keeping all `quantquips/` service layer code untouched.

**Architecture:** A single `nicegui_app.py` file defines all 5 pages as `@ui.page()` routes. A shared layout function renders the left drawer and header on every page. Blocking service calls (backtest, GA, yfinance) run in a thread executor so the async event loop stays responsive.

**Tech Stack:** NiceGUI 3.11.1, Plotly ≥ 5.18, Python 3.12, existing `quantquips/` services.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `nicegui_app.py` | Full NiceGUI application — all 5 pages, shared layout |
| Modify | `requirements-base.txt` | Add `nicegui>=3.11.1` |
| No change | `quantquips/` | All service layer files untouched |
| No change | `app.py` | Streamlit entry point kept as-is |

---

## Task 1: Install NiceGUI and verify imports

**Files:**
- Modify: `requirements-base.txt`

- [ ] **Step 1: Add nicegui to requirements**

In `requirements-base.txt`, append:
```
nicegui>=3.11.1
```

- [ ] **Step 2: Install into the venv**

```bash
uv pip install "nicegui>=3.11.1"
```

Expected output: `+ nicegui==...` in the install summary.

- [ ] **Step 3: Verify import**

```bash
uv run python -c "import nicegui; print(nicegui.__version__)"
```

Expected: a version string like `3.11.x`.

- [ ] **Step 4: Commit**

```bash
git add requirements-base.txt
git commit -m "deps: add nicegui>=3.11.1"
```

---

## Task 2: Scaffold nicegui_app.py with shared layout and routing

**Files:**
- Create: `nicegui_app.py`

- [ ] **Step 1: Create the file with shared layout helper**

```python
# nicegui_app.py
from __future__ import annotations

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


@ui.page("/")
def page_home() -> None:
    layout()
    ui.label("Home — coming in Task 3").classes("text-h6 q-pa-lg")


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
```

- [ ] **Step 2: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output (clean).

- [ ] **Step 3: Smoke-start the server**

```bash
uv run python nicegui_app.py &
sleep 3
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/
kill %1
```

Expected: `200`.

- [ ] **Step 4: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: scaffold nicegui_app with layout and stub routes"
```

---

## Task 3: Home page — market snapshot

**Files:**
- Modify: `nicegui_app.py` — replace `page_home()` stub

- [ ] **Step 1: Add imports at top of nicegui_app.py**

Add after the existing imports:
```python
import asyncio
from functools import partial

import pandas as pd
import plotly.express as px
import yfinance as yf
```

- [ ] **Step 2: Add the data helper**

Add before `page_home()`:
```python
def _download_period(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data
```

- [ ] **Step 3: Replace the page_home stub**

```python
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
```

- [ ] **Step 4: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: add home page market snapshot to nicegui_app"
```

---

## Task 4: Backtesting page

**Files:**
- Modify: `nicegui_app.py` — replace `page_backtest()` stub

- [ ] **Step 1: Add service imports at top of nicegui_app.py**

Add after the existing `yfinance` import:
```python
from datetime import date

from quantquips.backtest_service import BacktestResult, run_backtest
from quantquips.config import get_settings
from quantquips.data_service import get_history, load_ticker_lists
from quantquips.strategies import STRATEGIES
```

- [ ] **Step 2: Add ticker helper**

Add before `page_backtest()`:
```python
def _ticker_options() -> list[str]:
    try:
        tickers = load_ticker_lists(get_settings())
        if not tickers.empty and "Ticker" in tickers:
            return sorted(tickers["Ticker"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    return ["AAPL", "BHARTIARTL.NS"]
```

- [ ] **Step 3: Replace page_backtest stub**

```python
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
```

- [ ] **Step 4: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: add backtesting page to nicegui_app"
```

---

## Task 5: Genetic Algorithm page

**Files:**
- Modify: `nicegui_app.py` — replace `page_ga()` stub

- [ ] **Step 1: Add GA service import at top of nicegui_app.py**

Add after the `run_backtest` import line:
```python
from quantquips.ga_service import GaResult, run_ga_optimization
```

- [ ] **Step 2: Replace page_ga stub**

```python
@ui.page("/ga")
async def page_ga() -> None:
    layout()
    ui.label("Genetic Algorithm — SMA Crossover Optimiser").classes("text-h5 q-px-md q-pt-md")
    ui.label(
        "Evolves Short/Long SMA period pairs over multiple generations. Fitness = total return %."
    ).classes("text-caption text-grey-5 q-px-md q-mb-md")

    default_end = date.today()
    default_start = default_end.replace(year=default_end.year - 1)
    ticker_opts = _ticker_options()

    with ui.row().classes("w-full q-px-md q-gutter-md items-start"):
        # Left input panel
        with ui.card().classes("col-3 q-pa-md"):
            ui.label("Inputs").classes("text-subtitle1 text-bold q-mb-sm")
            ga_ticker = ui.select(ticker_opts, value=ticker_opts[0], label="Ticker").classes("w-full")
            ga_start = ui.date(value=default_start.isoformat()).classes("w-full")
            ga_end = ui.date(value=default_end.isoformat()).classes("w-full")
            ga_cash = ui.number(label="Starting cash", value=10000.0, min=100.0, step=1000.0).classes("w-full")
            ga_comm = ui.number(label="Commission", value=0.001, min=0.0, max=0.05, step=0.001, format="%.4f").classes("w-full")
            ga_refresh = ui.switch("Use latest Yahoo data", value=True)

            ui.separator().classes("q-my-sm")
            ui.label("GA Parameters").classes("text-caption text-grey-5")
            ga_pop = ui.slider(min=4, max=60, value=20, step=2).props("label-always").classes("w-full")
            ui.label().bind_text_from(ga_pop, "value", lambda v: f"Population: {v}")
            ga_gen = ui.slider(min=2, max=30, value=10).props("label-always").classes("w-full")
            ui.label().bind_text_from(ga_gen, "value", lambda v: f"Generations: {v}")
            ga_mut = ui.slider(min=0.05, max=0.5, value=0.2, step=0.05).props("label-always").classes("w-full")
            ui.label().bind_text_from(ga_mut, "value", lambda v: f"Mutation rate: {v:.2f}")

            ui.separator().classes("q-my-sm")
            ui.label("SMA Search Ranges").classes("text-caption text-grey-5")
            short_range = ui.range(min=2, max=100, value={"min": 2, "max": 30}).classes("w-full")
            ui.label().bind_text_from(short_range, "value", lambda v: f"Short: {v['min']}–{v['max']}")
            long_range = ui.range(min=5, max=300, value={"min": 10, "max": 100}).classes("w-full")
            ui.label().bind_text_from(long_range, "value", lambda v: f"Long: {v['min']}–{v['max']}")

            run_btn = ui.button("Run Optimisation", icon="psychology").props("color=primary").classes("w-full q-mt-sm")

        # Right result area
        with ui.column().classes("col q-gutter-md"):
            progress_bar = ui.linear_progress(value=0).classes("w-full")
            progress_label = ui.label("").classes("text-caption text-grey-5")
            result_area = ui.column().classes("w-full")

    async def _run_ga() -> None:
        result_area.clear()
        progress_bar.set_value(0)
        progress_label.set_text("Starting…")

        sr = short_range.value
        lr = long_range.value
        if lr["max"] <= sr["min"]:
            ui.notification("Long SMA range must exceed Short SMA minimum.", type="warning")
            return

        import asyncio as _asyncio
        queue: asyncio.Queue = asyncio.Queue()

        def _cb(current: int, total: int) -> None:
            asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, (current, total))

        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(
            None,
            partial(
                run_ga_optimization,
                ga_ticker.value,
                date.fromisoformat(ga_start.value),
                date.fromisoformat(ga_end.value),
                float(ga_cash.value),
                float(ga_comm.value),
                ga_refresh.value,
                int(ga_pop.value),
                int(ga_gen.value),
                float(ga_mut.value),
                (int(sr["min"]), int(sr["max"])),
                (int(lr["min"]), int(lr["max"])),
                _cb,
            ),
        )

        total_gens = int(ga_gen.value)
        while not fut.done():
            try:
                current, total = await asyncio.wait_for(asyncio.shield(queue.get()), timeout=0.2)
                progress_bar.set_value(current / total if total else 0)
                progress_label.set_text(f"Generation {current} / {total}")
            except asyncio.TimeoutError:
                pass

        try:
            ga_res: GaResult = await fut
        except Exception as exc:
            ui.notification(f"Optimisation failed: {exc}", type="negative")
            return

        progress_bar.set_value(1.0)
        progress_label.set_text("Optimisation complete.")

        with result_area:
            ui.label("Best Parameters Found").classes("text-subtitle1 text-bold")
            with ui.row().classes("q-gutter-md q-mb-sm"):
                for title, val in [
                    ("Short SMA period", str(ga_res.best_short)),
                    ("Long SMA period", str(ga_res.best_long)),
                    ("Best return", f"{ga_res.best_return_pct:.2f}%"),
                ]:
                    with ui.card().classes("q-pa-sm"):
                        ui.label(title).classes("text-caption text-grey-5")
                        ui.label(val).classes("text-h6 text-bold")

            hist_df = pd.DataFrame(ga_res.population_history)
            if not hist_df.empty:
                ui.label("Population History").classes("text-subtitle1 text-bold q-mt-sm")
                fig = px.scatter(
                    hist_df, x="short_period", y="long_period", color="return_pct",
                    color_continuous_scale="RdYlGn",
                    hover_data=["generation", "return_pct"],
                    labels={"short_period": "Short SMA", "long_period": "Long SMA", "return_pct": "Return %"},
                )
                fig.update_traces(marker_size=8)
                fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10))
                ui.plotly(fig).classes("w-full")

                best_per_gen = (
                    hist_df.sort_values("return_pct", ascending=False)
                    .groupby("generation").first().reset_index()
                    [["generation", "short_period", "long_period", "return_pct"]]
                )
                with ui.expansion("Best individual per generation", icon="expand_more").classes("w-full"):
                    ui.table(
                        columns=[{"name": c, "label": c, "field": c} for c in best_per_gen.columns],
                        rows=best_per_gen.to_dict("records"),
                    ).classes("w-full")

    run_btn.on("click", _run_ga)
```

- [ ] **Step 3: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: add genetic algorithm page to nicegui_app"
```

---

## Task 6: LLM page

**Files:**
- Modify: `nicegui_app.py` — replace `page_llm()` stub

- [ ] **Step 1: Replace page_llm stub**

```python
@ui.page("/llm")
async def page_llm() -> None:
    layout()
    ui.label("LLM Assistant").classes("text-h5 q-px-md q-pt-md")

    settings = get_settings()
    with ui.card().classes("q-ma-md q-pa-md"):
        if settings.llm_provider == "disabled":
            ui.icon("info", size="lg").classes("text-blue q-mb-sm")
            ui.label("LLM support is disabled.").classes("text-subtitle1 text-bold")
            ui.label(
                "Set the QUANTQUIPS_LLM_PROVIDER environment variable to 'mlx' or 'bifrost' "
                "to enable LLM support after the provider adapter is implemented."
            ).classes("text-body2 text-grey-5")
        else:
            ui.icon("warning", size="lg").classes("text-orange q-mb-sm")
            ui.label(f"Provider '{settings.llm_provider}' configured but not yet implemented.").classes("text-subtitle1")
            ui.label(
                "The LangChain v1 agent adapter for this provider has not been built yet."
            ).classes("text-body2 text-grey-5")
```

- [ ] **Step 2: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: add LLM placeholder page to nicegui_app"
```

---

## Task 7: About page

**Files:**
- Modify: `nicegui_app.py` — replace `page_about()` stub

- [ ] **Step 1: Replace page_about stub**

```python
@ui.page("/about")
async def page_about() -> None:
    layout()
    ui.label("About QuantQuips").classes("text-h5 q-px-md q-pt-md")

    with ui.card().classes("q-ma-md q-pa-md"):
        ui.label(
            "QuantQuips is a personal research workspace for exploring market data, "
            "running educational backtests, and experimenting with AI-assisted strategy analysis."
        ).classes("text-body1")
        ui.separator().classes("q-my-md")
        ui.label("Educational research only. This app does not place trades or provide financial advice.").classes(
            "text-caption text-orange"
        )

    ui.label("Our Team").classes("text-h6 q-px-md q-mt-md q-mb-sm")
    team = [
        ("Krishnatejaswi S", "LangChain Developer"),
        ("Vinayak C", "ML Engineer"),
        ("Bipin Raj C", "Python Developer"),
        ("Ananya Bhat", "Python Developer"),
    ]
    with ui.row().classes("q-px-md q-gutter-md"):
        for name, role in team:
            with ui.card().classes("q-pa-md items-center"):
                ui.avatar(name[0], color="primary", size="xl").classes("q-mb-sm")
                ui.label(name).classes("text-subtitle2 text-bold")
                ui.label(role).classes("text-caption text-grey-5")
```

- [ ] **Step 2: Compile check**

```bash
uv run python -m py_compile nicegui_app.py
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add nicegui_app.py
git commit -m "feat: add about page to nicegui_app"
```

---

## Task 8: End-to-end smoke test and PR

**Files:**
- No code changes — test and push only

- [ ] **Step 1: Install nicegui into venv**

```bash
uv pip install "nicegui>=3.11.1"
```

- [ ] **Step 2: Compile check all files**

```bash
uv run python -m py_compile nicegui_app.py app.py quantquips/backtest_service.py quantquips/ga_service.py quantquips/data_service.py quantquips/config.py quantquips/strategies.py
```

Expected: no output.

- [ ] **Step 3: Import smoke test**

```bash
uv run python -c "
import nicegui_app
print('nicegui_app imported OK')
"
```

Expected: `nicegui_app imported OK`

- [ ] **Step 4: Start server and check all routes respond**

```bash
uv run python nicegui_app.py &
sleep 4
for path in / /backtest /ga /llm /about; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080$path)
  echo "$path → $code"
done
kill %1
```

Expected: all 5 routes return `200`.

- [ ] **Step 5: Manual walkthrough**

Start the server with `uv run python nicegui_app.py` and open `http://localhost:8080` in a browser. Verify:
- [ ] Left drawer renders with all 5 nav links
- [ ] Home page loads Nifty 50 and Sensex charts (or shows graceful "no data" message)
- [ ] Backtesting page: select AAPL, Buy and Hold, run → metrics + equity curve render
- [ ] GA page: small run (population=6, generations=2) completes, scatter chart renders
- [ ] LLM page: disabled notice renders
- [ ] About page: team cards render

- [ ] **Step 6: Push branch and open stacked PR**

```bash
git push origin fix/sarang-pr12-rebased
gh pr create \
  --base fix/sarang-pr12-rebased \
  --head fix/sarang-pr12-rebased \
  --title "feat: NiceGUI UI — replace Streamlit frontend" \
  --body "Replaces app.py (Streamlit) with nicegui_app.py. All 5 pages ported. Service layer untouched. Stacked on PR #13."
```

> Note: push the nicegui branch as a new branch name, e.g. `feat/nicegui-ui`, stacked on top of `fix/sarang-pr12-rebased`.

```bash
git checkout -b feat/nicegui-ui
git push origin feat/nicegui-ui
gh pr create \
  --base fix/sarang-pr12-rebased \
  --head feat/nicegui-ui \
  --title "feat: NiceGUI UI — replace Streamlit frontend" \
  --body "$(cat <<'EOF'
## Summary

- Adds `nicegui_app.py` as the new UI entry point (NiceGUI 3.11.1 / Quasar / Material Design)
- All 5 pages ported: Home, Backtesting, Genetic Algorithm, LLM, About
- Blocking service calls run in thread executor — UI stays responsive during backtest/GA runs
- `app.py` (Streamlit) retained and untouched
- Stacked on PR #13

## Testing

\`\`\`bash
uv run python -m py_compile nicegui_app.py
uv run python nicegui_app.py  # then open localhost:8080
\`\`\`
EOF
)"
```

---

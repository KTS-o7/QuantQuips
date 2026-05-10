# QuantQuips — NiceGUI UI Redesign

**Date:** 2026-05-10  
**Status:** Approved  
**Branch target:** stacked on `fix/sarang-pr12-rebased` (PR #13)

---

## Goal

Replace the Streamlit frontend (`app.py`) with a NiceGUI-based UI (`nicegui_app.py`). The redesign removes the visual limitations of Streamlit (fixed layout, opinionated widget chrome, full-page reruns) and replaces them with a Material Design / Quasar-based UI that looks polished out of the box. All backend logic in `quantquips/` remains completely untouched.

---

## Architecture

```
nicegui_app.py          ← new entry point (NiceGUI)
app.py                  ← retained, no longer the default runner
quantquips/             ← untouched service layer
  backtest_service.py
  ga_service.py
  data_service.py
  config.py
  strategies.py
```

The NiceGUI app is a standard FastAPI application under the hood. For VPS deployment it is served via `uvicorn` or `gunicorn` with a `--workers 1` flag (NiceGUI uses shared in-process state).

### Entry points

```bash
# Development
uv run python nicegui_app.py

# VPS production
uvicorn nicegui_app:app --host 0.0.0.0 --port 8080
```

---

## Dependencies

Add to `requirements-base.txt` (or a new `requirements-nicegui.txt`):

```
nicegui>=1.4
plotly>=5.18
```

`nicegui` bundles FastAPI, uvicorn, and Quasar. No additional installs needed.

---

## Layout

- **Left drawer** (collapsible, 220 px): app logo/name + navigation links for all 5 pages.
- **Header bar**: app title, dark-mode toggle.
- **Main content area**: page-specific content, full width.
- **Theme**: Quasar dark mode enabled by default; user can toggle via header button.

Navigation is implemented with NiceGUI's `ui.left_drawer()` + `ui.link()` pointing to named routes (`/`, `/backtest`, `/ga`, `/llm`, `/about`). Each route is a separate `@ui.page()` function.

---

## Pages

### Home (`/`)

- Two `ui.card()` columns side by side (Nifty 50, Sensex).
- Each card: `ui.label()` showing last close + % change, then `ui.plotly(fig)` for the intraday chart.
- Data from `_download_period()` (same logic as current `app.py`).
- Error/empty states shown via `ui.notification()`.

### Backtesting (`/backtest`)

**Left panel (inputs, ~300 px wide):**
- `ui.select()` for ticker and strategy
- `ui.date()` for start/end
- `ui.number()` for cash and commission
- `ui.switch()` for "Use latest Yahoo data"
- Conditional `ui.number()` fields for SMA periods (shown only when SMA Crossover is selected)
- `ui.button('Run Backtest')` — primary style

**Right/main area:**
- Price preview chart (`ui.plotly`) rendered as soon as ticker/dates change (no button click needed)
- After run: results section with two rows of `ui.card()` metric tiles (Starting value, Ending value, Profit, Return, Max drawdown, Sharpe, Avg P&L, Trade count)
- Equity curve `ui.plotly()` chart
- Trades log in `ui.table()` inside a `ui.expansion()`

**Long-running work:** `run_backtest()` is called via `asyncio.get_event_loop().run_in_executor(None, ...)` so the UI stays responsive. A `ui.spinner()` is shown during execution.

### Genetic Algorithm (`/ga`)

**Left panel (inputs):**
- Same ticker/date/cash/commission inputs as Backtesting
- GA parameters: `ui.slider()` for population size, generations, mutation rate
- SMA search ranges: two `ui.range()` sliders for short and long SMA bounds
- `ui.button('Run Optimisation')` — primary style

**Main area:**
- `ui.linear_progress()` updated live via `progress_callback` as the GA runs (executor thread posts updates via `ui.update()` / `app.storage`)
- Status label showing current generation
- After completion: best-params metric cards
- `ui.plotly()` scatter chart of all evaluated individuals coloured by return %
- `ui.expansion()` table of best individual per generation

**Long-running work:** same executor pattern as Backtesting. Progress updates posted from the worker thread using `ui.run_javascript` or `asyncio` queues.

### LLM (`/llm`)

- If `settings.llm_provider == "disabled"`: show an info `ui.card()` explaining how to configure a provider.
- Otherwise: a basic `ui.chat_message()` / `ui.input()` chat layout (placeholder for future LangChain adapter).
- No actual LLM wiring in this PR — faithful port of the current stub.

### About (`/about`)

- `ui.card()` with project description and disclaimer.
- Team member grid using `ui.card()` + `ui.avatar()` + `ui.label()` for each person.

---

## Threading / Async Model

NiceGUI runs on an asyncio event loop. Blocking calls (`run_backtest`, `run_ga_optimization`, `yf.download`) must not block the loop.

Pattern for all blocking calls:

```python
import asyncio
from functools import partial

loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, partial(run_backtest, ...))
```

For the GA progress callback (called from a worker thread), use a `asyncio.Queue` to post progress updates back to the async handler, which polls it and updates `ui.linear_progress`.

---

## Error Handling

- All service calls wrapped in try/except.
- Errors surface as `ui.notification(message, type='negative')`.
- Empty data states shown as `ui.label()` info cards (not hard errors).

---

## What Is Not In Scope

- Removing `app.py` — it is kept so the Streamlit path still works.
- LangChain LLM integration — the LLM page remains a placeholder.
- Authentication / multi-user support.
- Any changes to `quantquips/` service layer.

---

## Testing

```bash
# Compile check
uv run python -m py_compile nicegui_app.py

# Import smoke test
uv run python -c "import nicegui_app"

# Manual: start the server and exercise all 5 pages
uv run python nicegui_app.py
```

End-to-end: navigate to each page, run a backtest, run a small GA (population=6, generations=2), confirm LLM disabled notice renders, confirm About page renders.

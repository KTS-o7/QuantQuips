# QuantQuips Personal Usability Plan

## Goal

Make QuantQuips reliable enough for personal daily experimentation with equities, backtests, parameter optimization, and an optional LLM assistant that can run through either local MLX models or an OpenAI-compatible Bifrost gateway.

This plan is intentionally ordered so the app becomes usable before the agent layer is added. The agent should call stable project tools, not scrape Streamlit state or execute arbitrary user code.

## Phase 1: Local App Reliability

- Move runtime configuration into environment-backed settings.
- Split dependencies into a small base install and optional LLM extras.
- Remove import-time LLM, PDF, and embedding loading from `app.py`.
- Handle empty or failed market-data downloads without crashing the whole app.
- Replace hardcoded local CSV paths with configurable data paths.

Verification:

- `streamlit run app.py` opens without local PDFs or model files.
- Home and Backtesting pages show useful errors when data is unavailable.

## Phase 2: Safe Backtesting MVP

- Move Backtrader execution into `quantquips.backtest_service`.
- Support built-in strategies first: buy-and-hold and SMA crossover.
- Let the user select ticker, date range, cash, commission, and strategy parameters.
- Return structured metrics: starting value, ending value, profit, return percentage, and trade count.
- Stop overwriting `strategies.py` from the UI.

Verification:

- Backtests run against local CSV data when present.
- Backtests can fetch missing ticker data from Yahoo Finance.
- Service functions can be imported and tested without Streamlit.

## Phase 3: Data Workflow

- Add a data service for local CSV cache reads and yfinance refreshes.
- Normalize CSV columns for Backtrader compatibility.
- Add ticker lists from `data/TickerList`.
- Add explicit cache refresh controls.

Verification:

- AAPL and BHARTIARTL.NS can be selected from repo data.
- Missing tickers can be downloaded into `data/companyData/<market>/`.

## Phase 4: LLM Provider Adapter

- Add `QUANTQUIPS_LLM_PROVIDER=disabled|mlx|bifrost`.
- Add a model factory for MLX and OpenAI-compatible Bifrost endpoints.
- Keep LLM initialization lazy and page-scoped.
- Use local RAG only when document folders exist.

Verification:

- App works when LLM provider is disabled.
- Bifrost mode can answer a simple prompt using configured base URL and key.
- MLX mode can answer a simple prompt when local MLX dependencies are installed.

## Phase 5: LangChain v1 Agent

- Use LangChain v1 `create_agent` over stable project tools.
- Expose tools for price history, backtesting, optimization, and finance-note retrieval.
- Add middleware for human approval before long optimizations or future write actions.
- Keep all responses educational, with no live trading execution.

Verification:

- Agent can answer "Backtest SMA crossover on AAPL for 2023".
- Agent summarizes the result and cites the exact parameters it used.
- Expensive or sensitive tool calls require explicit approval.

## Phase 6: Tests and Documentation

- Add focused tests for config, data loading, and backtest results.
- Document MLX and Bifrost setup separately.
- Add a demo screenshot once the Streamlit UI stabilizes.

Verification:

- `pytest` passes locally.
- README has a personal-use quickstart and provider configuration examples.

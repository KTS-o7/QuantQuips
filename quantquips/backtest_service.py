from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import backtrader as bt
import pandas as pd

from quantquips.data_service import get_history
from quantquips.strategies import STRATEGIES


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    start: date
    end: date
    starting_value: float
    ending_value: float
    profit: float
    return_pct: float
    trade_count: int
    max_drawdown_pct: float = 0.0
    sharpe: float = float("nan")
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Backtrader analyzers
# ---------------------------------------------------------------------------

class TradeCounter(bt.Analyzer):
    def start(self) -> None:
        self.trade_count = 0

    def notify_trade(self, trade) -> None:
        if trade.isclosed:
            self.trade_count += 1

    def get_analysis(self) -> dict[str, int]:
        return {"trade_count": self.trade_count}


class EquityCurveRecorder(bt.Analyzer):
    """Records the broker portfolio value at the close of every bar."""

    def start(self) -> None:
        self._dates: list[date] = []
        self._values: list[float] = []

    def next(self) -> None:
        bar_date = self.data.datetime.date(0)
        self._dates.append(bar_date)
        self._values.append(float(self.strategy.broker.getvalue()))

    def get_analysis(self) -> dict:
        return {"dates": self._dates, "values": self._values}


class TradeLogger(bt.Analyzer):
    """Records a full log of closed trades."""

    def start(self) -> None:
        self._records: list[dict] = []
        self._open_trades: dict[int, dict] = {}

    def notify_trade(self, trade) -> None:
        if trade.justopened:
            self._open_trades[trade.ref] = {
                "entry_date": self.data.datetime.date(0),
                "entry_price": trade.price,
                "size": trade.size,
            }
        if trade.isclosed:
            entry = self._open_trades.pop(trade.ref, {})
            self._records.append(
                {
                    "entry_date": entry.get("entry_date"),
                    "exit_date": self.data.datetime.date(0),
                    "entry_price": round(entry.get("entry_price", float("nan")), 4),
                    "exit_price": round(trade.price, 4),
                    "size": round(entry.get("size", trade.size), 6),
                    "pnl": round(trade.pnl, 4),
                }
            )

    def get_analysis(self) -> dict:
        return {"records": self._records}


# ---------------------------------------------------------------------------
# Post-run metrics
# ---------------------------------------------------------------------------

def _compute_max_drawdown(equity: pd.Series) -> float:
    """Return max peak-to-trough drawdown as a positive percentage."""
    if equity.empty or len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    return round(float(drawdown.min()), 4)  # most negative → largest drawdown magnitude


def _compute_approx_sharpe(equity: pd.Series) -> float:
    """Annualised Sharpe on daily returns, risk-free rate = 0."""
    if equity.empty or len(equity) < 2:
        return float("nan")
    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() == 0:
        return float("nan")
    sharpe = daily_returns.mean() / daily_returns.std() * math.sqrt(252)
    return round(float(sharpe), 4)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("No price data is available for the selected ticker and date range.")

    prepared = data.copy()
    prepared.columns = [str(column).lower() for column in prepared.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(prepared.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Price data is missing required columns: {missing_text}.")
    return prepared


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    ticker: str,
    strategy_name: str,
    start: date,
    end: date,
    cash: float,
    commission: float,
    strategy_params: dict[str, int] | None = None,
    refresh_data: bool = False,
) -> BacktestResult:
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    if cash <= 0:
        raise ValueError("Starting cash must be greater than zero.")
    if start >= end:
        raise ValueError("Start date must be before end date.")

    raw_data = get_history(
        ticker=ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        refresh=refresh_data,
    )
    data = _prepare_data(raw_data)

    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addstrategy(STRATEGIES[strategy_name], **(strategy_params or {}))
    cerebro.addanalyzer(TradeCounter, _name="trade_counter")
    cerebro.addanalyzer(EquityCurveRecorder, _name="equity")
    cerebro.addanalyzer(TradeLogger, _name="trade_log")

    starting_value = float(cerebro.broker.getvalue())
    run_results = cerebro.run()
    strat = run_results[0]
    ending_value = float(cerebro.broker.getvalue())
    profit = ending_value - starting_value
    return_pct = (profit / starting_value) * 100
    trade_count = strat.analyzers.trade_counter.get_analysis()["trade_count"]

    # Build equity curve Series
    eq_analysis = strat.analyzers.equity.get_analysis()
    equity_series = pd.Series(
        eq_analysis["values"],
        index=pd.to_datetime(eq_analysis["dates"]),
        name="Portfolio Value",
    )

    # Build trades DataFrame
    trade_records = strat.analyzers.trade_log.get_analysis()["records"]
    trades_df = pd.DataFrame(
        trade_records,
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "size", "pnl"],
    ) if trade_records else pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "size", "pnl"]
    )

    max_dd = _compute_max_drawdown(equity_series)
    sharpe = _compute_approx_sharpe(equity_series)

    return BacktestResult(
        ticker=ticker,
        strategy=strategy_name,
        start=start,
        end=end,
        starting_value=starting_value,
        ending_value=ending_value,
        profit=profit,
        return_pct=return_pct,
        trade_count=trade_count,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        equity_curve=equity_series,
        trades=trades_df,
    )

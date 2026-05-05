from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import backtrader as bt
import pandas as pd

from quantquips.data_service import get_history
from quantquips.strategies import STRATEGIES


@dataclass(frozen=True)
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


class TradeCounter(bt.Analyzer):
    def start(self) -> None:
        self.trade_count = 0

    def notify_trade(self, trade) -> None:
        if trade.isclosed:
            self.trade_count += 1

    def get_analysis(self) -> dict[str, int]:
        return {"trade_count": self.trade_count}


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
    cerebro.addanalyzer(TradeCounter, _name="trades")

    starting_value = float(cerebro.broker.getvalue())
    run_results = cerebro.run()
    strategy = run_results[0]
    ending_value = float(cerebro.broker.getvalue())
    profit = ending_value - starting_value
    return_pct = (profit / starting_value) * 100
    trade_count = strategy.analyzers.trades.get_analysis()["trade_count"]

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
    )

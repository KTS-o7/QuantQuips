from __future__ import annotations

import backtrader as bt


class BuyAndHoldStrategy(bt.Strategy):
    """Invest all available cash on the first bar and hold."""

    def __init__(self) -> None:
        self.has_ordered = False

    def next(self) -> None:
        if self.has_ordered:
            return

        self.order_target_percent(target=0.95)
        self.has_ordered = True


class SmaCrossoverStrategy(bt.Strategy):
    params = (
        ("short_period", 5),
        ("long_period", 20),
    )

    def __init__(self) -> None:
        short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.short_period,
        )
        long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.long_period,
        )
        self.crossover = bt.indicators.CrossOver(short_ma, long_ma)

    def next(self) -> None:
        if not self.position and self.crossover > 0:
            self.order_target_percent(target=0.95)
        elif self.position and self.crossover < 0:
            self.order_target_percent(target=0.0)


STRATEGIES = {
    "Buy and Hold": BuyAndHoldStrategy,
    "SMA Crossover": SmaCrossoverStrategy,
}

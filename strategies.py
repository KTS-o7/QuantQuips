import backtrader as bt


class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        self.log(f"Close, {self.dataclose[0]:.2f}")

        if self.dataclose[0] < self.dataclose[-1] and self.dataclose[-1] < self.dataclose[-2]:
            self.log(f"BUY CREATE, {self.dataclose[0]:.2f}")
            self.buy()
# trader.py
import backtrader as bt
import datetime
from strategies import TestStrategy
import matplotlib.pyplot as plt
from io import BytesIO
import sys

cerebro = bt.Cerebro()

# Get initial principal amount from command-line argument
initial_principal_amount = float(sys.argv[1])
cerebro.broker.setcash(initial_principal_amount)

data = bt.feeds.YahooFinanceCSVData(
    dataname='AAPL.csv',
    fromdate=datetime.datetime(2023, 1, 1),
    todate=datetime.datetime(2023, 12, 31),
    reverse=False)

cerebro.adddata(data)
cerebro.addstrategy(TestStrategy)

# Display initial portfolio value
print(f'Initial Portfolio Value: ${initial_principal_amount:.2f}')

cerebro.run()

# Get final portfolio value
final_portfolio_value = cerebro.broker.getvalue()
print(f'Ending Portfolio Value: ${final_portfolio_value:.2f}')

# Plot interactive graph
cerebro.plot(style='candlestick', barup='green', bardown='red')

# <span style="color:  #00FFFF">QUANTQUIPS</span>

<span style="color: lightblue;">Quantquips, a cutting-edge backtesting platform seamlessly crafted with Streamlit and powered by Python. Elevate your trading strategies with precision on our dynamic backtesting page, unleash the power of genetic algorithms to discover optimal parameters, and let our sophisticated LLM guide you through the intricate realm of algorithmic trading. Elevate your trading experience with Quantquips - where innovation meets precision.</span>

Set up a virtual environment in your system using the following commands for safer execution and deployment of the code.

```python
python3 -m venv stealthAlgo
```

```bash
source stealthAlgo/bin/activate
pip install <package-name>
```

To deactivate

```bash
deactivate
```

Packages required : refer the `requirements.txt` file

```python
pip install -r requirements.txt
```

Clone this repository using `git clone` and deploy using the command-

```python
streamlit run app.py
```

You can also just view the website on this link and interact with the application [link](https://www.youtube.com/watch?v=HIcSWuKMwOw)

## HOME PAGE

This page contains real time data and graphs of Nifty 50 and Sensex.
Hover over the graphs for an interactive experience.

## BACKTESTING PAGE

Here is a trial strategy for you to test out and view the plot and data for

```python
import backtrader as bt

# your Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()
```

make sure you add your strategy under TestStrategy class and add the line

```python
import backtrader as bt
```

A plot pops up and your output is displayed once the plot is closed.
If the plot does not pop up, check your system settings.

## GENETIC ALGORITHM PAGE

Enter a strategy which accepts the short term and long term duration as input parameters.
Then input the parameters and run

here's a sample strategy to run




```python
class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ('short_period', 50),
        ('long_period', 200),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:  # Short-term MA crosses above Long-term MA
            self.buy()
        elif self.crossover < 0:  # Short-term MA crosses below Long-term MA
            self.sell()
```




## LLM PAGE

Enter your queries and get AI generated answers for the same here.
Sample question - `what is a genetic algorithm?`

=======
## ABOUT US
We are a group of passionate and enthusiastic undergraduate students! Check out this page for more information about us. Explore Quantquips, where we blend technology and finance to redefine algorithmic trading.

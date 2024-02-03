import backtrader as bt
import datetime
import random
import numpy as np
best = []

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

def run_strategy(short_period, long_period):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MovingAverageCrossoverStrategy, short_period=short_period, long_period=long_period)
    data = bt.feeds.YahooFinanceCSVData(dataname='NVDA.csv', fromdate=datetime.datetime(2010, 1, 1), todate=datetime.datetime(2022, 1, 1))
    cerebro.adddata(data)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()

    #print(cerebro.broker.getvalue())
    return cerebro.broker.getvalue()  # or some other performance metric

def select_best(results):
    # Sort the results by performance
    max_tuple = max(results, key=lambda x: x[2])
    print(max_tuple)
    best.append(max_tuple)
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    # Select the top half of the population
    return [x[:2] for x in sorted_results[:int(POPULATION_SIZE // 2)]]

def crossover_and_mutate(population, mutation_rate):
    next_generation = []
    for _ in range(int(POPULATION_SIZE)):
        # Select two parents
        parent1, parent2 = random.sample(population, 2)
        # Perform crossover
        child = (parent1[0], parent2[1]) if random.random() < 0.5 else (parent2[0], parent1[1])
        # Perform mutation
        child = (int(child[0] + np.random.normal(0, mutation_rate)), int(child[1] + np.random.normal(0, mutation_rate)))
        next_generation.append(child)
    return next_generation

# Define the genetic algorithm parameters
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.05
# Generate the initial population
population = [(random.randint(15, 30), random.randint(40, 60)) for _ in range(int(POPULATION_SIZE))]

for _ in range(int(GENERATIONS)):
    # Evaluate the population
    results = [(short_period, long_period, run_strategy(short_period, long_period)) for short_period, long_period in population]
    # Select the best individuals for the next generation
    population = select_best(results)
    # Apply crossover and mutation to generate the next generation
    population = crossover_and_mutate(population, MUTATION_RATE)
max_tuple2 = max(best, key=lambda x: x[2])
print(max_tuple2)
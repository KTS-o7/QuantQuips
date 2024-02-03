import streamlit as st
import backtrader as bt
import datetime
import random
import numpy as np

def main():
    st.title("Trading Strategy Optimizer")

    st.header("Enter Your Trading Strategy")
    strategy_code = st.text_area("Python code for your trading strategy")

    st.header("Parameter Ranges")
    short_period_range = st.slider("Short Period Range", min_value=10, max_value=50, value=(15, 30))
    long_period_range = st.slider("Long Period Range", min_value=40, max_value=100, value=(40, 60))

    if st.button("Optimize Strategy"):
        results = run_genetic_algorithm(strategy_code, short_period_range, long_period_range)
        display_results(results)

def run_genetic_algorithm(strategy_code, short_period_range, long_period_range):
    best = []
    POPULATION_SIZE = 10
    GENERATIONS = 50
    MUTATION_RATE = 0.05

    # Generate the initial population
    population = [(random.randint(short_period_range[0], short_period_range[1]), 
                   random.randint(long_period_range[0], long_period_range[1])) 
                  for _ in range(POPULATION_SIZE)]

    for _ in range(GENERATIONS):
        # Evaluate the population
        results = [(short_period, long_period, 
                    evaluate_strategy(strategy_code, short_period, long_period)) 
                   for short_period, long_period in population]
        # Select the best individuals for the next generation
        population = select_best(results)
        # Apply crossover and mutation to generate the next generation
        population = crossover_and_mutate(population, MUTATION_RATE)
    max_tuple2 = max(best, key=lambda x: x[2])
    return {"best_params": max_tuple2[:2], "best_performance": max_tuple2[2]}

def evaluate_strategy(strategy_code, short_period, long_period):
    # Dynamically create a strategy class from the user-provided code
    strategy_locals = {}
    exec(strategy_code, {}, strategy_locals)
    StrategyClass = strategy_locals['MovingAverageCrossoverStrategy']
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(StrategyClass, short_period=short_period, long_period=long_period)
    data = bt.feeds.YahooFinanceCSVData(dataname='../data/companyData/IND/DLF.BO.csv', 
                                        fromdate=datetime.datetime(2010, 1, 1), 
                                        todate=datetime.datetime(2022, 1, 1))
    cerebro.adddata(data)
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()
    return cerebro.broker.getvalue()


def select_best(results):
    # Sort the results by performance
    max_tuple = max(results, key=lambda x: x[2])
    best.append(max_tuple)
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    # Select the top half of the population
    return [x[:2] for x in sorted_results[:int(len(results) // 2)]]

def crossover_and_mutate(population, mutation_rate):
    next_generation = []
    for _ in range(len(population)):
        # Select two parents
        parent1, parent2 = random.sample(population, 2)
        # Perform crossover
        child = (parent1[0], parent2[1]) if random.random() < 0.5 else (parent2[0], parent1[1])
        # Perform mutation
        child = (int(child[0] + np.random.normal(0, mutation_rate)), 
                 int(child[1] + np.random.normal(0, mutation_rate)))
        next_generation.append(child)
    return next_generation

def display_results(results):
    # Display the best parameters and performance
    st.subheader("Optimization Results")
    st.write("Best Parameters:")
    st.write(results["best_params"])
    st.write("Final Portfolio Value: $%.2f" % results["best_performance"])

if __name__ == "__main__":
    main()

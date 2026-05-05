"""Genetic algorithm optimiser for SMA Crossover strategy parameters.

Uses stdlib ``random`` only — no external optimisation libraries required.
Each individual is ``(short_period, long_period)``.
Fitness = ``run_backtest(...).return_pct``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from quantquips.backtest_service import run_backtest


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    short_period: int
    long_period: int
    fitness: float = float("-inf")

    def is_valid(self) -> bool:
        return self.short_period >= 2 and self.long_period > self.short_period


@dataclass
class GaResult:
    best_short: int
    best_long: int
    best_return_pct: float
    generations_run: int
    population_history: list[dict] = field(default_factory=list)
    """Each entry: {generation, short_period, long_period, return_pct}."""


# ---------------------------------------------------------------------------
# GA internals
# ---------------------------------------------------------------------------

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _random_individual(short_range: tuple[int, int], long_range: tuple[int, int]) -> Individual:
    short = random.randint(*short_range)
    long = random.randint(*long_range)
    if long <= short:
        long = short + random.randint(1, max(1, (long_range[1] - short_range[0]) // 4))
    long = _clamp(long, long_range[0], long_range[1])
    return Individual(short_period=short, long_period=long)


def _crossover(
    parent_a: Individual,
    parent_b: Individual,
    short_range: tuple[int, int],
    long_range: tuple[int, int],
) -> tuple[Individual, Individual]:
    """Single-point crossover — swap short or long gene with 50 % probability each."""
    child_a = Individual(
        short_period=parent_a.short_period if random.random() < 0.5 else parent_b.short_period,
        long_period=parent_a.long_period if random.random() < 0.5 else parent_b.long_period,
    )
    child_b = Individual(
        short_period=parent_b.short_period if random.random() < 0.5 else parent_a.short_period,
        long_period=parent_b.long_period if random.random() < 0.5 else parent_a.long_period,
    )
    # Clamp and enforce short < long
    for child in (child_a, child_b):
        child.short_period = _clamp(child.short_period, *short_range)
        child.long_period = _clamp(child.long_period, *long_range)
        if child.long_period <= child.short_period:
            child.long_period = _clamp(child.short_period + 1, *long_range)
    return child_a, child_b


def _mutate(
    individual: Individual,
    mutation_rate: float,
    short_range: tuple[int, int],
    long_range: tuple[int, int],
) -> Individual:
    short = individual.short_period
    long = individual.long_period
    if random.random() < mutation_rate:
        delta = random.choice([-2, -1, 1, 2])
        short = _clamp(short + delta, *short_range)
    if random.random() < mutation_rate:
        delta = random.choice([-3, -2, -1, 1, 2, 3])
        long = _clamp(long + delta, *long_range)
    if long <= short:
        long = _clamp(short + 1, *long_range)
    return Individual(short_period=short, long_period=long)


def _tournament_select(population: list[Individual], k: int = 3) -> Individual:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness)


def _evaluate(
    individual: Individual,
    fitness_fn: Callable[[int, int], float],
) -> Individual:
    try:
        individual.fitness = fitness_fn(individual.short_period, individual.long_period)
    except Exception:
        individual.fitness = float("-inf")
    return individual


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ga_optimization(
    ticker: str,
    start: date,
    end: date,
    cash: float = 10_000.0,
    commission: float = 0.001,
    refresh_data: bool = True,
    population_size: int = 20,
    generations: int = 10,
    mutation_rate: float = 0.2,
    short_range: tuple[int, int] = (2, 50),
    long_range: tuple[int, int] = (5, 200),
    progress_callback: Callable[[int, int], None] | None = None,
    seed: int | None = None,
) -> GaResult:
    """Run a genetic algorithm over SMA Crossover parameters.

    Args:
        ticker: Ticker symbol.
        start: Backtest start date.
        end: Backtest end date.
        cash: Starting capital.
        commission: Per-trade commission rate.
        refresh_data: Whether to fetch fresh data from Yahoo Finance.
        population_size: Number of individuals per generation.
        generations: Number of evolutionary generations.
        mutation_rate: Per-gene mutation probability (0–1).
        short_range: Inclusive (min, max) for the short SMA period.
        long_range: Inclusive (min, max) for the long SMA period.
        progress_callback: Optional callable(current_gen, total_gens) for UI updates.
        seed: Random seed for reproducibility.

    Returns:
        GaResult with best parameters and full population history.
    """
    if seed is not None:
        random.seed(seed)

    def fitness_fn(short: int, long: int) -> float:
        result = run_backtest(
            ticker=ticker,
            strategy_name="SMA Crossover",
            start=start,
            end=end,
            cash=cash,
            commission=commission,
            strategy_params={"short_period": short, "long_period": long},
            refresh_data=refresh_data,
        )
        return result.return_pct

    # Initialise population
    population = [
        _random_individual(short_range, long_range)
        for _ in range(population_size)
    ]
    # Evaluate generation 0
    population = [_evaluate(ind, fitness_fn) for ind in population]

    history: list[dict] = []
    for ind in population:
        history.append(
            {
                "generation": 0,
                "short_period": ind.short_period,
                "long_period": ind.long_period,
                "return_pct": ind.fitness,
            }
        )

    if progress_callback:
        progress_callback(0, generations)

    for gen in range(1, generations + 1):
        # Build next generation via tournament selection + crossover + mutation
        next_pop: list[Individual] = []
        while len(next_pop) < population_size:
            parent_a = _tournament_select(population)
            parent_b = _tournament_select(population)
            child_a, child_b = _crossover(parent_a, parent_b, short_range, long_range)
            child_a = _mutate(child_a, mutation_rate, short_range, long_range)
            child_b = _mutate(child_b, mutation_rate, short_range, long_range)
            next_pop.extend([child_a, child_b])

        # Trim to exact population size
        next_pop = next_pop[:population_size]
        # Evaluate
        next_pop = [_evaluate(ind, fitness_fn) for ind in next_pop]
        # Elitism: keep single best from previous generation
        best_prev = max(population, key=lambda ind: ind.fitness)
        worst_idx = min(range(len(next_pop)), key=lambda i: next_pop[i].fitness)
        next_pop[worst_idx] = best_prev

        population = next_pop

        for ind in population:
            history.append(
                {
                    "generation": gen,
                    "short_period": ind.short_period,
                    "long_period": ind.long_period,
                    "return_pct": ind.fitness,
                }
            )

        if progress_callback:
            progress_callback(gen, generations)

    best = max(population, key=lambda ind: ind.fitness)
    return GaResult(
        best_short=best.short_period,
        best_long=best.long_period,
        best_return_pct=best.fitness,
        generations_run=generations,
        population_history=history,
    )

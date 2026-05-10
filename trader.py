from __future__ import annotations

import argparse
from datetime import date

from quantquips.backtest_service import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a QuantQuips backtest from the command line.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL or BHARTIARTL.NS")
    parser.add_argument("--strategy", default="Buy and Hold", choices=["Buy and Hold", "SMA Crossover"])
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 12, 31))
    parser.add_argument("--short-period", type=int, default=5)
    parser.add_argument("--long-period", type=int, default=20)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = {}
    if args.strategy == "SMA Crossover":
        params = {
            "short_period": args.short_period,
            "long_period": args.long_period,
        }

    result = run_backtest(
        ticker=args.ticker,
        strategy_name=args.strategy,
        start=args.start,
        end=args.end,
        cash=args.cash,
        commission=args.commission,
        strategy_params=params,
        refresh_data=args.refresh_data,
    )

    print(f"Ticker: {result.ticker}")
    print(f"Strategy: {result.strategy}")
    print(f"Initial Portfolio Value: ${result.starting_value:,.2f}")
    print(f"Ending Portfolio Value: ${result.ending_value:,.2f}")
    print(f"Profit: ${result.profit:,.2f}")
    print(f"Return: {result.return_pct:.2f}%")
    print(f"Closed Trades: {result.trade_count}")


if __name__ == "__main__":
    main()

# QuantQuips

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-informational.svg)](VERSION)

QuantQuips is a Streamlit-based backtesting workspace for traders, students, and developers who want to prototype algorithmic trading ideas in Python. It combines Backtrader strategy execution, genetic algorithm parameter search, market index charts, and a local LLM-assisted finance Q&A page.

> This project is an educational tool. It is not financial advice and should not be used as the only basis for live trading decisions.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Page Guide](#page-guide)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Home dashboard with Nifty 50 and Sensex price charts from Yahoo Finance.
- Backtesting page for running custom Backtrader strategies.
- Genetic Algorithm page for searching moving-average strategy parameters.
- LLM page for finance-related Q&A with local document retrieval.
- About page with project/team information.

## Demo

A video walkthrough is available here: [QuantQuips demo video](https://www.youtube.com/watch?v=HIcSWuKMwOw).

There is not currently a public live deployment. If you deploy one, replace the video-only reference above with the hosted Streamlit URL and add a screenshot in this section.

## Requirements

- Python 3.11
- pip
- macOS, Linux, or Windows with a shell that can activate a Python virtual environment
- Optional for the LLM page: a local CTransformers-compatible model file named `mistral-7b-openorca.Q4_0.gguf`
- Optional for the LLM page: PDF documents under `data/data/`
- Optional for CSV backtests: downloaded market CSV files under `data/companyData/`

The Python dependencies are pinned in [requirements.txt](requirements.txt).

## Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/KTS-o7/QuantQuips.git
   cd QuantQuips
   ```

2. Create and activate a virtual environment.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies.

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Add optional local assets if you plan to use the LLM or local CSV backtesting flows.

   ```text
   data/data/                         # PDF files for retrieval-augmented Q&A
   data/companyData/US/AAPL.csv       # Example local CSV path used by trader.py
   mistral-7b-openorca.Q4_0.gguf      # Local LLM model file
   ```

## Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

Deactivate the virtual environment when finished:

```bash
deactivate
```

## Page Guide

### Home

The Home page downloads current session data for:

- Nifty 50: `^NSEI`
- Sensex: `^BSESN`

It displays line charts, recent price rows, and a simple bullish/bearish market condition based on the current period's percentage change.

### Backtesting

Use the Backtesting page to paste a Backtrader strategy class named `TestStrategy`. The app writes the strategy to `strategies.py`, then runs `trader.py` with the initial principal amount entered in the UI.

Example strategy:

```python
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
```

Current data note: `trader.py` expects a local Yahoo Finance CSV at `data/companyData/US/AAPL.csv`. Update that path in `trader.py` or add the CSV before running this page.

### Genetic Algorithm

Use the Genetic Algorithm page to search for moving-average crossover parameters. Paste a strategy class named `MovingAverageCrossoverStrategy` that accepts `short_period` and `long_period` parameters, choose parameter ranges, enter initial cash, and provide a Yahoo Finance CSV path.

Example strategy:

```python
import backtrader as bt


class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ("short_period", 50),
        ("long_period", 200),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.short_period,
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.long_period,
        )
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.sell()
```

The page reports the best short period, long period, estimated result, and estimated profit from the generated population.

### LLM

The LLM page builds a retrieval chain over PDF files in `data/data/` and uses the local `mistral-7b-openorca.Q4_0.gguf` model through CTransformers.

Example prompts:

- `What is a genetic algorithm?`
- `How does a moving average crossover strategy work?`
- `What risks should I consider before backtesting a strategy?`

If the model file or PDF directory is missing, add those assets before using this page.

## Project Structure

```text
.
+-- app.py                         # Streamlit application and page routing
+-- trader.py                      # Backtrader runner used by the Backtesting page
+-- strategies.py                  # Strategy file overwritten by the Backtesting page
+-- requirements.txt               # Pinned Python dependencies
+-- data/
|   +-- TickerList/                # Static ticker lists
|   +-- companyData/collectData.py # Data collection helper
+-- VERSION                        # Project version
+-- LICENSE                        # MIT license
```

## Contributing

Contributions are welcome. For changes that affect behavior, include a short description of the scenario tested and any required local data/model assets.

Recommended workflow:

```bash
git checkout -b feature/your-change
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Before opening a pull request, confirm the affected page starts successfully and update this README if setup steps, data paths, or user workflows change.

## License

QuantQuips is released under the [MIT License](LICENSE).

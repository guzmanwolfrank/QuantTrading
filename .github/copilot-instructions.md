# AI Coding Agent Instructions for QuantTrading

## Project Overview
**QuantTrading** is a quantitative algorithmic backtesting suite for forex and equities trading. It contains multiple specialized backtesting engines and trading tools, focusing on finding exponential returns through strategy iteration and risk-adjusted performance analysis.

### Core Principle
> All backtests use **Open as entry and Close as exit** as long as the signal is triggered. Complex orders, size parlays, and stops are calculated for live deployment, not backtesting.

---

## Architecture

### Two Primary Backtest Engines

#### 1. **AlgoHaus Backtester** (`Algorithmic Backtests/BacktestEngine/`)
- **Purpose**: Professional-grade forex backtester with modern GUI
- **Tech Stack**: CustomTkinter UI, Pandas, Plotly, QuantStats
- **Key Files**: `algoHausBTE.py`, multiple `.ipynb` notebooks
- **Strategies Included**:
  - Opening Range Breakout (first 30-minute range trades)
  - VWAP Crossover (volume-weighted momentum)
  - Pivot Point Reversal (support/resistance bounces)
- **Output**: Interactive HTML reports with Plotly charts, equity curves, P&L analysis
- **Specialized for**: Forex trading with pip calculations, leverage, margin requirements

#### 2. **MatrixSys Trading System** (`Algorithmic Backtests/MatrixSys/`)
- **Purpose**: Modular, extensible backtesting framework
- **Structure**: Jupyter-based pipeline with separate concerns
  - `strategies/`: Trading strategy implementations
  - `utils/`: Data processing, metrics, visualization, reporting
  - `main.ipynb`: Orchestration notebook
- **Supported Strategies**:
  - Bollinger Bounce (band squeeze + RSI confirmation)
  - Moving Average Cross (fast/slow MA crossovers)
  - Support Resistance (level-based bounces)
- **Output**: HTML reports, performance dashboards, trade lists

### Data Flow Architecture
```
Data (CSV/Parquet/yfinance) 
  → DataProcessor.load_data()
  → Strategy.generate_signals()
  → PerformanceAnalyzer.calculate_metrics()
  → Visualizer.create_dashboard()
  → ReportGenerator.generate_report()
```

---

## Critical Developer Patterns

### Strategy Implementation Pattern
All strategies follow this interface in `MatrixSys/trading_system/strategies/`:

```python
class StrategyName:
    def __init__(self, param1=default, param2=default):
        self.name = "Strategy Name"
        # Store parameters
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator columns (RSI, BB, MA, etc.)"""
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL: Return df with columns:
        - 'Signal': 1 (long), -1 (short), 0 (flat)
        - 'Entry_Price': Entry price level
        - 'Stop_Loss': SL price level  
        - 'Take_Profit': TP price level
        """
        return df
    
    def _apply_position_management(self, df: pd.DataFrame) -> None:
        """Apply exit rules (timeout bars, etc.)"""
        pass
```

### Forex Calculator Pattern
For forex backtests (AlgoHaus), always use the `ForexCalculator` class:

```python
ForexCalculator.PIP_VALUES  # Hardcoded pip sizes by pair
ForexCalculator.USD_MAJORS  # Set of USD-denominated pairs
ForexCalculator.calculate_pip_value_in_usd()  # Gets pip value in USD
ForexCalculator.calculate_margin_required()  # Margin calc with leverage
```

**Pair naming convention**: `'EUR/USD'`, `'USD/JPY'` (slash-separated, specific case)

### Performance Metrics Pattern
The `PerformanceAnalyzer` class calculates:
- **Trading Metrics**: Win rate, profit factor, total trades
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Distribution Metrics**: Skewness, kurtosis
- **Drawdown Metrics**: Peak-to-trough analysis

All metrics assume **252 trading days/year** and **2% annual risk-free rate**.

---

## Essential Integration Points

### 1. Data Sources
- **yfinance**: Primary source for equity/forex data (EURUSD=X format for forex)
- **Oanda API**: Used in production data fetching (not in backtests)
- **CSV/Parquet**: Historical data stored in `data/` folders by pair
- **Column Names**: Backtests expect `Open`, `High`, `Low`, `Close`, `Volume` (capitalized)

### 2. Exit Logic (Critical!)
ALL backtests implement three exit conditions:
1. **Stop Loss (SL)**: Price breaches stop level
2. **Take Profit (TP)**: Price reaches take-profit level
3. **Timeout**: Held for N bars without hitting SL/TP

Example from AlgoHaus:
```python
if signal == 'BUY':
    if bar['low'] <= stop_level:
        exit_price = stop_level; exit_reason = 'SL'
    elif bar['high'] >= take_level:
        exit_price = take_level; exit_reason = 'TP'
```

### 3. HTML Report Generation
Both engines generate interactive HTML reports with:
- Plotly charts (equity curve, drawdown, P&L distribution)
- Trade table with entry/exit prices, times, P&L
- Performance metrics dashboard
- Recommendations based on metrics (low win rate, high drawdown, etc.)

---

## Common Workflows

### Adding a New Strategy
1. Create `strategies/my_strategy.py` in MatrixSys
2. Implement class with `calculate_indicators()` and `generate_signals()`
3. Return DataFrame with `Signal`, `Entry_Price`, `Stop_Loss`, `Take_Profit` columns
4. Add to `main.ipynb` strategy registry
5. Run backtest via `BacktestEngine.run_backtest(strategy_name="my_strategy")`

### Running a Backtest
```python
backtest = BacktestEngine(
    strategy_name="bollinger_bounce",
    symbol="EURUSD=X",
    start_date="2020-01-01",
    end_date="2024-01-03"
)
metrics, figures, report = backtest.run_backtest()
```

### Optimizing Parameters
MatrixSys strategies include `optimize_parameters()` method:
- Grids parameter ranges
- Evaluates Sharpe ratio for each combination
- Returns best parameters

---

## Project-Specific Conventions

### 1. Signal Values
- `1` = Long signal (buy)
- `-1` = Short signal (sell)
- `0` = No position (flat)

### 2. Risk Management
- **Position sizing**: Usually fixed unit size, occasionally volatility-adjusted
- **Leverage**: AlgoHaus supports 1-500x (hardcoded options)
- **Margin checks**: Skip trades if insufficient margin available

### 3. Data Validation
- Minimum 30 periods required for indicator calculation
- Missing values handled with forward-fill by default
- Volume validation for certain strategies

### 4. Timezone/Datetime
- All datetimes in UTC
- Backtests run on daily or intraday (minute) data
- Time-in-trade calculated in hours

---

## File Organization

```
QuantTrading/
├── Algorithmic Backtests/
│   ├── BacktestEngine/          # AlgoHaus GUI-based forex backtester
│   │   ├── algoHausBTE.py       # Main application
│   │   ├── *.ipynb              # Notebook versions
│   │   └── data/                # Forex historical data
│   └── MatrixSys/               # Modular backtesting framework
│       └── trading_system/
│           ├── main.ipynb       # Orchestration
│           ├── strategies/      # Strategy implementations
│           └── utils/           # Shared utilities
└── Tools/                       # Standalone trading tools
    ├── Wolf Pivot Calculator/   # NQ pivot calculation
    ├── Wolf RSI Filter/         # RSI < 30 screener
    └── Range/                   # Range analysis tools
```

---

## Testing & Validation

### Common Issues
1. **Look-ahead bias**: Ensure indicators use only past data (use `.shift()` or `.iloc[i-1]`)
2. **Data alignment**: After signal generation, check for NaN values in first N rows
3. **Slippage**: AlgoHaus estimates slippage using first 2 minutes of trading range
4. **Pip value mismatches**: Verify pair is in `ForexCalculator.PIP_VALUES` dict

### Quick Validation Checklist
- [ ] Signal column has only values: 1, -1, 0
- [ ] No future data used in indicator calculation
- [ ] Stop loss and take profit levels are set for all trades
- [ ] DataFrame has required columns after processing
- [ ] HTML report generates without errors

---

## Key Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
yfinance>=0.2.0
plotly>=5.13.0
customtkinter (AlgoHaus GUI)
scipy (performance metrics)
matplotlib/seaborn (visualizations)
```

---

## What to Avoid
- ❌ Using `Close` prices to generate signals that will be evaluated at the same bar
- ❌ Hardcoding pair-specific parameters; use dictionaries or strategy parameters
- ❌ Mixing different timeframes in a single backtest
- ❌ Assuming position exists without checking signal column
- ❌ Modifying original DataFrame; always `.copy()` first

---

## Recommended Reading Order
1. Main README.md (project overview)
2. MatrixSys Readme.md (modular framework design)
3. AlgoHaus README.md (forex-specific conventions)
4. A strategy implementation (`bollinger_bounce.py` or `moving_average_cross.py`)
5. `PerformanceAnalyzer` class for metrics understanding

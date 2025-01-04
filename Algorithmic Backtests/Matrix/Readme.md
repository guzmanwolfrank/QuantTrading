# Modular Trading System

A flexible and extensible backtesting framework for trading strategies with customizable components and interactive visualization.

## Features

- Modular strategy implementation
- Dynamic strategy loading
- Comprehensive performance metrics
- Interactive visualizations using Seaborn
- HTML report generation with mobile-friendly design
- Jupyter Lab integration
- Tkinter GUI interface
- Standalone executable support

## Project Structure

```
trading_system/
├── main.ipynb              # Main notebook for strategy execution
├── readme.md              # This documentation
├── requirements.txt       # Project dependencies
├── strategies/           # Trading strategies
│   ├── __init__.py
│   ├── bollinger_bounce.py
│   └── support_resistance.py
└── utils/               # Utility functions and classes
    ├── __init__.py
    ├── data_processor.py
    ├── performance_metrics.py
    ├── risk_manager.py
    ├── visualizer.py
    └── report_generator.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Lab Interface

1. Start Jupyter Lab:
```bash
jupyter lab
```

2. Open `main.ipynb` and run the backtest:
```python
backtest = BacktestEngine(
    strategy_name="bollinger_bounce",
    symbol="EURUSD=X",
    start_date="2020-01-01",
    end_date="2024-01-03"
)

metrics, figures, report = backtest.run_backtest()
```

### Adding New Strategies

1. Create a new strategy file in the `strategies` folder:
```python
# strategies/my_strategy.py

class MyStrategy:
    def __init__(self, param1=default1, param2=default2):
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, df):
        # Add your strategy logic here
        df['Signal'] = 0
        # Generate signals based on your strategy
        return df
```

2. Import and use your strategy:
```python
backtest = BacktestEngine(
    strategy_name="my_strategy",
    symbol="EURUSD=X",
    start_date="2020-01-01",
    end_date="2024-01-03"
)
```

## Performance Metrics

The system calculates various performance metrics including:

- Total and Annual Returns
- Sharpe and Sortino Ratios
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Win Rate
- Profit Factor
- Trade Statistics

## Visualization

The system generates several visualizations:

- Equity Curve
- Drawdown Chart
- Returns Distribution
- Trade Entry/Exit Points
- Performance Metrics Dashboard

## Report Generation

HTML reports include:

- Strategy Overview
- Performance Metrics
- Interactive Charts
- Trade List
- Risk Analytics
- Mobile-Friendly Design

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dependencies

- pandas
- numpy
- yfinance
- seaborn
- matplotlib
- scipy
- jupyterlab
- tkinter

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various open-source trading frameworks
- Built with Python and its amazing ecosystem

## TODO

- [ ] Add more strategy templates
- [ ] Implement portfolio backtesting
- [ ] Add machine learning integration
- [ ] Enhance risk management features
- [ ] Add real-time data support
- [ ] Implement strategy optimization
- [ ] Add more visualization options
- [ ] Create comprehensive documentation
- [ ] Add unit tests
- [ ] Implement paper trading mode

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
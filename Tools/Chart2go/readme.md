# NQ Futures Dashboard

A professional-grade, real-time dashboard for NASDAQ 100 E-mini Futures (NQ) analysis featuring dual candlestick charts, accurate trading metrics, and zero JavaScript dependencies.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-Minimal-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

![chart2go](https://github.com/user-attachments/assets/c14e8fc7-e41d-40e8-bf9b-d3b72d309e9f)


### Dual Chart System
- **Weekly Overview Chart** - Last 100 15-minute candles showing recent trend
- **Intraday Session Chart** - Today's complete trading session (8:30 AM - 4:00 PM)
- **Professional Layout** - Price axis on right, time axis at bottom (industry standard)

### Professional Trading Metrics
- **ATR (14-period)** - Average True Range with proper calculation
- **Volume Analysis** - Actual futures contracts traded (not dollar volume)
- **Range Analysis** - Daily, hourly, and 15-minute High-Low ranges
- **Moving Averages** - 20, 50, and 200-day simple moving averages

### Technical Implementation
- **Pure CSS Charts** - No JavaScript dependencies for maximum reliability
- **Real-time Data** - Yahoo Finance API integration with fallback sample data
- **Responsive Design** - Works perfectly on desktop and mobile devices
- **Interactive Tooltips** - Hover over candlesticks for detailed OHLC data

## 📊 Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NQ FUTURES DASHBOARD                        │
│                 NASDAQ 100 E-mini Futures                      │
├─────────────────────────────────┬───────────────────────────────┤
│                                 │  Current Price: $18,234.50   │
│        WEEKLY CHART             │  Volume: 1,247 contracts     │
│     (Last 100 candles)          │  ATR (14): 187.32 pts        │
│                                 │  Daily Range: 156.78 pts     │
│  [Candlestick Chart]           │  Hourly Range: 89.45 pts     │
│                                 │  15min Range: 23.67 pts      │
│                                 │                               │
│  MA20: $18,156 | MA50: $18,089  │                               │
├─────────────────────────────────────────────────────────────────┤
│                    TODAY'S INTRADAY SESSION                     │
│                  (8:30 AM - 4:00 PM EST)                       │
│                                                                 │
│                [Full Trading Day Chart]                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection for data fetching

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nq-futures-dashboard.git
   cd nq-futures-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install yfinance pandas numpy
   ```
   *Note: The script will auto-install missing dependencies*

3. **Run the dashboard**
   ```bash
   python nq_dashboard.py
   ```

4. **Open the generated HTML file**
   - The script will output a clickable file path
   - Open in any modern web browser
   - No server required - runs locally

## 📈 Technical Details

### Data Sources
- **Primary**: Yahoo Finance API (`yfinance`)
- **Ticker**: `NQ=F` (NASDAQ 100 E-mini Futures)
- **Intervals**: 15-minute, daily
- **Fallback**: Realistic sample data when API unavailable

### Calculations

#### ATR (Average True Range)
```python
True Range = Max(
    High - Low,
    |High - Previous Close|,
    |Low - Previous Close|
)
ATR = 14-period Simple Moving Average of True Range
```

#### Range Calculations
- **Daily Range**: 20-day average of (High - Low) from daily data
- **Hourly Range**: Max High - Min Low of last 4 fifteen-minute periods
- **15min Range**: Current candle High - Low

#### Volume
- Actual futures contracts traded (not notional dollar volume)
- Represents real market participation and liquidity

### Chart Features
- **Candlestick Colors**: Green (bullish), Red (bearish)
- **Price Scaling**: Dynamic based on data range
- **Time Labels**: Smart positioning to avoid crowding
- **Hover Tooltips**: OHLC data for each period

## 🎨 Customization

### Modify Chart Appearance
Edit the CSS section in `generate_html()` function:

```python
# Change candlestick colors
color = "#your_color" if is_bullish else "#your_color"

# Adjust chart height
height: 500px;  # Change this value

# Modify color scheme
background: #your_background_color;
```

### Add New Metrics
Extend the dashboard by adding functions in the metrics section:

```python
def calculate_new_metric(data):
    # Your calculation here
    return result

# Add to stats panel in HTML template
```

### Change Time Periods
Modify data fetching parameters:

```python
# For different time ranges
minute_data = ticker_obj.history(period="5d", interval="15m")  # 5 days
daily_data = ticker_obj.history(period="2y", interval="1d")    # 2 years
```

## 📱 Browser Compatibility

- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Safari 12+
- ✅ Edge 79+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## 🔧 Troubleshooting

### Common Issues

**"Too Many Requests" Error**
```
Error fetching data: Too Many Requests. Rate limited.
```
*Solution: Wait a few minutes and try again. The script will use sample data automatically.*

**Missing Dependencies**
```
ModuleNotFoundError: No module named 'yfinance'
```
*Solution: Run `pip install yfinance pandas numpy` or let the script auto-install.*

**Charts Not Displaying**
- Ensure you're opening the HTML file in a web browser
- Check that the file path doesn't contain special characters
- Try a different browser if issues persist

## 📊 Sample Output

The dashboard generates timestamped HTML files:
```
nq_dashboard_20240610_153045.html
```

Each file is completely self-contained with:
- All CSS styling embedded
- No external dependencies
- Responsive design
- Professional appearance

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-indicator`
3. **Make your changes**: Add new indicators, improve styling, fix bugs
4. **Test thoroughly**: Ensure all functionality works
5. **Submit a pull request**: Describe your changes clearly

### Ideas for Contributions
- Additional technical indicators (RSI, MACD, Bollinger Bands)
- Multiple timeframe analysis
- Alert system for price levels
- Export functionality (PDF, PNG)
- Real-time auto-refresh capability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This dashboard is for educational and informational purposes only. It is not financial advice. Trading futures involves substantial risk and is not suitable for all investors. Past performance is not indicative of future results.

## 🏗️ Architecture

```
nq_dashboard.py
├── Data Fetching (Yahoo Finance API)
├── Technical Calculations (ATR, Ranges, Moving Averages)
├── Chart Generation (Pure CSS Candlesticks)
├── HTML Template (Responsive Design)
└── File Output (Self-contained HTML)
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nq-futures-dashboard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/guzmanwolfrank/nq-futures-dashboard/discussions)
- **Email**: wolfrankny@gmail.com

## 🌟 Acknowledgments

- Yahoo Finance for providing free market data
- The Python community for excellent libraries
- Professional traders who inspired the metrics selection

---

**Made with ❤️ for the trading community**

*If this project helped you, please consider giving it a ⭐ on GitHub!*

# HTML Trading Journal

A Python-based trading journal application with Quantower API integration that generates interactive HTML reports for trade analysis and performance tracking.

## Features

- **Quantower Integration**: Connect directly to Quantower trading platform via API
- **Interactive GUI**: User-friendly Tkinter interface for easy trade management
- **HTML Report Generation**: Create detailed, interactive HTML reports of trading performance
- **Trade Import/Export**: Import trades from Quantower or manually input trade data
- **Performance Analytics**: Track P&L, win rates, and other key trading metrics
- **Date Range Filtering**: Analyze trades within specific time periods
- **Offline Capability**: Works without internet connection for local trade analysis

## Requirements

- Python 3.7+
- tkinter (usually included with Python)
- requests
- pathlib (included with Python 3.4+)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/html-trading-journal.git
cd html-trading-journal
```

2. Install required dependencies:
```bash
pip install requests
```

3. Run the application:
```bash
python trading_journal.py
```

## Quantower API Setup

1. **Enable API in Quantower**:
   - Open Quantower platform
   - Navigate to Settings > Connections > API
   - Enable API access and note the host/port (default: localhost:8080)

2. **Configure Connection**:
   - Launch the trading journal application
   - Go to Settings > API Configuration
   - Enter your Quantower API details:
     - Host: `localhost` (or your Quantower server IP)
     - Port: `8080` (default)
     - API Key: (if required by your setup)

3. **Test Connection**:
   - Click "Test Connection" to verify the API is working
   - If successful, you can now import trades directly from Quantower

## Usage

### Connecting to Quantower
1. Start Quantower platform with API enabled
2. Launch the trading journal application
3. Configure API settings in the application
4. Test the connection to ensure proper setup

### Importing Trades
- **From Quantower**: Use the "Import from Quantower" button to fetch trades automatically
- **Manual Entry**: Add trades manually using the trade entry form
- **Date Range**: Specify date ranges for targeted trade imports

### Generating Reports
1. Select the trades you want to include in the report
2. Choose date range (optional)
3. Click "Generate HTML Report"
4. The interactive HTML report will open in your default browser

### Report Features
- Trade summary and statistics
- P&L charts and visualizations
- Individual trade details
- Performance metrics (win rate, average trade, etc.)
- Exportable format for sharing or archiving

## Configuration

The application stores configuration in a local JSON file including:
- API connection settings
- Default report preferences
- Trade categories and tags
- Export/import paths

## API Reference

### QuantowerAPI Class Methods

```python
# Initialize API connection
api = QuantowerAPI()

# Set connection credentials
api.set_credentials(api_key="your_key", host="localhost", port="8080")

# Test connection
is_connected = api.test_connection()

# Fetch trades
trades = api.get_trades(start_date=datetime.now() - timedelta(days=30))
```

## Troubleshooting

### Common Issues

**Connection Failed**:
- Ensure Quantower is running and API is enabled
- Check firewall settings
- Verify host/port configuration

**No Trades Imported**:
- Check date range settings
- Ensure trades exist in the specified period
- Verify API permissions

**HTML Report Not Generated**:
- Check write permissions in output directory
- Ensure sufficient disk space
- Verify trade data is not empty

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and analysis purposes only. Trading involves substantial risk of loss. The authors are not responsible for any trading decisions made using this software.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [Wiki](../../wiki) for detailed documentation
- Review existing issues for similar problems

## Roadmap

- [ ] Support for additional trading platforms
- [ ] Advanced charting capabilities
- [ ] Machine learning performance predictions
- [ ] Mobile-responsive HTML reports
- [ ] Database integration for large datasets
- [ ] Real-time trade monitoring

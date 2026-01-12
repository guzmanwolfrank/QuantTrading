# Wolf's AlgoHaus Backtester v6.0 - Professional Edition
# Created for: Wolf Guzman
# Features: Real Data Validation, Risk Management, QuantStats Analysis, Retro DOS Metrics
# Professional Forex Backtesting with Parquet Data

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import webbrowser
import os
import tempfile
import inspect
import pathlib
import threading
import queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ======================================================================
# 1. FOREX CALCULATOR
# ======================================================================
class ForexCalculator:
    """Handle all forex calculations - Wolf Guzman's Trading System"""
   
    LEVERAGE_OPTIONS = [1, 10, 20, 30, 50, 100, 200, 500]
   
    PIP_VALUES = {
        # Major Pairs
        'EUR/USD': 0.0001, 'GBP/USD': 0.0001, 'USD/JPY': 0.01,
        'USD/CHF': 0.0001, 'USD/CAD': 0.0001, 'AUD/USD': 0.0001,
        'NZD/USD': 0.0001,
        # Cross Pairs - JPY
        'EUR/JPY': 0.01, 'GBP/JPY': 0.01, 'AUD/JPY': 0.01,
        'NZD/JPY': 0.01, 'CHF/JPY': 0.01, 'CAD/JPY': 0.01,
        # Cross Pairs - EUR
        'EUR/GBP': 0.0001, 'EUR/CHF': 0.0001, 'EUR/AUD': 0.0001,
        # Cross Pairs - GBP
        'GBP/CHF': 0.0001,
        # Cross Pairs - AUD
        'AUD/CHF': 0.0001, 'AUD/NZD': 0.0001,
        # Cross Pairs - NZD
        'NZD/CHF': 0.0001,
        # Cross Pairs - CAD
        'CAD/CHF': 0.0001,
        # Exotic
        'USD/THB': 0.01
    }
    
    # Hardcoded date ranges from Wolf's data for faster loading
    DATA_RANGES = {
        'AUD/CHF': ('2014-12-09', '2025-10-06'),
        'AUD/JPY': ('2014-12-10', '2025-10-06'),
        'AUD/NZD': ('2014-12-10', '2025-10-06'),
        'AUD/USD': ('2014-12-09', '2025-10-06'),
        'CAD/CHF': ('2014-12-10', '2025-10-06'),
        'CHF/JPY': ('2014-12-08', '2025-10-06'),
        'EUR/AUD': ('2014-12-09', '2025-10-06'),
        'EUR/CHF': ('2014-12-10', '2025-10-06'),
        'EUR/GBP': ('2014-12-08', '2025-10-06'),
        'EUR/JPY': ('2014-12-10', '2025-10-06'),
        'EUR/USD': ('2014-12-09', '2025-10-06'),
        'GBP/CHF': ('2014-12-10', '2025-10-06'),
        'GBP/USD': ('2014-12-10', '2025-10-06'),
        'NZD/CHF': ('2014-12-09', '2025-10-06'),
        'NZD/JPY': ('2014-12-08', '2025-10-06'),
        'NZD/USD': ('2014-12-10', '2025-10-06'),
        'USD/CAD': ('2014-12-10', '2025-10-06'),
        'USD/CHF': ('2014-12-09', '2025-10-06'),
        'USD/JPY': ('2014-12-09', '2025-10-06'),
        'USD/THB': ('2014-12-03', '2025-10-06'),
    }
    
    USD_MAJORS = {
        'EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD', 
        'USD/JPY', 'USD/CHF', 'USD/CAD', 'USD/THB'
    }
    
    @staticmethod
    def is_usd_major(pair):
        return pair in ForexCalculator.USD_MAJORS
   
    @staticmethod
    def calculate_pip_value_in_usd(pair, unit_size, current_price, conversion_rate=1.0):
        """Calculate pip value in USD for given position size"""
        pip_size = ForexCalculator.PIP_VALUES.get(pair, 0.0001) 
        
        if pair.endswith('/USD'):
            # Direct quote: EUR/USD, GBP/USD, etc.
            pip_value = pip_size * unit_size
        elif pair.startswith('USD/'):
            # Indirect quote: USD/JPY, USD/CHF, etc.
            pip_value = (pip_size / current_price) * unit_size
        else:
            # Cross pair: EUR/GBP, GBP/JPY, etc.
            pip_value_in_base = pip_size * unit_size
            pip_value = pip_value_in_base * conversion_rate 
        
        return pip_value
   
    @staticmethod
    def calculate_margin_required(pair, unit_size, current_price, leverage, conversion_rate=1.0):
        """Calculate margin required for position"""
        if pair.endswith('/USD'):
            position_value = unit_size * current_price
        elif pair.startswith('USD/'):
            position_value = unit_size
        else:
            position_value = unit_size * conversion_rate 
        
        margin_required = position_value / leverage
        return margin_required
    
    @staticmethod
    def calculate_position_size(balance, risk_pct, sl_pips, pair, price, conversion_rate=1.0):
        """Calculate position size based on risk percentage"""
        risk_amount = balance * (risk_pct / 100)
        pip_val_per_unit = ForexCalculator.calculate_pip_value_in_usd(pair, 1, price, conversion_rate)
        if pip_val_per_unit <= 0:
            return 0
        size = risk_amount / (sl_pips * pip_val_per_unit)
        return max(1000, int(round(size / 1000)) * 1000)

# ======================================================================
# 2. DATA LOADING WITH VALIDATION
# ======================================================================
def detect_available_pairs(base_folder: pathlib.Path):
    """Scan for available forex pair folders and return valid pairs"""
    pairs = set()
    
    # Check if base folder exists
    if not base_folder.exists():
        logging.error(f"Base folder does not exist: {base_folder}")
        return []
    
    logging.info(f"Scanning for pairs in: {base_folder}")
    
    # Scan for subfolder names like EUR_USD, GBP_USD, etc.
    for subfolder in base_folder.iterdir():
        if subfolder.is_dir() and subfolder.name not in ['README.TXT', '__pycache__']:
            folder_name = subfolder.name
            # Check if folder name matches pair pattern (XXX_XXX)
            if '_' in folder_name and len(folder_name.split('_')) == 2:
                parts = folder_name.split('_')
                # Verify both parts are 3 letters (currency codes)
                if len(parts[0]) == 3 and len(parts[1]) == 3:
                    # Verify folder contains parquet files
                    parquet_files = list(subfolder.glob("*.parquet"))
                    if parquet_files:
                        pair = folder_name.replace('_', '/')
                        pairs.add(pair)
                        logging.info(f"Found pair: {pair} with {len(parquet_files)} parquet file(s)")
                    else:
                        logging.warning(f"Folder {folder_name} has no parquet files")
    
    if not pairs:
        logging.warning("No valid pairs found!")
    else:
        logging.info(f"Total pairs found: {len(pairs)}")
    
    return sorted(list(pairs))

def get_data_date_range(pair_name: str, base_folder: pathlib.Path):
    """Get actual date range from parquet file in pair's subfolder"""
    
    # First, try hardcoded values for instant response
    if pair_name in ForexCalculator.DATA_RANGES:
        start_str, end_str = ForexCalculator.DATA_RANGES[pair_name]
        from datetime import datetime
        start = datetime.strptime(start_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_str, '%Y-%m-%d').date()
        logging.info(f"{pair_name}: Using cached date range {start} to {end}")
        return start, end
    
    # Fallback: read from file (slower but accurate if data updated)
    try:
        # Convert pair name to folder name (EUR/USD -> EUR_USD)
        pair_folder_name = pair_name.replace('/', '_')
        pair_folder = base_folder / pair_folder_name
        
        # Check if pair folder exists
        if not pair_folder.exists() or not pair_folder.is_dir():
            logging.warning(f"Folder not found: {pair_folder}")
            return None, None
        
        # Get parquet files from pair's folder
        parquet_files = list(pair_folder.glob("*.parquet"))
        
        if not parquet_files:
            logging.warning(f"No parquet files in {pair_folder}")
            return None, None
       
        df = pd.read_parquet(parquet_files[0], engine='pyarrow')
        
        # Find datetime column
        datetime_col = None
        for col in df.columns:
            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
            df = df.dropna(subset=[datetime_col])
            df[datetime_col] = df[datetime_col].dt.tz_localize(None)
            start = df[datetime_col].min().date()
            end = df[datetime_col].max().date()
            logging.info(f"{pair_name}: Data from {start} to {end} (read from file)")
            return start, end
        
        return None, None
    except Exception as e:
        logging.error(f"Error getting date range for {pair_name}: {e}")
        return None, None

def load_pair_data(pair_name: str, base_folder: pathlib.Path, start_date: datetime, end_date: datetime, timeframe: str):
    """Load and validate parquet data from pair-specific subfolder"""
    
    # Convert pair name to folder name (EUR/USD -> EUR_USD)
    pair_folder_name = pair_name.replace('/', '_')
    
    # Construct the full path to the pair's folder
    pair_folder = base_folder / pair_folder_name
    
    logging.info(f"=" * 50)
    logging.info(f"Loading pair: {pair_name}")
    logging.info(f"Base folder: {base_folder}")
    logging.info(f"Pair folder name: {pair_folder_name}")
    logging.info(f"Full pair folder path: {pair_folder}")
    logging.info(f"Pair folder exists: {pair_folder.exists()}")
    logging.info(f"Is directory: {pair_folder.is_dir() if pair_folder.exists() else 'N/A'}")
    
    # Check if pair folder exists
    if not pair_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {pair_folder}")
    
    if not pair_folder.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {pair_folder}")
    
    # Get all parquet files in the pair's folder
    parquet_files = list(pair_folder.glob("*.parquet"))
    logging.info(f"Parquet files found: {len(parquet_files)}")
    
    if not parquet_files:
        # List what IS in the folder
        folder_contents = list(pair_folder.iterdir())
        logging.error(f"Folder contents: {[f.name for f in folder_contents]}")
        raise FileNotFoundError(f"No PARQUET files found in {pair_folder}")
    
    # Use the first parquet file
    parquet_path = parquet_files[0]
    logging.info(f"Loading file: {parquet_path.name}")
    
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    if df.empty:
        raise ValueError(f"PARQUET file is empty: {parquet_path}")
   
    # Map columns
    cols_lower = [c.strip().lower() for c in df.columns]
    col_map = {
        'datetime': ['datetime', 'date', 'time', 'timestamp', 'date_time', 'index'],
        'open': ['open', 'o'],
        'high': ['high', 'h'],
        'low': ['low', 'l'],
        'close': ['close', 'c', 'last'],
        'volume': ['volume', 'vol', 'v']
    }
   
    rename = {}
    for target, aliases in col_map.items():
        for alias in aliases:
            if any(alias in col for col in cols_lower):
                orig = next(col for col in df.columns if alias in col.lower())
                rename[orig] = target
                break
        else:
            if target != 'volume':
                raise KeyError(f"Column for '{target}' not found in {parquet_path}")
   
    df = df.rename(columns=rename)
    
    if 'volume' not in df.columns:
        df['volume'] = 1000
   
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['datetime'])
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    df = df.drop_duplicates(subset='datetime', keep='first')
    df = df.sort_values('datetime')
   
    # Get actual data range
    actual_start = df['datetime'].min().date()
    actual_end = df['datetime'].max().date()
    
    logging.info(f"Data range: {actual_start} to {actual_end}")
    
    # Validate user date range
    if start_date.date() < actual_start:
        logging.warning(f"Start date {start_date.date()} before data start {actual_start}, adjusting to {actual_start}")
        start_date = datetime.combine(actual_start, datetime.min.time())
    
    if end_date.date() > actual_end:
        logging.warning(f"End date {end_date.date()} after data end {actual_end}, adjusting to {actual_end}")
        end_date = datetime.combine(actual_end, datetime.min.time())
   
    df = df.set_index('datetime')
    user_start = max(pd.Timestamp(start_date.date()), df.index.min())
    user_end = min(pd.Timestamp(end_date.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59), df.index.max())
    df = df.loc[user_start:user_end].copy()
   
    if df.empty:
        raise ValueError(f"No data in range {start_date.date()} to {end_date.date()}")
   
    # Resample if needed
    if timeframe != '1min':
        rule = {'5min': '5T', '15min': '15T', '1hr': '1H', '1Day': '1D'}.get(timeframe, '1T')
        logging.info(f"Resampling to {timeframe}")
        df = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
   
    df = df.reset_index()
    df['date'] = df['datetime'].dt.date
   
    # Calculate previous day levels
    daily = df.groupby('date').agg({
        'high': 'max', 'low': 'min', 'close': 'last'
    })
    daily.columns = ['day_high', 'day_low', 'day_close']
    daily['prev_high'] = daily['day_high'].shift(1)
    daily['prev_low'] = daily['day_low'].shift(1)
    daily['prev_close'] = daily['day_close'].shift(1)
   
    daily = daily.reset_index()
    df = pd.merge(df, daily[['date', 'prev_high', 'prev_low', 'prev_close']], on='date', how='left')
    df[['prev_high', 'prev_low', 'prev_close']] = df[['prev_high', 'prev_low', 'prev_close']].ffill()
   
    logging.info(f"Loaded {len(df)} bars")
    return df, actual_start, actual_end

# ======================================================================
# 3. TRADING STRATEGIES
# ======================================================================
class TradingStrategies:
    @staticmethod
    def vwap_crossover_strategy(df, sl_pips, tp_pips, pip_value):
        """VWAP Crossover Strategy - Wolf Guzman"""
        df = df.copy()
        df['tpv'] = df['volume'] * (df['high'] + df['low'] + df['close']) / 3
        df['cumvol'] = df.groupby('date')['volume'].cumsum()
        df['cumtpv'] = df.groupby('date')['tpv'].cumsum()
        df['vwap'] = df['cumtpv'] / df['cumvol']
        
        df['prev_close'] = df['close'].shift(1)
        df['prev_vwap'] = df['vwap'].shift(1)
        
        # FIXED: Use pandas operations to avoid numpy dtype promotion issues
        df['signal'] = None
        buy_condition = (df['prev_close'] <= df['prev_vwap']) & (df['close'] > df['vwap'])
        sell_condition = (df['prev_close'] >= df['prev_vwap']) & (df['close'] < df['vwap'])
        df.loc[buy_condition, 'signal'] = 'BUY'
        df.loc[sell_condition, 'signal'] = 'SELL'
        
        entries = df[df['signal'].notna()].copy()
        trades = []
        
        for idx, row in entries.iterrows():
            remaining_data = df[df.index > idx].reset_index(drop=True)
            if len(remaining_data) > 0:
                trades.append({
                    'datetime': row['datetime'],
                    'entry_price': row['close'],
                    'signal': row['signal'],
                    'day_data': remaining_data
                })
        
        return trades
    
    @staticmethod
    def opening_range_strategy(df, sl_pips, tp_pips, pip_value):
        """Opening Range Breakout Strategy - Wolf Guzman"""
        df = df.copy()
        trades = []
        
        for date in df['date'].unique():
            day_data = df[df['date'] == date].reset_index(drop=True)
            if len(day_data) < 31: 
                continue
            
            opening_range = day_data.iloc[:30]
            or_high = opening_range['high'].max()
            or_low = opening_range['low'].min()
            or_mid = (or_high + or_low) / 2
            
            if len(day_data) > 30:
                entry_bar = day_data.iloc[30]
                signal = 'BUY' if entry_bar['close'] > or_mid else 'SELL'
                
                trades.append({
                    'datetime': entry_bar['datetime'],
                    'entry_price': entry_bar['close'],
                    'signal': signal,
                    'day_data': day_data[31:].reset_index(drop=True)
                })
        
        return trades

# ======================================================================
# 4. ENHANCED BACKTESTER
# ======================================================================
class EnhancedBacktester:
    def __init__(self, df, initial_balance=10000, pip_value=0.0001, leverage=50, risk_percent=1.0):
        self.df = df
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.leverage = leverage
        self.risk_percent = risk_percent
        self.results = None
        
    def run_backtest(self, strategy_func, sl_pips, tp_pips, pair_name):
        """Run backtest with proper PnL calculation and progress updates"""
        logging.info(f"Generating trades using {strategy_func.__name__}...")
        trades = strategy_func(self.df, sl_pips, tp_pips, self.pip_value)
        
        if not trades:
            self.results = pd.DataFrame()
            return "No trades generated.", {}
        
        logging.info(f"Processing {len(trades)} potential trades...")
        results = []
        current_balance = self.initial_balance
        trade_number = 1
        
        is_usd_major = ForexCalculator.is_usd_major(pair_name)
        
        # Process trades in batches for progress reporting
        batch_size = max(1, len(trades) // 10)  # 10 progress updates
        
        for idx, t in enumerate(trades):
            if idx % batch_size == 0:
                logging.info(f"Processing trade {idx}/{len(trades)} ({idx*100//len(trades)}%)")
            
            entry_price = t['entry_price']
            signal = t['signal']
            bars = t['day_data']
            
            if bars.empty:
                continue
            
            # Calculate position size based on risk
            unit_size = ForexCalculator.calculate_position_size(
                current_balance, self.risk_percent, sl_pips, pair_name, entry_price
            )
            
            if unit_size < 1000:
                continue
            
            # Calculate margin
            margin_required = ForexCalculator.calculate_margin_required(
                pair_name, unit_size, entry_price, self.leverage, 1.0
            )
            
            if margin_required > current_balance * 0.8:
                continue
            
            # Calculate pip value for this position
            pip_value_usd = ForexCalculator.calculate_pip_value_in_usd(
                pair_name, unit_size, entry_price, 1.0
            )
            
            # Set stop and target levels
            if signal == 'BUY':
                stop_level = entry_price - (sl_pips * self.pip_value)
                take_level = entry_price + (tp_pips * self.pip_value)
            else:
                stop_level = entry_price + (sl_pips * self.pip_value)
                take_level = entry_price - (tp_pips * self.pip_value)
            
            # Find exit - optimized with early break
            exit_idx = None
            exit_reason = 'Timeout'
            exit_price = None
            
            for i, (idx_val, bar) in enumerate(bars.iterrows()):
                if signal == 'BUY':
                    if bar['low'] <= stop_level:
                        exit_idx = i
                        exit_price = stop_level
                        exit_reason = 'SL'
                        break
                    elif bar['high'] >= take_level:
                        exit_idx = i
                        exit_price = take_level
                        exit_reason = 'TP'
                        break
                else:  # SELL
                    if bar['high'] >= stop_level:
                        exit_idx = i
                        exit_price = stop_level
                        exit_reason = 'SL'
                        break
                    elif bar['low'] <= take_level:
                        exit_idx = i
                        exit_price = take_level
                        exit_reason = 'TP'
                        break
            
            if exit_price is None:
                exit_idx = len(bars) - 1
                exit_price = bars.iloc[-1]['close']
                exit_reason = 'Timeout'
            
            # Calculate PnL correctly
            if signal == 'BUY':
                pips_pnl = (exit_price - entry_price) / self.pip_value
            else:
                pips_pnl = (entry_price - exit_price) / self.pip_value
            
            monetary_pnl = pips_pnl * pip_value_usd
            
            entry_time = t['datetime']
            exit_time = bars.iloc[exit_idx]['datetime'] if exit_idx is not None else bars.iloc[-1]['datetime']
            
            time_in_trade = exit_time - entry_time
            hours_in_trade = time_in_trade.total_seconds() / 3600
            current_balance += monetary_pnl
            
            results.append({
                'trade_number': f"{trade_number:05d}",
                'entry_time': entry_time,
                'exit_time': exit_time,
                'time_in_trade_hours': round(hours_in_trade, 2),
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pips_pnl': round(pips_pnl, 2),
                'monetary_pnl': round(monetary_pnl, 2),
                'unit_size': unit_size,
                'margin_used': round(margin_required, 2),
                'balance': round(current_balance, 2),
                'pip_value_usd': round(pip_value_usd, 4)
            })
            
            trade_number += 1
        
        logging.info(f"Backtest complete: {len(results)} trades executed")
        self.results = pd.DataFrame(results)
        
        if not self.results.empty:
            total_pnl = self.results['monetary_pnl'].sum()
            total_pips = self.results['pips_pnl'].sum()
            win_rate = (self.results['pips_pnl'] > 0).mean() * 100
            
            summary = (f"TRADES: {len(self.results)}\n"
                      f"WIN RATE: {win_rate:.1f}%\n"
                      f"P&L: ${total_pnl:,.2f}\n"
                      f"PIPS: {total_pips:,.1f}\n"
                      f"FINAL BALANCE: ${current_balance:,.2f}")
            
            if not is_usd_major:
                summary += "\n⚠️ Cross pair - approximate pip values"
            
            metrics = self.calculate_metrics()
        else:
            summary = "No trades executed"
            metrics = {}
        
        return summary, metrics
    
    def calculate_metrics(self):
        """Calculate comprehensive trading metrics"""
        trades_df = self.results
        if trades_df.empty:
            return {}
        
        total_trades = len(trades_df)
        winning_trades = (trades_df['monetary_pnl'] > 0).sum()
        losing_trades = (trades_df['monetary_pnl'] < 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['monetary_pnl'].sum()
        total_pips = trades_df['pips_pnl'].sum()
        
        wins = trades_df[trades_df['monetary_pnl'] > 0]['monetary_pnl']
        losses = trades_df[trades_df['monetary_pnl'] < 0]['monetary_pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = abs(losses.min()) if len(losses) > 0 else 0
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        final_balance = trades_df['balance'].iloc[-1]
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Equity curve and drawdown
        equity_curve = self.initial_balance + trades_df['monetary_pnl'].cumsum()
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()
        
        # Risk metrics
        if len(trades_df) > 1:
            daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['monetary_pnl'].sum()
            daily_returns_pct = daily_returns / self.initial_balance
            
            sharpe = (daily_returns_pct.mean() * 252) / (daily_returns_pct.std() * np.sqrt(252)) if daily_returns_pct.std() > 0 else 0
            
            negative_returns = daily_returns_pct[daily_returns_pct < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino = (daily_returns_pct.mean() * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            # Best trading day
            best_day = daily_returns.max()
            best_day_date = daily_returns.idxmax()
        else:
            sharpe = 0
            sortino = 0
            best_day = 0
            best_day_date = None
        
        # Consecutive wins/losses
        trades_df['win'] = trades_df['monetary_pnl'] > 0
        trades_df['streak'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['win']].groupby('streak').size()
        loss_streaks = trades_df[~trades_df['win']].groupby('streak').size()
        
        max_consecutive_wins = win_streaks.max() if len(win_streaks) > 0 else 0
        max_consecutive_losses = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_%': round(win_rate, 2),
            'total_pnl_$': round(total_pnl, 2),
            'total_pips': round(total_pips, 1),
            'avg_win_$': round(avg_win, 2),
            'avg_loss_$': round(avg_loss, 2),
            'largest_win_$': round(largest_win, 2),
            'largest_loss_$': round(largest_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_%': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'total_return_%': round(total_return, 2),
            'final_balance_$': round(final_balance, 2),
            'best_day_$': round(best_day, 2) if best_day_date else 0,
            'best_day_date': str(best_day_date) if best_day_date else 'N/A',
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            'avg_time_in_trade_hrs': round(trades_df['time_in_trade_hours'].mean(), 2)
        }

# ======================================================================
# 5. HTML REPORT GENERATOR - QUANTSTATS PROFESSIONAL STYLE
# ======================================================================
class HTMLReportGenerator:
    @staticmethod
    def generate_report(metrics, trades_df, strategy_name, timeframe, pair, initial_balance, 
                       leverage, sl_pips, tp_pips, risk_pct, start_date, end_date, df=None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        # Generate all charts
        charts_html = HTMLReportGenerator.generate_quantstats_charts(trades_df, initial_balance, metrics)
        
        # Metrics table
        metrics_table = HTMLReportGenerator.generate_metrics_table(metrics, initial_balance)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wolf Guzman - {pair} Backtest Report</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background: #ffffff; 
                    color: #2c3e50; 
                    line-height: 1.6;
                    padding: 0;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px 20px;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{ 
                    font-size: 32px; 
                    margin-bottom: 10px;
                    font-weight: 600;
                }}
                
                .header .subtitle {{
                    font-size: 16px;
                    opacity: 0.9;
                    margin: 5px 0;
                }}
                
                .parameters {{
                    background: #f8f9fa;
                    border-left: 4px solid #667eea;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 5px;
                }}
                
                .parameters h3 {{
                    color: #2c3e50;
                    margin-bottom: 15px;
                    font-size: 18px;
                    font-weight: 600;
                }}
                
                .param-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                }}
                
                .param-item {{
                    color: #495057;
                    font-size: 14px;
                    padding: 10px;
                    background: white;
                    border-radius: 5px;
                    border: 1px solid #e9ecef;
                }}
                
                .param-label {{
                    font-weight: 600;
                    color: #667eea;
                    display: block;
                    margin-bottom: 5px;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                
                .metrics-overview {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .metrics-overview h2 {{
                    color: #2c3e50;
                    font-size: 24px;
                    margin-bottom: 20px;
                    font-weight: 600;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                
                .metric-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                    transition: transform 0.2s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                
                .metric-label {{
                    color: #6c757d;
                    font-size: 12px;
                    text-transform: uppercase;
                    font-weight: 600;
                    margin-bottom: 8px;
                }}
                
                .metric-value {{
                    font-size: 24px;
                    font-weight: 700;
                    color: #2c3e50;
                }}
                
                .metric-value.positive {{ color: #28a745; }}
                .metric-value.negative {{ color: #dc3545; }}
                .metric-value.neutral {{ color: #667eea; }}
                
                .charts-section {{
                    margin: 30px 0;
                }}
                
                .chart-row {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .chart-title {{
                    color: #2c3e50;
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #e9ecef;
                }}
                
                .chart-full {{
                    grid-column: 1 / -1;
                }}
                
                .trade-log {{
                    background: white;
                    padding: 25px;
                    margin-top: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .trade-log h2 {{
                    color: #2c3e50;
                    font-size: 24px;
                    margin-bottom: 20px;
                    font-weight: 600;
                }}
                
                .trade-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                    margin-top: 15px;
                }}
                
                .trade-table thead {{
                    background: #f8f9fa;
                }}
                
                .trade-table th {{
                    padding: 12px 8px;
                    text-align: left;
                    font-weight: 600;
                    color: #495057;
                    border-bottom: 2px solid #dee2e6;
                    font-size: 12px;
                    text-transform: uppercase;
                }}
                
                .trade-table td {{
                    padding: 10px 8px;
                    border-bottom: 1px solid #e9ecef;
                    color: #495057;
                }}
                
                .trade-table tbody tr:hover {{
                    background: #f8f9fa;
                }}
                
                .positive-trade {{ color: #28a745; font-weight: 600; }}
                .negative-trade {{ color: #dc3545; font-weight: 600; }}
                
                .collapsible {{
                    background: #667eea;
                    color: white;
                    cursor: pointer;
                    padding: 15px;
                    border: none;
                    text-align: center;
                    font-size: 16px;
                    font-weight: 600;
                    border-radius: 5px;
                    margin-top: 20px;
                    width: 100%;
                    transition: background 0.3s;
                }}
                
                .collapsible:hover {{
                    background: #764ba2;
                }}
                
                .content {{
                    padding: 0;
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.3s ease-out;
                    background: white;
                }}
                
                @media (max-width: 1200px) {{
                    .chart-row {{ grid-template-columns: 1fr; }}
                }}
                
                @media print {{
                    .collapsible {{ display: none; }}
                    .content {{ max-height: none !important; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Wolf Guzman - Trading Performance Report</h1>
                    <div class="subtitle">Professional Forex Backtesting Analysis</div>
                    <div class="subtitle">Generated: {timestamp}</div>
                </div>
                
                <div class="parameters">
                    <h3>Backtest Parameters</h3>
                    <div class="param-grid">
                        <div class="param-item">
                            <span class="param-label">Trading Pair</span>
                            {pair}
                        </div>
                        <div class="param-item">
                            <span class="param-label">Strategy</span>
                            {strategy_name}
                        </div>
                        <div class="param-item">
                            <span class="param-label">Timeframe</span>
                            {timeframe}
                        </div>
                        <div class="param-item">
                            <span class="param-label">Start Date</span>
                            {start_date}
                        </div>
                        <div class="param-item">
                            <span class="param-label">End Date</span>
                            {end_date}
                        </div>
                        <div class="param-item">
                            <span class="param-label">Initial Balance</span>
                            ${initial_balance:,.2f}
                        </div>
                        <div class="param-item">
                            <span class="param-label">Leverage</span>
                            {leverage}:1
                        </div>
                        <div class="param-item">
                            <span class="param-label">Stop Loss</span>
                            {sl_pips} pips
                        </div>
                        <div class="param-item">
                            <span class="param-label">Take Profit</span>
                            {tp_pips} pips
                        </div>
                        <div class="param-item">
                            <span class="param-label">Risk per Trade</span>
                            {risk_pct}%
                        </div>
                    </div>
                </div>
                
                {metrics_table}
                
                <div class="charts-section">
                    {charts_html}
                </div>
                
                <div class="trade-log">
                    <h2>Detailed Trade Log</h2>
                    <button class="collapsible">Click to View All Trades</button>
                    <div class="content">
                        {HTMLReportGenerator.generate_trade_table(trades_df)}
                    </div>
                </div>
            </div>
            
            <script>
                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {{
                    coll[i].addEventListener("click", function() {{
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.maxHeight) {{
                            content.style.maxHeight = null;
                        }} else {{
                            content.style.maxHeight = content.scrollHeight + "px";
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
        """
        
        temp_dir = tempfile.gettempdir()
        filename = f"WolfGuzman_{pair.replace('/', '-')}_{strategy_name}_{timestamp}.html"
        report_path = os.path.join(temp_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return report_path
    
    @staticmethod
    def generate_metrics_table(metrics, initial_balance):
        """Generate professional metrics overview"""
        
        def format_value(key, val):
            if isinstance(val, float):
                if '%' in key or 'pct' in key.lower():
                    color_class = 'positive' if val > 0 else 'negative' if val < 0 else 'neutral'
                    return f'<span class="{color_class}">{val:+.2f}%</span>'
                elif '$' in key or 'balance' in key.lower() or 'pnl' in key.lower():
                    color_class = 'positive' if val > 0 else 'negative' if val < 0 else 'neutral'
                    return f'<span class="{color_class}">${val:,.2f}</span>'
                else:
                    return f'{val:.2f}'
            return str(val)
        
        html = """
        <div class="metrics-overview">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
        """
        
        key_metrics = {
            'Initial Balance': f'${initial_balance:,.2f}',
            'Final Balance': format_value('final_balance_$', metrics.get('final_balance_$', 0)),
            'Total Return': format_value('total_return_%', metrics.get('total_return_%', 0)),
            'Total P&L': format_value('total_pnl_$', metrics.get('total_pnl_$', 0)),
            'Total Trades': metrics.get('total_trades', 0),
            'Win Rate': format_value('win_rate_%', metrics.get('win_rate_%', 0)),
            'Profit Factor': f'{metrics.get("profit_factor", 0):.2f}',
            'Sharpe Ratio': f'{metrics.get("sharpe_ratio", 0):.2f}',
            'Sortino Ratio': f'{metrics.get("sortino_ratio", 0):.2f}',
            'Max Drawdown': format_value('max_drawdown_%', metrics.get('max_drawdown_%', 0)),
            'Avg Win': format_value('avg_win_$', metrics.get('avg_win_$', 0)),
            'Avg Loss': format_value('avg_loss_$', metrics.get('avg_loss_$', 0)),
            'Largest Win': format_value('largest_win_$', metrics.get('largest_win_$', 0)),
            'Largest Loss': format_value('largest_loss_$', metrics.get('largest_loss_$', 0)),
            'Best Day': format_value('best_day_$', metrics.get('best_day_$', 0)),
            'Consecutive Wins': metrics.get('max_consecutive_wins', 0),
            'Consecutive Losses': metrics.get('max_consecutive_losses', 0),
            'Avg Trade Duration': f'{metrics.get("avg_time_in_trade_hrs", 0):.1f}h',
        }
        
        for label, value in key_metrics.items():
            html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    @staticmethod
    def generate_quantstats_charts(trades_df, initial_balance, metrics):
        """Generate comprehensive QuantStats-style charts in 2-column layout"""
        if trades_df.empty:
            return "<p>No trades to display</p>"
        
        # Config for all charts (defined once)
        chart_config = dict(displayModeBar=False)
        
        html = ""
        
        # Calculate equity curve
        equity = initial_balance + trades_df['monetary_pnl'].cumsum()
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax * 100
        
        # 1. EQUITY CURVE (Full Width)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=trades_df['exit_time'],
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#667eea', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        fig1.add_hline(y=initial_balance, line_dash="dash", line_color="gray", 
                       annotation_text="Initial Balance", annotation_position="right")
        fig1.update_layout(
            template='plotly_white',
            title={'text': 'Equity Curve', 'font': {'size': 18, 'color': '#2c3e50'}},
            xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Portfolio Value ($)', 'gridcolor': '#e9ecef'},
            height=400,
            hovermode='x unified'
        )
        
        equity_html = fig1.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container chart-full">
                <div class="chart-title">Equity Curve</div>
                {equity_html}
            </div>
        </div>
        '''
        
        # 2. DRAWDOWN ANALYSIS (Full Width)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=trades_df['exit_time'],
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#dc3545', width=2),
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.2)'
        ))
        fig2.update_layout(
            template='plotly_white',
            title={'text': 'Underwater Plot (Drawdown)', 'font': {'size': 18, 'color': '#2c3e50'}},
            xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Drawdown (%)', 'gridcolor': '#e9ecef'},
            height=350,
            hovermode='x unified'
        )
        
        drawdown_html = fig2.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container chart-full">
                <div class="chart-title">Drawdown Analysis</div>
                {drawdown_html}
            </div>
        </div>
        '''
        
        # 3. RETURNS DISTRIBUTION + DAILY RETURNS (2 columns)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=trades_df['monetary_pnl'],
            nbinsx=40,
            marker=dict(
                color=trades_df['monetary_pnl'],
                colorscale=[[0, '#dc3545'], [0.5, '#6c757d'], [1, '#28a745']],
                line=dict(color='white', width=1)
            ),
            name='Returns'
        ))
        fig3.add_vline(x=0, line_dash="dash", line_color="gray")
        fig3.update_layout(
            template='plotly_white',
            title={'text': 'Returns Distribution', 'font': {'size': 16, 'color': '#2c3e50'}},
            xaxis={'title': 'P&L ($)', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Frequency', 'gridcolor': '#e9ecef'},
            height=350,
            showlegend=False
        )
        
        returns_dist_html = fig3.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        # Daily Returns
        daily_returns = trades_df.groupby(trades_df['exit_time'].dt.date)['monetary_pnl'].sum()
        fig4 = go.Figure()
        colors = ['#28a745' if x >= 0 else '#dc3545' for x in daily_returns.values]
        fig4.add_trace(go.Bar(
            x=daily_returns.index,
            y=daily_returns.values,
            marker_color=colors,
            name='Daily P&L'
        ))
        fig4.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        fig4.update_layout(
            template='plotly_white',
            title={'text': 'Daily Returns', 'font': {'size': 16, 'color': '#2c3e50'}},
            xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'P&L ($)', 'gridcolor': '#e9ecef'},
            height=350,
            showlegend=False
        )
        
        daily_returns_html = fig4.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Returns Distribution</div>
                {returns_dist_html}
            </div>
            <div class="chart-container">
                <div class="chart-title">Daily Returns</div>
                {daily_returns_html}
            </div>
        </div>
        '''
        
        # 4. WIN/LOSS DISTRIBUTION + ROLLING SHARPE (2 columns)
        wins = trades_df[trades_df['monetary_pnl'] > 0]['monetary_pnl']
        losses = trades_df[trades_df['monetary_pnl'] < 0]['monetary_pnl']
        
        fig5 = go.Figure()
        if len(wins) > 0:
            fig5.add_trace(go.Histogram(
                x=wins,
                name='Wins',
                marker=dict(color='#28a745', line=dict(color='white', width=1)),
                opacity=0.7,
                nbinsx=25
            ))
        if len(losses) > 0:
            fig5.add_trace(go.Histogram(
                x=losses,
                name='Losses',
                marker=dict(color='#dc3545', line=dict(color='white', width=1)),
                opacity=0.7,
                nbinsx=25
            ))
        fig5.update_layout(
            template='plotly_white',
            title={'text': 'Win vs Loss Distribution', 'font': {'size': 16, 'color': '#2c3e50'}},
            barmode='overlay',
            xaxis={'title': 'P&L ($)', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Frequency', 'gridcolor': '#e9ecef'},
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        win_loss_html = fig5.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        # Rolling Sharpe (30-trade window)
        if len(trades_df) > 30:
            returns_pct = trades_df['monetary_pnl'] / initial_balance
            rolling_sharpe = returns_pct.rolling(window=30).mean() / returns_pct.rolling(window=30).std() * np.sqrt(252)
            
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(
                x=trades_df['exit_time'][29:],
                y=rolling_sharpe[29:],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='#667eea', width=2)
            ))
            fig6.add_hline(y=1, line_dash="dash", line_color="#28a745", 
                          annotation_text="Good (>1)", annotation_position="right")
            fig6.add_hline(y=0, line_dash="dash", line_color="gray")
            fig6.update_layout(
                template='plotly_white',
                title={'text': 'Rolling Sharpe Ratio (30-trade)', 'font': {'size': 16, 'color': '#2c3e50'}},
                xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
                yaxis={'title': 'Sharpe Ratio', 'gridcolor': '#e9ecef'},
                height=350
            )
            
            # Generate HTML outside f-string to avoid dict issues
            sharpe_html = fig6.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        else:
            sharpe_html = '<p style="text-align:center;padding:50px;color:#6c757d;">Insufficient data for rolling Sharpe calculation (need 30+ trades)</p>'
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Win vs Loss Distribution</div>
                {win_loss_html}
            </div>
            <div class="chart-container">
                <div class="chart-title">Rolling Sharpe Ratio</div>
                {sharpe_html}
            </div>
        </div>
        '''
        
        # 5. MONTHLY RETURNS + TRADE DURATION (2 columns)
        if len(trades_df) >= 5:
            trades_copy = trades_df.copy()
            trades_copy['month'] = pd.to_datetime(trades_copy['exit_time']).dt.to_period('M')
            monthly_pnl = trades_copy.groupby('month')['monetary_pnl'].sum()
            
            colors_monthly = ['#28a745' if x >= 0 else '#dc3545' for x in monthly_pnl.values]
            
            fig7 = go.Figure()
            fig7.add_trace(go.Bar(
                x=[str(m) for m in monthly_pnl.index],
                y=monthly_pnl.values,
                marker_color=colors_monthly,
                name='Monthly P&L'
            ))
            fig7.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
            fig7.update_layout(
                template='plotly_white',
                title={'text': 'Monthly Returns', 'font': {'size': 16, 'color': '#2c3e50'}},
                xaxis={'title': 'Month', 'gridcolor': '#e9ecef'},
                yaxis={'title': 'P&L ($)', 'gridcolor': '#e9ecef'},
                height=350,
                showlegend=False
            )
            
            # Generate HTML outside f-string to avoid dict issues
            monthly_html = fig7.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        else:
            monthly_html = '<p style="text-align:center;padding:50px;color:#6c757d;">Insufficient data for monthly analysis</p>'
        
        # Trade Duration
        fig8 = go.Figure()
        fig8.add_trace(go.Histogram(
            x=trades_df['time_in_trade_hours'],
            nbinsx=30,
            marker=dict(color='#667eea', line=dict(color='white', width=1))
        ))
        fig8.update_layout(
            template='plotly_white',
            title={'text': 'Trade Duration Distribution', 'font': {'size': 16, 'color': '#2c3e50'}},
            xaxis={'title': 'Hours in Trade', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Frequency', 'gridcolor': '#e9ecef'},
            height=350,
            showlegend=False
        )
        
        duration_html = fig8.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Monthly Returns</div>
                {monthly_html}
            </div>
            <div class="chart-container">
                <div class="chart-title">Trade Duration Analysis</div>
                {duration_html}
            </div>
        </div>
        '''
        
        # 6. CUMULATIVE WINS/LOSSES (Full Width)
        trades_df_copy = trades_df.copy()
        trades_df_copy['cumulative_wins'] = (trades_df_copy['monetary_pnl'] > 0).cumsum()
        trades_df_copy['cumulative_losses'] = (trades_df_copy['monetary_pnl'] < 0).cumsum()
        
        fig9 = make_subplots(specs=[[{"secondary_y": False}]])
        fig9.add_trace(go.Scatter(
            x=trades_df_copy['exit_time'],
            y=trades_df_copy['cumulative_wins'],
            mode='lines',
            name='Cumulative Wins',
            line=dict(color='#28a745', width=2)
        ))
        fig9.add_trace(go.Scatter(
            x=trades_df_copy['exit_time'],
            y=trades_df_copy['cumulative_losses'],
            mode='lines',
            name='Cumulative Losses',
            line=dict(color='#dc3545', width=2)
        ))
        fig9.update_layout(
            template='plotly_white',
            title={'text': 'Cumulative Wins vs Losses', 'font': {'size': 18, 'color': '#2c3e50'}},
            xaxis={'title': 'Date', 'gridcolor': '#e9ecef'},
            yaxis={'title': 'Count', 'gridcolor': '#e9ecef'},
            height=350,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        cumulative_html = fig9.to_html(full_html=False, include_plotlyjs=False, config=chart_config)
        
        html += f'''
        <div class="chart-row">
            <div class="chart-container chart-full">
                <div class="chart-title">Cumulative Wins vs Losses</div>
                {cumulative_html}
            </div>
        </div>
        '''
        
        return html
    
    @staticmethod
    def generate_trade_table(trades_df):
        """Generate clean trade log table"""
        if trades_df.empty:
            return "<p>No trades to display</p>"
        
        html = '''
        <table class="trade-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Entry Time</th>
                    <th>Exit Time</th>
                    <th>Signal</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Pips</th>
                    <th>P&L ($)</th>
                    <th>Balance</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
        '''
        
        for _, row in trades_df.iterrows():
            pnl_class = 'positive-trade' if row['monetary_pnl'] > 0 else 'negative-trade'
            html += f'''
                <tr>
                    <td>{row['trade_number']}</td>
                    <td>{row['entry_time'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{row['exit_time'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{row['signal']}</td>
                    <td>{row['entry_price']:.5f}</td>
                    <td>{row['exit_price']:.5f}</td>
                    <td class="{pnl_class}">{row['pips_pnl']:+.1f}</td>
                    <td class="{pnl_class}">${row['monetary_pnl']:+,.2f}</td>
                    <td>${row['balance']:,.2f}</td>
                    <td>{row['exit_reason']}</td>
                </tr>
            '''
        
        html += '''
            </tbody>
        </table>
        '''
        
        return html

# ======================================================================
# 6. UI
# ======================================================================
class BacktesterUI:
    def __init__(self, master):
        self.master = master
        master.title("🐺 Wolf Guzman - AlgoHaus Backtester v6.0")
        master.geometry("1400x900")
        master.minsize(1200, 700)
        
        # Default to Wolf's data folder
        default_path = pathlib.Path(r"D:\compressedworld\AlgoHaus\OandaHistoricalData\1MinCharts")
        self.data_folder = default_path if default_path.exists() else pathlib.Path.cwd() / "data" 
        self.df = None
        
        # Variables
        self.selected_pair = tk.StringVar(master, value="EUR/USD")
        self.selected_timeframe = tk.StringVar(master, value="1hr")
        self.selected_strategy = tk.StringVar(master, value="vwap_crossover_strategy")
        self.initial_balance = tk.DoubleVar(master, value=10000.0)
        self.leverage = tk.IntVar(master, value=50)
        self.sl_pips = tk.IntVar(master, value=30)
        self.tp_pips = tk.IntVar(master, value=60)
        self.risk_percent = tk.DoubleVar(master, value=1.0)
        
        today = date.today()
        self.end_date_var = tk.StringVar(master, value=today.strftime("%Y-%m-%d"))
        self.start_date_var = tk.StringVar(master, value=(today - timedelta(days=365)).strftime("%Y-%m-%d"))
        
        self.status_text = tk.StringVar(master, value="Ready - Wolf Guzman's Trading System v6.0")
        self.metrics_data = {}
        self.trades_df = pd.DataFrame()
        
        self.setup_ui()
        self.refresh_available_pairs()
        
        # Trigger initial pair info update to set dates
        self.master.after(500, self.update_pair_info)
        
        # Show initial status
        if self.data_folder.exists():
            self.update_status(f"✅ Ready - Wolf's Trading System | Data folder: {self.data_folder.name}", '#00ff41')
        else:
            self.update_status(f"⚠️ Data folder not found - Please select folder", '#ff9f43')

    def setup_ui(self):
        main_container = ctk.CTkFrame(self.master, corner_radius=0)
        main_container.pack(fill='both', expand=True)
        
        left_panel = ctk.CTkFrame(main_container, corner_radius=15, width=450)
        left_panel.pack(side='left', fill='both', padx=(20, 10), pady=20)
        left_panel.pack_propagate(False)
        
        right_panel = ctk.CTkFrame(main_container, corner_radius=15)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 20), pady=20)
        
        # Left Panel - Title
        title_frame = ctk.CTkFrame(left_panel, corner_radius=10, fg_color="#1a1a1a")
        title_frame.pack(fill='x', padx=20, pady=(20, 15))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🐺 WOLF GUZMAN",
            font=ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
            text_color="#00ff41"
        )
        title_label.pack(pady=5)
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="AlgoHaus Backtester v6.0",
            font=ctk.CTkFont(family="Helvetica", size=12),
            text_color="#888888"
        )
        subtitle_label.pack(pady=5)
        
        controls_scroll = ctk.CTkScrollableFrame(left_panel, corner_radius=10, fg_color="transparent")
        controls_scroll.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Configuration Section
        config_frame = ctk.CTkFrame(controls_scroll, corner_radius=10)
        config_frame.pack(fill='x', pady=(0, 15))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Configuration",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#00ff41",
            anchor="w"
        ).pack(fill='x', padx=15, pady=(10, 5))
        
        # Data folder
        folder_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        folder_frame.pack(fill='x', padx=15, pady=5)
        
        ctk.CTkLabel(folder_frame, text="Data Folder:", width=100).pack(side='left')
        
        # Display folder path
        folder_display = str(self.data_folder)
        if len(folder_display) > 35:
            folder_display = "..." + folder_display[-32:]
        
        self.folder_label = ctk.CTkLabel(
            folder_frame,
            text=folder_display,
            text_color="#888888",
            anchor="w"
        )
        self.folder_label.pack(side='left', expand=True, fill='x', padx=10)
        ctk.CTkButton(
            folder_frame,
            text="Browse",
            command=self.select_data_folder,
            width=70,
            height=28
        ).pack(side='right')
        
        # Input fields
        self.pair_combo = self.create_input_field(config_frame, "Trading Pair:", self.selected_pair, 
                               is_combobox=True, values=[])
        
        # Pair info display - no scrollbar, white text, auto-sized
        info_section = ctk.CTkLabel(
            config_frame,
            text="Pair Information:",
            font=ctk.CTkFont(family="Helvetica", size=11, weight="bold"),
            text_color="#888888",
            anchor="w"
        )
        info_section.pack(fill='x', padx=15, pady=(10, 2))
        
        self.pair_info_frame = ctk.CTkFrame(config_frame, fg_color="#252525", corner_radius=5)
        self.pair_info_frame.pack(fill='x', padx=15, pady=5)
        
        self.pair_info_label = ctk.CTkLabel(
            self.pair_info_frame,
            text='Select a pair to view details...',
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color="#ffffff",
            anchor="nw",
            justify="left"
        )
        self.pair_info_label.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add trace to update pair info
        self.selected_pair.trace('w', self.update_pair_info)
        
        self.create_input_field(config_frame, "Timeframe:", self.selected_timeframe,
                               is_combobox=True, values=["1min", "5min", "15min", "1hr", "1Day"])
        self.create_input_field(config_frame, "Start Date:", self.start_date_var)
        self.create_input_field(config_frame, "End Date:", self.end_date_var)
        
        # Strategy Section
        strategy_frame = ctk.CTkFrame(controls_scroll, corner_radius=10)
        strategy_frame.pack(fill='x', pady=(0, 15))
        
        ctk.CTkLabel(
            strategy_frame,
            text="💡 Strategy & Risk",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#00ff41",
            anchor="w"
        ).pack(fill='x', padx=15, pady=(10, 5))
        
        strategies = [name for name, obj in inspect.getmembers(TradingStrategies) if inspect.isfunction(obj)]
        self.create_input_field(strategy_frame, "Strategy:", self.selected_strategy,
                               is_combobox=True, values=strategies)
        self.create_input_field(strategy_frame, "Stop Loss (Pips):", self.sl_pips)
        self.create_input_field(strategy_frame, "Take Profit (Pips):", self.tp_pips)
        
        # Account Section
        account_frame = ctk.CTkFrame(controls_scroll, corner_radius=10)
        account_frame.pack(fill='x', pady=(0, 15))
        
        ctk.CTkLabel(
            account_frame,
            text="💰 Account Settings",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#00ff41",
            anchor="w"
        ).pack(fill='x', padx=15, pady=(10, 5))
        
        self.create_input_field(account_frame, "Initial Balance ($):", self.initial_balance)
        self.create_input_field(account_frame, "Leverage:", self.leverage,
                               is_combobox=True, values=[str(x) for x in ForexCalculator.LEVERAGE_OPTIONS])
        self.create_input_field(account_frame, "Risk % per Trade:", self.risk_percent)
        
        # Run Button
        run_button = ctk.CTkButton(
            left_panel,
            text="🚀 RUN BACKTEST",
            command=self.start_backtest_thread,
            height=45,
            corner_radius=8,
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            fg_color="#00ff41",
            hover_color="#32ff7e",
            text_color="#000000"
        )
        run_button.pack(fill='x', padx=20, pady=(10, 20))
        
        # Right Panel - Summary
        summary_frame = ctk.CTkFrame(right_panel, corner_radius=10)
        summary_frame.pack(fill='x', padx=15, pady=(15, 10))
        
        ctk.CTkLabel(
            summary_frame,
            text="📈 Summary",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#00ff41",
            anchor="w"
        ).pack(fill='x', padx=15, pady=(10, 5))
        
        self.summary_textbox = ctk.CTkTextbox(
            summary_frame,
            height=150,
            corner_radius=8,
            font=ctk.CTkFont(family="Courier New", size=12),
            fg_color="#1a1a1a",
            text_color="#00ff41"
        )
        self.summary_textbox.pack(fill='both', padx=15, pady=(5, 15))
        
        # Metrics Section
        metrics_frame = ctk.CTkFrame(right_panel, corner_radius=10)
        metrics_frame.pack(fill='both', expand=True, padx=15, pady=(0, 10))
        
        ctk.CTkLabel(
            metrics_frame,
            text="📊 Detailed Metrics",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#00ff41",
            anchor="w"
        ).pack(fill='x', padx=15, pady=(10, 5))
        
        self.metrics_scroll = ctk.CTkScrollableFrame(
            metrics_frame,
            corner_radius=8,
            fg_color="#1a1a1a"
        )
        self.metrics_scroll.pack(fill='both', expand=True, padx=15, pady=(5, 15))
        
        # Report Button
        self.report_button = ctk.CTkButton(
            right_panel,
            text="📋 Generate Professional Report",
            command=self.generate_report,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            fg_color="#434446",
            hover_color="#595C5E",
            state="disabled"
        )
        self.report_button.pack(fill='x', padx=15, pady=(0, 15))
        
        # Status Bar
        status_frame = ctk.CTkFrame(self.master, corner_radius=0, height=35, fg_color="#111111")
        status_frame.pack(side='bottom', fill='x')
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_text,
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color="#888888",
            anchor="w"
        )
        self.status_label.pack(side='left', padx=20, pady=8)

    def create_input_field(self, parent, label_text, variable, is_combobox=False, values=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill='x', padx=15, pady=5)
        
        label = ctk.CTkLabel(frame, text=label_text, width=120, anchor="w")
        label.pack(side='left')
        
        if is_combobox:
            widget = ctk.CTkComboBox(frame, variable=variable, values=values or [])
        else:
            widget = ctk.CTkEntry(frame, textvariable=variable)
        widget.pack(side='left', expand=True, fill='x', padx=(10, 0))
        
        return widget

    def refresh_available_pairs(self):
        """Refresh available pairs from data folder"""
        try:
            self.update_status("Scanning for pairs...", '#ffa502')
            pairs = detect_available_pairs(self.data_folder)
            
            if pairs:
                # Update the combobox directly
                if hasattr(self, 'pair_combo') and self.pair_combo:
                    self.pair_combo.configure(values=pairs)
                    # Set first pair if current selection is invalid
                    if self.selected_pair.get() not in pairs:
                        self.selected_pair.set(pairs[0])
                
                self.update_status(f"✅ Found {len(pairs)} pairs: {', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''}", '#00ff41')
            else:
                self.update_status(f"⚠️ No pairs found in {self.data_folder}", '#ff4757')
        except Exception as e:
            logging.error(f"Error refreshing pairs: {e}")
            self.update_status(f"❌ Error scanning: {str(e)}", '#ff4757')

    def update_pair_info(self, *args):
        """Update pair information display and automatically set date ranges"""
        pair = self.selected_pair.get()
        if not pair:
            return
        
        # Get actual date range from the pair's data
        start, end = get_data_date_range(pair, self.data_folder)
        
        # Get pip value and calculate margin info
        pip_value = ForexCalculator.PIP_VALUES.get(pair, 0.0001)
        
        # Example calculations
        example_price = 1.10 if pair.startswith('EUR') else 1.30
        leverage = self.leverage.get()
        
        margin = ForexCalculator.calculate_margin_required(pair, 10000, example_price, leverage)
        pip_val_usd = ForexCalculator.calculate_pip_value_in_usd(pair, 10000, example_price)
        
        # Show which subfolder will be used
        pair_folder = pair.replace('/', '_')
        folder_path = self.data_folder / pair_folder
        
        if start and end:
            # Calculate total days
            total_days = (end - start).days
            
            info_text = f"""PAIR: {pair}  |  FOLDER: {pair_folder}

Data Available: {start} to {end}
Total Days: {total_days:,}

Pip Value: {pip_value}
Per 10k Units: ${pip_val_usd:.2f} per pip
Margin (10k @ {leverage}:1): ${margin:.2f}"""
            
            # AUTOMATICALLY UPDATE DATE FIELDS TO MATCH PAIR'S DATA RANGE
            self.start_date_var.set(str(start))
            self.end_date_var.set(str(end))
            
            # Visual feedback that dates were auto-set
            self.update_status(f"✅ {pair}: {total_days:,} days | Dates: {start} to {end}", '#00ff41')
        else:
            info_text = f"""PAIR: {pair}  |  FOLDER: {pair_folder}

⚠️ No data found
Path: {folder_path}"""
            self.update_status(f"⚠️ No data found for {pair} in {pair_folder}", '#ff9f43')
        
        self.pair_info_label.configure(text=info_text)

    def update_status(self, text, color='#888888'):
        self.status_text.set(text)
        self.status_label.configure(text_color=color)

    def select_data_folder(self):
        new_folder = filedialog.askdirectory(
            title="Select Main Data Folder (containing pair subfolders)",
            initialdir=str(self.data_folder)
        )
        if new_folder:
            self.data_folder = pathlib.Path(new_folder)
            folder_text = str(self.data_folder)
            if len(folder_text) > 35:
                folder_text = "..." + folder_text[-32:]
            self.folder_label.configure(text=folder_text)
            self.refresh_available_pairs()
            self.update_status(f"Data folder updated", '#00ff41')

    def start_backtest_thread(self):
        self.update_status("Running backtest...", '#ffa502')
        
        self.summary_textbox.delete("0.0", "end")
        self.trades_df = pd.DataFrame()
        
        for widget in self.metrics_scroll.winfo_children():
            widget.destroy()
        
        self.report_button.configure(state="disabled")
        
        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            
            if start_date >= end_date:
                raise ValueError("Start date must be before End date.")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            self.update_status("Error: Invalid input.", '#ff4757')
            return

        self.q = queue.Queue()
        threading.Thread(target=self.run_backtest_task, 
                        args=(start_date, end_date), 
                        daemon=True).start()
        self.master.after(100, self.check_queue)

    def run_backtest_task(self, start_date, end_date):
        try:
            pair = self.selected_pair.get()
            timeframe = self.selected_timeframe.get()
            strategy_name = self.selected_strategy.get()
            
            # Debug: Log the exact path being used
            pair_folder = pair.replace('/', '_')
            expected_path = self.data_folder / pair_folder
            logging.info(f"Attempting to load pair: {pair}")
            logging.info(f"Base folder: {self.data_folder}")
            logging.info(f"Expected subfolder: {expected_path}")
            
            df, actual_start, actual_end = load_pair_data(pair, self.data_folder, start_date, end_date, timeframe)
            self.df = df
            self.actual_start = actual_start
            self.actual_end = actual_end
            
            strategy_func = getattr(TradingStrategies, strategy_name)
            
            backtester = EnhancedBacktester(
                df, 
                initial_balance=self.initial_balance.get(), 
                pip_value=ForexCalculator.PIP_VALUES.get(pair, 0.0001), 
                leverage=self.leverage.get(),
                risk_percent=self.risk_percent.get()
            )
            
            summary, metrics = backtester.run_backtest(
                strategy_func, 
                self.sl_pips.get(), 
                self.tp_pips.get(), 
                pair
            )
            
            self.q.put(('success', summary, metrics, backtester.results))

        except Exception as e:
            logging.error(f"Backtest error: {e}", exc_info=True)
            self.q.put(('error', str(e)))

    def check_queue(self):
        try:
            result_type, *data = self.q.get_nowait()
            
            if result_type == 'success':
                summary, metrics, trades_df = data
                self.update_results_ui(summary, metrics, trades_df)
                self.update_status("✅ Backtest completed!", '#00ff41')
            elif result_type == 'error':
                error_msg = data[0]
                messagebox.showerror("Backtest Error", error_msg)
                self.update_status("❌ Backtest failed.", '#ff4757')

        except queue.Empty:
            self.master.after(100, self.check_queue)

    def update_results_ui(self, summary, metrics, trades_df):
        self.trades_df = trades_df
        self.metrics_data = metrics
        
        self.summary_textbox.delete("0.0", "end")
        self.summary_textbox.insert("0.0", summary)
        
        for widget in self.metrics_scroll.winfo_children():
            widget.destroy()
        
        # Create metrics cards
        row_frame = None
        for i, (key, value) in enumerate(metrics.items()):
            if i % 2 == 0:
                row_frame = ctk.CTkFrame(self.metrics_scroll, fg_color="transparent")
                row_frame.pack(fill='x', pady=5)
            
            card = ctk.CTkFrame(row_frame, corner_radius=8, fg_color="#252525", width=250)
            card.pack(side='left', expand=True, fill='x', padx=5)
            card.pack_propagate(False)
            
            display_key = key.replace('_', ' ').title()
            
            if isinstance(value, (int, float)):
                if value > 0 and ('pnl' in key.lower() or 'return' in key.lower()):
                    color = '#00ff41'
                elif value < 0 and ('pnl' in key.lower() or 'return' in key.lower()):
                    color = '#ff4757'
                else:
                    color = '#00ff41'
            else:
                color = '#888888'
            
            label = ctk.CTkLabel(
                card,
                text=display_key,
                font=ctk.CTkFont(family="Helvetica", size=11),
                text_color="#888888",
                anchor="w"
            )
            label.pack(fill='x', padx=10, pady=(8, 2))
            
            value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)
            value_label = ctk.CTkLabel(
                card,
                text=value_str,
                font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
                text_color=color,
                anchor="w"
            )
            value_label.pack(fill='x', padx=10, pady=(2, 8))
        
        self.report_button.configure(state="normal")

    def generate_report(self):
        if self.trades_df.empty:
            messagebox.showwarning("Report Warning", "No trades to generate report.")
            return

        try:
            self.update_status("Generating professional report...", '#ffa502')
            
            report_path = HTMLReportGenerator.generate_report(
                self.metrics_data, 
                self.trades_df, 
                self.selected_strategy.get(),
                self.selected_timeframe.get(), 
                self.selected_pair.get(), 
                self.initial_balance.get(),
                self.leverage.get(),
                self.sl_pips.get(),
                self.tp_pips.get(),
                self.risk_percent.get(),
                self.start_date_var.get(),
                self.end_date_var.get(),
                df=self.df
            )
            
            messagebox.showinfo("Report Generated", f"Professional report saved!")
            webbrowser.open_new_tab('file://' + os.path.realpath(report_path))
            self.update_status("📊 Report generated successfully!", '#00ff41')
        
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {e}")
            self.update_status("❌ Error generating report.", '#ff4757')

# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    app = ctk.CTk()
    
    width = 1400
    height = 900
    x = (app.winfo_screenwidth() // 2) - (width // 2)
    y = (app.winfo_screenheight() // 2) - (height // 2)
    app.geometry(f'{width}x{height}+{x}+{y}')
    
    backtester = BacktesterUI(app)
    app.mainloop()

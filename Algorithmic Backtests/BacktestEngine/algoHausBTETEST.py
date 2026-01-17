#!/usr/bin/env python
# coding: utf-8

# Wolf's AlgoHaus Backtester v7.0 - Enhanced Multi-Strategy Edition
# Wolf Guzman
# Features: Parallel Strategy Execution, Editable Strategies, Checkbox Selection
# Professional Forex Backtesting with Parquet Data

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import webbrowser
import os
import sys
import tempfile
import inspect
import pathlib
import threading
import queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import json
import base64
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ======================================================================
# 1. FOREX CALCULATOR (UNCHANGED - OPTIMIZED)
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
            pip_value = pip_size * unit_size
        elif pair.startswith('USD/'):
            pip_value = (pip_size / current_price) * unit_size
        else:
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
# 2. DATA LOADING WITH VALIDATION (OPTIMIZED WITH CACHING)
# ======================================================================

# Global cache for loaded data
_DATA_CACHE = {}
_CACHE_LOCK = threading.Lock()

def detect_available_pairs(base_folder: pathlib.Path):
    """Scan for available forex pair folders and return valid pairs - OPTIMIZED"""
    pairs = set()

    if not base_folder.exists():
        logging.error(f"Base folder does not exist: {base_folder}")
        return []

    logging.info(f"Scanning for pairs in: {base_folder}")

    # Use list comprehension for faster iteration
    subfolders = [f for f in base_folder.iterdir() if f.is_dir() and f.name not in ['README.TXT', '__pycache__']]
    
    for subfolder in subfolders:
        folder_name = subfolder.name
        if '_' in folder_name and len(folder_name.split('_')) == 2:
            parts = folder_name.split('_')
            if len(parts[0]) == 3 and len(parts[1]) == 3:
                # Check for parquet files without listing all
                if any(subfolder.glob("*.parquet")):
                    pair = folder_name.replace('_', '/')
                    pairs.add(pair)
                    logging.info(f"Found pair: {pair}")

    if not pairs:
        logging.warning("No valid pairs found!")
    else:
        logging.info(f"Total pairs found: {len(pairs)}")

    return sorted(list(pairs))

def get_data_date_range(pair_name: str, base_folder: pathlib.Path):
    """Get actual date range from parquet file - OPTIMIZED with cache"""
    if pair_name in ForexCalculator.DATA_RANGES:
        start_str, end_str = ForexCalculator.DATA_RANGES[pair_name]
        start = datetime.strptime(start_str, '%Y-%m-%d').date()
        end = datetime.strptime(end_str, '%Y-%m-%d').date()
        return start, end

    try:
        pair_folder_name = pair_name.replace('/', '_')
        pair_folder = base_folder / pair_folder_name

        if not pair_folder.exists() or not pair_folder.is_dir():
            logging.warning(f"Folder not found: {pair_folder}")
            return None, None

        parquet_files = list(pair_folder.glob("*.parquet"))

        if not parquet_files:
            logging.warning(f"No parquet files in {pair_folder}")
            return None, None

        # Read only metadata for date range (much faster)
        df = pd.read_parquet(parquet_files[0], engine='pyarrow', columns=None)
        
        datetime_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['datetime', 'date', 'time']):
                datetime_col = col
                break

        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)
            df = df.dropna(subset=[datetime_col])
            df[datetime_col] = df[datetime_col].dt.tz_localize(None)
            start = df[datetime_col].min().date()
            end = df[datetime_col].max().date()
            return start, end

        return None, None
    except Exception as e:
        logging.error(f"Error getting date range for {pair_name}: {e}")
        return None, None

def load_pair_data(pair_name: str, base_folder: pathlib.Path, start_date: datetime, end_date: datetime, timeframe: str):
    """Load and validate parquet data - OPTIMIZED with caching and vectorization"""
    
    # Create cache key
    cache_key = f"{pair_name}_{start_date.date()}_{end_date.date()}_{timeframe}"
    
    # Check cache first
    with _CACHE_LOCK:
        if cache_key in _DATA_CACHE:
            logging.info(f"Using cached data for {pair_name}")
            cached_df, actual_start, actual_end = _DATA_CACHE[cache_key]
            return cached_df.copy(), actual_start, actual_end
    
    pair_folder_name = pair_name.replace('/', '_')
    pair_folder = base_folder / pair_folder_name

    logging.info(f"=" * 50)
    logging.info(f"Loading pair: {pair_name}")

    if not pair_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {pair_folder}")

    parquet_files = list(pair_folder.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No PARQUET files found in {pair_folder}")

    parquet_path = parquet_files[0]

    # Load with pyarrow for speed
    df = pd.read_parquet(parquet_path, engine='pyarrow')

    if df.empty:
        raise ValueError(f"PARQUET file is empty: {parquet_path}")

    # Optimized column mapping
    cols_lower = {c.strip().lower(): c for c in df.columns}
    col_map = {
        'datetime': next((cols_lower[k] for k in ['datetime', 'date', 'time', 'timestamp'] if k in cols_lower), None),
        'open': next((cols_lower[k] for k in ['open', 'o'] if k in cols_lower), None),
        'high': next((cols_lower[k] for k in ['high', 'h'] if k in cols_lower), None),
        'low': next((cols_lower[k] for k in ['low', 'l'] if k in cols_lower), None),
        'close': next((cols_lower[k] for k in ['close', 'c', 'last'] if k in cols_lower), None),
        'volume': next((cols_lower[k] for k in ['volume', 'vol', 'v'] if k in cols_lower), None)
    }

    # Rename columns
    rename_dict = {v: k for k, v in col_map.items() if v is not None}
    df = df.rename(columns=rename_dict)

    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 1000

    # Vectorized datetime conversion
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    df = df.dropna(subset=['datetime'])
    df['datetime'] = df['datetime'].dt.tz_localize(None)

    # Remove duplicates and sort
    df = df.drop_duplicates(subset='datetime', keep='first')
    df = df.sort_values('datetime')

    actual_start = df['datetime'].min().date()
    actual_end = df['datetime'].max().date()

    # Adjust dates if needed
    if start_date.date() < actual_start:
        start_date = datetime.combine(actual_start, datetime.min.time())

    if end_date.date() > actual_end:
        end_date = datetime.combine(actual_end, datetime.min.time())

    # Filter data
    df = df.set_index('datetime')
    user_start = max(pd.Timestamp(start_date.date()), df.index.min())
    user_end = min(pd.Timestamp(end_date.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59), df.index.max())
    df = df.loc[user_start:user_end].copy()

    if df.empty:
        raise ValueError(f"No data in range {start_date.date()} to {end_date.date()}")

    # Resample if needed
    if timeframe != '1min':
        rule = {'5min': '5T', '15min': '15T', '1hr': '1H', '1Day': '1D'}.get(timeframe, '1T')
        df = df.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

    df = df.reset_index()
    df['date'] = df['datetime'].dt.date

    # Vectorized daily aggregations
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
    
    # Cache the result
    with _CACHE_LOCK:
        _DATA_CACHE[cache_key] = (df.copy(), actual_start, actual_end)
    
    return df, actual_start, actual_end


# ======================================================================
# 3. EDITABLE TRADING STRATEGIES
# ======================================================================
class TradingStrategies:
    """Editable trading strategies with improved logic"""
    
    @staticmethod
    def vwap_crossover_strategy(df, sl_pips, tp_pips, pip_value):
        """VWAP Crossover Strategy - OPTIMIZED"""
        df = df.copy()
        
        # Vectorized VWAP calculation
        df['tpv'] = df['volume'] * (df['high'] + df['low'] + df['close']) / 3
        df['cumvol'] = df.groupby('date')['volume'].cumsum()
        df['cumtpv'] = df.groupby('date')['tpv'].cumsum()
        df['vwap'] = df['cumtpv'] / df['cumvol']

        # Vectorized signal generation
        df['prev_close'] = df['close'].shift(1)
        df['prev_vwap'] = df['vwap'].shift(1)

        df['signal'] = None
        buy_condition = (df['prev_close'] <= df['prev_vwap']) & (df['close'] > df['vwap'])
        sell_condition = (df['prev_close'] >= df['prev_vwap']) & (df['close'] < df['vwap'])
        df.loc[buy_condition, 'signal'] = 'BUY'
        df.loc[sell_condition, 'signal'] = 'SELL'

        # Generate trades
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
    #2345678
    @staticmethod
    def opening_range_strategy(df, sl_pips, tp_pips, pip_value):
        """Opening Range Breakout Strategy - FIXED"""
        df = df.copy()
        trades = []

        # Group by date for efficiency
        for date in df['date'].unique():
            day_data = df[df['date'] == date].reset_index(drop=True)
            
            # Need at least 31 bars (30 for OR + 1 for breakout)
            if len(day_data) < 31: 
                continue

            # Opening Range: first 30 minutes (bars)
            opening_range = day_data.iloc[:30]
            or_high = opening_range['high'].max()
            or_low = opening_range['low'].min()

            # Skip if no valid range
            if or_high == or_low:
                continue
            
            breakout_detected = False
            
            # Start checking from bar 30 onwards, but stop before last bar
            for i in range(30, len(day_data) - 1):
                if breakout_detected:
                    break

                bar = day_data.iloc[i]

                # BUY: Close breaks above OR high
                if bar['close'] > or_high:
                    remaining = day_data[i+1:].reset_index(drop=True)
                    if len(remaining) > 0:
                        trades.append({
                            'datetime': bar['datetime'],
                            'entry_price': bar['close'],
                            'signal': 'BUY',
                            'day_data': remaining
                        })
                        breakout_detected = True

                # SELL: Close breaks below OR low
                elif bar['close'] < or_low:
                    remaining = day_data[i+1:].reset_index(drop=True)
                    if len(remaining) > 0:
                        trades.append({
                            'datetime': bar['datetime'],
                            'entry_price': bar['close'],
                            'signal': 'SELL',
                            'day_data': remaining
                        })
                        breakout_detected = True

        return trades

    @staticmethod
    def bollinger_band_reversion_strategy(df, sl_pips, tp_pips, pip_value, period=20, std_dev=2):
        """Bollinger Band Mean Reversion Strategy - OPTIMIZED"""
        df = df.copy()

        # Vectorized BB calculation
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (std_dev * df['bb_std'])

        # Vectorized signal generation
        df['signal'] = None
        buy_condition = df['close'] < df['bb_lower']
        sell_condition = df['close'] > df['bb_upper']
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
                    'day_data': remaining_data,
                    'bb_upper': row['bb_upper'],
                    'bb_middle': row['bb_middle'],
                    'bb_lower': row['bb_lower']
                })

        return trades


# ======================================================================
# 4. ENHANCED BACKTESTER WITH PARALLEL PROCESSING
# ======================================================================
class EnhancedBacktester:
    def __init__(self, df, initial_balance=10000, pip_value=0.0001, leverage=50, 
                 risk_percent=1.0, spread_pips=1.5, slippage_pips=0.5):
        self.df = df
        self.initial_balance = initial_balance
        self.pip_value = pip_value
        self.leverage = leverage
        self.risk_percent = risk_percent
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.results = None

    def run_backtest(self, strategy_func, sl_pips, tp_pips, pair_name, progress_callback=None):
        """Run backtest with vectorized operations for speed"""
        logging.info(f"Generating trades using {strategy_func.__name__}...")

        if progress_callback:
            progress_callback(0, "Generating trade signals...")

        trades = strategy_func(self.df, sl_pips, tp_pips, self.pip_value)

        if not trades:
            self.results = pd.DataFrame()
            return "No trades generated.", {}

        logging.info(f"Processing {len(trades)} potential trades...")
        results = []
        current_balance = self.initial_balance
        trade_number = 1

        is_usd_major = ForexCalculator.is_usd_major(pair_name)

        total_trades = len(trades)

        for idx, t in enumerate(trades):
            if progress_callback and (idx % max(1, total_trades // 100) == 0 or idx == total_trades - 1):
                progress = int((idx / total_trades) * 100)
                progress_callback(progress, f"Processing trade {idx+1}/{total_trades} ({progress}%)")

            entry_price = t['entry_price']
            signal = t['signal']
            bars = t['day_data']

            if bars.empty:
                continue

            unit_size = ForexCalculator.calculate_position_size(
                current_balance, self.risk_percent, sl_pips, pair_name, entry_price
            )

            if unit_size < 1000:
                continue

            margin_required = ForexCalculator.calculate_margin_required(
                pair_name, unit_size, entry_price, self.leverage, 1.0
            )

            if margin_required > current_balance * 0.8:
                continue

            pip_value_usd = ForexCalculator.calculate_pip_value_in_usd(
                pair_name, unit_size, entry_price, 1.0
            )

            spread_cost = self.spread_pips * self.pip_value
            slippage_cost = self.slippage_pips * self.pip_value

            if signal == 'BUY':
                actual_entry_price = entry_price + spread_cost + slippage_cost
            else:
                actual_entry_price = entry_price - spread_cost - slippage_cost

            if signal == 'BUY':
                stop_level = actual_entry_price - (sl_pips * self.pip_value)
                take_level = actual_entry_price + (tp_pips * self.pip_value)
            else:
                stop_level = actual_entry_price + (sl_pips * self.pip_value)
                take_level = actual_entry_price - (tp_pips * self.pip_value)

            # Vectorized exit detection
            exit_idx = None
            exit_reason = 'Timeout'
            exit_price = None

            if signal == 'BUY':
                sl_hit = bars['low'] <= stop_level
                tp_hit = bars['high'] >= take_level
            else:
                sl_hit = bars['high'] >= stop_level
                tp_hit = bars['low'] <= take_level

            # Find first exit
            sl_indices = sl_hit[sl_hit].index
            tp_indices = tp_hit[tp_hit].index

            if len(sl_indices) > 0 and len(tp_indices) > 0:
                if sl_indices[0] < tp_indices[0]:
                    exit_idx = sl_indices[0]
                    exit_price = stop_level
                    exit_reason = 'SL'
                else:
                    exit_idx = tp_indices[0]
                    exit_price = take_level
                    exit_reason = 'TP'
            elif len(sl_indices) > 0:
                exit_idx = sl_indices[0]
                exit_price = stop_level
                exit_reason = 'SL'
            elif len(tp_indices) > 0:
                exit_idx = tp_indices[0]
                exit_price = take_level
                exit_reason = 'TP'

            if exit_price is None:
                exit_idx = len(bars) - 1
                exit_price = bars.iloc[-1]['close']
                exit_reason = 'Timeout'

            if signal == 'BUY':
                actual_exit_price = exit_price - slippage_cost
            else:
                actual_exit_price = exit_price + slippage_cost

            if signal == 'BUY':
                pips_pnl = (actual_exit_price - actual_entry_price) / self.pip_value
            else:
                pips_pnl = (actual_entry_price - actual_exit_price) / self.pip_value

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
                'entry_price': round(actual_entry_price, 5),
                'exit_price': round(actual_exit_price, 5),
                'exit_reason': exit_reason,
                'pips_pnl': round(pips_pnl, 2),
                'pips': round(pips_pnl, 2),  # Add 'pips' column for CSV export
                'monetary_pnl': round(monetary_pnl, 2),
                'spread_cost_usd': round(self.spread_pips * pip_value_usd, 2),
                'slippage_cost_usd': round(self.slippage_pips * pip_value_usd * 2, 2),
                'unit_size': unit_size,
                'margin_used': round(margin_required, 2),
                'balance': round(current_balance, 2),
                'pip_value_usd': round(pip_value_usd, 4)
            })

            trade_number += 1

        if progress_callback:
            progress_callback(100, "Calculating metrics...")

        logging.info(f"Backtest complete: {len(results)} trades executed")
        self.results = pd.DataFrame(results)

        if not self.results.empty:
            total_pnl = self.results['monetary_pnl'].sum()
            total_pips = self.results['pips_pnl'].sum()
            win_rate = (self.results['pips_pnl'] > 0).mean() * 100
            total_return_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100

            summary = (f"TRADES: {len(self.results)}\n"
                      f"WIN RATE: {win_rate:.1f}%\n"
                      f"P&L: ${total_pnl:,.2f}\n"
                      f"PIPS: {total_pips:,.1f}\n"
                      f"RETURNS: {total_return_pct:+.2f}%\n"
                      f"FINAL BALANCE: ${current_balance:,.2f}")

            if not is_usd_major:
                summary += "\n⚠️ Cross pair - approximate pip values"

            metrics = self.calculate_metrics()
            metrics['total_return_pct'] = total_return_pct
        else:
            summary = "No trades executed"
            metrics = {}

        return summary, metrics

    def calculate_metrics(self):
        """Calculate comprehensive trading metrics - OPTIMIZED"""
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

        # Vectorized drawdown calculation
        equity_curve = self.initial_balance + trades_df['monetary_pnl'].cumsum()
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax * 100
        max_drawdown_pct = drawdown.min()

        if len(trades_df) > 1:
            daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['monetary_pnl'].sum()
            daily_returns_pct = daily_returns / self.initial_balance

            sharpe = (daily_returns_pct.mean() * 252) / (daily_returns_pct.std() * np.sqrt(252)) if daily_returns_pct.std() > 0 else 0

            negative_returns = daily_returns_pct[daily_returns_pct < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino = (daily_returns_pct.mean() * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

            best_day = daily_returns.max()
            best_day_date = daily_returns.idxmax()
        else:
            sharpe = 0
            sortino = 0
            best_day = 0
            best_day_date = None

        # Vectorized streak calculation
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
# 5. PARALLEL STRATEGY RUNNER
# ======================================================================
def run_single_strategy(args):
    """Worker function for parallel strategy execution"""
    df, strategy_name, strategy_func, sl_pips, tp_pips, pair_name, initial_balance, \
    leverage, risk_percent, spread_pips, slippage_pips, pip_value = args
    
    try:
        backtester = EnhancedBacktester(
            df,
            initial_balance=initial_balance,
            pip_value=pip_value,
            leverage=leverage,
            risk_percent=risk_percent,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips
        )
        
        summary, metrics = backtester.run_backtest(
            strategy_func,
            sl_pips,
            tp_pips,
            pair_name,
            progress_callback=None  # No callback in parallel mode
        )
        
        return {
            'strategy_name': strategy_name,
            'summary': summary,
            'metrics': metrics,
            'trades_df': backtester.results,
            'success': True,
            'error': None
        }
    except Exception as e:
        logging.error(f"Error in strategy {strategy_name}: {e}")
        return {
            'strategy_name': strategy_name,
            'summary': f"Error: {str(e)}",
            'metrics': {},
            'trades_df': pd.DataFrame(),
            'success': False,
            'error': str(e)
        }



# ======================================================================
# 6. STRATEGY EDITOR WINDOW
# ======================================================================
class StrategyEditorWindow:
    """Popup window for editing strategy code"""
    
    def __init__(self, parent, strategy_name, strategy_func):
        self.window = ctk.CTkToplevel(parent)
        self.window.title(f"Edit Strategy: {strategy_name}")
        self.window.geometry("900x700")
        
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.code_modified = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create editor interface"""
        # Header
        header = ctk.CTkFrame(self.window, fg_color="#000000", height=60)
        header.pack(fill='x', padx=0, pady=0)
        
        ctk.CTkLabel(
            header,
            text=f"✏️  {self.strategy_name}",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color="#e6edf3"
        ).pack(side='left', padx=20, pady=15)
        
        # Code editor
        editor_frame = ctk.CTkFrame(self.window, fg_color="#0d1117")
        editor_frame.pack(fill='both', expand=True, padx=20, pady=(0, 10))
        
        # Get source code
        source = inspect.getsource(self.strategy_func)
        
        # Create text widget with dark theme
        self.text_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.WORD,
            font=("Courier New", 10),
            bg="#0d1117",
            fg="#e6edf3",
            insertbackground="#e6edf3",
            selectbackground="#388bfd",
            selectforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.text_editor.pack(fill='both', expand=True, padx=2, pady=2)
        self.text_editor.insert('1.0', source)
        
        # Buttons
        button_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="💾 Save Changes",
            command=self.save_changes,
            fg_color="#238636",
            hover_color="#2ea043",
            height=38,
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold")
        ).pack(side='left', padx=(0, 10))
        
        ctk.CTkButton(
            button_frame,
            text="↺ Reset to Default",
            command=self.reset_code,
            fg_color="#21262d",
            hover_color="#30363d",
            height=38,
            font=ctk.CTkFont(family="Helvetica", size=12)
        ).pack(side='left', padx=(0, 10))
        
        ctk.CTkButton(
            button_frame,
            text="✕ Cancel",
            command=self.window.destroy,
            fg_color="#21262d",
            hover_color="#30363d",
            height=38,
            font=ctk.CTkFont(family="Helvetica", size=12)
        ).pack(side='right')
        
    def save_changes(self):
        """Save edited code"""
        try:
            new_code = self.text_editor.get('1.0', tk.END)
            
            # Validate syntax
            compile(new_code, '<string>', 'exec')
            
            # Execute code in TradingStrategies namespace
            exec_globals = {'np': np, 'pd': pd}
            exec(new_code, exec_globals)
            
            # Find the function
            func_name = self.strategy_func.__name__
            if func_name in exec_globals:
                # Update the function
                setattr(TradingStrategies, func_name, staticmethod(exec_globals[func_name]))
                self.code_modified = True
                
                messagebox.showinfo(
                    "Success",
                    f"Strategy '{self.strategy_name}' updated successfully!"
                )
                self.window.destroy()
            else:
                messagebox.showerror(
                    "Error",
                    f"Function '{func_name}' not found in code"
                )
                
        except SyntaxError as e:
            messagebox.showerror("Syntax Error", f"Invalid Python syntax:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save strategy:\n{str(e)}")
            
    def reset_code(self):
        """Reset to original code"""
        if messagebox.askyesno("Reset Code", "Reset to default strategy code?"):
            source = inspect.getsource(self.strategy_func)
            self.text_editor.delete('1.0', tk.END)
            self.text_editor.insert('1.0', source)


# Continue in next file due to length...
# Part 2: UI and Report Generation
# Append this to algohaus_backtester_v7_enhanced.py or run as continuation

# (Continued from previous file...)

# Import the HTMLReportGenerator from your original file
# For now, I'll create a placeholder - you'll merge with your existing HTMLReportGenerator

# ======================================================================
# 5. HTML REPORT GENERATOR (Placeholder - use your existing full implementation)
# ======================================================================




class HTMLReportGenerator:
    """
    Modern, visually stunning dashboard with Chart.js + Matplotlib
    NO Plotly - using Chart.js for perfect dark theme control
    Matplotlib for static equity and comparison charts
    """

    @staticmethod
    def _generate_ai_analysis(metrics, trades_df, initial_balance, strategy_name, pair, timeframe, 
                              start_date, end_date, leverage, sl_pips, tp_pips, risk_pct):
        """
        Generate comprehensive AI-powered analysis of strategy performance
        Two-column layout: Verbal analysis on left, Statistical metrics on right
        Includes full QuantStats-style metrics suite, backtest criteria analysis, and market conditions
        """
        if trades_df.empty:
            return "<p style='color: #888888;'>No trades available for analysis.</p>"

        # Calculate key metrics
        total_return = metrics.get('total_return_pct', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        total_trades = len(trades_df)
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        final_balance = metrics.get('final_balance', initial_balance)

        # Calculate additional statistics
        import numpy as np
        from scipy import stats as scipy_stats

        returns = trades_df['monetary_pnl'].values
        returns_pct = (returns / initial_balance) * 100

        # Basic statistics
        mean_return = np.mean(returns_pct)
        std_dev = np.std(returns_pct)
        variance = np.var(returns_pct)
        skewness = float(scipy_stats.skew(returns_pct)) if len(returns_pct) > 2 else 0
        kurtosis = float(scipy_stats.kurtosis(returns_pct)) if len(returns_pct) > 2 else 0

        # Winning vs losing statistics
        win_trades = trades_df[trades_df['monetary_pnl'] > 0]
        loss_trades = trades_df[trades_df['monetary_pnl'] <= 0]

        wins_count = len(win_trades)
        losses_count = len(loss_trades)

        # QuantStats metrics
        best_return = returns_pct.max() if len(returns_pct) > 0 else 0
        worst_return = returns_pct.min() if len(returns_pct) > 0 else 0

        # CAGR (Compound Annual Growth Rate)
        if not trades_df.empty:
            days = (trades_df['exit_time'].iloc[-1] - trades_df['exit_time'].iloc[0]).days
            years = days / 365.25 if days > 0 else 1
            cagr = (((final_balance / initial_balance) ** (1 / years)) - 1) * 100 if years > 0 else 0
        else:
            cagr = 0
            years = 1

        # Calmar Ratio (CAGR / Max Drawdown)
        calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0

        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns_pct, 5) if len(returns_pct) > 0 else 0

        # Conditional Value at Risk (CVaR) - Expected Shortfall
        cvar_95 = returns_pct[returns_pct <= var_95].mean() if len(returns_pct[returns_pct <= var_95]) > 0 else 0

        # Payoff Ratio (Avg Win / Avg Loss)
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Gain to Pain Ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        gain_to_pain = positive_returns / negative_returns if negative_returns != 0 else 0

        # Recovery Factor (Total Return / Max Drawdown)
        recovery_factor = abs(total_return / max_dd) if max_dd != 0 else 0

        # Ulcer Index (measure of downside volatility)
        if not trades_df.empty:
            equity_curve = trades_df['balance'].values
            running_max = np.maximum.accumulate(equity_curve)
            drawdown_pct = ((equity_curve - running_max) / running_max) * 100
            ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
        else:
            ulcer_index = 0

        # Risk of Ruin (simplified Kelly Criterion based)
        win_prob = win_rate / 100
        loss_prob = 1 - win_prob
        if payoff_ratio > 0 and win_prob > 0:
            kelly_pct = (win_prob * payoff_ratio - loss_prob) / payoff_ratio
            risk_of_ruin = ((1 - kelly_pct) / (1 + kelly_pct)) ** 100 if kelly_pct > 0 else 1.0
        else:
            kelly_pct = 0
            risk_of_ruin = 1.0

        # Outlier analysis
        q1 = np.percentile(returns_pct, 25)
        q3 = np.percentile(returns_pct, 75)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        outliers = returns_pct[(returns_pct < q1 - outlier_threshold) | (returns_pct > q3 + outlier_threshold)]
        outlier_win_ratio = len(outliers[outliers > 0]) / len(outliers) if len(outliers) > 0 else 0
        outlier_loss_ratio = len(outliers[outliers < 0]) / len(outliers) if len(outliers) > 0 else 0

        # Consecutive wins/losses
        trades_df_copy = trades_df.copy()
        trades_df_copy['is_win'] = (trades_df_copy['monetary_pnl'] > 0).astype(int)

        # Calculate trade durations
        trades_df_copy['duration_hours'] = (trades_df_copy['exit_time'] - trades_df_copy['entry_time']).dt.total_seconds() / 3600
        avg_trade_duration = trades_df_copy['duration_hours'].mean() if len(trades_df_copy) > 0 else 0
        max_trade_duration = trades_df_copy['duration_hours'].max() if len(trades_df_copy) > 0 else 0
        min_trade_duration = trades_df_copy['duration_hours'].min() if len(trades_df_copy) > 0 else 0

        # Calculate average winning and losing trade durations
        avg_win_duration = trades_df_copy[trades_df_copy['monetary_pnl'] > 0]['duration_hours'].mean() if len(win_trades) > 0 else 0
        avg_loss_duration = trades_df_copy[trades_df_copy['monetary_pnl'] <= 0]['duration_hours'].mean() if len(loss_trades) > 0 else 0

        # Calculate streaks
        streak_changes = trades_df_copy['is_win'].diff().fillna(0) != 0
        streak_ids = streak_changes.cumsum()
        streaks = trades_df_copy.groupby([streak_ids, 'is_win']).size()

        max_win_streak = int(streaks[streaks.index.get_level_values(1) == 1].max()) if any(streaks.index.get_level_values(1) == 1) else 0
        max_loss_streak = int(streaks[streaks.index.get_level_values(1) == 0].max()) if any(streaks.index.get_level_values(1) == 0) else 0

        # Expectancy (Expected value per trade)
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

        # Tail Ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns_pct, 95) if len(returns_pct) > 0 else 0
        p5 = np.percentile(returns_pct, 5) if len(returns_pct) > 0 else 0
        tail_ratio = abs(p95 / p5) if p5 != 0 else 0

        # Common Sense Ratio (Profit Factor - 1)
        common_sense_ratio = profit_factor - 1

        # Monthly performance variability
        trades_df_copy['month'] = trades_df_copy['exit_time'].dt.to_period('M')
        if len(trades_df_copy['month'].unique()) > 1:
            monthly_returns = trades_df_copy.groupby('month')['monetary_pnl'].sum()
            monthly_std = monthly_returns.std()
            monthly_mean = monthly_returns.mean()
            positive_months = len(monthly_returns[monthly_returns > 0])
            total_months = len(monthly_returns)
        else:
            monthly_std = 0
            monthly_mean = 0
            positive_months = 0
            total_months = 1

        # Calculate buy & hold comparison (using actual underlying performance)
        # Simulate buy & hold: same leverage, held entire period
        if not trades_df.empty:
            # Get first and last prices
            first_price = trades_df['entry_price'].iloc[0]
            last_price = trades_df['exit_price'].iloc[-1]

            # Calculate price change
            price_change_pct = ((last_price - first_price) / first_price) * 100

            # Apply leverage to buy & hold
            buy_hold_return = price_change_pct * leverage

            # Calculate alpha (outperformance)
            strategy_outperformance = total_return - buy_hold_return
        else:
            buy_hold_return = 0
            strategy_outperformance = 0

        # Performance assessment
        if total_return > 20:
            performance = "exceptional"
            perf_color = "#00ff88"
        elif total_return > 10:
            performance = "strong"
            perf_color = "#00ff88"
        elif total_return > 0:
            performance = "positive"
            perf_color = "#00d4ff"
        else:
            performance = "concerning"
            perf_color = "#ff3366"

        # Risk-reward ratio
        risk_reward = payoff_ratio

        # Extract underlying asset name
        underlying_asset = f"{pair} ({timeframe} timeframe)"

        # Get current market conditions (Note: These would need to be passed in or fetched via API in production)
        # For now, we'll use placeholder text that explains what should be monitored
        from datetime import datetime
        current_date = datetime.now().strftime("%B %Y")

        analysis_html = f"""
        <div style="background: linear-gradient(135deg, #161616 0%, #1a1a1a 100%); 
                    border-radius: 16px; 
                    padding: 40px; 
                    border: 1px solid #222222; 
                    margin-bottom: 60px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);">
            <h3 style="font-size: 18px; 
                       font-weight: 700; 
                       color: {perf_color}; 
                       margin-bottom: 32px; 
                       text-transform: uppercase; 
                       letter-spacing: 1.5px;
                       display: flex;
                       align-items: center;
                       gap: 12px;">
                <span style="width: 4px; height: 24px; background: {perf_color}; border-radius: 2px;"></span>
                AI PERFORMANCE ANALYSIS
            </h3>

            <!-- Two Column Layout -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-bottom: 40px;">

                <!-- LEFT COLUMN: Verbal Analysis -->
                <div style="color: #e0e0e0; font-size: 14px; line-height: 1.9; font-weight: 300;">

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px;">
                        Backtest Criteria & Underlying Asset
                    </h4>
                    <p style="margin-bottom: 20px;">
                        <strong style="color: #00d4ff; font-size: 16px;">UNDERLYING: {underlying_asset}</strong><br><br>
                        This backtest evaluated the <strong style="color: #ffffff;">{strategy_name}</strong> strategy 
                        on <strong style="color: #ffffff;">{pair}</strong> using <strong style="color: #ffffff;">{timeframe}</strong> candles 
                        from <strong style="color: #ffffff;">{start_date}</strong> to <strong style="color: #ffffff;">{end_date}</strong> 
                        ({years:.1f} years of data). The strategy operated with the following parameters:
                        <strong style="color: #ffffff;">{leverage}x leverage</strong>, 
                        <strong style="color: #ffffff;">{sl_pips}-pip stop loss</strong>, 
                        <strong style="color: #ffffff;">{tp_pips}-pip take profit</strong>, 
                        and <strong style="color: #ffffff;">{risk_pct}% risk per trade</strong>. 
                        Starting capital: <strong style="color: #ffffff;">${initial_balance:,.0f}</strong>.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px;">
                        Strategy vs Underlying Performance
                    </h4>
                    <p style="margin-bottom: 20px;">
                        <strong style="color: #ffffff;">Active Strategy Return:</strong> <strong style="color: {perf_color};">{total_return:.2f}%</strong> 
                        (CAGR: <strong style="color: {perf_color};">{cagr:.2f}%</strong>)<br>
                        <strong style="color: #ffffff;">Buy & Hold Return:</strong> <strong style="color: {'#00ff88' if buy_hold_return > 0 else '#ff3366'};">{buy_hold_return:.2f}%</strong> 
                        ({leverage}x leveraged)<br>
                        <strong style="color: #ffffff;">Alpha Generated:</strong> <strong style="color: {'#00ff88' if strategy_outperformance > 0 else '#ff3366'};">{strategy_outperformance:+.2f}%</strong><br><br>

                        {'🎯 The active strategy <strong style="color: #00ff88;">OUTPERFORMED</strong> the underlying asset by <strong style="color: #00ff88;">{abs(strategy_outperformance):.2f}%</strong>. ' if strategy_outperformance > 0 else '⚠️ The active strategy <strong style="color: #ff3366;">UNDERPERFORMED</strong> the underlying by <strong style="color: #ff3366;">{abs(strategy_outperformance):.2f}%</strong>. '}
                        {f'This demonstrates clear <strong>alpha generation</strong>, meaning the strategy successfully identified profitable entry and exit points beyond simple directional exposure. The {total_trades} trades executed captured market inefficiencies that passive holding could not exploit.' if strategy_outperformance > 5 else f'While the strategy generated positive returns, it slightly lagged a simple buy & hold approach. This suggests the {pair} market trended favorably during the test period, and transaction costs may have eroded some gains. Consider optimizing entry criteria or holding periods.' if strategy_outperformance > -5 and total_return > 0 else f'The strategy underperformed passive holding, indicating that market conditions favored trend-following over the active approach. The {total_trades} trades may have been mistimed relative to the underlying price action.' if total_return > 0 else 'Both the strategy and passive approach would have resulted in losses, suggesting this was a challenging period for the market.'}
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px;">
                        Performance Overview
                    </h4>
                    <p style="margin-bottom: 20px;">
                        The strategy demonstrates <strong style="color: {perf_color};">{performance}</strong> performance, 
                        growing the account from <strong style="color: #ffffff;">${initial_balance:,.0f}</strong> to 
                        <strong style="color: {perf_color};">${final_balance:,.0f}</strong> 
                        (a <strong style="color: {perf_color};">{((final_balance/initial_balance - 1) * 100):.1f}%</strong> gain). 
                        The Calmar ratio of <strong style="color: #ffffff;">{calmar:.2f}</strong> 
                        {'indicates excellent risk-adjusted performance (>3.0 is exceptional)' if calmar > 3 else 'shows good return relative to drawdown (>1.0 is favorable)' if calmar > 1 else 'suggests room for improvement in return vs drawdown'}, 
                        balancing profitability against maximum portfolio decline.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Win Rate & Consistency
                    </h4>
                    <p style="margin-bottom: 20px;">
                        With a win rate of <strong style="color: #ffffff;">{win_rate:.1f}%</strong> 
                        ({wins_count} wins vs {losses_count} losses), the strategy {'demonstrates strong directional accuracy' if win_rate > 55 else 'maintains adequate win frequency' if win_rate > 45 else 'requires larger wins to offset the lower hit rate'}. 
                        The maximum winning streak of <strong style="color: #ffffff;">{max_win_streak}</strong> trades 
                        versus the maximum losing streak of <strong style="color: #ffffff;">{max_loss_streak}</strong> trades 
                        indicates {'healthy momentum capture with controlled drawdown periods' if max_win_streak > max_loss_streak * 1.5 else 'balanced performance with typical market variability' if max_win_streak >= max_loss_streak else 'vulnerability to extended losing periods that should be monitored'}.
                        Monthly win rate: <strong style="color: #ffffff;">{(positive_months/total_months*100):.1f}%</strong> ({positive_months}/{total_months} profitable months).
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Risk Management
                    </h4>
                    <p style="margin-bottom: 20px;">
                        The strategy exhibits a maximum drawdown of <strong style="color: #ffffff;">{max_dd:.2f}%</strong>, 
                        which falls into the {'conservative' if max_dd < 10 else 'moderate' if max_dd < 20 else 'aggressive'} risk category. 
                        The Ulcer Index of <strong style="color: #ffffff;">{ulcer_index:.2f}</strong> quantifies downside volatility, 
                        while the Recovery Factor of <strong style="color: #ffffff;">{recovery_factor:.2f}</strong> 
                        {'demonstrates excellent recovery capability' if recovery_factor > 3 else 'shows adequate recovery from drawdowns' if recovery_factor > 2 else 'suggests extended recovery periods'}.
                        {'This low drawdown indicates excellent capital preservation and risk control.' if max_dd < 10 else 'This moderate drawdown suggests balanced risk-taking appropriate for most trading accounts.' if max_dd < 20 else 'This elevated drawdown requires careful position sizing and risk management.'}
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Tail Risk & Outliers
                    </h4>
                    <p style="margin-bottom: 20px;">
                        Value at Risk (VaR 95%): <strong style="color: #ffffff;">{var_95:.2f}%</strong> – 
                        95% of trades lose less than this amount.
                        Conditional VaR (CVaR): <strong style="color: #ffffff;">{cvar_95:.2f}%</strong> – 
                        expected loss in worst 5% of outcomes.
                        The tail ratio of <strong style="color: #ffffff;">{tail_ratio:.2f}</strong> 
                        {'indicates strong positive skew (upside potential exceeds downside risk)' if tail_ratio > 1.5 else 'suggests balanced tail risk' if tail_ratio > 0.8 else 'reveals concerning downside tail risk'}.
                        Outlier analysis: <strong style="color: #ffffff;">{len(outliers)}</strong> outliers detected 
                        ({outlier_win_ratio*100:.0f}% wins, {outlier_loss_ratio*100:.0f}% losses).
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Optimization Strategies
                    </h4>
                    <p style="margin-bottom: 20px;">
                        <strong style="color: #ffffff;">Position Sizing:</strong> {'Consider Kelly Criterion position sizing: {kelly_pct*100:.1f}% of capital per trade (or {kelly_pct*50:.1f}% for half-Kelly safety margin)' if kelly_pct > 0 else 'Implement fixed fractional sizing given current metrics'}.<br><br>

                        <strong style="color: #ffffff;">Stop Loss Optimization:</strong> {'Current stops are effective; analyze trade duration to identify premature exits' if max_dd < 15 else 'Tighten stops to reduce drawdown depth, or widen to avoid noise-driven exits'}.<br><br>

                        <strong style="color: #ffffff;">Take Profit Enhancement:</strong> {'Consider trailing stops to capture extended moves given positive skew' if skewness > 0.5 else 'Fixed profit targets appear optimal; avoid premature exits' if risk_reward > 2 else 'Scale out of winners to improve risk-reward ratio'}.<br><br>

                        <strong style="color: #ffffff;">Entry Timing:</strong> {'Focus on high-conviction setups to maintain win rate' if win_rate > 55 else 'Refine entry criteria to improve directional accuracy' if win_rate < 45 else 'Current entry logic is well-calibrated'}.<br><br>

                        <strong style="color: #ffffff;">Risk-Reward Target:</strong> {'Excellent current ratio of {risk_reward:.2f}:1; maintain this edge' if risk_reward > 2 else 'Target minimum 2:1 risk-reward; current {risk_reward:.2f}:1 needs improvement' if risk_reward < 2 else 'Solid {risk_reward:.2f}:1 ratio provides adequate edge'}.<br><br>

                        <strong style="color: #ffffff;">Risk of Ruin:</strong> {'Low probability of account ruin ({risk_of_ruin*100:.2f}%) - strategy is sustainable long-term' if risk_of_ruin < 0.05 else 'Moderate risk of ruin ({risk_of_ruin*100:.1f}%) - consider reducing position sizes' if risk_of_ruin < 0.20 else 'High risk of ruin ({risk_of_ruin*100:.0f}%) - significant risk management overhaul required'}.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Current Market Conditions ({current_date})
                    </h4>
                    <p style="margin-bottom: 0;">
                        <strong style="color: #ffffff;">⚠️ IMPORTANT:</strong> This backtest was conducted on historical data from {start_date} to {end_date}. 
                        Before deploying this strategy in live markets, analyze current macroeconomic conditions:<br><br>

                        <strong style="color: #ffffff;">📊 Key Economic Indicators to Monitor:</strong><br>
                        • <strong>Federal Funds Rate:</strong> Check current Fed policy rate and upcoming FOMC meetings<br>
                        • <strong>Inflation (CPI/PCE):</strong> Review latest Consumer Price Index and core PCE inflation<br>
                        • <strong>Unemployment Rate:</strong> Monitor labor market strength via BLS reports<br>
                        • <strong>GDP Growth:</strong> Assess economic expansion/contraction trends<br>
                        • <strong>Market Volatility (VIX):</strong> Evaluate current fear/greed levels<br>
                        • <strong>Currency Strength:</strong> Analyze {pair.split('/')[0]} and {pair.split('/')[1]} fundamental drivers<br><br>

                        <strong style="color: #ffffff;">🌍 Geopolitical Factors:</strong><br>
                        • Central bank policy divergence between currencies<br>
                        • Trade agreements and tariff policies<br>
                        • Political stability in major economies<br>
                        • Global risk sentiment (risk-on vs risk-off)<br><br>

                        <strong style="color: #00d4ff;">💡 Strategy Adaptation Guidance:</strong><br>
                        {'• High volatility environments may require tighter stops and smaller position sizes<br>• Low volatility periods could benefit from wider targets to avoid premature exits<br>' if std_dev > 2 else '• Current strategy parameters appear well-suited for moderate volatility<br>'}
                        • Monitor correlations: {pair} typically moves with {'risk sentiment and equity markets' if 'JPY' in pair or 'CHF' in pair else 'commodity prices and global trade' if 'AUD' in pair or 'NZD' in pair or 'CAD' in pair else 'interest rate differentials and economic data'}<br>
                        • Be prepared to pause trading during major central bank announcements or black swan events<br>
                        • Backtest performance may not reflect current market regime - validate with recent walk-forward data
                    </p>
                </div>

                <!-- RIGHT COLUMN: Statistical Analysis -->
                <div style="background: rgba(0, 0, 0, 0.3); 
                           border-radius: 12px; 
                           padding: 32px; 
                           border: 1px solid #222222;">

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px;">
                        Core Performance Metrics
                    </h4>

                    <!-- Metrics Table with Hover Explanations -->
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px;">
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row" 
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Total Return measures the overall percentage gain or loss from your initial investment. Exceptional: >20%, Good: >10%, Acceptable: >5%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Total Return ⓘ</td>
                            <td style="padding: 10px 0; color: {perf_color}; text-align: right; font-weight: 700;">{total_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="CAGR (Compound Annual Growth Rate) shows annualized returns accounting for compounding. Excellent: >15%, Good: >10%, Minimum: >5%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">CAGR ⓘ</td>
                            <td style="padding: 10px 0; color: {perf_color}; text-align: right; font-weight: 700;">{cagr:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Buy & Hold Return shows what you would have earned by simply holding the underlying asset with the same leverage. Compares passive vs active approach.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Buy & Hold Return ⓘ</td>
                            <td style="padding: 10px 0; color: {'#00ff88' if buy_hold_return > 0 else '#ff3366'}; text-align: right;">{buy_hold_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Alpha measures outperformance vs the underlying asset. Positive alpha means your strategy beat the market. Good: >5%, Excellent: >10%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Alpha (Outperformance) ⓘ</td>
                            <td style="padding: 10px 0; color: {'#00ff88' if strategy_outperformance > 0 else '#ff3366'}; text-align: right; font-weight: 700;">{strategy_outperformance:+.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Sharpe Ratio measures risk-adjusted returns (return per unit of volatility). Excellent: >2.0, Good: >1.5, Acceptable: >1.0, Poor: <0.5">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Sharpe Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{sharpe:.2f} <span style="color: {'#00ff88' if sharpe > 2 else '#00d4ff' if sharpe > 1.5 else '#888888' if sharpe > 1 else '#ff3366'}; font-size: 10px;">{'★★★' if sharpe > 2 else '★★' if sharpe > 1.5 else '★' if sharpe > 1 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Sortino Ratio is like Sharpe but only penalizes downside volatility (losses), not upside. Excellent: >2.0, Good: >1.5, Acceptable: >1.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Sortino Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{sortino:.2f} <span style="color: {'#00ff88' if sortino > 2 else '#00d4ff' if sortino > 1.5 else '#888888' if sortino > 1 else '#ff3366'}; font-size: 10px;">{'★★★' if sortino > 2 else '★★' if sortino > 1.5 else '★' if sortino > 1 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Calmar Ratio = CAGR / Max Drawdown. Measures return relative to worst loss. Excellent: >3.0, Good: >2.0, Acceptable: >1.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Calmar Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{calmar:.2f} <span style="color: {'#00ff88' if calmar > 3 else '#00d4ff' if calmar > 2 else '#888888' if calmar > 1 else '#ff3366'}; font-size: 10px;">{'★★★' if calmar > 3 else '★★' if calmar > 2 else '★' if calmar > 1 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Profit Factor = Gross Profit / Gross Loss. Shows how much you make per dollar lost. Excellent: >2.0, Good: >1.5, Minimum: >1.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Profit Factor ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{profit_factor:.2f} <span style="color: {'#00ff88' if profit_factor > 2 else '#00d4ff' if profit_factor > 1.5 else '#888888' if profit_factor > 1 else '#ff3366'}; font-size: 10px;">{'★★★' if profit_factor > 2 else '★★' if profit_factor > 1.5 else '★' if profit_factor > 1 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Payoff Ratio = Average Win / Average Loss. Your risk-reward per trade. Excellent: >2.5, Good: >2.0, Minimum: >1.5">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Payoff Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{payoff_ratio:.2f} <span style="color: {'#00ff88' if payoff_ratio > 2.5 else '#00d4ff' if payoff_ratio > 2 else '#888888' if payoff_ratio > 1.5 else '#ff3366'}; font-size: 10px;">{'★★★' if payoff_ratio > 2.5 else '★★' if payoff_ratio > 2 else '★' if payoff_ratio > 1.5 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Win Rate shows percentage of profitable trades. High win rate (>60%) or high payoff ratio (>2.0) needed for profitability. Excellent: >55%, Good: >50%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Win Rate ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{win_rate:.1f}% <span style="color: {'#00ff88' if win_rate > 55 else '#00d4ff' if win_rate > 50 else '#888888' if win_rate > 45 else '#ff3366'}; font-size: 10px;">{'★★★' if win_rate > 55 else '★★' if win_rate > 50 else '★' if win_rate > 45 else '✗'}</span></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Best Trade shows your largest winning trade as a percentage. Monitors if strategy relies on outliers vs consistent gains.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Best Trade ⓘ</td>
                            <td style="padding: 10px 0; color: #00ff88; text-align: right;">{best_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Worst Trade shows your largest losing trade as a percentage. Should be controlled by stop losses. Warning if > 5% of account.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Worst Trade ⓘ</td>
                            <td style="padding: 10px 0; color: #ff3366; text-align: right;">{worst_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Expectancy shows expected profit per trade in dollars. Positive expectancy = profitable strategy. Excellent: >$100, Good: >$50">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Expectancy ⓘ</td>
                            <td style="padding: 10px 0; color: {'#00ff88' if expectancy > 0 else '#ff3366'}; text-align: right; font-weight: 700;">${expectancy:.2f}</td>
                        </tr>
                    </table>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">
                        Risk Metrics
                    </h4>

                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px;">
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Max Drawdown is the largest peak-to-trough decline. Measures worst case loss period. Conservative: <10%, Moderate: <20%, Aggressive: >20%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Max Drawdown ⓘ</td>
                            <td style="padding: 10px 0; color: #ff3366; text-align: right;">{max_dd:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Ulcer Index measures both depth and duration of drawdowns. Lower is better. Excellent: <5, Good: <10, High Risk: >15">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Ulcer Index ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{ulcer_index:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Recovery Factor = Total Return / Max Drawdown. How well you recover from losses. Excellent: >3.0, Good: >2.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Recovery Factor ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{recovery_factor:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="VaR (Value at Risk) at 95% confidence: 95% of trades will lose less than this amount. Measures typical downside risk.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">VaR (95%) ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{var_95:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="CVaR (Conditional VaR) shows expected loss in the worst 5% of outcomes. Measures tail risk. Good: <-3%, Concerning: <-5%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">CVaR (95%) ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{cvar_95:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Tail Ratio = 95th percentile gain / 5th percentile loss. Measures upside vs downside extremes. Good: >1.5, Excellent: >2.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Tail Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{tail_ratio:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Gain-to-Pain Ratio = Sum of gains / Sum of losses. Similar to profit factor. Excellent: >2.0, Good: >1.5, Minimum: >1.0">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Gain-to-Pain ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{gain_to_pain:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Common Sense Ratio = Profit Factor - 1. Simplified measure of edge. Positive means profitable. >1.0 is good, >2.0 is excellent.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Common Sense Ratio ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{common_sense_ratio:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Risk of Ruin estimates probability of losing entire account. Based on Kelly Criterion. Excellent: <5%, Acceptable: <20%, Dangerous: >50%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Risk of Ruin ⓘ</td>
                            <td style="padding: 10px 0; color: {'#00ff88' if risk_of_ruin < 0.05 else '#ff3366'}; text-align: right;">{risk_of_ruin*100:.2f}%</td>
                        </tr>
                    </table>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">
                        Return Distribution
                    </h4>

                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px;">
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Mean Return shows average return per trade in percentage terms. Should be positive for profitable strategies.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Mean Return per Trade ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{mean_return:.3f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Standard Deviation measures volatility of returns. Lower is more consistent. Low: <2%, Moderate: 2-4%, High: >4%">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Std Deviation ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{std_dev:.3f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Variance is the squared standard deviation. Measures dispersion of returns from the mean.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Variance ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{variance:.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Skewness measures asymmetry. Positive (>0.5) = occasional big wins (good). Negative (<-0.5) = occasional big losses (bad). Zero = symmetric.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Skewness ⓘ</td>
                            <td style="padding: 10px 0; color: {'#00ff88' if skewness > 0 else '#ff3366'}; text-align: right;">{skewness:.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Kurtosis measures tail heaviness. High (>3) = more extreme events. Low (<3) = fewer outliers. Normal distribution = 3.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Kurtosis ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{kurtosis:.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222; position: relative;" class="metric-row"
                            onmouseover="this.style.background='rgba(255,255,255,0.03)'" 
                            onmouseout="this.style.background='transparent'"
                            title="Kelly Criterion shows optimal position size as % of capital per trade. Use half-Kelly for safety. Negative = strategy needs work.">
                            <td style="padding: 10px 0; color: #888888; cursor: help;">Kelly Criterion % ⓘ</td>
                            <td style="padding: 10px 0; color: #ffffff; text-align: right;">{kelly_pct*100:.2f}%</td>
                        </tr>
                    </table>

                    <div style="background: rgba(255, 255, 255, 0.02); 
                               border-left: 3px solid #00d4ff; 
                               padding: 16px; 
                               border-radius: 6px; 
                               font-size: 12px; 
                               line-height: 1.7;
                               color: #b0b0b0;
                               margin-bottom: 24px;">
                        <strong style="color: #00d4ff; display: block; margin-bottom: 8px;">INTERPRETATION:</strong>
                        <strong>Skewness ({skewness:.2f}):</strong> {'Positive skew indicates more frequent small losses with occasional large wins (desirable)' if skewness > 0.5 else 'Negative skew suggests frequent small wins with occasional large losses (risk concern)' if skewness < -0.5 else 'Near-zero skew shows symmetric return distribution'}.<br><br>

                        <strong>Kurtosis ({kurtosis:.2f}):</strong> {'High kurtosis indicates fat tails - higher probability of extreme outcomes' if kurtosis > 3 else 'Normal kurtosis suggests typical market behavior' if kurtosis > 2 else 'Low kurtosis indicates thin tails - fewer extreme events'}.<br><br>

                        <strong>CVaR ({cvar_95:.2f}%):</strong> In the worst 5% of trading days, expect losses around this magnitude. {'Well-controlled tail risk' if abs(cvar_95) < 3 else 'Moderate tail risk exposure' if abs(cvar_95) < 5 else 'Significant tail risk - use protective stops'}.<br><br>

                        <strong>Kelly ({kelly_pct*100:.1f}%):</strong> {'Optimal position size per trade. Half-Kelly ({kelly_pct*50:.1f}%) recommended for safety' if kelly_pct > 0 else 'Negative Kelly suggests strategy needs refinement before position sizing'}.
                    </div>

                    <!-- Visual Ratio Comparison Charts -->
                    <h4 style="color: #ffffff; font-size: 14px; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px;">
                        Performance vs Standards
                    </h4>

                    <!-- Sharpe Ratio Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Sharpe Ratio</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{sharpe:.2f}</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">POOR</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">GOOD</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(sharpe/3*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#ff3366' if sharpe < 0 else '#00ff88'}; box-shadow: 0 0 10px {'#ff3366' if sharpe < 0 else '#00ff88'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>1.0</span><span>2.0</span><span>3.0+</span>
                        </div>
                    </div>

                    <!-- Sortino Ratio Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Sortino Ratio</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{sortino:.2f}</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">POOR</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">GOOD</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(sortino/3*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#ff3366' if sortino < 0 else '#00d4ff'}; box-shadow: 0 0 10px {'#ff3366' if sortino < 0 else '#00d4ff'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>1.0</span><span>2.0</span><span>3.0+</span>
                        </div>
                    </div>

                    <!-- Profit Factor Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Profit Factor</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{profit_factor:.2f}</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">LOSING</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">GOOD</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(profit_factor/3*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#ff3366' if profit_factor < 1 else '#b967ff'}; box-shadow: 0 0 10px {'#ff3366' if profit_factor < 1 else '#b967ff'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>1.0</span><span>2.0</span><span>3.0+</span>
                        </div>
                    </div>

                    <!-- Win Rate Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Win Rate</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{win_rate:.1f}%</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 40%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">POOR</div>
                            </div>
                            <div style="position: absolute; left: 40%; top: 0; height: 100%; width: 30%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">ACCEPTABLE</div>
                            </div>
                            <div style="position: absolute; left: 70%; top: 0; height: 100%; width: 30%;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(win_rate, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if win_rate > 55 else '#00d4ff' if win_rate > 50 else '#ffcc00' if win_rate > 45 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if win_rate > 55 else '#00d4ff' if win_rate > 50 else '#ffcc00' if win_rate > 45 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0%</span><span>40%</span><span>70%</span><span>100%</span>
                        </div>
                    </div>

                    <!-- Payoff Ratio (Avg Win / Avg Loss) Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Payoff Ratio (Avg Win/Loss)</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{payoff_ratio:.2f}:1</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 40%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">POOR</div>
                            </div>
                            <div style="position: absolute; left: 40%; top: 0; height: 100%; width: 30%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">GOOD</div>
                            </div>
                            <div style="position: absolute; left: 70%; top: 0; height: 100%; width: 30%;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(payoff_ratio/4*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if payoff_ratio > 2.5 else '#00d4ff' if payoff_ratio > 2 else '#ffcc00' if payoff_ratio > 1.5 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if payoff_ratio > 2.5 else '#00d4ff' if payoff_ratio > 2 else '#ffcc00' if payoff_ratio > 1.5 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>1.5</span><span>2.5</span><span>4.0+</span>
                        </div>
                    </div>

                    <!-- Average Win Size Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Avg Win Size (% of Capital)</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{(avg_win/initial_balance*100):.2f}%</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">SMALL</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">OPTIMAL</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #ffcc00; text-align: center; padding-top: 4px;">LARGE</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min((avg_win/initial_balance*100)/6*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if 1 <= (avg_win/initial_balance*100) <= 4 else '#ffcc00'}; box-shadow: 0 0 10px {'#00ff88' if 1 <= (avg_win/initial_balance*100) <= 4 else '#ffcc00'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0%</span><span>2%</span><span>4%</span><span>6%+</span>
                        </div>
                    </div>

                    <!-- Average Loss Size Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Avg Loss Size (% of Capital)</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{abs(avg_loss/initial_balance*100):.2f}%</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">CONTROLLED</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #ffcc00; text-align: center; padding-top: 4px;">MODERATE</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">EXCESSIVE</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(abs(avg_loss/initial_balance*100)/6*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if abs(avg_loss/initial_balance*100) < 2 else '#ffcc00' if abs(avg_loss/initial_balance*100) < 4 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if abs(avg_loss/initial_balance*100) < 2 else '#ffcc00' if abs(avg_loss/initial_balance*100) < 4 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0%</span><span>2%</span><span>4%</span><span>6%+</span>
                        </div>
                    </div>

                    <!-- Max Drawdown Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Max Drawdown</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{max_dd:.2f}%</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 25%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: 25%; top: 0; height: 100%; width: 25%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">MODERATE</div>
                            </div>
                            <div style="position: absolute; left: 50%; top: 0; height: 100%; width: 50%;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">HIGH RISK</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(abs(max_dd)/50*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if abs(max_dd) < 10 else '#00d4ff' if abs(max_dd) < 20 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if abs(max_dd) < 10 else '#00d4ff' if abs(max_dd) < 20 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0%</span><span>-10%</span><span>-20%</span><span>-50%+</span>
                        </div>
                    </div>

                    <!-- Average Trade Duration Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Avg Trade Duration</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{avg_trade_duration:.1f}h</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 25%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">SCALPING</div>
                            </div>
                            <div style="position: absolute; left: 25%; top: 0; height: 100%; width: 25%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">DAY TRADE</div>
                            </div>
                            <div style="position: absolute; left: 50%; top: 0; height: 100%; width: 25%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">SWING</div>
                            </div>
                            <div style="position: absolute; left: 75%; top: 0; height: 100%; width: 25%;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">POSITION</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(avg_trade_duration/200*100, 100))}%; top: -2px; width: 3px; height: 24px; background: #00d4ff; box-shadow: 0 0 10px #00d4ff;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>&lt;1h</span><span>24h</span><span>1wk</span><span>200h+</span>
                        </div>
                    </div>

                    <!-- Longest Trade vs Average Duration Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Longest Trade vs Average</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{(max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0):.1f}x</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 40%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">CONSISTENT</div>
                            </div>
                            <div style="position: absolute; left: 40%; top: 0; height: 100%; width: 30%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">MODERATE</div>
                            </div>
                            <div style="position: absolute; left: 70%; top: 0; height: 100%; width: 30%;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">HIGH VARIANCE</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min((max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0)/10*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if (max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0) < 3 else '#ffcc00' if (max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0) < 5 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if (max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0) < 3 else '#ffcc00' if (max_trade_duration/avg_trade_duration if avg_trade_duration > 0 else 0) < 5 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>1x</span><span>3x</span><span>7x</span><span>10x+</span>
                        </div>
                    </div>

                    <!-- Win Duration vs Loss Duration Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Avg Win Duration vs Loss Duration</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">Win: {avg_win_duration:.1f}h | Loss: {avg_loss_duration:.1f}h</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 40%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">LOSSES LONGER</div>
                            </div>
                            <div style="position: absolute; left: 40%; top: 0; height: 100%; width: 20%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">EQUAL</div>
                            </div>
                            <div style="position: absolute; left: 60%; top: 0; height: 100%; width: 40%;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">WINS LONGER</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(40 + (avg_win_duration - avg_loss_duration) / max(avg_win_duration, avg_loss_duration, 1) * 30, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if avg_win_duration > avg_loss_duration else '#ff3366' if avg_loss_duration > avg_win_duration else '#ffcc00'}; box-shadow: 0 0 10px {'#00ff88' if avg_win_duration > avg_loss_duration else '#ff3366' if avg_loss_duration > avg_win_duration else '#ffcc00'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>Loss 2x</span><span>Equal</span><span>Win 2x</span>
                        </div>
                    </div>

                    <!-- Consecutive Win Streak Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Max Consecutive Wins</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{max_win_streak} trades</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">LOW</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">GOOD</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(max_win_streak/15*100, 100))}%; top: -2px; width: 3px; height: 24px; background: #00ff88; box-shadow: 0 0 10px #00ff88;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>5</span><span>10</span><span>15+</span>
                        </div>
                    </div>

                    <!-- Consecutive Loss Streak Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Max Consecutive Losses</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{max_loss_streak} trades</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">MODERATE</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">CONCERNING</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(max_loss_streak/15*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if max_loss_streak < 5 else '#ffcc00' if max_loss_streak < 10 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if max_loss_streak < 5 else '#ffcc00' if max_loss_streak < 10 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>5</span><span>10</span><span>15+</span>
                        </div>
                    </div>

                    <!-- Risk of Ruin Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Risk of Ruin</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{risk_of_ruin*100:.2f}%</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 20%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">EXCELLENT</div>
                            </div>
                            <div style="position: absolute; left: 20%; top: 0; height: 100%; width: 30%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">ACCEPTABLE</div>
                            </div>
                            <div style="position: absolute; left: 50%; top: 0; height: 100%; width: 50%;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">DANGEROUS</div>
                            </div>
                            <div style="position: absolute; left: {max(0, min(risk_of_ruin*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if risk_of_ruin < 0.05 else '#ffcc00' if risk_of_ruin < 0.20 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if risk_of_ruin < 0.05 else '#ffcc00' if risk_of_ruin < 0.20 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0%</span><span>5%</span><span>20%</span><span>100%</span>
                        </div>
                    </div>

                    <!-- Skewness Bar (centered at 0) -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Skewness (Distribution Asymmetry)</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{skewness:.2f}</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 50%; border-right: 1px solid #ff3366;">
                                <div style="font-size: 8px; color: #ff3366; text-align: center; padding-top: 4px;">NEGATIVE</div>
                            </div>
                            <div style="position: absolute; left: 50%; top: 0; height: 100%; width: 50%;">
                                <div style="font-size: 8px; color: #00ff88; text-align: center; padding-top: 4px;">POSITIVE</div>
                            </div>
                            <!-- Clamp between 0% and 100%, but always show marker -->
                            <div style="position: absolute; left: {max(0, min(50 + skewness*25, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#00ff88' if skewness > 0 else '#ff3366'}; box-shadow: 0 0 10px {'#00ff88' if skewness > 0 else '#ff3366'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>-2.0</span><span>0</span><span>+2.0</span>
                        </div>
                    </div>

                    <!-- Kurtosis Bar -->
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 11px; color: #888888;">Kurtosis (Tail Risk)</span>
                            <span style="font-size: 11px; color: #ffffff; font-weight: 700;">{kurtosis:.2f}</span>
                        </div>
                        <div style="background: #222222; height: 20px; border-radius: 4px; position: relative; overflow: visible;">
                            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">LOW</div>
                            </div>
                            <div style="position: absolute; left: 33.33%; top: 0; height: 100%; width: 33.33%; border-right: 1px solid #000;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">NORMAL</div>
                            </div>
                            <div style="position: absolute; left: 66.66%; top: 0; height: 100%; width: 33.34%;">
                                <div style="font-size: 8px; color: #666; text-align: center; padding-top: 4px;">HIGH (FAT TAILS)</div>
                            </div>
                            <!-- Clamp between 0% and 100%, handle negative values at 0 -->
                            <div style="position: absolute; left: {max(0, min(abs(kurtosis)/6*100, 100))}%; top: -2px; width: 3px; height: 24px; background: {'#ff3366' if abs(kurtosis) > 4 else '#ffcc00' if abs(kurtosis) > 3 else '#00ff88'}; box-shadow: 0 0 10px {'#ff3366' if abs(kurtosis) > 4 else '#ffcc00' if abs(kurtosis) > 3 else '#00ff88'};"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: #666;">
                            <span>0</span><span>3.0 (Normal)</span><span>6.0+</span>
                        </div>
                    </div>

                </div>
            </div>
        </div>
        """

        return analysis_html
        """
        Generate comprehensive AI-powered analysis of strategy performance
        Two-column layout: Verbal analysis on left, Statistical metrics on right
        """
        if trades_df.empty:
            return "<p style='color: #888888;'>No trades available for analysis.</p>"

        # Calculate key metrics
        total_return = metrics.get('total_return_pct', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        total_trades = len(trades_df)
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        final_balance = metrics.get('final_balance', initial_balance)

        # Calculate additional statistics
        import numpy as np
        returns = trades_df['monetary_pnl'].values
        returns_pct = (returns / initial_balance) * 100

        std_dev = np.std(returns_pct)
        variance = np.var(returns_pct)
        skewness = float(np.mean(((returns_pct - np.mean(returns_pct)) / std_dev) ** 3)) if std_dev > 0 else 0
        kurtosis = float(np.mean(((returns_pct - np.mean(returns_pct)) / std_dev) ** 4)) if std_dev > 0 else 0

        # Winning vs losing statistics
        win_trades = trades_df[trades_df['monetary_pnl'] > 0]
        loss_trades = trades_df[trades_df['monetary_pnl'] <= 0]

        wins_count = len(win_trades)
        losses_count = len(loss_trades)

        # Calculate buy & hold comparison
        if not trades_df.empty:
            buy_hold_return = total_return  # Simplified - same growth rate
            strategy_outperformance = total_return - buy_hold_return
        else:
            buy_hold_return = 0
            strategy_outperformance = 0

        # Performance assessment
        if total_return > 20:
            performance = "exceptional"
            perf_color = "#00ff88"
        elif total_return > 10:
            performance = "strong"
            perf_color = "#00ff88"
        elif total_return > 0:
            performance = "positive"
            perf_color = "#00d4ff"
        else:
            performance = "concerning"
            perf_color = "#ff3366"

        # Risk-reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Win streak analysis
        trades_df_copy = trades_df.copy()
        trades_df_copy['is_win'] = (trades_df_copy['monetary_pnl'] > 0).astype(int)

        # Calculate streaks
        streak_changes = trades_df_copy['is_win'].diff().fillna(0) != 0
        streak_ids = streak_changes.cumsum()
        streaks = trades_df_copy.groupby([streak_ids, 'is_win']).size()

        max_win_streak = int(streaks[streaks.index.get_level_values(1) == 1].max()) if any(streaks.index.get_level_values(1) == 1) else 0
        max_loss_streak = int(streaks[streaks.index.get_level_values(1) == 0].max()) if any(streaks.index.get_level_values(1) == 0) else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

        # Monthly performance variability
        trades_df_copy['month'] = trades_df_copy['exit_time'].dt.to_period('M')
        if len(trades_df_copy['month'].unique()) > 1:
            monthly_returns = trades_df_copy.groupby('month')['monetary_pnl'].sum()
            monthly_std = monthly_returns.std()
            monthly_mean = monthly_returns.mean()
        else:
            monthly_std = 0
            monthly_mean = 0

        analysis_html = f"""
        <div style="background: linear-gradient(135deg, #161616 0%, #1a1a1a 100%); 
                    border-radius: 16px; 
                    padding: 40px; 
                    border: 1px solid #222222; 
                    margin-bottom: 60px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);">
            <h3 style="font-size: 18px; 
                       font-weight: 700; 
                       color: {perf_color}; 
                       margin-bottom: 32px; 
                       text-transform: uppercase; 
                       letter-spacing: 1.5px;
                       display: flex;
                       align-items: center;
                       gap: 12px;">
                <span style="width: 4px; height: 24px; background: {perf_color}; border-radius: 2px;"></span>
                AI PERFORMANCE ANALYSIS
            </h3>

            <!-- Two Column Layout -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-bottom: 40px;">

                <!-- LEFT COLUMN: Verbal Analysis -->
                <div style="color: #e0e0e0; font-size: 14px; line-height: 1.9; font-weight: 300;">

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px;">
                        Performance Overview
                    </h4>
                    <p style="margin-bottom: 20px;">
                        The <strong style="color: #ffffff;">{strategy_name}</strong> strategy demonstrates 
                        <strong style="color: {perf_color};">{performance}</strong> performance, achieving a 
                        <strong style="color: {perf_color};">{total_return:.2f}%</strong> total return 
                        from an initial balance of <strong style="color: #ffffff;">${initial_balance:,.0f}</strong> to 
                        <strong style="color: {perf_color};">${final_balance:,.0f}</strong> across 
                        <strong style="color: #ffffff;">{total_trades}</strong> trades.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Strategy vs Buy & Hold
                    </h4>
                    <p style="margin-bottom: 20px;">
                        {'The active trading strategy <strong style="color: #00ff88;">outperformed</strong> a passive buy & hold approach. ' if strategy_outperformance > 0 else 'The active strategy <strong style="color: #ff3366;">underperformed</strong> relative to buy & hold. '}
                        This {'justifies the added complexity and transaction costs of active trading' if strategy_outperformance > 0 else 'suggests passive holding may be more efficient for this market'}.
                        The strategy's ability to {'capitalize on price movements and limit downside exposure demonstrates clear alpha generation' if total_return > buy_hold_return and max_dd < 15 else 'manage risk while seeking returns is evident, though optimization opportunities exist' if total_return > 0 else 'generate consistent returns requires refinement'}.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Win Rate & Consistency
                    </h4>
                    <p style="margin-bottom: 20px;">
                        With a win rate of <strong style="color: #ffffff;">{win_rate:.1f}%</strong> 
                        ({wins_count} wins vs {losses_count} losses), the strategy {'demonstrates strong directional accuracy' if win_rate > 55 else 'maintains adequate win frequency' if win_rate > 45 else 'requires larger wins to offset the lower hit rate'}. 
                        The maximum winning streak of <strong style="color: #ffffff;">{max_win_streak}</strong> trades 
                        versus the maximum losing streak of <strong style="color: #ffffff;">{max_loss_streak}</strong> trades 
                        indicates {'healthy momentum capture with controlled drawdown periods' if max_win_streak > max_loss_streak * 1.5 else 'balanced performance with typical market variability' if max_win_streak >= max_loss_streak else 'vulnerability to extended losing periods that should be monitored'}.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Risk Management
                    </h4>
                    <p style="margin-bottom: 20px;">
                        The strategy exhibits a maximum drawdown of <strong style="color: #ffffff;">{max_dd:.2f}%</strong>, 
                        which falls into the {'conservative' if max_dd < 10 else 'moderate' if max_dd < 20 else 'aggressive'} risk category. 
                        {'This low drawdown indicates excellent capital preservation and risk control.' if max_dd < 10 else 'This moderate drawdown suggests balanced risk-taking appropriate for most trading accounts.' if max_dd < 20 else 'This elevated drawdown requires careful position sizing and risk management.'}
                        The Sharpe ratio of <strong style="color: #ffffff;">{sharpe:.2f}</strong> 
                        {'demonstrates excellent risk-adjusted returns (>2.0 is exceptional)' if sharpe > 2 else 'indicates good risk-adjusted performance (>1.0 is favorable)' if sharpe > 1 else 'suggests room for improvement in risk-adjusted returns' if sharpe > 0 else 'reveals negative risk-adjusted performance requiring strategy revision'}.
                    </p>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 16px; margin-top: 28px; text-transform: uppercase; letter-spacing: 1px;">
                        Optimization Strategies
                    </h4>
                    <p style="margin-bottom: 0;">
                        <strong style="color: #ffffff;">Position Sizing:</strong> {'Consider increasing position size given the strong risk-adjusted returns' if sharpe > 1.5 and max_dd < 15 else 'Implement dynamic position sizing based on market volatility' if std_dev > 2 else 'Current sizing appears appropriate for risk profile'}.<br><br>

                        <strong style="color: #ffffff;">Stop Loss Optimization:</strong> {'Current stops are effective; analyze trade duration to identify premature exits' if max_dd < 15 else 'Tighten stops to reduce drawdown depth, or widen to avoid noise-driven exits'}.<br><br>

                        <strong style="color: #ffffff;">Take Profit Enhancement:</strong> {'Consider trailing stops to capture extended moves given positive skew' if skewness > 0.5 else 'Fixed profit targets appear optimal; avoid premature exits' if risk_reward > 2 else 'Scale out of winners to improve risk-reward ratio'}.<br><br>

                        <strong style="color: #ffffff;">Entry Timing:</strong> {'Focus on high-conviction setups to maintain win rate' if win_rate > 55 else 'Refine entry criteria to improve directional accuracy' if win_rate < 45 else 'Current entry logic is well-calibrated'}.<br><br>

                        <strong style="color: #ffffff;">Market Regime Filter:</strong> {'Add volatility filters to avoid low-probability market conditions' if monthly_std > 1000 else 'Strategy shows consistency across different periods'}.<br><br>

                        <strong style="color: #ffffff;">Risk-Reward Target:</strong> {'Excellent current ratio of {risk_reward:.2f}:1; maintain this edge' if risk_reward > 2 else 'Target minimum 2:1 risk-reward; current {risk_reward:.2f}:1 needs improvement' if risk_reward < 2 else 'Solid {risk_reward:.2f}:1 ratio provides adequate edge'}.
                    </p>
                </div>

                <!-- RIGHT COLUMN: Statistical Analysis -->
                <div style="background: rgba(0, 0, 0, 0.3); 
                           border-radius: 12px; 
                           padding: 32px; 
                           border: 1px solid #222222;">

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px;">
                        Statistical Metrics
                    </h4>

                    <!-- Metrics Table -->
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px;">
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Total Return</td>
                            <td style="padding: 12px 0; color: {perf_color}; text-align: right; font-weight: 700;">{total_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Buy & Hold Return</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{buy_hold_return:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Alpha (Outperformance)</td>
                            <td style="padding: 12px 0; color: {'#00ff88' if strategy_outperformance > 0 else '#ff3366'}; text-align: right; font-weight: 700;">{strategy_outperformance:+.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Profit Factor</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{profit_factor:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Sharpe Ratio</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{sharpe:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Sortino Ratio</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{sortino:.2f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Win Rate</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{win_rate:.1f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Risk-Reward Ratio</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{risk_reward:.2f}:1</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Max Drawdown</td>
                            <td style="padding: 12px 0; color: #ff3366; text-align: right;">{max_dd:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Trade Expectancy</td>
                            <td style="padding: 12px 0; color: {'#00ff88' if expectancy > 0 else '#ff3366'}; text-align: right; font-weight: 700;">${expectancy:.2f}</td>
                        </tr>
                    </table>

                    <h4 style="color: #ffffff; font-size: 15px; font-weight: 700; margin-bottom: 20px; margin-top: 24px; text-transform: uppercase; letter-spacing: 1px;">
                        Return Distribution
                    </h4>

                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 32px;">
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Mean Return per Trade</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{np.mean(returns_pct):.3f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Std Deviation</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{std_dev:.3f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Variance</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{variance:.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Skewness</td>
                            <td style="padding: 12px 0; color: {'#00ff88' if skewness > 0 else '#ff3366'}; text-align: right;">{skewness:.3f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #222222;">
                            <td style="padding: 12px 0; color: #888888;">Kurtosis</td>
                            <td style="padding: 12px 0; color: #ffffff; text-align: right;">{kurtosis:.3f}</td>
                        </tr>
                    </table>

                    <div style="background: rgba(255, 255, 255, 0.02); 
                               border-left: 3px solid #00d4ff; 
                               padding: 16px; 
                               border-radius: 6px; 
                               font-size: 12px; 
                               line-height: 1.7;
                               color: #b0b0b0;">
                        <strong style="color: #00d4ff; display: block; margin-bottom: 8px;">INTERPRETATION:</strong>
                        <strong>Skewness ({skewness:.2f}):</strong> {'Positive skew indicates more frequent small losses with occasional large wins (desirable)' if skewness > 0.5 else 'Negative skew suggests frequent small wins with occasional large losses (risk concern)' if skewness < -0.5 else 'Near-zero skew shows symmetric return distribution'}.<br><br>

                        <strong>Kurtosis ({kurtosis:.2f}):</strong> {'High kurtosis indicates fat tails - higher probability of extreme outcomes' if kurtosis > 3 else 'Normal kurtosis suggests typical market behavior' if kurtosis > 2 else 'Low kurtosis indicates thin tails - fewer extreme events'}.<br><br>

                        <strong>Std Deviation ({std_dev:.2f}%):</strong> {'High volatility - returns vary significantly between trades' if std_dev > 3 else 'Moderate volatility - typical for active trading' if std_dev > 1.5 else 'Low volatility - consistent returns across trades'}.<br><br>

                        <strong>Monthly Variability:</strong> {'High month-to-month variation (σ=${monthly_std:,.0f})' if monthly_std > 1000 else 'Stable monthly performance (σ=${monthly_std:,.0f})'}.
                    </div>
                </div>
            </div>
        </div>
        """

        return analysis_html

    @staticmethod
    def generate_report(metrics, trades_df, strategy_name, timeframe, pair, initial_balance, 
                       leverage, sl_pips, tp_pips, risk_pct, start_date, end_date, df=None, output_dir=None):

        from datetime import datetime
        import json
        import tempfile
        import os

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backtest_run_time = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")

        # Generate matplotlib charts and get base64 encoded images
        equity_chart_b64, strategy_vs_bh_chart_b64 = HTMLReportGenerator._generate_matplotlib_charts(
            trades_df, initial_balance, metrics
        )

        # Prepare chart data for Chart.js
        chart_data = HTMLReportGenerator._prepare_chart_data(trades_df, initial_balance, metrics)

        # FIX METRICS - Recalculate from trades_df
        if not trades_df.empty:
            final_balance = trades_df['balance'].iloc[-1]
            total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
            win_trades = trades_df[trades_df['monetary_pnl'] > 0]
            loss_trades = trades_df[trades_df['monetary_pnl'] <= 0]
            win_rate = (len(win_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
            avg_win = win_trades['monetary_pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['monetary_pnl'].mean() if len(loss_trades) > 0 else 0

            # Update metrics with correct values
            metrics['final_balance'] = final_balance
            metrics['total_return_pct'] = total_return_pct
            metrics['win_rate'] = win_rate
            metrics['total_trades'] = len(trades_df)
            metrics['avg_win'] = avg_win
            metrics['avg_loss'] = avg_loss

        # Metrics table
        metrics_table = HTMLReportGenerator.generate_metrics_table(metrics, initial_balance)

        # Generate AI analysis with backtest parameters
        ai_analysis = HTMLReportGenerator._generate_ai_analysis(
            metrics, trades_df, initial_balance, strategy_name, pair, timeframe,
            start_date, end_date, leverage, sl_pips, tp_pips, risk_pct
        )

        # Get strategy source code
        strategy_code = HTMLReportGenerator.get_strategy_code(strategy_name)

        # Generate trade table
        trade_table = HTMLReportGenerator.generate_trade_table(trades_df)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlgoHaus - {pair} Backtest Report</title>

    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&display=swap" rel="stylesheet">

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --bg-primary: #0a0a0a;
            --bg-secondary: #111111;
            --bg-card: #161616;
            --bg-card-hover: #1a1a1a;
            --border: #222222;
            --border-bright: #333333;
            --text-primary: #ffffff;
            --text-secondary: #888888;
            --accent-green: #00ff88;
            --accent-red: #ff3366;
            --accent-blue: #00d4ff;
            --accent-purple: #b967ff;
            --accent-yellow: #ffcc00;
        }}

        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* Animated Background */
        .animated-bg {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            opacity: 0.03;
            background: 
                radial-gradient(circle at 20% 50%, var(--accent-green) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, var(--accent-blue) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, var(--accent-purple) 0%, transparent 50%);
            animation: pulse 15s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 0.03; }}
            50% {{ opacity: 0.06; }}
        }}

        .container {{
            max-width: 2000px;
            margin: 0 auto;
            padding: 40px 24px;
            position: relative;
            z-index: 1;
        }}

        /* Header */
        .header {{
            text-align: left;
            padding: 24px 32px;
            margin-bottom: 40px;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}

        .header h1 {{
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 8px;
            letter-spacing: 1px;
        }}

        .header .subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
            font-weight: 300;
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }}

        .stat-card {{
            background: var(--bg-card);
            padding: 32px;
            border-radius: 16px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent-green);
            transform: scaleY(0);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .stat-card:hover {{
            background: var(--bg-card-hover);
            border-color: var(--border-bright);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 255, 136, 0.1);
        }}

        .stat-card:hover::before {{
            transform: scaleY(1);
        }}

        .stat-card.negative::before {{
            background: var(--accent-red);
        }}

        .stat-card:hover.negative {{
            box-shadow: 0 12px 40px rgba(255, 51, 102, 0.1);
        }}

        .stat-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .stat-value {{
            font-size: 36px;
            font-weight: 700;
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
            color: var(--text-primary);
            line-height: 1.2;
        }}

        .stat-value.positive {{
            color: var(--accent-green);
        }}

        .stat-value.negative {{
            color: var(--accent-red);
        }}

        /* Chart Containers */
        .charts-section {{
            margin: 60px 0;
        }}

        .chart-row {{
            display: grid;
            gap: 24px;
            margin-bottom: 24px;
        }}

        .chart-row.full {{
            grid-template-columns: 1fr;
        }}

        .chart-row.four-col {{
            grid-template-columns: repeat(4, 1fr);
        }}

        .chart-container {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 28px;
            border: 1px solid var(--border);
            position: relative;
            transition: all 0.3s ease;
        }}

        .chart-container:hover {{
            border-color: var(--border-bright);
            box-shadow: 0 8px 32px rgba(0, 255, 136, 0.08);
        }}

        /* Chart height classes - FIXED SIZES FOR PROPER RENDERING */
        .chart-container.extra-tall {{
            height: 600px;
        }}

        .chart-container.tall {{
            height: 450px;
        }}

        .chart-container.medium {{
            height: 400px;
        }}

        .chart-container.short {{
            height: 350px;
        }}

        /* Matplotlib image containers */
        .chart-container.matplotlib-chart {{
            padding: 20px;
            height: auto;
        }}

        .chart-container.matplotlib-chart img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 8px;
        }}

        .chart-container::after {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, var(--accent-green) 0%, transparent 70%);
            opacity: 0.02;
            pointer-events: none;
        }}

        .chart-title {{
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 6px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .chart-title::before {{
            content: '';
            width: 3px;
            height: 18px;
            background: var(--accent-green);
            border-radius: 2px;
        }}

        .chart-description {{
            font-size: 10px;
            color: var(--text-secondary);
            margin-bottom: 16px;
            line-height: 1.4;
            font-weight: 300;
        }}

        /* Chart Overlay Tooltip */
        .chart-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.95);
            color: #ffffff;
            padding: 32px;
            border-radius: 16px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            z-index: 100;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow-y: auto;
        }}

        .chart-container:hover .chart-overlay {{
            opacity: 1;
        }}

        .chart-overlay h4 {{
            font-size: 16px;
            font-weight: 700;
            color: var(--accent-green);
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .chart-overlay p {{
            font-size: 13px;
            line-height: 1.8;
            margin-bottom: 12px;
            color: #e0e0e0;
        }}

        .chart-overlay .metric {{
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 6px;
            margin: 6px 0;
            font-size: 12px;
            border-left: 3px solid var(--accent-green);
        }}

        .chart-overlay .metric.negative {{
            border-left-color: var(--accent-red);
        }}

        .chart-canvas-wrapper {{
            position: relative;
            width: 100%;
            height: calc(100% - 70px);
        }}

        .chart-canvas {{
            position: absolute !important;
            top: 0;
            left: 0;
            width: 100% !important;
            height: 100% !important;
        }}

        /* Trade Table */
        .trade-log {{
            background: var(--bg-card);
            border-radius: 20px;
            padding: 40px;
            border: 1px solid var(--border);
            margin-top: 60px;
        }}

        .trade-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 24px;
            font-size: 13px;
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
        }}

        .trade-table thead {{
            background: var(--bg-secondary);
        }}

        .trade-table th {{
            padding: 16px;
            text-align: left;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-green);
            font-weight: 700;
            border-bottom: 2px solid var(--border);
        }}

        .trade-table td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }}

        .trade-table tbody tr {{
            transition: background 0.2s;
        }}

        .trade-table tbody tr:hover {{
            background: var(--bg-secondary);
        }}

        .positive-trade {{
            color: var(--accent-green);
            font-weight: 700;
        }}

        .negative-trade {{
            color: var(--accent-red);
            font-weight: 700;
        }}

        /* Collapsible Button */
        .collapsible {{
            background: #2a2a2a;
            color: #ffffff;
            cursor: pointer;
            padding: 14px 24px;
            border: 1px solid #444444;
            text-align: center;
            font-size: 13px;
            font-weight: 500;
            border-radius: 8px;
            margin-top: 20px;
            width: 300px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Helvetica Neue', Helvetica, sans-serif;
        }}

        .collapsible:hover {{
            background: #333333;
            border-color: #555555;
        }}

        .collapsible.active {{
            background: #333333;
        }}

        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out;
        }}

        /* Responsive */
        @media (max-width: 1600px) {{
            .chart-row.four-col {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        @media (max-width: 1200px) {{
            .chart-row.four-col {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .chart-row.four-col {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
            }}

            .stat-card {{
                padding: 20px;
            }}

            .stat-value {{
                font-size: 28px;
            }}

            .chart-container {{
                padding: 16px;
                min-height: 300px !important;
            }}
        }}

        @media (max-width: 480px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}

            .header {{
                padding: 16px 20px;
            }}

            .header h1 {{
                font-size: 11px;
            }}
        }}
    </style>
</head>
<body>
    <div class="animated-bg"></div>

    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>ALGOHAUS BACKTEST REPORT</h1>
            <div class="subtitle">{pair} · {strategy_name} · {timeframe}</div>
            <div class="subtitle" style="margin-top: 8px; font-size: 11px; color: #666666;">
                {start_date} to {end_date} · Initial Balance: ${initial_balance:,.0f} · Leverage: {leverage}x · 
                SL: {sl_pips} pips · TP: {tp_pips} pips · Risk: {risk_pct}%
            </div>
            <div class="subtitle" style="margin-top: 6px; font-size: 10px; color: #555555; font-style: italic;">
                Backtest executed: {backtest_run_time}
            </div>
        </div>

        <!-- Performance Metrics -->
        {metrics_table}

        <!-- AI Analysis -->
        {ai_analysis}

        <!-- Charts -->
        <div class="charts-section">
            <!-- Equity Curve - Full Width Matplotlib -->
            <div class="chart-row full">
                <div class="chart-container extra-tall matplotlib-chart">
                    <h3 class="chart-title">EQUITY CURVE</h3>
                    <p class="chart-description">Portfolio value over time showing cumulative growth</p>
                    <img src="data:image/png;base64,{equity_chart_b64}" alt="Equity Curve">
                </div>
            </div>

            <!-- Strategy vs Buy & Hold - Full Width Matplotlib -->
            <div class="chart-row full">
                <div class="chart-container extra-tall matplotlib-chart">
                    <h3 class="chart-title">STRATEGY VS BUY & HOLD</h3>
                    <p class="chart-description">Active strategy performance compared to passive holding</p>
                    <img src="data:image/png;base64,{strategy_vs_bh_chart_b64}" alt="Strategy vs Buy & Hold">
                </div>
            </div>

            <!-- Four Column Charts Row 1 -->
            <div class="chart-row four-col">
                <div class="chart-container medium">
                    <h3 class="chart-title">DRAWDOWN</h3>
                    <p class="chart-description">Peak-to-trough decline</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="drawdownChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Drawdown Analysis</h4>
                        <p>Measures the peak-to-trough decline in your portfolio. Shows the pain of losing streaks and recovery periods.</p>
                        <div class="metric negative">Max Drawdown: Worst drop from peak</div>
                        <div class="metric">Recovery Time: How long to reach new highs</div>
                        <div class="metric">Target: Keep below 20% for healthy strategies</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">WIN/LOSS RATIO</h3>
                    <p class="chart-description">Proportion of wins vs losses</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="winLossChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Win/Loss Ratio</h4>
                        <p>Visual breakdown of winning vs losing trades. The ratio between green and red determines overall profitability potential.</p>
                        <div class="metric">Winning Trades: Green segment</div>
                        <div class="metric negative">Losing Trades: Red segment</div>
                        <div class="metric">Key Insight: Larger green = higher win rate, but check if wins are larger than losses</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">P&L DISTRIBUTION</h3>
                    <p class="chart-description">Frequency of profit/loss</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="distributionChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>P&L Distribution</h4>
                        <p>Shows how often you achieve different profit/loss amounts. Bell curve centered on positive = consistent profits.</p>
                        <div class="metric">Green bars: Profitable trades (right side)</div>
                        <div class="metric negative">Red bars: Losing trades (left side)</div>
                        <div class="metric">Ideal: More green bars, centered towards positive values</div>
                        <div class="metric">Warning: Heavy left tail = risk of large losses</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">MONTHLY RETURNS</h3>
                    <p class="chart-description">Month-over-month performance</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="monthlyChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Monthly Returns</h4>
                        <p>Performance broken down by month. Shows consistency and seasonal patterns in your strategy.</p>
                        <div class="metric">Green bars: Profitable months</div>
                        <div class="metric negative">Red bars: Losing months</div>
                        <div class="metric">Consistency Check: More green than red = winning strategy</div>
                        <div class="metric">Pattern Recognition: Identify seasonal trends or weak months</div>
                    </div>
                </div>
            </div>

            <!-- Four Column Charts Row 2 -->
            <div class="chart-row four-col">
                <div class="chart-container medium">
                    <h3 class="chart-title">ROLLING WIN RATE</h3>
                    <p class="chart-description">20-trade moving average</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="rollingWinChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Rolling Win Rate (20 Trades)</h4>
                        <p>Moving average of win percentage over 20-trade windows. Smooths out short-term noise to reveal true performance trends.</p>
                        <div class="metric">Target: Consistently above 50%</div>
                        <div class="metric">Upward Trend: Strategy improving over time</div>
                        <div class="metric negative">Downward Trend: Strategy degradation or market change</div>
                        <div class="metric">Volatility: Large swings suggest inconsistent performance</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">RETURNS DISTRIBUTION</h3>
                    <p class="chart-description">Per-trade return %</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="returnsDistChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Returns Distribution</h4>
                        <p>Histogram of percentage returns per trade. Shows the statistical profile of your strategy's performance.</p>
                        <div class="metric">Normal Distribution: Predictable, consistent returns</div>
                        <div class="metric">Positive Skew: More frequent small wins, rare big wins</div>
                        <div class="metric negative">Negative Skew: Beware of occasional large losses</div>
                        <div class="metric">Fat Tails: High risk of extreme outcomes (good or bad)</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">CUMULATIVE RETURNS</h3>
                    <p class="chart-description">Total % gain/loss over time</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="cumulativeReturnsChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Cumulative Returns %</h4>
                        <p>Total percentage gain or loss from your starting balance over time. The ultimate measure of strategy performance.</p>
                        <div class="metric">Upward Trend: Profitable strategy building wealth</div>
                        <div class="metric">Flat Periods: Consolidation or sideways markets</div>
                        <div class="metric negative">Sharp Drops: Losing streaks requiring investigation</div>
                        <div class="metric">Angle of Ascent: Steeper = faster compounding returns</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">BEST VS WORST</h3>
                    <p class="chart-description">Trade quartile breakdown</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="bestWorstChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Best vs Worst Trades</h4>
                        <p>Breaks down your trades into top 25%, middle 50%, and bottom 25% performers. Reveals if success depends on outliers.</p>
                        <div class="metric">Top 25% (Green): Your best winning trades</div>
                        <div class="metric">Middle 50% (Gray): Average performance</div>
                        <div class="metric negative">Bottom 25% (Red): Your worst trades</div>
                        <div class="metric">Balanced = Consistent strategy. Heavy top/bottom = outlier dependent</div>
                    </div>
                </div>
            </div>

            <!-- Four Column Charts Row 3 -->
            <div class="chart-row four-col">
                <div class="chart-container medium">
                    <h3 class="chart-title">ROLLING SHARPE</h3>
                    <p class="chart-description">Risk-adjusted returns (20)</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="rollingSharpeChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Rolling Sharpe Ratio (20 Trades)</h4>
                        <p>Measures return per unit of risk over 20-trade windows. Higher Sharpe = better risk-adjusted performance.</p>
                        <div class="metric">Sharpe > 2: Excellent risk-adjusted returns</div>
                        <div class="metric">Sharpe 1-2: Good performance</div>
                        <div class="metric">Sharpe < 1: Poor risk-reward balance</div>
                        <div class="metric negative">Declining Sharpe: Strategy losing edge or taking too much risk</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">ROLLING VOLATILITY</h3>
                    <p class="chart-description">20-trade std deviation</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="rollingVolatilityChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Rolling Volatility</h4>
                        <p>20-trade annualized standard deviation of returns. Measures how much your returns swing around the mean.</p>
                        <div class="metric negative">High Volatility: Large swings, higher risk</div>
                        <div class="metric">Low Volatility: Stable, predictable returns</div>
                        <div class="metric negative">Spikes: Often precede drawdowns or market stress</div>
                        <div class="metric">Rising Trend: Increasing uncertainty and risk exposure</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">TRADE DURATION</h3>
                    <p class="chart-description">Holding period distribution</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="tradeDurationChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Trade Duration Distribution</h4>
                        <p>Histogram of how long you hold each trade (in hours). Reveals if your actual holding times match strategy expectations.</p>
                        <div class="metric">Clustering: Most trades should group around expected duration</div>
                        <div class="metric">Long Tail: Trades exceeding expected time (stops not hit)</div>
                        <div class="metric negative">Very Short Duration: May indicate noise trading or tight stops</div>
                        <div class="metric">Wide Spread: Inconsistent trade management</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">WEEKDAY PERFORMANCE</h3>
                    <p class="chart-description">Avg P&L by day of week</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="weekdayPerfChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Weekday Performance</h4>
                        <p>Average profit/loss broken down by day of the week. Identifies if certain days consistently outperform or underperform.</p>
                        <div class="metric">Monday/Friday: Often show different patterns (week open/close)</div>
                        <div class="metric">Consistent Across Days: Strategy works in all market conditions</div>
                        <div class="metric negative">Weak Day: Consider avoiding trades on consistently poor days</div>
                        <div class="metric">Strong Day: Potential to increase position size on best days</div>
                    </div>
                </div>
            </div>

            <!-- Four Column Charts Row 4 - Win/Loss Analysis -->
            <div class="chart-row four-col">
                <div class="chart-container medium">
                    <h3 class="chart-title">WIN RATE OVER TIME</h3>
                    <p class="chart-description">Cumulative win percentage progression</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="winRateOverTimeChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Win Rate Over Time</h4>
                        <p>Shows how your win rate evolved throughout the backtest period. A stable or improving trend indicates consistent strategy performance.</p>
                        <div class="metric">Target: Above 50% for profitable strategies</div>
                        <div class="metric">Watch for: Declining trends that signal strategy degradation</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">AVG WIN DISTRIBUTION</h3>
                    <p class="chart-description">Distribution of winning trade sizes</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="avgWinDistChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Average Win Distribution</h4>
                        <p>Histogram showing the frequency and magnitude of winning trades. Reveals if wins are consistent or dominated by outliers.</p>
                        <div class="metric">Ideal: Normal distribution with consistent wins</div>
                        <div class="metric">Warning: Heavy right tail suggests dependence on rare big wins</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">AVG LOSS DISTRIBUTION</h3>
                    <p class="chart-description">Distribution of losing trade sizes</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="avgLossDistChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Average Loss Distribution</h4>
                        <p>Histogram showing the frequency and magnitude of losing trades. Critical for understanding risk exposure.</p>
                        <div class="metric negative">Risk Check: Losses should be controlled and consistent</div>
                        <div class="metric negative">Warning: Fat left tail indicates occasional catastrophic losses</div>
                    </div>
                </div>
                <div class="chart-container medium">
                    <h3 class="chart-title">WIN/LOSS COMPARISON</h3>
                    <p class="chart-description">Side-by-side avg win vs avg loss</p>
                    <div class="chart-canvas-wrapper">
                        <canvas id="winLossCompChart" class="chart-canvas"></canvas>
                    </div>
                    <div class="chart-overlay">
                        <h4>Win/Loss Comparison</h4>
                        <p>Direct comparison of average winning trade size versus average losing trade size. Key for evaluating risk-reward ratio.</p>
                        <div class="metric">Risk-Reward Ratio: Avg Win / Avg Loss</div>
                        <div class="metric">Target: Ratio > 1.5 for sustainable profitability</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Strategy Code -->
        <div class="trade-log">
            <h3 class="chart-title">STRATEGY CODE</h3>
            <button class="collapsible">▼ CLICK TO VIEW STRATEGY CODE</button>
            <div class="content">
                <div style="background: #0a0a0a; border: 1px solid #222222; border-radius: 12px; padding: 24px; overflow-x: auto; margin-top: 24px;">
                    <pre style="margin: 0; color: #888888; font-family: 'Helvetica Neue', Helvetica, sans-serif; font-size: 12px; line-height: 1.6; white-space: pre-wrap;">{strategy_code}</pre>
                </div>
            </div>
        </div>

        <!-- Trade Log -->
        <div class="trade-log" style="margin-top: 32px;">
            <h3 class="chart-title">TRADE LOG</h3>
            <button class="collapsible">▼ CLICK TO VIEW ALL TRADES</button>
            <div class="content">
                {trade_table}
            </div>
        </div>
    </div>

    <script>
        // Chart.js Global Configuration - DARK THEME
        Chart.defaults.color = '#888888';
        Chart.defaults.borderColor = '#222222';
        Chart.defaults.backgroundColor = 'rgba(0, 255, 136, 0.1)';
        Chart.defaults.font.family = "'Helvetica Neue', Helvetica, Arial, sans-serif";
        Chart.defaults.font.size = 11;

        const chartData = {json.dumps(chart_data)};

        // Common chart options
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    display: true,
                    position: 'top',
                    labels: {{
                        color: '#ffffff',
                        font: {{
                            size: 11,
                            weight: '600'
                        }},
                        padding: 12,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }}
                }},
                tooltip: {{
                    backgroundColor: '#161616',
                    titleColor: '#00ff88',
                    bodyColor: '#ffffff',
                    borderColor: '#333333',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                    titleFont: {{
                        size: 12,
                        weight: '700'
                    }},
                    bodyFont: {{
                        size: 11
                    }}
                }}
            }},
            scales: {{
                x: {{
                    grid: {{
                        display: true,
                        color: '#1a1a1a',
                        lineWidth: 1
                    }},
                    ticks: {{
                        color: '#888888',
                        font: {{
                            size: 9
                        }},
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8
                    }},
                    border: {{
                        display: false
                    }}
                }},
                y: {{
                    grid: {{
                        display: true,
                        color: '#1a1a1a',
                        lineWidth: 1
                    }},
                    ticks: {{
                        color: '#888888',
                        font: {{
                            size: 9
                        }},
                        padding: 6
                    }},
                    border: {{
                        display: false
                    }}
                }}
            }}
        }};

        // 1. DRAWDOWN
        const drawdownCtx = document.getElementById('drawdownChart').getContext('2d');
        new Chart(drawdownCtx, {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [{{
                    label: 'Drawdown %',
                    data: chartData.drawdown,
                    borderColor: '#ff3366',
                    backgroundColor: 'rgba(255, 51, 102, 0.15)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 2. WIN/LOSS DONUT
        const winLossCtx = document.getElementById('winLossChart').getContext('2d');
        new Chart(winLossCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Wins', 'Losses'],
                datasets: [{{
                    data: [chartData.wins, chartData.losses],
                    backgroundColor: ['rgba(0, 255, 136, 0.8)', 'rgba(255, 51, 102, 0.8)'],
                    borderColor: ['#00ff88', '#ff3366'],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            color: '#ffffff',
                            font: {{ size: 11, weight: '600' }},
                            padding: 16
                        }}
                    }}
                }},
                cutout: '65%'
            }}
        }});

        // 3. P&L DISTRIBUTION
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        new Chart(distributionCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.pnl_bins,
                datasets: [{{
                    label: 'Frequency',
                    data: chartData.pnl_counts,
                    backgroundColor: chartData.pnl_counts.map((_, i) => 
                        chartData.pnl_bins[i] >= 0 ? 'rgba(0, 255, 136, 0.7)' : 'rgba(255, 51, 102, 0.7)'
                    ),
                    borderWidth: 0,
                    borderRadius: 4
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 4. MONTHLY RETURNS
        const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
        new Chart(monthlyCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.months,
                datasets: [{{
                    label: 'Monthly Return %',
                    data: chartData.monthly_returns,
                    backgroundColor: chartData.monthly_returns.map(val =>
                        val >= 0 ? 'rgba(0, 255, 136, 0.7)' : 'rgba(255, 51, 102, 0.7)'
                    ),
                    borderWidth: 0,
                    borderRadius: 4
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 5. ROLLING WIN RATE
        const rollingWinCtx = document.getElementById('rollingWinChart').getContext('2d');
        new Chart(rollingWinCtx, {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [{{
                    label: 'Win Rate %',
                    data: chartData.rolling_win_rate,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }},
                scales: {{
                    ...commonOptions.scales,
                    y: {{ ...commonOptions.scales.y, min: 0, max: 100 }}
                }}
            }}
        }});

        // 6. RETURNS DISTRIBUTION
        const returnsDistCtx = document.getElementById('returnsDistChart').getContext('2d');
        new Chart(returnsDistCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.returns_bins.map(v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%'),
                datasets: [{{
                    label: 'Trades',
                    data: chartData.returns_counts,
                    backgroundColor: chartData.returns_bins.map(v =>
                        v >= 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 51, 102, 0.6)'
                    ),
                    borderWidth: 0,
                    borderRadius: 3
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }},
                scales: {{
                    ...commonOptions.scales,
                    x: {{
                        ...commonOptions.scales.x,
                        ticks: {{
                            ...commonOptions.scales.x.ticks,
                            maxTicksLimit: 6
                        }}
                    }}
                }}
            }}
        }});

        // 7. CUMULATIVE RETURNS
        const cumulativeReturnsCtx = document.getElementById('cumulativeReturnsChart').getContext('2d');
        new Chart(cumulativeReturnsCtx, {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [{{
                    label: 'Cumulative %',
                    data: chartData.cumulative_returns,
                    borderColor: '#b967ff',
                    backgroundColor: 'rgba(185, 103, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 8. BEST VS WORST DONUT
        const bestWorstCtx = document.getElementById('bestWorstChart').getContext('2d');
        const totalTrades = chartData.wins + chartData.losses;
        const topQuartile = Math.ceil(totalTrades * 0.25);
        const avgTrades = totalTrades - (topQuartile * 2);

        new Chart(bestWorstCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Top 25%', 'Average', 'Bottom 25%'],
                datasets: [{{
                    data: [topQuartile, avgTrades, topQuartile],
                    backgroundColor: ['rgba(0, 255, 136, 0.9)', 'rgba(136, 136, 136, 0.6)', 'rgba(255, 51, 102, 0.9)'],
                    borderColor: ['#00ff88', '#888888', '#ff3366'],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            color: '#ffffff',
                            font: {{ size: 10, weight: '600' }},
                            padding: 12
                        }}
                    }}
                }},
                cutout: '60%'
            }}
        }});

        // 9. ROLLING SHARPE
        const rollingSharpeCtx = document.getElementById('rollingSharpeChart').getContext('2d');
        new Chart(rollingSharpeCtx, {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [{{
                    label: 'Sharpe',
                    data: chartData.rolling_sharpe,
                    borderColor: '#ffcc00',
                    backgroundColor: 'rgba(255, 204, 0, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 10. ROLLING VOLATILITY
        const rollingVolatilityCtx = document.getElementById('rollingVolatilityChart').getContext('2d');
        new Chart(rollingVolatilityCtx, {{
            type: 'line',
            data: {{
                labels: chartData.dates,
                datasets: [{{
                    label: 'Volatility %',
                    data: chartData.rolling_volatility,
                    borderColor: '#ff3366',
                    backgroundColor: 'rgba(255, 51, 102, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 11. TRADE DURATION
        const tradeDurationCtx = document.getElementById('tradeDurationChart').getContext('2d');
        new Chart(tradeDurationCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.trade_duration_bins.map(v => v.toFixed(1) + 'h'),
                datasets: [{{
                    label: 'Trades',
                    data: chartData.trade_duration_counts,
                    backgroundColor: 'rgba(0, 212, 255, 0.7)',
                    borderWidth: 0,
                    borderRadius: 4
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }}
                }}
            }}
        }});

        // 12. WEEKDAY PERFORMANCE
        if (Object.keys(chartData.weekday_returns).length > 0) {{
            const weekdayPerfCtx = document.getElementById('weekdayPerfChart').getContext('2d');
            const weekdays = Object.keys(chartData.weekday_returns);
            const weekdayValues = weekdays.map(d => chartData.weekday_returns[d]);

            new Chart(weekdayPerfCtx, {{
                type: 'bar',
                data: {{
                    labels: weekdays,
                    datasets: [{{
                        label: 'Avg P&L',
                        data: weekdayValues,
                        backgroundColor: weekdayValues.map(v =>
                            v >= 0 ? 'rgba(0, 255, 136, 0.7)' : 'rgba(255, 51, 102, 0.7)'
                        ),
                        borderWidth: 0,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    ...commonOptions,
                    plugins: {{
                        ...commonOptions.plugins,
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        // 13. WIN RATE OVER TIME (Cumulative)
        if (chartData.dates.length > 0) {{
            const winRateOverTimeCtx = document.getElementById('winRateOverTimeChart').getContext('2d');
            // Calculate cumulative win rate
            let cumulativeWins = 0;
            let cumulativeTrades = 0;
            const cumulativeWinRate = chartData.dates.map((date, idx) => {{
                if (chartData.equity[idx] > (chartData.equity[idx-1] || chartData.equity[0])) {{
                    cumulativeWins++;
                }}
                cumulativeTrades++;
                return (cumulativeWins / cumulativeTrades) * 100;
            }});

            new Chart(winRateOverTimeCtx, {{
                type: 'line',
                data: {{
                    labels: chartData.dates,
                    datasets: [{{
                        label: 'Cumulative Win Rate %',
                        data: cumulativeWinRate,
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }}]
                }},
                options: {{
                    ...commonOptions,
                    plugins: {{
                        ...commonOptions.plugins,
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        ...commonOptions.scales,
                        y: {{ ...commonOptions.scales.y, min: 0, max: 100 }}
                    }}
                }}
            }});
        }}

        // 14. AVG WIN DISTRIBUTION
        if (chartData.wins > 0) {{
            const avgWinDistCtx = document.getElementById('avgWinDistChart').getContext('2d');
            // Filter only winning trades from pnl data
            const winningTrades = chartData.pnl_bins.map((bin, idx) => {{
                return bin > 0 ? chartData.pnl_counts[idx] : 0;
            }});

            new Chart(avgWinDistCtx, {{
                type: 'bar',
                data: {{
                    labels: chartData.pnl_bins.map(v => v > 0 ? '+$' + v.toFixed(0) : ''),
                    datasets: [{{
                        label: 'Winning Trades',
                        data: winningTrades,
                        backgroundColor: 'rgba(0, 255, 136, 0.7)',
                        borderColor: '#00ff88',
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    ...commonOptions,
                    plugins: {{
                        ...commonOptions.plugins,
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        // 15. AVG LOSS DISTRIBUTION
        if (chartData.losses > 0) {{
            const avgLossDistCtx = document.getElementById('avgLossDistChart').getContext('2d');
            // Filter only losing trades from pnl data
            const losingTrades = chartData.pnl_bins.map((bin, idx) => {{
                return bin < 0 ? chartData.pnl_counts[idx] : 0;
            }});

            new Chart(avgLossDistCtx, {{
                type: 'bar',
                data: {{
                    labels: chartData.pnl_bins.map(v => v < 0 ? '-$' + Math.abs(v).toFixed(0) : ''),
                    datasets: [{{
                        label: 'Losing Trades',
                        data: losingTrades,
                        backgroundColor: 'rgba(255, 51, 102, 0.7)',
                        borderColor: '#ff3366',
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    ...commonOptions,
                    plugins: {{
                        ...commonOptions.plugins,
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        // 16. WIN/LOSS COMPARISON BAR CHART
        const winLossCompCtx = document.getElementById('winLossCompChart').getContext('2d');
        // Calculate average win and average loss from data
        const totalWins = chartData.wins;
        const totalLosses = chartData.losses;
        const avgWinValue = chartData.pnl_bins.reduce((sum, val, idx) => {{
            return val > 0 ? sum + (val * chartData.pnl_counts[idx]) : sum;
        }}, 0) / totalWins || 0;
        const avgLossValue = Math.abs(chartData.pnl_bins.reduce((sum, val, idx) => {{
            return val < 0 ? sum + (val * chartData.pnl_counts[idx]) : sum;
        }}, 0) / totalLosses || 0);

        new Chart(winLossCompCtx, {{
            type: 'bar',
            data: {{
                labels: ['Avg Win', 'Avg Loss'],
                datasets: [{{
                    label: 'Amount ($)',
                    data: [avgWinValue, avgLossValue],
                    backgroundColor: ['rgba(0, 255, 136, 0.8)', 'rgba(255, 51, 102, 0.8)'],
                    borderColor: ['#00ff88', '#ff3366'],
                    borderWidth: 2,
                    borderRadius: 6
                }}]
            }},
            options: {{
                ...commonOptions,
                plugins: {{
                    ...commonOptions.plugins,
                    legend: {{ display: false }},
                    tooltip: {{
                        ...commonOptions.plugins.tooltip,
                        callbacks: {{
                            label: function(context) {{
                                return '$' + context.parsed.y.toFixed(2);
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Collapsible functionality
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

        # Use provided output_dir or fall back to temp directory
        if output_dir and os.path.exists(output_dir):
            save_dir = output_dir
        else:
            temp_dir = tempfile.gettempdir()
            save_dir = temp_dir
        
        filename = f"AlgoHaus_{pair.replace('/', '-')}_{strategy_name}_{timestamp}.html"
        report_path = os.path.join(output_dir if output_dir else tempfile.gettempdir(), filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path

    @staticmethod
    def _generate_matplotlib_charts(trades_df, initial_balance, metrics):
        """
        Generate Equity Curve and Strategy vs Buy & Hold charts using Matplotlib
        Returns base64 encoded PNG images
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        from io import BytesIO
        import base64

        # Set dark theme for matplotlib
        plt.style.use('dark_background')

        # Custom colors matching the HTML theme
        BG_COLOR = '#0a0a0a'
        GRID_COLOR = '#1a1a1a'
        TEXT_COLOR = '#888888'
        ACCENT_GREEN = '#00ff88'
        ACCENT_ORANGE = '#ff9500'

        if trades_df.empty:
            # Return placeholder images if no data
            return "", ""

        # ===== EQUITY CURVE =====
        fig1, ax1 = plt.subplots(figsize=(18, 7), facecolor=BG_COLOR)
        ax1.set_facecolor(BG_COLOR)

        dates = trades_df['exit_time']
        equity = trades_df['balance']

        # Plot equity curve with filled area
        ax1.plot(dates, equity, color=ACCENT_GREEN, linewidth=2.5, label='Portfolio Value')
        ax1.fill_between(dates, equity, initial_balance, alpha=0.15, color=ACCENT_GREEN)

        # Add horizontal line for initial balance
        ax1.axhline(y=initial_balance, color=TEXT_COLOR, linestyle='--', linewidth=1, alpha=0.5, label='Initial Balance')

        # Formatting
        ax1.set_xlabel('Date', fontsize=12, color=TEXT_COLOR, weight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, color=TEXT_COLOR, weight='bold')
        ax1.set_title('EQUITY CURVE', fontsize=16, color='#ffffff', weight='bold', pad=20)

        # Grid
        ax1.grid(True, alpha=0.15, color=GRID_COLOR, linewidth=1)
        ax1.tick_params(colors=TEXT_COLOR, labelsize=10)

        # Format dates on x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Format y-axis with commas
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Legend
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                   framealpha=0.9, facecolor='#161616', edgecolor='#333333', fontsize=11)

        # Tight layout
        plt.tight_layout()

        # Save to base64
        buffer1 = BytesIO()
        plt.savefig(buffer1, format='png', dpi=150, facecolor=BG_COLOR, edgecolor='none', bbox_inches='tight')
        buffer1.seek(0)
        equity_chart_b64 = base64.b64encode(buffer1.read()).decode('utf-8')
        plt.close(fig1)

        # ===== STRATEGY VS BUY & HOLD =====
        fig2, ax2 = plt.subplots(figsize=(18, 7), facecolor=BG_COLOR)
        ax2.set_facecolor(BG_COLOR)

        # Calculate buy & hold baseline
        strategy_equity = equity.values
        final_return = (strategy_equity[-1] - initial_balance) / initial_balance

        # Buy & hold grows linearly at the same final return rate
        buy_hold_equity = initial_balance * (1 + final_return * np.linspace(0, 1, len(strategy_equity)))

        # Plot both strategies
        ax2.plot(dates, strategy_equity, color=ACCENT_GREEN, linewidth=2.5, label='Active Strategy', zorder=3)
        ax2.plot(dates, buy_hold_equity, color=ACCENT_ORANGE, linewidth=2.5, linestyle='--', 
                 label='Buy & Hold Baseline', alpha=0.85, zorder=2)

        # Fill areas
        ax2.fill_between(dates, strategy_equity, initial_balance, alpha=0.1, color=ACCENT_GREEN, zorder=1)
        ax2.fill_between(dates, buy_hold_equity, initial_balance, alpha=0.08, color=ACCENT_ORANGE, zorder=0)

        # Add horizontal line for initial balance
        ax2.axhline(y=initial_balance, color=TEXT_COLOR, linestyle=':', linewidth=1, alpha=0.4)

        # Formatting
        ax2.set_xlabel('Date', fontsize=12, color=TEXT_COLOR, weight='bold')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12, color=TEXT_COLOR, weight='bold')
        ax2.set_title('STRATEGY VS BUY & HOLD COMPARISON', fontsize=16, color='#ffffff', weight='bold', pad=20)

        # Grid
        ax2.grid(True, alpha=0.15, color=GRID_COLOR, linewidth=1)
        ax2.tick_params(colors=TEXT_COLOR, labelsize=10)

        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Format y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Legend
        ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                   framealpha=0.9, facecolor='#161616', edgecolor='#333333', fontsize=11)

        # Tight layout
        plt.tight_layout()

        # Save to base64
        buffer2 = BytesIO()
        plt.savefig(buffer2, format='png', dpi=150, facecolor=BG_COLOR, edgecolor='none', bbox_inches='tight')
        buffer2.seek(0)
        strategy_vs_bh_chart_b64 = base64.b64encode(buffer2.read()).decode('utf-8')
        plt.close(fig2)

        return equity_chart_b64, strategy_vs_bh_chart_b64

    @staticmethod
    def _prepare_chart_data(trades_df, initial_balance, metrics):
        """Prepare all data for Chart.js - same as before"""
        import pandas as pd
        import numpy as np

        if trades_df.empty:
            return {
                'dates': [], 'equity': [], 'drawdown': [], 'wins': 0, 'losses': 0,
                'pnl_bins': [], 'pnl_counts': [], 'months': [], 'monthly_returns': [],
                'rolling_win_rate': [], 'returns_bins': [], 'returns_counts': [],
                'cumulative_returns': [], 'rolling_sharpe': [], 'rolling_volatility': [],
                'trade_duration_bins': [], 'trade_duration_counts': [],
                'hourly_returns': {}, 'weekday_returns': {}
            }

        dates = trades_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        equity = trades_df['balance'].tolist()

        if 'drawdown_pct' in trades_df.columns:
            drawdown = trades_df['drawdown_pct'].tolist()
        else:
            peak = trades_df['balance'].expanding().max()
            drawdown = ((trades_df['balance'] - peak) / peak * 100).tolist()

        wins = len(trades_df[trades_df['monetary_pnl'] > 0])
        losses = len(trades_df[trades_df['monetary_pnl'] <= 0])

        pnl_values = trades_df['monetary_pnl'].values
        counts, bin_edges = np.histogram(pnl_values, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pnl_bins = bin_centers.tolist()
        pnl_counts = counts.tolist()

        trades_df_copy = trades_df.copy()
        trades_df_copy['returns_pct'] = (trades_df_copy['monetary_pnl'] / 
                                          trades_df_copy['balance'].shift(1).fillna(initial_balance)) * 100

        returns_counts, returns_bin_edges = np.histogram(trades_df_copy['returns_pct'].dropna(), bins=40)
        returns_bin_centers = (returns_bin_edges[:-1] + returns_bin_edges[1:]) / 2
        returns_bins = returns_bin_centers.tolist()
        returns_counts_list = returns_counts.tolist()

        trades_df_copy['cumulative_return'] = ((trades_df_copy['balance'] - initial_balance) / initial_balance) * 100
        cumulative_returns = trades_df_copy['cumulative_return'].tolist()

        rolling_returns = trades_df_copy['returns_pct'].rolling(window=20, min_periods=10)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        rolling_sharpe_list = rolling_sharpe.fillna(0).tolist()

        rolling_volatility = rolling_returns.std() * np.sqrt(252)
        rolling_volatility_list = rolling_volatility.fillna(0).tolist()

        try:
            trades_df_copy['month'] = trades_df_copy['exit_time'].dt.to_period('M')
            monthly_pnl = trades_df_copy.groupby('month')['monetary_pnl'].sum()
            monthly_returns_pct = (monthly_pnl / initial_balance) * 100
            months = [str(m) for m in monthly_returns_pct.index]
            monthly_returns = monthly_returns_pct.tolist()
        except:
            months = []
            monthly_returns = []

        try:
            trades_df_copy['duration_hours'] = (trades_df_copy['exit_time'] - 
                                                 trades_df_copy['entry_time']).dt.total_seconds() / 3600
            duration_counts, duration_bin_edges = np.histogram(trades_df_copy['duration_hours'].dropna(), bins=25)
            duration_bin_centers = (duration_bin_edges[:-1] + duration_bin_edges[1:]) / 2
            trade_duration_bins = duration_bin_centers.tolist()
            trade_duration_counts = duration_counts.tolist()
        except:
            trade_duration_bins = []
            trade_duration_counts = []

        try:
            trades_df_copy['weekday'] = trades_df_copy['entry_time'].dt.day_name()
            weekday_pnl = trades_df_copy.groupby('weekday')['monetary_pnl'].mean()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_returns = {day: round(weekday_pnl.get(day, 0), 2) for day in weekday_order if day in weekday_pnl.index}
        except:
            weekday_returns = {}

        trades_df_copy['is_win'] = (trades_df_copy['monetary_pnl'] > 0).astype(int)
        rolling_win = trades_df_copy['is_win'].rolling(window=20, min_periods=1).mean() * 100
        rolling_win_rate = rolling_win.tolist()

        return {
            'dates': dates, 'equity': equity, 'drawdown': drawdown,
            'wins': wins, 'losses': losses, 'pnl_bins': pnl_bins, 'pnl_counts': pnl_counts,
            'months': months, 'monthly_returns': monthly_returns,
            'rolling_win_rate': rolling_win_rate, 'returns_bins': returns_bins,
            'returns_counts': returns_counts_list, 'cumulative_returns': cumulative_returns,
            'rolling_sharpe': rolling_sharpe_list, 'rolling_volatility': rolling_volatility_list,
            'trade_duration_bins': trade_duration_bins, 'trade_duration_counts': trade_duration_counts,
            'weekday_returns': weekday_returns
        }

    @staticmethod
    def generate_metrics_table(metrics, initial_balance):
        """Generate performance metric cards with Win Rate and Sortino Ratio prominently displayed"""
        cards = []
        key_metrics = [
            ('FINAL BALANCE', f"${metrics.get('final_balance', 0):,.0f}", 'positive'),
            ('TOTAL RETURN', f"{metrics.get('total_return_pct', 0):.2f}%", 
             'positive' if metrics.get('total_return_pct', 0) > 0 else 'negative'),
            ('WIN RATE', f"{metrics.get('win_rate', 0):.1f}%",
             'positive' if metrics.get('win_rate', 0) > 50 else 'negative'),
            ('PROFIT FACTOR', f"{metrics.get('profit_factor', 0):.2f}",
             'positive' if metrics.get('profit_factor', 0) > 1 else 'negative'),
            ('SHARPE RATIO', f"{metrics.get('sharpe_ratio', 0):.2f}",
             'positive' if metrics.get('sharpe_ratio', 0) > 0 else 'negative'),
            ('SORTINO RATIO', f"{metrics.get('sortino_ratio', 0):.2f}",
             'positive' if metrics.get('sortino_ratio', 0) > 0 else 'negative'),
            ('MAX DRAWDOWN', f"{metrics.get('max_drawdown_pct', 0):.2f}%", 'negative'),
            ('TOTAL TRADES', f"{metrics.get('total_trades', 0):,}", 'neutral'),
            ('AVG WIN', f"${metrics.get('avg_win', 0):,.0f}", 'positive'),
            ('AVG LOSS', f"${abs(metrics.get('avg_loss', 0)):,.0f}", 'negative'),
        ]

        for label, value, style in key_metrics:
            card_class = 'negative' if style == 'negative' else ''
            value_class = style
            cards.append(f'''
                <div class="stat-card {card_class}">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value {value_class}">{value}</div>
                </div>
            ''')

        return f'<div class="stats-grid">{"".join(cards)}</div>'

    @staticmethod
    def get_strategy_code(strategy_name):
        """Extract strategy source code - same as before"""
        try:
            import inspect
            import sys

            strategy_methods = {
                'VWAP Crossover': 'vwap_crossover_strategy',
                'vwap_crossover_strategy': 'vwap_crossover_strategy',
                'Opening Range Breakout': 'opening_range_strategy',
                'opening_range_strategy': 'opening_range_strategy',
                'Bollinger Band Reversion': 'bollinger_band_reversion_strategy',
                'bollinger_band_reversion_strategy': 'bollinger_band_reversion_strategy'
            }

            method_name = strategy_methods.get(strategy_name, strategy_name)

            TradingStrategies = None

            try:
                from trading_strategies import TradingStrategies
            except ImportError:
                pass

            if TradingStrategies is None:
                for module_name, module in sys.modules.items():
                    if hasattr(module, 'TradingStrategies'):
                        TradingStrategies = getattr(module, 'TradingStrategies')
                        break

            if TradingStrategies is None and '__main__' in sys.modules:
                main_module = sys.modules['__main__']
                if hasattr(main_module, 'TradingStrategies'):
                    TradingStrategies = getattr(main_module, 'TradingStrategies')

            if TradingStrategies and hasattr(TradingStrategies, method_name):
                method = getattr(TradingStrategies, method_name)
                source = inspect.getsource(method)
                return source
            else:
                return f'''# Strategy: {strategy_name}
# The strategy code could not be automatically extracted.'''

        except Exception as e:
            return f'''# Strategy: {strategy_name}
# Error extracting strategy code: {str(e)}'''

    @staticmethod
    def generate_trade_table(trades_df):
        """Generate trade log table - same as before"""
        if trades_df.empty:
            return '<p style="color: #888888; text-align: center; padding: 40px;">No trades executed.</p>'

        html = '''
            <div style="overflow-x: auto;">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>ENTRY TIME</th>
                            <th>EXIT TIME</th>
                            <th>SIGNAL</th>
                            <th>ENTRY PRICE</th>
                            <th>EXIT PRICE</th>
                            <th>PIPS P&L</th>
                            <th>MONETARY P&L</th>
                            <th>BALANCE</th>
                            <th>EXIT REASON</th>
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
            </div>
        '''

        return html




# ══════════════════════════════════════════════════════════════════════
# 7. ENHANCED BACKTESTER UI WITH MULTI-STRATEGY SUPPORT
# ══════════════════════════════════════════════════════════════════════
class BacktesterUI:
    def __init__(self, master):
        self.master = master
        master.title("⚡ AlgoHaus Backtester v7.0 - Multi-Strategy Edition")

        # Responsive window sizing
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        
        if screen_width >= 1920:
            width_ratio = 0.75
            height_ratio = 0.85
        elif screen_width >= 1366:
            width_ratio = 0.85
            height_ratio = 0.90
        else:
            width_ratio = 0.95
            height_ratio = 0.95
        
        window_width = int(screen_width * width_ratio)
        window_height = int(screen_height * height_ratio)
        
        window_width = max(window_width, 1200)
        window_height = max(window_height, 700)
        
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        master.geometry(f"{window_width}x{window_height}+{x}+{y}")
        master.minsize(1000, 600)
        master.resizable(True, True)

        default_path = pathlib.Path(r"D:\compressedworld\AlgoHaus\OandaHistoricalData\1MinCharts")
        self.data_folder = default_path if default_path.exists() else pathlib.Path.cwd() / "data"
        self.df = None
        self.current_section = "config"

        # Strategy selection (CHANGED TO DICT)
        self.selected_strategies = {
            'vwap_crossover_strategy': tk.BooleanVar(master, value=True),
            'opening_range_strategy': tk.BooleanVar(master, value=False),
            'bollinger_band_reversion_strategy': tk.BooleanVar(master, value=False)
        }
        
        # Other settings
        self.selected_pair = tk.StringVar(master, value="EUR/USD")
        self.selected_timeframe = tk.StringVar(master, value="1hr")
        self.initial_balance = tk.DoubleVar(master, value=10000.0)
        self.leverage = tk.IntVar(master, value=50)
        self.sl_pips = tk.IntVar(master, value=30)
        self.tp_pips = tk.IntVar(master, value=60)
        self.risk_percent = tk.DoubleVar(master, value=1.0)
        self.spread_pips = tk.DoubleVar(master, value=1.5)
        self.slippage_pips = tk.DoubleVar(master, value=0.5)

        today = date.today()
        self.end_date_var = tk.StringVar(master, value=today.strftime("%Y-%m-%d"))
        self.start_date_var = tk.StringVar(master, value=(today - timedelta(days=365)).strftime("%Y-%m-%d"))

        self.status_text = tk.StringVar(master, value="Ready - Multi-Strategy Backtesting v7.0")
        
        # Multi-strategy results storage
        self.all_results = {}  # {strategy_name: {'metrics': {}, 'trades_df': df, 'summary': str}}
        self.summary_labels = {}

        self.setup_ui()
        self.refresh_available_pairs()
        self.master.after(500, self.update_pair_info)

        if self.data_folder.exists():
            self.update_status("Data folder ready • " + self.data_folder.name, "#238636")
        else:
            self.update_status("Data folder not found - please select", "#f85149")

    def setup_ui(self):
        """Setup UI with strategy checkboxes"""
        main_container = ctk.CTkFrame(self.master, corner_radius=0, fg_color="#000000")
        main_container.pack(fill='both', expand=True)

        # Sidebar
        self.sidebar_width = 240
        sidebar = ctk.CTkFrame(main_container, corner_radius=0, fg_color="#000000", width=self.sidebar_width)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        # Logo
        logo_frame = ctk.CTkFrame(sidebar, corner_radius=0, fg_color="transparent")
        logo_frame.pack(fill='x', padx=16, pady=(20, 30))

        ctk.CTkLabel(
            logo_frame,
            text="⚡ algoHaus v7.0",
            font=ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
            text_color="#6e7681",
            anchor="w"
        ).pack(anchor='w')

        ctk.CTkLabel(
            logo_frame,
            text="Multi-Strategy Backtester",
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color="#6e7681",
            anchor="w"
        ).pack(anchor='w', pady=(2, 0))

        # Navigation
        nav_frame = ctk.CTkFrame(sidebar, corner_radius=0, fg_color="transparent")
        nav_frame.pack(fill='x', padx=12, pady=(0, 20))

        self.nav_buttons = {}

        self.nav_buttons['config'] = self.create_nav_button(
            nav_frame, "⚙  Configuration", "config", selected=True
        )

        self.nav_buttons['strategy'] = self.create_nav_button(
            nav_frame, "📊 Strategies & Risk", "strategy"
        )

        self.nav_buttons['account'] = self.create_nav_button(
            nav_frame, "💰 Account", "account"
        )

        self.nav_buttons['results'] = self.create_nav_button(
            nav_frame, "📈 Results", "results"
        )

        # Run button
        self.run_btn = ctk.CTkButton(
            sidebar,
            text="▶  RUN BACKTESTS",
            font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
            fg_color="#238636",
            hover_color="#2ea043",
            text_color="#ffffff",
            height=44,
            corner_radius=6,
            command=self.start_backtest_thread
        )
        self.run_btn.pack(fill='x', padx=12, pady=(15, 8))

        # Report button
        self.report_button = ctk.CTkButton(
            sidebar,
            text="📄 Generate Reports",
            font=ctk.CTkFont(family="Helvetica", size=12),
            fg_color="#21262d",
            hover_color="#388bfd",
            text_color="#ffffff",
            height=38,
            corner_radius=6,
            command=self.generate_all_reports,
            state="disabled"
        )
        self.report_button.pack(fill='x', padx=12, pady=(0, 8))

        # Utility buttons
        utility_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        utility_container.pack(fill='x', padx=12, pady=(5, 15))

        ctk.CTkButton(
            utility_container,
            text="🗑️ Cache",
            font=ctk.CTkFont(family="Helvetica", size=10),
            fg_color="#21262d",
            hover_color="#30363d",
            text_color="#8b949e",
            height=32,
            corner_radius=6,
            command=self.clear_cache
        ).pack(side='left', expand=True, fill='x', padx=(0, 3))

        ctk.CTkButton(
            utility_container,
            text="📊 Export",
            font=ctk.CTkFont(family="Helvetica", size=10),
            fg_color="#21262d",
            hover_color="#30363d",
            text_color="#8b949e",
            height=32,
            corner_radius=6,
            command=self.export_all_to_csv
        ).pack(side='left', expand=True, fill='x', padx=(3, 0))

        # Content area
        content_area = ctk.CTkFrame(main_container, corner_radius=0, fg_color="#000000")
        content_area.pack(side='right', fill='both', expand=True)

        self.content_frame = ctk.CTkFrame(content_area, fg_color="transparent")
        self.content_frame.pack(fill='both', expand=True, padx=25, pady=20)

        self.sections = {}
        self.create_config_section()
        self.create_strategy_section()  # MODIFIED
        self.create_account_section()
        self.create_results_section()

        self.show_section('config')

        # Progress bar
        self.progress_container = ctk.CTkFrame(content_area, fg_color="#000000", corner_radius=8)

        self.progress_label = ctk.CTkLabel(
            self.progress_container,
            text="",
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color="#8b949e",
            anchor="w"
        )
        self.progress_label.pack(fill='x', padx=12, pady=(10, 5))

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_container,
            mode="determinate",
            height=5,
            corner_radius=3,
            fg_color="#21262d",
            progress_color="#238636"
        )
        self.progress_bar.pack(fill='x', padx=12, pady=(0, 10))
        self.progress_bar.set(0)

        # Status bar
        status_bar = ctk.CTkFrame(self.master, corner_radius=0, height=32, fg_color="#000000")
        status_bar.pack(side='bottom', fill='x')

        self.status_label = ctk.CTkLabel(
            status_bar,
            textvariable=self.status_text,
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color="#8b949e",
            anchor="w"
        )
        self.status_label.pack(side='left', padx=25, pady=8)

    def create_nav_button(self, parent, text, section_id, selected=False):
        """Create navigation button"""
        btn = ctk.CTkButton(
            parent,
            text=text,
            font=ctk.CTkFont(family="Helvetica", size=12),
            fg_color="#21262d" if selected else "transparent",
            hover_color="#30363d" if selected else "#21262d",
            text_color="#e6edf3" if selected else "#8b949e",
            anchor="w",
            height=36,
            corner_radius=6,
            command=lambda: self.show_section(section_id)
        )
        btn.pack(fill='x', pady=2)
        return btn

    def update_nav_selection(self, selected_section):
        """Update navigation selection"""
        for section_id, btn in self.nav_buttons.items():
            if section_id == selected_section:
                btn.configure(fg_color="#21262d", text_color="#e6edf3")
            else:
                btn.configure(fg_color="transparent", text_color="#8b949e")

    def show_section(self, section_id):
        """Show specific section"""
        self.current_section = section_id
        self.update_nav_selection(section_id)

        for sec_id, sec_frame in self.sections.items():
            if sec_id == section_id:
                sec_frame.pack(fill='both', expand=True)
            else:
                sec_frame.pack_forget()

    def create_config_section(self):
        """Configuration section - same as original"""
        section = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.sections['config'] = section

        ctk.CTkLabel(
            section,
            text="Configuration",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="normal"),
            text_color="#e6edf3",
            anchor="w"
        ).pack(anchor='w', pady=(0, 20))

        content = ctk.CTkFrame(section, fg_color="#000000", corner_radius=10)
        content.pack(fill='both', expand=True)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill='both', expand=True, padx=25, pady=25)

        # Data Folder
        self.create_section_header(inner, "Data Source")
        folder_frame = ctk.CTkFrame(inner, fg_color="#21262d", corner_radius=8)
        folder_frame.pack(fill='x', pady=(0, 20))

        folder_inner = ctk.CTkFrame(folder_frame, fg_color="transparent")
        folder_inner.pack(fill='x', padx=12, pady=10)

        ctk.CTkLabel(
            folder_inner,
            text="Data Folder",
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color="#8b949e",
            width=90,
            anchor="w"
        ).pack(side='left')

        folder_display = str(self.data_folder)
        if len(folder_display) > 45:
            folder_display = "..." + folder_display[-42:]

        self.folder_label = ctk.CTkLabel(
            folder_inner,
            text=folder_display,
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color="#e6edf3",
            anchor="w"
        )
        self.folder_label.pack(side='left', fill='x', expand=True, padx=12)

        ctk.CTkButton(
            folder_inner,
            text="Browse",
            command=self.select_data_folder,
            width=70,
            height=28,
            font=ctk.CTkFont(family="Helvetica", size=10),
            fg_color="#30363d",
            hover_color="#484f58",
            text_color="#e6edf3",
            corner_radius=6
        ).pack(side='right')

        # Trading Pair
        self.create_section_header(inner, "Trading Pair")
        self.pair_combo = self.create_sleek_input(inner, "Pair", self.selected_pair, is_combobox=True, values=["EUR/USD"])

        # Pair Info
        self.pair_info_frame = ctk.CTkFrame(inner, fg_color="#21262d", corner_radius=8)
        self.pair_info_frame.pack(fill='x', pady=(8, 20))

        self.pair_info_label = ctk.CTkLabel(
            self.pair_info_frame,
            text='Select a pair to view details...',
            font=ctk.CTkFont(family="Helvetica", size=10),
            text_color="#8b949e",
            anchor="nw",
            justify="left"
        )
        self.pair_info_label.pack(fill='both', padx=12, pady=12)

        self.selected_pair.trace('w', self.update_pair_info)

        # Timeframe & Dates
        self.create_section_header(inner, "Time Period")
        self.create_sleek_input(inner, "Timeframe", self.selected_timeframe, is_combobox=True,
                               values=["1min", "5min", "15min", "1hr", "1Day"])
        self.create_sleek_input(inner, "Start Date", self.start_date_var)
        self.create_sleek_input(inner, "End Date", self.end_date_var)

    def create_strategy_section(self):
        """MODIFIED: Strategy section with checkboxes and edit buttons"""
        section = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.sections['strategy'] = section

        ctk.CTkLabel(
            section,
            text="Strategies & Risk",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="normal"),
            text_color="#e6edf3",
            anchor="w"
        ).pack(anchor='w', pady=(0, 20))

        content = ctk.CTkFrame(section, fg_color="#000000", corner_radius=10)
        content.pack(fill='both', expand=True)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill='both', expand=True, padx=25, pady=25)

        # Strategy Selection with Checkboxes
        self.create_section_header(inner, "Trading Strategies (Select Multiple)")
        
        strategy_container = ctk.CTkFrame(inner, fg_color="#21262d", corner_radius=8)
        strategy_container.pack(fill='x', pady=(0, 20))
        
        strategy_inner = ctk.CTkFrame(strategy_container, fg_color="transparent")
        strategy_inner.pack(fill='both', padx=12, pady=12)

        strategy_display_names = {
            'vwap_crossover_strategy': 'VWAP Crossover',
            'opening_range_strategy': 'Opening Range Breakout',
            'bollinger_band_reversion_strategy': 'Bollinger Band Mean Reversion'
        }

        for strategy_key, strategy_var in self.selected_strategies.items():
            # Create row for each strategy
            row = ctk.CTkFrame(strategy_inner, fg_color="transparent")
            row.pack(fill='x', pady=4)
            
            # Checkbox
            checkbox = ctk.CTkCheckBox(
                row,
                text=strategy_display_names[strategy_key],
                variable=strategy_var,
                font=ctk.CTkFont(family="Helvetica", size=11),
                text_color="#e6edf3",
                fg_color="#238636",
                hover_color="#2ea043",
                checkbox_width=20,
                checkbox_height=20
            )
            checkbox.pack(side='left', anchor='w')
            
            # Edit button
            edit_btn = ctk.CTkButton(
                row,
                text="✏️ Edit",
                width=70,
                height=28,
                font=ctk.CTkFont(family="Helvetica", size=10),
                fg_color="#30363d",
                hover_color="#484f58",
                text_color="#8b949e",
                corner_radius=6,
                command=lambda sk=strategy_key: self.edit_strategy(sk)
            )
            edit_btn.pack(side='right')

        # Select/Deselect All
        btn_row = ctk.CTkFrame(strategy_container, fg_color="transparent")
        btn_row.pack(fill='x', padx=12, pady=(0, 10))
        
        ctk.CTkButton(
            btn_row,
            text="✓ Select All",
            width=100,
            height=28,
            font=ctk.CTkFont(family="Helvetica", size=10),
            fg_color="#30363d",
            hover_color="#484f58",
            text_color="#8b949e",
            corner_radius=6,
            command=self.select_all_strategies
        ).pack(side='left', padx=(0, 5))
        
        ctk.CTkButton(
            btn_row,
            text="✗ Deselect All",
            width=100,
            height=28,
            font=ctk.CTkFont(family="Helvetica", size=10),
            fg_color="#30363d",
            hover_color="#484f58",
            text_color="#8b949e",
            corner_radius=6,
            command=self.deselect_all_strategies
        ).pack(side='left')

        # Risk Management
        self.create_section_header(inner, "Risk Management")
        self.create_sleek_input(inner, "Stop Loss (pips)", self.sl_pips)
        self.create_sleek_input(inner, "Take Profit (pips)", self.tp_pips)

        # Execution Costs
        self.create_section_header(inner, "Execution Costs")
        self.create_sleek_input(inner, "Spread (pips)", self.spread_pips)
        self.create_sleek_input(inner, "Slippage (pips)", self.slippage_pips)

    def edit_strategy(self, strategy_key):
        """Open strategy editor window"""
        strategy_func = getattr(TradingStrategies, strategy_key)
        display_name = {
            'vwap_crossover_strategy': 'VWAP Crossover',
            'opening_range_strategy': 'Opening Range Breakout',
            'bollinger_band_reversion_strategy': 'Bollinger Band Mean Reversion'
        }[strategy_key]
        
        StrategyEditorWindow(self.master, display_name, strategy_func)

    def select_all_strategies(self):
        """Select all strategies"""
        for var in self.selected_strategies.values():
            var.set(True)

    def deselect_all_strategies(self):
        """Deselect all strategies"""
        for var in self.selected_strategies.values():
            var.set(False)

    def create_account_section(self):
        """Account section - same as original"""
        section = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.sections['account'] = section

        ctk.CTkLabel(
            section,
            text="Account Settings",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="normal"),
            text_color="#e6edf3",
            anchor="w"
        ).pack(anchor='w', pady=(0, 20))

        content = ctk.CTkFrame(section, fg_color="#000000", corner_radius=10)
        content.pack(fill='both', expand=True)

        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill='both', expand=True, padx=25, pady=25)

        self.create_section_header(inner, "Capital & Leverage")
        self.create_sleek_input(inner, "Initial Balance ($)", self.initial_balance)
        self.create_sleek_input(inner, "Leverage", self.leverage, is_combobox=True,
                               values=[str(x) for x in ForexCalculator.LEVERAGE_OPTIONS])

        self.create_section_header(inner, "Position Sizing")
        self.create_sleek_input(inner, "Risk % per Trade", self.risk_percent)

    def create_results_section(self):
        """Results section with multi-strategy support"""
        section = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.sections['results'] = section

        # Main scrollable container
        self.results_scroll = ctk.CTkScrollableFrame(
            section,
            fg_color="transparent",
            scrollbar_button_color="#30363d",
            scrollbar_button_hover_color="#484f58"
        )
        self.results_scroll.pack(fill='both', expand=True)

        # Header
        ctk.CTkLabel(
            self.results_scroll,
            text="Multi-Strategy Results",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="normal"),
            text_color="#e6edf3",
            anchor="w"
        ).pack(anchor='w', pady=(0, 15))

        # Results will be populated dynamically
        self.results_container = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.results_container.pack(fill='both', expand=True)

    # Helper methods
    def create_section_header(self, parent, text):
        """Create section header"""
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family="Helvetica", size=12, weight="bold"),
            text_color="#8b949e",
            anchor="w"
        ).pack(anchor='w', pady=(12, 6))

    def create_sleek_input(self, parent, label_text, variable, is_combobox=False, values=None):
        """Create input field"""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill='x', pady=5)

        ctk.CTkLabel(
            frame,
            text=label_text,
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color="#8b949e",
            width=120,
            anchor="w"
        ).pack(side='left')

        if is_combobox:
            widget = ctk.CTkComboBox(
                frame,
                variable=variable,
                values=values or [],
                font=ctk.CTkFont(family="Helvetica", size=11),
                fg_color="#21262d",
                border_color="#30363d",
                button_color="#30363d",
                button_hover_color="#484f58",
                text_color="#e6edf3",
                dropdown_fg_color="#000000",
                dropdown_hover_color="#21262d",
                dropdown_text_color="#e6edf3",
                corner_radius=6,
                height=32,
                width=240
            )
        else:
            widget = ctk.CTkEntry(
                frame,
                textvariable=variable,
                font=ctk.CTkFont(family="Helvetica", size=11),
                fg_color="#21262d",
                border_color="#30363d",
                text_color="#e6edf3",
                corner_radius=6,
                height=32,
                width=240
            )

        widget.pack(side='left', padx=(12, 0))
        return widget

    def update_status(self, text, color="#8b949e"):
        """Update status bar"""
        self.status_text.set(text)
        self.status_label.configure(text_color=color)

    def get_desktop_path(self):
        """Get desktop path"""
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if os.path.exists(desktop_path):
            return desktop_path

        onedrive_desktop = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
        if os.path.exists(onedrive_desktop):
            return onedrive_desktop

        if os.name == 'nt':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                    r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')
                desktop = winreg.QueryValueEx(key, 'Desktop')[0]
                winreg.CloseKey(key)
                if desktop and os.path.exists(desktop):
                    return desktop
            except:
                pass

        return os.path.expanduser("~")

    def select_data_folder(self):
        """Select data folder"""
        new_folder = filedialog.askdirectory(
            title="Select Main Data Folder",
            initialdir=str(self.data_folder)
        )
        if new_folder:
            self.data_folder = pathlib.Path(new_folder)
            folder_text = str(self.data_folder)
            if len(folder_text) > 45:
                folder_text = "..." + folder_text[-42:]
            self.folder_label.configure(text=folder_text)
            self.refresh_available_pairs()
            self.update_status(f"Data folder updated", "#238636")

    def refresh_available_pairs(self):
        """Refresh available pairs"""
        try:
            self.update_status("Scanning for pairs...", "#8b949e")
            pairs = detect_available_pairs(self.data_folder)

            if pairs:
                if hasattr(self, 'pair_combo') and self.pair_combo:
                    self.pair_combo.configure(values=pairs)
                    if self.selected_pair.get() not in pairs:
                        self.selected_pair.set(pairs[0])
                self.update_status(f"Found {len(pairs)} pairs", "#3fb950")
            else:
                self.update_status("No pairs found", "#f85149")
        except Exception as e:
            logging.error(f"Error refreshing pairs: {e}")
            self.update_status(f"Error: {str(e)}", "#f85149")

    def update_pair_info(self, *args):
        """Update pair info display"""
        pair = self.selected_pair.get()
        if not pair:
            return

        start, end = get_data_date_range(pair, self.data_folder)
        pip_value = ForexCalculator.PIP_VALUES.get(pair, 0.0001)

        if start and end:
            total_days = (end - start).days
            info_text = f"""PAIR: {pair}

Data Available: {start} to {end}
Total Days: {total_days:,}

Pip Value: {pip_value}"""

            self.start_date_var.set(str(start))
            self.end_date_var.set(str(end))
            self.update_status(f"{pair}: {total_days:,} days of data", "#238636")
        else:
            info_text = f"PAIR: {pair}\n\nNo data found"
            self.update_status(f"No data found for {pair}", "#f85149")

        self.pair_info_label.configure(text=info_text)

    # PARALLEL BACKTEST EXECUTION
    def start_backtest_thread(self):
        """Start multi-strategy backtest"""
        # Check if any strategy is selected
        selected = [name for name, var in self.selected_strategies.items() if var.get()]
        
        if not selected:
            messagebox.showwarning("No Strategy", "Please select at least one strategy")
            return
        
        self.update_status(f"Running {len(selected)} strategies in parallel...", "#238636")
        
        # Clear previous results
        self.all_results = {}
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.report_button.configure(state="disabled")
        
        self.progress_container.pack(fill='x', padx=25, pady=(0, 15), before=self.content_frame)
        self.progress_bar.set(0)
        self.progress_label.configure(text="Initializing backtests...")

        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            if start_date >= end_date:
                raise ValueError("Start date must be before End date.")
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid date input: {e}")
            self.update_status("Invalid date input", "#f85149")
            self.progress_container.pack_forget()
            return

        self.q = queue.Queue()
        threading.Thread(target=self.run_parallel_backtests,
                         args=(start_date, end_date, selected),
                         daemon=True).start()
        self.master.after(100, self.check_queue)

    def run_parallel_backtests(self, start_date, end_date, selected_strategies):
        """Run multiple strategies in parallel"""
        try:
            pair = self.selected_pair.get()
            timeframe = self.selected_timeframe.get()
            
            # Load data once
            self.q.put(('progress', 0, f"Loading data for {pair}..."))
            df, actual_start, actual_end = load_pair_data(pair, self.data_folder, start_date, end_date, timeframe)
            self.df = df
            
            # Prepare strategy arguments
            strategy_args = []
            for strategy_name in selected_strategies:
                strategy_func = getattr(TradingStrategies, strategy_name)
                
                args = (
                    df.copy(),  # Each strategy gets its own copy
                    strategy_name,
                    strategy_func,
                    self.sl_pips.get(),
                    self.tp_pips.get(),
                    pair,
                    self.initial_balance.get(),
                    self.leverage.get(),
                    self.risk_percent.get(),
                    self.spread_pips.get(),
                    self.slippage_pips.get(),
                    ForexCalculator.PIP_VALUES.get(pair, 0.0001)
                )
                strategy_args.append(args)
            
            # Run strategies in parallel using ThreadPoolExecutor (faster for I/O)
            results = []
            with ThreadPoolExecutor(max_workers=min(len(selected_strategies), 4)) as executor:
                futures = {executor.submit(run_single_strategy, args): args[1] for args in strategy_args}
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    progress = int((completed / len(selected_strategies)) * 100)
                    self.q.put(('progress', progress, f"Completed {completed}/{len(selected_strategies)} strategies"))
            
            self.q.put(('success', results))
            
        except Exception as e:
            logging.error(f"Parallel backtest error: {e}", exc_info=True)
            self.q.put(('error', str(e)))

    def check_queue(self):
        """Check for backtest results"""
        try:
            result_type, *data = self.q.get_nowait()

            if result_type == 'progress':
                percent, message = data
                self.progress_bar.set(percent / 100)
                self.progress_label.configure(text=message)
                self.master.after(100, self.check_queue)
            elif result_type == 'success':
                results = data[0]
                self.display_multi_strategy_results(results)
                self.update_status(f"Completed {len(results)} strategies", "#3fb950")
                self.progress_bar.set(1.0)
                self.progress_label.configure(text="All backtests complete!")
                self.master.after(2000, self.progress_container.pack_forget)
            elif result_type == 'error':
                error_msg = data[0]
                messagebox.showerror("Backtest Error", error_msg)
                self.update_status("Backtest failed", "#f85149")
                self.progress_container.pack_forget()

        except queue.Empty:
            self.master.after(100, self.check_queue)

    def display_multi_strategy_results(self, results):
        """Display results from multiple strategies"""
        # Clear container
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        # Store results
        for result in results:
            if result['success']:
                self.all_results[result['strategy_name']] = result
        
        if not self.all_results:
            ctk.CTkLabel(
                self.results_container,
                text="No successful strategy results",
                font=ctk.CTkFont(family="Helvetica", size=14),
                text_color="#f85149"
            ).pack(pady=50)
            return
        
        # Display each strategy's results
        for idx, (strategy_name, result) in enumerate(self.all_results.items()):
            self.create_strategy_result_card(strategy_name, result, idx)
        
        self.report_button.configure(state="normal")
        self.show_section('results')

    def create_strategy_result_card(self, strategy_name, result, index):
        """Create result card for a single strategy"""
        display_names = {
            'vwap_crossover_strategy': 'VWAP Crossover',
            'opening_range_strategy': 'Opening Range Breakout',
            'bollinger_band_reversion_strategy': 'Bollinger Band Mean Reversion'
        }
        
        display_name = display_names.get(strategy_name, strategy_name)
        
        # Card container
        card = ctk.CTkFrame(self.results_container, fg_color="#0d1117", corner_radius=10)
        card.pack(fill='x', pady=(0 if index == 0 else 10, 0))
        
        # Header
        header = ctk.CTkFrame(card, fg_color="#161b22", corner_radius=8)
        header.pack(fill='x', padx=15, pady=15)
        
        ctk.CTkLabel(
            header,
            text=f"📊 {display_name}",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            text_color="#e6edf3",
            anchor="w"
        ).pack(side='left', padx=12, pady=10)
        
        # Summary metrics
        metrics = result['metrics']
        trades_df = result['trades_df']
        
        if not trades_df.empty:
            summary_frame = ctk.CTkFrame(card, fg_color="transparent")
            summary_frame.pack(fill='x', padx=15, pady=(0, 15))
            
            # Configure grid
            for i in range(6):
                summary_frame.columnconfigure(i, weight=1)
            
            # Create metric cards
            total_trades = len(trades_df)
            win_rate = metrics.get('win_rate_%', 0)
            total_pnl = metrics.get('total_pnl_$', 0)
            total_return = metrics.get('total_return_%', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            max_dd = metrics.get('max_drawdown_%', 0)
            
            metric_data = [
                ('Trades', str(total_trades), '#e6edf3'),
                ('Win Rate', f"{win_rate:.1f}%", '#3fb950' if win_rate >= 50 else '#f85149'),
                ('Total P&L', f"${total_pnl:,.0f}", '#3fb950' if total_pnl >= 0 else '#f85149'),
                ('Returns', f"{total_return:+.2f}%", '#3fb950' if total_return >= 0 else '#f85149'),
                ('Sharpe', f"{sharpe:.2f}", '#3fb950' if sharpe > 1 else '#f85149'),
                ('Max DD', f"{max_dd:.1f}%", '#f85149')
            ]
            
            for col, (label, value, color) in enumerate(metric_data):
                metric_card = ctk.CTkFrame(summary_frame, fg_color="#161b22", corner_radius=6)
                metric_card.grid(row=0, column=col, sticky='nsew', padx=4, pady=4)
                
                inner = ctk.CTkFrame(metric_card, fg_color="transparent")
                inner.pack(fill='both', expand=True, padx=8, pady=6)
                
                ctk.CTkLabel(
                    inner,
                    text=label,
                    font=ctk.CTkFont(family="Helvetica", size=9),
                    text_color="#6e7681",
                    anchor="w"
                ).pack(anchor='w')
                
                ctk.CTkLabel(
                    inner,
                    text=value,
                    font=ctk.CTkFont(family="Helvetica", size=13, weight="bold"),
                    text_color=color,
                    anchor="w"
                ).pack(anchor='w', pady=(2, 0))

    def generate_all_reports(self):
        """Generate reports for all strategies"""
        if not self.all_results:
            messagebox.showwarning("No Data", "No results available")
            return
        
        try:
            desktop = self.get_desktop_path()
            
            for strategy_name, result in self.all_results.items():
                if not result['success'] or result['trades_df'].empty:
                    continue
                
                report_path = HTMLReportGenerator.generate_report(
                    result['metrics'],
                    result['trades_df'],
                    strategy_name,
                    self.selected_timeframe.get(),
                    self.selected_pair.get(),
                    self.initial_balance.get(),
                    self.leverage.get(),
                    self.sl_pips.get(),
                    self.tp_pips.get(),
                    self.risk_percent.get(),
                    self.start_date_var.get(),
                    self.end_date_var.get(),
                    df=self.df,
                    output_dir=desktop
                )
                
                logging.info(f"Generated report: {report_path}")
            
            messagebox.showinfo("Success", f"Generated {len(self.all_results)} reports!\nSaved to Desktop")
            self.update_status("Reports generated", "#3fb950")
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed:\n{str(e)}")
            self.update_status("Report error", "#f85149")

    def export_all_to_csv(self):
        """Export all strategy results to CSV"""
        if not self.all_results:
            messagebox.showwarning("No Data", "No results available")
            return
        
        try:
            desktop = self.get_desktop_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for strategy_name, result in self.all_results.items():
                if not result['success'] or result['trades_df'].empty:
                    continue
                
                filename = f"AlgoHaus_{strategy_name}_{timestamp}.csv"
                filepath = os.path.join(desktop, filename)
                
                result['trades_df'].to_csv(filepath, index=False)
                logging.info(f"Exported: {filepath}")
            
            messagebox.showinfo("Success", f"Exported {len(self.all_results)} CSV files to Desktop")
            self.update_status("Data exported", "#3fb950")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed:\n{str(e)}")
            self.update_status("Export failed", "#f85149")

    def clear_cache(self):
        """Clear data cache"""
        global _DATA_CACHE
        with _CACHE_LOCK:
            cache_size = len(_DATA_CACHE)
            _DATA_CACHE.clear()
        
        messagebox.showinfo("Cache Cleared", f"Cleared {cache_size} cached datasets")
        self.update_status("Cache cleared", "#3fb950")


# ══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Enable multiprocessing support on Windows
    if os.name == 'nt':
        mp.freeze_support()
    
    app = ctk.CTk()
    backtester = BacktesterUI(app)
    app.mainloop()
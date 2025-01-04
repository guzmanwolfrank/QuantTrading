# utils/__init__.py

from .performance_metrics import PerformanceAnalyzer
from .visualizer import Visualizer
from .risk_manager import RiskManager
from .report_generator import ReportGenerator
from .data_processor import DataProcessor

__all__ = [
    'PerformanceAnalyzer',
    'Visualizer',
    'RiskManager',
    'ReportGenerator',
    'DataProcessor'
]

# Utility function defaults and configurations
DEFAULT_CONFIGS = {
    'risk_management': {
        'max_position_size': 0.02,  # 2% per trade
        'max_correlation': 0.7,
        'max_drawdown': 0.20,  # 20% maximum drawdown
        'risk_free_rate': 0.02,  # 2% risk-free rate for calculations
        'position_sizing': {
            'method': 'fixed_fractional',
            'risk_per_trade': 0.01  # 1% risk per trade
        }
    },
    'performance_metrics': {
        'rolling_window': 252,  # Trading days in a year
        'risk_metrics': {
            'var_confidence': 0.95,
            'calculation_method': 'historical'
        },
        'benchmark': '^GSPC'  # S&P 500 as default benchmark
    },
    'visualization': {
        'default_figsize': (12, 8),
        'style': 'seaborn',
        'color_scheme': {
            'lines': ['#1f77b4', '#ff7f0e', '#2ca02c'],
            'fills': ['#a6cee3', '#fdbf6f', '#b2df8a']
        }
    },
    'reporting': {
        'template': 'default',
        'include_sections': [
            'summary',
            'performance_metrics',
            'risk_metrics',
            'trade_analysis',
            'visualizations'
        ]
    },
    'data_processing': {
        'default_timeframe': '1D',
        'fill_method': 'ffill',
        'min_data_points': 252,
        'adjust_prices': True
    }
}

# Version information
__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# System requirements
MINIMUM_PYTHON_VERSION = '3.8.0'
REQUIRED_PACKAGES = [
    'pandas>=1.5.0',
    'numpy>=1.21.0',
    'yfinance>=0.2.0',
    'seaborn>=0.12.0',
    'matplotlib>=3.5.0',
    'scipy>=1.7.0'
]

def get_strategy_info(strategy_name):
    """Get information about a specific strategy"""
    from .strategies import AVAILABLE_STRATEGIES
    return AVAILABLE_STRATEGIES.get(strategy_name, None)

def list_available_strategies():
    """List all available strategies with descriptions"""
    from .strategies import AVAILABLE_STRATEGIES
    return {name: info['description'] 
            for name, info in AVAILABLE_STRATEGIES.items()}

def validate_parameters(strategy_name, parameters):
    """Validate parameters for a specific strategy"""
    from .strategies import AVAILABLE_STRATEGIES
    
    strategy_info = AVAILABLE_STRATEGIES.get(strategy_name)
    if not strategy_info:
        raise ValueError(f"Strategy {strategy_name} not found")
        
    valid_params = {}
    for param_name, param_value in parameters.items():
        param_info = strategy_info['parameters'].get(param_name)
        if not param_info:
            raise ValueError(f"Invalid parameter {param_name} for strategy {strategy_name}")
            
        # Type checking
        if not isinstance(param_value, param_info['type']):
            raise TypeError(f"Parameter {param_name} must be of type {param_info['type']}")
            
        # Range checking
        if 'range' in param_info:
            min_val, max_val = param_info['range']
            if param_value < min_val or param_value > max_val:
                raise ValueError(
                    f"Parameter {param_name} must be between {min_val} and {max_val}"
                )
                
        valid_params[param_name] = param_value
        
    return valid_params
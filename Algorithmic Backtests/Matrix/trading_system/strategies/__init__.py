# strategies/__init__.py

from .bollinger_bounce import BollingerBounceStrategy
from .moving_average_cross import MovingAverageCrossStrategy
from .support_resistance import SupportResistanceStrategy

__all__ = [
    'BollingerBounceStrategy',
    'MovingAverageCrossStrategy',
    'SupportResistanceStrategy'
]

# Strategy metadata for dynamic loading
AVAILABLE_STRATEGIES = {
    'bollinger_bounce': {
        'class': BollingerBounceStrategy,
        'description': 'Trading strategy based on Bollinger Bands bounces',
        'parameters': {
            'bb_period': {
                'type': int,
                'default': 20,
                'range': (10, 50),
                'description': 'Period for Bollinger Bands calculation'
            },
            'bb_std': {
                'type': float,
                'default': 2.0,
                'range': (1.0, 3.0),
                'description': 'Number of standard deviations for bands'
            }
        }
    },
    'moving_average_cross': {
        'class': MovingAverageCrossStrategy,
        'description': 'Moving average crossover strategy',
        'parameters': {
            'fast_period': {
                'type': int,
                'default': 10,
                'range': (5, 20),
                'description': 'Period for fast moving average'
            },
            'slow_period': {
                'type': int,
                'default': 30,
                'range': (20, 50),
                'description': 'Period for slow moving average'
            },
            'exit_bars': {
                'type': int,
                'default': 5,
                'range': (1, 10),
                'description': 'Number of bars to hold position'
            }
        }
    },
    'support_resistance': {
        'class': SupportResistanceStrategy,
        'description': 'Support and resistance level trading strategy',
        'parameters': {
            'window': {
                'type': int,
                'default': 20,
                'range': (10, 50),
                'description': 'Window for support/resistance calculation'
            },
            'threshold': {
                'type': float,
                'default': 0.02,
                'range': (0.01, 0.05),
                'description': 'Threshold for level identification'
            }
        }
    }
}
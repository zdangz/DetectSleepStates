"""
Sleep State Detection Package

A modular package for detecting sleep states using accelerometer data
and machine learning models.
"""

from . import config
from . import data_loader
from . import feature_engineering
from . import model
from . import utils

__version__ = "1.0.0"
__all__ = [
    "config",
    "data_loader",
    "feature_engineering",
    "model",
    "utils",
]

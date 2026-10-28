"""Standalone Ascend 950PR Polar Express operator."""

from ._loader import load_library

load_library()

from .ops import polar_step

__all__ = ["polar_step"]
__version__ = "0.1.1"

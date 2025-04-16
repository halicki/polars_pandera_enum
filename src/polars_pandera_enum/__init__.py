"""Polars Pandera Enum integration package.

This package provides a Pydantic-compatible wrapper for Polars DataFrames
that are validated with Pandera schemas.
"""

from .type_integration import PolarsDataFrame

__all__ = ["PolarsDataFrame"]
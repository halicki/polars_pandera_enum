# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_file.py::test_function`
- Run tests in watch mode: `pytest-watch`
- Lint code: `ruff check .`
- Type check: `mypy src tests`

## Code Style Guidelines
- Imports: Group standard library, third-party, and local imports with a blank line between groups
- Types: Use type annotations for all function parameters and return values
- Naming: Use snake_case for variables/functions and PascalCase for classes
- Pandera schemas: Define field constraints with pa.Field(), document with docstrings
- Error handling: Use pytest.raises for expected exceptions in tests
- Format docstrings with triple quotes and a brief description
- When working with Polars DataFrames, use pandera.typing.polars types
- Follow the class-based schema pattern for data validation
- Maintain separation between schema definition and data processing logic
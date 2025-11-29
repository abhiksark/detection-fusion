.PHONY: format lint check test clean install dev-install help

# Default target
help:
	@echo "DetectionFusion Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Formatting:"
	@echo "  make format     - Format code and sort imports with ruff"
	@echo "  make lint       - Check code style without fixing"
	@echo "  make check      - Run all checks (lint + type check)"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run tests"
	@echo "  make test-cov   - Run tests with coverage"
	@echo ""
	@echo "Installation:"
	@echo "  make install    - Install package"
	@echo "  make dev        - Install package with dev dependencies"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean      - Remove build artifacts"

# Format code with ruff (includes import sorting)
format:
	ruff check --fix detection_fusion tests examples
	ruff format detection_fusion tests examples

# Check code style without fixing
lint:
	ruff check detection_fusion tests examples
	ruff format --check detection_fusion tests examples

# Run type checking
typecheck:
	mypy detection_fusion

# Run all checks
check: lint typecheck

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=detection_fusion --cov-report=term-missing

# Install package
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev]"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

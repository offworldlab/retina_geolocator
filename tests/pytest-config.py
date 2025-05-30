# pytest.ini
"""
[pytest]
# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test output options
addopts = 
    -v
    --tb=short
    --strict-markers
    --cov=initial_guess
    --cov=least_squares_geolocator
    --cov-report=html
    --cov-report=term-missing
    --cov-branch

# Test markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests

# Ignore warnings from scipy
filterwarnings =
    ignore::DeprecationWarning
    ignore::RuntimeWarning:scipy.*
"""

# conftest.py
"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
from initial_guess import Position3D, Sensor, IoO, Measurement


@pytest.fixture
def basic_sensor_configuration():
    """Provide basic 2-sensor configuration for tests"""
    sensors = [
        Sensor("s1", Position3D(0, 0, 100), "ioo1"),
        Sensor("s2", Position3D(50000, 0, 150), "ioo1")
    ]
    ioos = [
        IoO("ioo1", Position3D(25000, 40000, 200))
    ]
    return sensors, ioos


@pytest.fixture
def sample_measurements():
    """Provide sample measurements for tests"""
    return [
        Measurement(1234567890, "s1", 58.44, -25.45, 5.52),
        Measurement(1234567890, "s2", 62.31, -18.72, 6.12)
    ]


@pytest.fixture
def synthetic_target():
    """Provide synthetic target position and velocity"""
    position = np.array([30000, 20000, 5000])
    velocity = np.array([100, -50, 0])
    return position, velocity


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility"""
    np.random.seed(42)


# Custom pytest markers for organizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# requirements_test.txt
"""
# Testing requirements
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
coverage>=7.0.0
numpy>=1.20.0
scipy>=1.7.0
"""

# Makefile for easy test running
"""
# Makefile

.PHONY: test test-coverage test-unit test-integration test-verbose clean

# Run all tests
test:
	python -m pytest

# Run tests with coverage
test-coverage:
	python -m pytest --cov --cov-report=html --cov-report=term

# Run only unit tests
test-unit:
	python -m pytest -m unit

# Run only integration tests  
test-integration:
	python -m pytest -m integration

# Run tests with verbose output
test-verbose:
	python -m pytest -vv

# Run specific test file
test-file:
	python -m pytest $(FILE)

# Run specific test class
test-class:
	python -m pytest -k $(CLASS)

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install test dependencies
install-test:
	pip install -r requirements_test.txt

# Run tests and open coverage report
test-open:
	python -m pytest --cov --cov-report=html
	open htmlcov/index.html
"""
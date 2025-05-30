# Bistatic Passive Radar Geolocation

A modular Python package for geolocating targets using Time Difference of Arrival (TDOA) and Frequency Difference of Arrival (FDOA) measurements from bistatic passive radar systems.

## Features

- **Least Squares Optimization**: Robust geolocation using scipy's least squares solver
- **Multiple Initial Guess Strategies**: Flexible starting point estimation for better convergence
- **2D/3D Position Estimation**: Support for fixed altitude or full 3D positioning
- **Velocity Estimation**: Optional target velocity estimation using Doppler measurements
- **Modular Design**: Easy to extend with new strategies and algorithms
- **Comprehensive Testing**: Full test suite with >90% code coverage

## Installation

### Requirements

```bash
python >= 3.7
numpy >= 1.20.0
scipy >= 1.7.0
```

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bistatic-radar-geolocation.git
cd bistatic-radar-geolocation

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with test dependencies
pip install -r requirements.txt
pip install -r requirements_test.txt
```

## Quick Start

```python
from least_squares_geolocator import BistaticRadarGeolocator, parse_measurement_json
from initial_guess import Position3D, Sensor, IoO

# Define sensor configuration
sensors = [
    Sensor("sensor1", Position3D(0, 0, 100), "ioo1"),
    Sensor("sensor2", Position3D(50000, 0, 150), "ioo1")
]

ioos = [
    IoO("ioo1", Position3D(25000, 40000, 200))
]

# Create geolocator
geolocator = BistaticRadarGeolocator(sensors, ioos, freq_hz=100e6)

# Parse measurements from JSON
meas1_json = '{"timestamp":1748609918201,"delay":[58.44],"doppler":[-25.45],"snr":[5.52]}'
meas2_json = '{"timestamp":1748609918201,"delay":[62.31],"doppler":[-18.72],"snr":[6.12]}'

measurements = [
    parse_measurement_json(meas1_json, "sensor1"),
    parse_measurement_json(meas2_json, "sensor2")
]

# Perform geolocation
result = geolocator.geolocate(measurements, estimate_velocity=True)

print(f"Position: {result['position']} m")
print(f"Velocity: {result['velocity']} m/s")
print(f"Success: {result['success']}")
```

## Architecture

### Core Components

#### 1. Data Classes (`initial_guess.py`)
- `Position3D`: 3D position in East-North-Up (ENU) coordinates
- `Sensor`: Radar sensor configuration with position and associated IoO
- `IoO`: Illuminator of Opportunity (e.g., FM transmitter, TV broadcaster)
- `Measurement`: Single detection with timestamp, delays, Doppler, and SNR

#### 2. Initial Guess Strategies (`initial_guess.py`)
- `EllipseCenterStrategy`: Default strategy using midpoint between ellipse centers
- `WeightedEllipseCenterStrategy`: SNR-weighted version for better accuracy
- `IsodopplerIntersectionStrategy`: Optimized for near-stationary targets
- `GridSearchStrategy`: Coarse grid search to avoid local minima

#### 3. Least Squares Geolocator (`least_squares_geolocator.py`)
- Main optimization engine using scipy's least squares
- Supports multiple optimization methods (TRF, Dogbox, LM)
- Automatic bounds generation
- Measurement weighting by SNR
- Covariance estimation and error ellipse calculation

## Usage Examples

### Basic 2D Geolocation (Fixed Altitude)

```python
# Create geolocator with fixed altitude
geolocator = BistaticRadarGeolocator(
    sensors, 
    ioos, 
    altitude_assumption=5000  # Fix at 5km
)

# Geolocate without velocity estimation
result = geolocator.geolocate(measurements, estimate_velocity=False)
```

### 3D Position + Velocity Estimation

```python
# Full 3D problem
geolocator = BistaticRadarGeolocator(sensors, ioos)

# Estimate both position and velocity
result = geolocator.geolocate(
    measurements, 
    estimate_velocity=True,
    use_fdoa=True
)
```

### Using Different Initial Guess Strategies

```python
# Method 1: Set strategy by name
geolocator.set_initial_guess_strategy('weighted', altitude_assumption=1000)

# Method 2: Create custom strategy
from initial_guess import GridSearchStrategy

grid_strategy = GridSearchStrategy(
    grid_bounds={'x': (0, 100000), 'y': (0, 100000), 'z': (0, 20000)},
    grid_points=20
)
geolocator.set_initial_guess_strategy(grid_strategy)

# Method 3: Pass strategy at initialization
geolocator = BistaticRadarGeolocator(
    sensors, 
    ioos,
    initial_guess_strategy=grid_strategy
)
```

### Constrained Optimization

```python
# Define bounds for target position
bounds = (
    np.array([0, 0, 0, -500, -500, -100]),      # Lower bounds [x,y,z,vx,vy,vz]
    np.array([100000, 100000, 20000, 500, 500, 100])  # Upper bounds
)

result = geolocator.geolocate(
    measurements,
    estimate_velocity=True,
    bounds=bounds,
    method='trf'  # Trust Region Reflective algorithm
)
```

### Custom Measurement Weights

```python
# Weight measurements by SNR squared
weights = np.array([m.snr**2 for m in measurements])

result = geolocator.geolocate(
    measurements,
    measurement_weights=weights
)
```

## Understanding the Results

The `geolocate()` method returns a comprehensive dictionary:

```python
{
    'success': bool,                # Optimization success flag
    'position': np.ndarray,         # Target position [x,y,z] in meters
    'velocity': np.ndarray,         # Target velocity [vx,vy,vz] in m/s (if estimated)
    'covariance': np.ndarray,       # Full state covariance matrix
    'position_covariance': np.ndarray,  # Position-only covariance
    'error_ellipse': {              # 2D error ellipse (95% confidence)
        'semi_major_m': float,      # Semi-major axis in meters
        'semi_minor_m': float,      # Semi-minor axis in meters  
        'angle_deg': float          # Orientation angle in degrees
    },
    'tdoa_residuals_km': np.ndarray,   # TDOA residuals in km
    'fdoa_residuals_hz': np.ndarray,   # FDOA residuals in Hz (if used)
    'cost': float,                  # Final cost function value
    'nfev': int,                    # Number of function evaluations
    'message': str,                 # Optimization termination message
    'initial_guess': np.ndarray,    # Initial state estimate used
    'final_state': np.ndarray       # Final optimized state vector
}
```

## Creating Custom Initial Guess Strategies

Extend the `InitialGuessStrategy` base class:

```python
from initial_guess import InitialGuessStrategy
import numpy as np

class MyCustomStrategy(InitialGuessStrategy):
    def __init__(self, **kwargs):
        # Initialize your strategy parameters
        self.param = kwargs.get('param', default_value)
    
    def compute(self, measurements, sensors, ioos, **kwargs):
        # Implement your initial guess logic
        # Return numpy array of appropriate dimension
        
        # For 2D position only: return np.array([x, y])
        # For 3D position only: return np.array([x, y, z])
        # For 2D position + velocity: return np.array([x, y, vx, vy])
        # For 3D position + velocity: return np.array([x, y, z, vx, vy, vz])
        
        return initial_guess
```

## Testing

### Run All Tests

```bash
# Using unittest
python run_tests.py

# Using pytest
pytest
```

### Run with Coverage

```bash
# Using unittest
python run_tests.py  # Coverage included by default

# Using pytest
pytest --cov --cov-report=html
```

### Run Specific Tests

```bash
# Test specific module
pytest test_initial_guess.py

# Test specific class
python run_tests.py --class TestEllipseCenterStrategy

# Test by pattern
pytest -k "ellipse"
```

### View Coverage Report

```bash
# After running tests with coverage
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Algorithm Details

### Bistatic Geometry

The system uses bistatic radar geometry where:
- **Bistatic Range**: `R_b = R_tx + R_rx - R_baseline`
- **R_tx**: Distance from IoO to target
- **R_rx**: Distance from target to sensor
- **R_baseline**: Direct path distance from IoO to sensor

### TDOA (Time Difference of Arrival)

Measures the time difference between:
1. Direct signal path: IoO → Sensor
2. Reflected path: IoO → Target → Sensor

This provides the bistatic range sum.

### FDOA (Frequency Difference of Arrival)

Doppler shift caused by target motion:
```
f_d = -(f_c / c) * v · (û_tx + û_rx)
```
Where:
- `f_c`: Carrier frequency
- `c`: Speed of light
- `v`: Target velocity vector
- `û_tx`, `û_rx`: Unit vectors from IoO to target and target to sensor

### Least Squares Optimization

Minimizes the cost function:
```
J = Σ w_i * (r_i)²
```
Where:
- `r_i`: Residuals (measured - predicted values)
- `w_i`: Measurement weights (default: SNR-based)

## Performance Considerations

### Convergence Tips

1. **Multiple Sensors**: Use 3+ sensors for reliable 3D+velocity estimation
2. **Good Geometry**: Avoid colinear sensor arrangements
3. **Time Diversity**: Multiple measurements over time improve velocity estimates
4. **Initial Guess**: Choose appropriate strategy based on scenario
5. **Bounds**: Use physical constraints to improve convergence

### Computational Efficiency

- Grid search strategy is slowest but most robust
- Ellipse center strategies are fast but may converge to local minima
- Use measurement weights to emphasize high-quality data
- Consider fixing altitude for ground targets (2D problem)

## Limitations

1. **Observability**: 2 sensors with TDOA+FDOA cannot uniquely determine 3D position + velocity
2. **Local Minima**: Non-convex problem may have multiple solutions
3. **Noise Sensitivity**: Performance degrades with measurement noise
4. **Near-Field**: Assumes far-field conditions (planar wavefronts)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`python run_tests.py`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{bistatic_radar_geolocation,
  title = {Bistatic Passive Radar Geolocation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/bistatic-radar-geolocation}
}
```

## Acknowledgments

- Based on bistatic radar theory and least squares optimization
- Uses scipy for numerical optimization
- Inspired by passive radar research in the signal processing community

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com].
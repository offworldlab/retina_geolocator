"""
Initial guess strategies for bistatic radar geolocation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable
from abc import ABC, abstractmethod


@dataclass
class Position3D:
    """3D position in ENU coordinates"""
    east: float
    north: float
    up: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.east, self.north, self.up])
    
    def to_2d_array(self) -> np.ndarray:
        return np.array([self.east, self.north])


@dataclass
class Sensor:
    """Sensor configuration"""
    id: str
    position: Position3D
    ioo_id: str  # Associated IoO ID
    

@dataclass
class IoO:
    """Illuminator of Opportunity"""
    id: str
    position: Position3D


@dataclass
class Measurement:
    """Single measurement from a sensor"""
    timestamp: int
    sensor_id: str
    delay_km: float  # Bistatic delay in km
    doppler_hz: float  # Doppler shift in Hz
    snr: float


class InitialGuessStrategy(ABC):
    """Abstract base class for initial guess strategies"""
    
    @abstractmethod
    def compute(self, measurements: List[Measurement], 
                sensors: Dict[str, Sensor], 
                ioos: Dict[str, IoO],
                **kwargs) -> np.ndarray:
        """Compute initial guess for the state vector"""
        pass


class EllipseCenterStrategy(InitialGuessStrategy):
    """
    Initial guess as midpoint between ellipse centers
    This is the default strategy you specified
    """
    
    def __init__(self, altitude_assumption: Optional[float] = None,
                 estimate_velocity: bool = False,
                 is_3d: bool = True):
        self.altitude_assumption = altitude_assumption
        self.estimate_velocity = estimate_velocity
        self.is_3d = is_3d
    
    def compute(self, measurements: List[Measurement], 
                sensors: Dict[str, Sensor], 
                ioos: Dict[str, IoO],
                **kwargs) -> np.ndarray:
        """
        Compute initial guess as midpoint between ellipse centers
        """
        centers = []
        
        for meas in measurements:
            sensor = sensors[meas.sensor_id]
            ioo = ioos[sensor.ioo_id]
            
            sensor_2d = sensor.position.to_2d_array()
            ioo_2d = ioo.position.to_2d_array()
            
            # Center of the ellipse for this sensor-IoO pair
            center = 0.5 * (sensor_2d + ioo_2d)
            centers.append(center)
        
        # Average of all centers
        initial_guess_2d = np.mean(centers, axis=0)
        
        # Determine altitude
        if self.altitude_assumption is not None:
            initial_altitude = self.altitude_assumption
        else:
            # Average altitude of all stations
            all_altitudes = [s.position.up for s in sensors.values()]
            all_altitudes.extend([i.position.up for i in ioos.values()])
            initial_altitude = np.mean(all_altitudes)
        
        # Build state vector based on configuration
        if self.is_3d:
            pos_guess = np.array([initial_guess_2d[0], initial_guess_2d[1], initial_altitude])
        else:
            pos_guess = initial_guess_2d
        
        if self.estimate_velocity:
            vel_guess = np.zeros(3 if self.is_3d else 2)
            return np.concatenate([pos_guess, vel_guess])
        else:
            return pos_guess


class WeightedEllipseCenterStrategy(InitialGuessStrategy):
    """
    Weighted average of ellipse centers based on SNR
    """
    
    def __init__(self, altitude_assumption: Optional[float] = None,
                 estimate_velocity: bool = False,
                 is_3d: bool = True):
        self.altitude_assumption = altitude_assumption
        self.estimate_velocity = estimate_velocity
        self.is_3d = is_3d
    
    def compute(self, measurements: List[Measurement], 
                sensors: Dict[str, Sensor], 
                ioos: Dict[str, IoO],
                **kwargs) -> np.ndarray:
        """
        Compute SNR-weighted initial guess
        """
        centers = []
        weights = []
        
        for meas in measurements:
            sensor = sensors[meas.sensor_id]
            ioo = ioos[sensor.ioo_id]
            
            sensor_2d = sensor.position.to_2d_array()
            ioo_2d = ioo.position.to_2d_array()
            
            center = 0.5 * (sensor_2d + ioo_2d)
            centers.append(center)
            weights.append(meas.snr)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        initial_guess_2d = np.average(centers, axis=0, weights=weights)
        
        # Handle altitude
        if self.altitude_assumption is not None:
            initial_altitude = self.altitude_assumption
        else:
            all_altitudes = [s.position.up for s in sensors.values()]
            all_altitudes.extend([i.position.up for i in ioos.values()])
            initial_altitude = np.mean(all_altitudes)
        
        # Build state vector
        if self.is_3d:
            pos_guess = np.array([initial_guess_2d[0], initial_guess_2d[1], initial_altitude])
        else:
            pos_guess = initial_guess_2d
        
        if self.estimate_velocity:
            vel_guess = np.zeros(3 if self.is_3d else 2)
            return np.concatenate([pos_guess, vel_guess])
        else:
            return pos_guess


class IsodopplerIntersectionStrategy(InitialGuessStrategy):
    """
    Use intersection of isodoppler curves for targets with zero Doppler
    Good for nearly stationary targets
    """
    
    def __init__(self, altitude_assumption: Optional[float] = None,
                 estimate_velocity: bool = False,
                 is_3d: bool = True):
        self.altitude_assumption = altitude_assumption
        self.estimate_velocity = estimate_velocity
        self.is_3d = is_3d
    
    def compute(self, measurements: List[Measurement], 
                sensors: Dict[str, Sensor], 
                ioos: Dict[str, IoO],
                **kwargs) -> np.ndarray:
        """
        For near-zero Doppler, target is likely on the bistatic bisector
        """
        # Find measurement with smallest Doppler
        min_doppler_meas = min(measurements, key=lambda m: abs(m.doppler_hz))
        
        sensor = sensors[min_doppler_meas.sensor_id]
        ioo = ioos[sensor.ioo_id]
        
        # Start from bisector point
        sensor_pos = sensor.position.to_2d_array()
        ioo_pos = ioo.position.to_2d_array()
        
        # For zero Doppler, target is on perpendicular bisector
        midpoint = 0.5 * (sensor_pos + ioo_pos)
        baseline = ioo_pos - sensor_pos
        perpendicular = np.array([-baseline[1], baseline[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # Estimate distance using delay
        bistatic_range = min_doppler_meas.delay_km
        baseline_length = np.linalg.norm(baseline) / 1000.0  # to km
        
        # Approximate distance from midpoint (simplified)
        d = (bistatic_range**2 - baseline_length**2) / (4 * bistatic_range)
        
        initial_guess_2d = midpoint + d * perpendicular * 1000  # back to meters
        
        # Handle altitude
        if self.altitude_assumption is not None:
            initial_altitude = self.altitude_assumption
        else:
            all_altitudes = [s.position.up for s in sensors.values()]
            all_altitudes.extend([i.position.up for i in ioos.values()])
            initial_altitude = np.mean(all_altitudes)
        
        # Build state vector
        if self.is_3d:
            pos_guess = np.array([initial_guess_2d[0], initial_guess_2d[1], initial_altitude])
        else:
            pos_guess = initial_guess_2d
        
        if self.estimate_velocity:
            # Use Doppler to estimate initial velocity direction
            if abs(min_doppler_meas.doppler_hz) > 1.0:
                # Rough velocity magnitude estimate from Doppler
                vel_mag = abs(min_doppler_meas.doppler_hz) * 3e8 / kwargs.get('freq_hz', 100e6)
                vel_direction = perpendicular if min_doppler_meas.doppler_hz > 0 else -perpendicular
                vel_2d = vel_mag * vel_direction
                
                if self.is_3d:
                    vel_guess = np.array([vel_2d[0], vel_2d[1], 0])
                else:
                    vel_guess = vel_2d
            else:
                vel_guess = np.zeros(3 if self.is_3d else 2)
            
            return np.concatenate([pos_guess, vel_guess])
        else:
            return pos_guess


class GridSearchStrategy(InitialGuessStrategy):
    """
    Perform a coarse grid search to find the best initial guess
    More computationally expensive but can avoid local minima
    """
    
    def __init__(self, grid_bounds: Dict[str, tuple], grid_points: int = 10,
                 altitude_assumption: Optional[float] = None,
                 estimate_velocity: bool = False,
                 is_3d: bool = True):
        self.grid_bounds = grid_bounds
        self.grid_points = grid_points
        self.altitude_assumption = altitude_assumption
        self.estimate_velocity = estimate_velocity
        self.is_3d = is_3d
    
    def compute(self, measurements: List[Measurement], 
                sensors: Dict[str, Sensor], 
                ioos: Dict[str, IoO],
                **kwargs) -> np.ndarray:
        """
        Grid search over position space to minimize TDOA residuals
        """
        from scipy.optimize import least_squares
        
        # Create grid
        x_range = np.linspace(self.grid_bounds['x'][0], self.grid_bounds['x'][1], self.grid_points)
        y_range = np.linspace(self.grid_bounds['y'][0], self.grid_bounds['y'][1], self.grid_points)
        
        if self.is_3d and self.altitude_assumption is None:
            z_range = np.linspace(self.grid_bounds['z'][0], self.grid_bounds['z'][1], self.grid_points)
        else:
            z_range = [self.altitude_assumption if self.altitude_assumption else 0]
        
        # Evaluate TDOA residuals at each grid point
        min_cost = float('inf')
        best_pos = None
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    pos = np.array([x, y, z])
                    cost = 0
                    
                    # Compute TDOA residuals
                    for meas in measurements:
                        sensor = sensors[meas.sensor_id]
                        ioo = ioos[sensor.ioo_id]
                        
                        # Predicted delay
                        R_tx_target = np.linalg.norm(pos - ioo.position.to_array()) / 1000.0
                        R_target_rx = np.linalg.norm(pos - sensor.position.to_array()) / 1000.0
                        R_baseline = np.linalg.norm(sensor.position.to_array() - ioo.position.to_array()) / 1000.0
                        
                        predicted_delay = R_tx_target + R_target_rx - R_baseline
                        residual = meas.delay_km - predicted_delay
                        cost += residual**2
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_pos = pos
        
        # Build state vector
        if self.is_3d:
            pos_guess = best_pos
        else:
            pos_guess = best_pos[:2]
        
        if self.estimate_velocity:
            vel_guess = np.zeros(3 if self.is_3d else 2)
            return np.concatenate([pos_guess, vel_guess])
        else:
            return pos_guess


def create_initial_guess_strategy(strategy_name: str = 'ellipse_center', 
                                  **kwargs) -> InitialGuessStrategy:
    """
    Factory function to create initial guess strategies
    
    Args:
        strategy_name: Name of the strategy ('ellipse_center', 'weighted', 'isodoppler', 'grid')
        **kwargs: Additional arguments for the strategy
    
    Returns:
        InitialGuessStrategy instance
    """
    strategies = {
        'ellipse_center': EllipseCenterStrategy,
        'weighted': WeightedEllipseCenterStrategy,
        'isodoppler': IsodopplerIntersectionStrategy,
        'grid': GridSearchStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)

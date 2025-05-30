import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json


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
    

class BistaticRadarGeolocator:
    """Least squares geolocator for bistatic passive radar"""
    
    def __init__(self, sensors: List[Sensor], ioos: List[IoO], 
                 freq_hz: float = 100e6, altitude_assumption: Optional[float] = None):
        """
        Initialize geolocator
        
        Args:
            sensors: List of sensor configurations
            ioos: List of IoO configurations
            freq_hz: Carrier frequency in Hz (default 100 MHz)
            altitude_assumption: Fixed altitude if 2D solution desired
        """
        self.sensors = {s.id: s for s in sensors}
        self.ioos = {i.id: i for i in ioos}
        self.freq_hz = freq_hz
        self.c = 299792.458  # Speed of light in km/s
        self.altitude_assumption = altitude_assumption
        
    def compute_distances(self, target_pos: np.ndarray, sensor: Sensor, ioo: IoO) -> Tuple[float, float, float]:
        """
        Compute distances for bistatic geometry
        
        Returns:
            (R_tx_target, R_target_rx, R_baseline) in km
        """
        target_3d = np.array([target_pos[0], target_pos[1], 
                              target_pos[2] if len(target_pos) > 2 else self.altitude_assumption])
        
        sensor_pos = sensor.position.to_array()
        ioo_pos = ioo.position.to_array()
        
        R_tx_target = np.linalg.norm(target_3d - ioo_pos) / 1000.0  # m to km
        R_target_rx = np.linalg.norm(target_3d - sensor_pos) / 1000.0  # m to km
        R_baseline = np.linalg.norm(sensor_pos - ioo_pos) / 1000.0  # m to km
        
        return R_tx_target, R_target_rx, R_baseline
    
    def predict_tdoa(self, target_pos: np.ndarray, sensor: Sensor, ioo: IoO) -> float:
        """Predict bistatic delay (TDOA) in km"""
        R_tx_target, R_target_rx, R_baseline = self.compute_distances(target_pos, sensor, ioo)
        return R_tx_target + R_target_rx - R_baseline
    
    def predict_fdoa(self, target_pos: np.ndarray, target_vel: np.ndarray, 
                     sensor: Sensor, ioo: IoO) -> float:
        """
        Predict Doppler shift (FDOA) in Hz
        
        Args:
            target_pos: Target position [x, y] or [x, y, z]
            target_vel: Target velocity [vx, vy] or [vx, vy, vz]
        """
        target_3d = np.array([target_pos[0], target_pos[1], 
                              target_pos[2] if len(target_pos) > 2 else self.altitude_assumption])
        target_vel_3d = np.array([target_vel[0], target_vel[1], 
                                  target_vel[2] if len(target_vel) > 2 else 0])
        
        sensor_pos = sensor.position.to_array()
        ioo_pos = ioo.position.to_array()
        
        # Unit vectors
        tx_to_target = target_3d - ioo_pos
        target_to_rx = sensor_pos - target_3d
        
        R_tx_target = np.linalg.norm(tx_to_target)
        R_target_rx = np.linalg.norm(target_to_rx)
        
        if R_tx_target > 0 and R_target_rx > 0:
            u_tx = tx_to_target / R_tx_target
            u_rx = target_to_rx / R_target_rx
            
            # Bistatic Doppler
            doppler = -(self.freq_hz / self.c) * np.dot(target_vel_3d, u_tx + u_rx) / 1000.0  # m/s to km/s
            return doppler
        else:
            return 0.0
    
    def residual_function(self, state: np.ndarray, measurements: List[Measurement], 
                          use_fdoa: bool = True) -> np.ndarray:
        """
        Compute residuals for least squares
        
        Args:
            state: [x, y] or [x, y, z] for position only, 
                   [x, y, vx, vy] or [x, y, z, vx, vy, vz] for position + velocity
            measurements: List of measurements
            use_fdoa: Whether to include FDOA residuals
        """
        residuals = []
        
        # Determine state structure
        if len(state) == 2:  # 2D position only
            pos = state
            vel = None
        elif len(state) == 3:  # 3D position only
            pos = state
            vel = None
        elif len(state) == 4:  # 2D position + velocity
            pos = state[:2]
            vel = state[2:]
        elif len(state) == 6:  # 3D position + velocity
            pos = state[:3]
            vel = state[3:]
        else:
            raise ValueError(f"Invalid state dimension: {len(state)}")
        
        for meas in measurements:
            sensor = self.sensors[meas.sensor_id]
            ioo = self.ioos[sensor.ioo_id]
            
            # TDOA residual
            predicted_delay = self.predict_tdoa(pos, sensor, ioo)
            residuals.append(meas.delay_km - predicted_delay)
            
            # FDOA residual (if velocity is being estimated)
            if use_fdoa and vel is not None:
                predicted_doppler = self.predict_fdoa(pos, vel, sensor, ioo)
                residuals.append((meas.doppler_hz - predicted_doppler) / 100.0)  # Scale for numerical stability
        
        return np.array(residuals)
    
    def compute_initial_guess(self, measurements: List[Measurement]) -> np.ndarray:
        """Compute initial guess as midpoint between ellipse centers"""
        centers = []
        
        for meas in measurements:
            sensor = self.sensors[meas.sensor_id]
            ioo = self.ioos[sensor.ioo_id]
            
            sensor_2d = sensor.position.to_2d_array()
            ioo_2d = ioo.position.to_2d_array()
            
            center = 0.5 * (sensor_2d + ioo_2d)
            centers.append(center)
        
        # Average of all centers
        initial_guess_2d = np.mean(centers, axis=0)
        
        if self.altitude_assumption is not None:
            initial_altitude = self.altitude_assumption
        else:
            # Average altitude of all stations
            all_altitudes = [s.position.up for s in self.sensors.values()]
            all_altitudes.extend([i.position.up for i in self.ioos.values()])
            initial_altitude = np.mean(all_altitudes)
        
        return np.array([initial_guess_2d[0], initial_guess_2d[1], initial_altitude])
    
    def geolocate(self, measurements: List[Measurement], 
                  estimate_velocity: bool = False,
                  use_fdoa: bool = True,
                  initial_guess: Optional[np.ndarray] = None,
                  bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                  **kwargs) -> Dict:
        """
        Perform geolocation using least squares
        
        Args:
            measurements: List of measurements from sensors
            estimate_velocity: Whether to estimate target velocity
            use_fdoa: Whether to use FDOA measurements
            initial_guess: Initial state estimate (overrides default)
            bounds: (lower_bounds, upper_bounds) for constrained optimization
            **kwargs: Additional arguments for scipy.optimize.least_squares
            
        Returns:
            Dictionary with solution details
        """
        if not measurements:
            raise ValueError("No measurements provided")
        
        # Determine problem dimensionality
        is_3d = self.altitude_assumption is None
        
        # Compute initial guess
        if initial_guess is None:
            pos_guess = self.compute_initial_guess(measurements)
            
            if estimate_velocity:
                vel_guess = np.zeros(3 if is_3d else 2)
                if is_3d:
                    initial_guess = np.concatenate([pos_guess, vel_guess])
                else:
                    initial_guess = np.concatenate([pos_guess[:2], vel_guess])
            else:
                initial_guess = pos_guess if is_3d else pos_guess[:2]
        
        # Perform least squares optimization
        result = least_squares(
            self.residual_function,
            initial_guess,
            args=(measurements, use_fdoa and estimate_velocity),
            bounds=bounds if bounds else (-np.inf, np.inf),
            **kwargs
        )
        
        # Extract solution
        if estimate_velocity:
            if is_3d:
                position = result.x[:3]
                velocity = result.x[3:]
            else:
                position = np.array([result.x[0], result.x[1], self.altitude_assumption])
                velocity = np.array([result.x[2], result.x[3], 0])
        else:
            if is_3d:
                position = result.x
                velocity = None
            else:
                position = np.array([result.x[0], result.x[1], self.altitude_assumption])
                velocity = None
        
        # Compute covariance if possible
        try:
            J = result.jac
            cov = np.linalg.inv(J.T @ J) * np.var(result.fun)
            position_cov = cov[:3, :3] if is_3d else cov[:2, :2]
        except:
            cov = None
            position_cov = None
        
        return {
            'success': result.success,
            'position': position,
            'velocity': velocity,
            'covariance': cov,
            'position_covariance': position_cov,
            'residuals': result.fun,
            'cost': result.cost,
            'optimality': result.optimality,
            'nfev': result.nfev,
            'message': result.message,
            'initial_guess': initial_guess,
            'final_state': result.x
        }


# Example usage
def parse_measurement_json(json_data: str, sensor_id: str) -> Measurement:
    """Parse measurement from JSON format"""
    data = json.loads(json_data)
    return Measurement(
        timestamp=data['timestamp'],
        sensor_id=sensor_id,
        delay_km=data['delay'][0],
        doppler_hz=data['doppler'][0],
        snr=data['snr'][0]
    )


# Example configuration
if __name__ == "__main__":
    # Define sensors and IoOs
    sensors = [
        Sensor("sensor1", Position3D(0, 0, 100), "ioo1"),
        Sensor("sensor2", Position3D(50000, 0, 150), "ioo1")  # Shared IoO
    ]
    
    ioos = [
        IoO("ioo1", Position3D(25000, 40000, 200))
    ]
    
    # Create geolocator
    geolocator = BistaticRadarGeolocator(sensors, ioos, freq_hz=100e6)
    
    # Example measurements
    meas1_json = '{"timestamp":1748609918201,"delay":[58.44],"doppler":[-25.45],"snr":[5.52]}'
    meas2_json = '{"timestamp":1748609918201,"delay":[62.31],"doppler":[-18.72],"snr":[6.12]}'
    
    measurements = [
        parse_measurement_json(meas1_json, "sensor1"),
        parse_measurement_json(meas2_json, "sensor2")
    ]
    
    # Perform geolocation
    result = geolocator.geolocate(measurements, estimate_velocity=True)
    
    print(f"Success: {result['success']}")
    print(f"Position: {result['position']} m")
    if result['velocity'] is not None:
        print(f"Velocity: {result['velocity']} m/s")
    print(f"Final cost: {result['cost']}")

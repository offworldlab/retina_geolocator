"""
Unit tests for least squares geolocator
"""

import unittest
import numpy as np
import json
from unittest.mock import Mock, patch

from least_squares_geolocator import (
    BistaticRadarGeolocator,
    parse_measurement_json,
    create_measurement_from_dict
)
from initial_guess import (
    Position3D, Sensor, IoO, Measurement,
    EllipseCenterStrategy
)


class TestBistaticRadarGeolocator(unittest.TestCase):
    """Test main geolocator class"""
    
    def setUp(self):
        # Create test configuration
        self.sensors = [
            Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            Sensor("s2", Position3D(50000, 0, 150), "ioo1")
        ]
        self.ioos = [
            IoO("ioo1", Position3D(25000, 40000, 200))
        ]
        self.geolocator = BistaticRadarGeolocator(
            self.sensors, 
            self.ioos,
            freq_hz=100e6,
            altitude_assumption=None
        )
        
        # Create test measurements
        self.measurements = [
            Measurement(1234567890, "s1", 58.44, -25.45, 5.52),
            Measurement(1234567890, "s2", 62.31, -18.72, 6.12)
        ]
    
    def test_initialization(self):
        # Test proper initialization
        self.assertEqual(len(self.geolocator.sensors), 2)
        self.assertEqual(len(self.geolocator.ioos), 1)
        self.assertEqual(self.geolocator.freq_hz, 100e6)
        self.assertIsNone(self.geolocator.altitude_assumption)
        self.assertIsInstance(self.geolocator.initial_guess_strategy, EllipseCenterStrategy)
    
    def test_compute_distances(self):
        sensor = self.geolocator.sensors["s1"]
        ioo = self.geolocator.ioos["ioo1"]
        target_pos = np.array([20000, 30000, 5000])
        
        R_tx, R_rx, R_base = self.geolocator.compute_distances(target_pos, sensor, ioo)
        
        # Check that distances are positive
        self.assertGreater(R_tx, 0)
        self.assertGreater(R_rx, 0)
        self.assertGreater(R_base, 0)
        
        # Verify baseline distance
        expected_baseline = np.linalg.norm(
            sensor.position.to_array() - ioo.position.to_array()
        ) / 1000.0
        self.assertAlmostEqual(R_base, expected_baseline, places=6)
    
    def test_predict_tdoa(self):
        sensor = self.geolocator.sensors["s1"]
        ioo = self.geolocator.ioos["ioo1"]
        target_pos = np.array([20000, 30000, 5000])
        
        tdoa = self.geolocator.predict_tdoa(target_pos, sensor, ioo)
        
        # TDOA should be positive for targets outside baseline
        self.assertGreater(tdoa, 0)
        
        # Verify TDOA calculation
        R_tx, R_rx, R_base = self.geolocator.compute_distances(target_pos, sensor, ioo)
        expected_tdoa = R_tx + R_rx - R_base
        self.assertAlmostEqual(tdoa, expected_tdoa, places=6)
    
    def test_predict_fdoa(self):
        sensor = self.geolocator.sensors["s1"]
        ioo = self.geolocator.ioos["ioo1"]
        target_pos = np.array([20000, 30000, 5000])
        target_vel = np.array([100, 50, 0])  # Moving target
        
        fdoa = self.geolocator.predict_fdoa(target_pos, target_vel, sensor, ioo)
        
        # FDOA should be non-zero for moving target
        self.assertNotEqual(fdoa, 0)
        
        # Test zero velocity
        fdoa_zero = self.geolocator.predict_fdoa(target_pos, np.zeros(3), sensor, ioo)
        self.assertEqual(fdoa_zero, 0)
    
    def test_residual_function_position_only(self):
        # Test 3D position only
        state = np.array([20000, 30000, 5000])
        residuals = self.geolocator.residual_function(state, self.measurements, use_fdoa=False)
        
        # Should have one residual per measurement
        self.assertEqual(len(residuals), len(self.measurements))
    
    def test_residual_function_with_velocity(self):
        # Test 3D position + velocity
        state = np.array([20000, 30000, 5000, 100, 50, 0])
        residuals = self.geolocator.residual_function(state, self.measurements, use_fdoa=True)
        
        # Should have TDOA + FDOA residuals
        self.assertEqual(len(residuals), 2 * len(self.measurements))
    
    def test_residual_function_invalid_state(self):
        # Test invalid state dimension
        state = np.array([1, 2, 3, 4, 5])  # Invalid dimension
        
        with self.assertRaises(ValueError):
            self.geolocator.residual_function(state, self.measurements)
    
    def test_residual_function_with_weights(self):
        state = np.array([20000, 30000, 5000])
        weights = np.array([1.0, 2.0])  # Different weights for measurements
        
        unweighted = self.geolocator.residual_function(state, self.measurements, use_fdoa=False)
        weighted = self.geolocator.residual_function(state, self.measurements, use_fdoa=False, 
                                                      measurement_weights=weights)
        
        # Weighted residuals should be different
        self.assertFalse(np.array_equal(unweighted, weighted))
        
        # Check weighting is applied correctly
        expected = unweighted * weights
        np.testing.assert_array_almost_equal(weighted, expected)


class TestGeolocationMethods(unittest.TestCase):
    """Test geolocation with different configurations"""
    
    def setUp(self):
        # Simple 2-sensor configuration
        self.sensors = [
            Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            Sensor("s2", Position3D(50000, 0, 100), "ioo1")
        ]
        self.ioos = [
            IoO("ioo1", Position3D(25000, 0, 100))
        ]
        
        # Known target position for synthetic measurements
        self.true_position = np.array([30000, 20000, 5000])
        self.true_velocity = np.array([100, -50, 0])
    
    def create_synthetic_measurements(self, position, velocity=None, add_noise=False):
        """Create synthetic measurements from known position/velocity"""
        measurements = []
        
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        for sensor in self.sensors:
            ioo = self.ioos[0]
            
            # Calculate true TDOA
            tdoa = geolocator.predict_tdoa(position, sensor, ioo)
            
            # Calculate true FDOA if velocity provided
            if velocity is not None:
                fdoa = geolocator.predict_fdoa(position, velocity, sensor, ioo)
            else:
                fdoa = 0.0
            
            # Add noise if requested
            if add_noise:
                tdoa += np.random.normal(0, 0.01)  # 10m std dev
                fdoa += np.random.normal(0, 0.1)   # 0.1 Hz std dev
            
            measurements.append(
                Measurement(1234567890, sensor.id, tdoa, fdoa, 10.0)
            )
        
        return measurements
    
    def test_2d_position_only(self):
        # Test 2D geolocation with fixed altitude
        geolocator = BistaticRadarGeolocator(
            self.sensors, 
            self.ioos,
            altitude_assumption=5000  # Fix altitude at 5km
        )
        
        measurements = self.create_synthetic_measurements(self.true_position)
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        self.assertTrue(result['success'])
        self.assertIsNone(result['velocity'])
        
        # Check position accuracy (should be exact for noiseless case)
        estimated_pos = result['position']
        self.assertAlmostEqual(estimated_pos[0], self.true_position[0], delta=100)
        self.assertAlmostEqual(estimated_pos[1], self.true_position[1], delta=100)
        self.assertEqual(estimated_pos[2], 5000)  # Fixed altitude
    
    def test_3d_position_only(self):
        # Test 3D geolocation
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        measurements = self.create_synthetic_measurements(self.true_position)
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        self.assertTrue(result['success'])
        self.assertIsNone(result['velocity'])
        
        # With only 2 sensors, 3D position is underdetermined
        # But should still converge to something reasonable
        self.assertEqual(len(result['position']), 3)
    
    def test_position_and_velocity(self):
        # Test position + velocity estimation
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        measurements = self.create_synthetic_measurements(
            self.true_position, 
            self.true_velocity
        )
        result = geolocator.geolocate(measurements, estimate_velocity=True)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['velocity'])
        self.assertEqual(len(result['velocity']), 3)
    
    def test_with_noise(self):
        # Test robustness to measurement noise
        geolocator = BistaticRadarGeolocator(
            self.sensors, 
            self.ioos,
            altitude_assumption=self.true_position[2]
        )
        
        measurements = self.create_synthetic_measurements(
            self.true_position,
            add_noise=True
        )
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        self.assertTrue(result['success'])
        
        # Should still be reasonably close despite noise
        estimated_pos = result['position']
        error_distance = np.linalg.norm(estimated_pos - self.true_position)
        self.assertLess(error_distance, 1000)  # Within 1km
    
    def test_constrained_optimization(self):
        # Test with bounds
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        measurements = self.create_synthetic_measurements(self.true_position)
        
        # Set bounds around true position
        bounds = (
            np.array([25000, 15000, 0]),
            np.array([35000, 25000, 10000])
        )
        
        result = geolocator.geolocate(
            measurements, 
            estimate_velocity=False,
            bounds=bounds
        )
        
        self.assertTrue(result['success'])
        
        # Check solution is within bounds
        pos = result['position']
        np.testing.assert_array_less(bounds[0] - 1e-6, pos)
        np.testing.assert_array_less(pos, bounds[1] + 1e-6)
    
    def test_covariance_computation(self):
        # Test that covariance is computed
        geolocator = BistaticRadarGeolocator(
            self.sensors, 
            self.ioos,
            altitude_assumption=5000
        )
        
        measurements = self.create_synthetic_measurements(self.true_position)
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        self.assertIsNotNone(result['covariance'])
        self.assertIsNotNone(result['position_covariance'])
        self.assertIsNotNone(result['error_ellipse'])
        
        # Check error ellipse parameters
        ellipse = result['error_ellipse']
        self.assertIn('semi_major_m', ellipse)
        self.assertIn('semi_minor_m', ellipse)
        self.assertIn('angle_deg', ellipse)
        self.assertGreater(ellipse['semi_major_m'], 0)
        self.assertGreater(ellipse['semi_minor_m'], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_parse_measurement_json(self):
        json_data = '{"timestamp":1748609918201,"delay":[58.44],"doppler":[-25.45],"snr":[5.52]}'
        sensor_id = "s1"
        
        meas = parse_measurement_json(json_data, sensor_id)
        
        self.assertEqual(meas.timestamp, 1748609918201)
        self.assertEqual(meas.sensor_id, "s1")
        self.assertEqual(meas.delay_km, 58.44)
        self.assertEqual(meas.doppler_hz, -25.45)
        self.assertEqual(meas.snr, 5.52)
    
    def test_create_measurement_from_dict(self):
        data = {
            'timestamp': 1234567890,
            'delay': 50.5,
            'doppler': -20.3,
            'snr': 10.2
        }
        
        meas = create_measurement_from_dict(data, "s1")
        
        self.assertEqual(meas.timestamp, 1234567890)
        self.assertEqual(meas.sensor_id, "s1")
        self.assertEqual(meas.delay_km, 50.5)


class TestStrategyIntegration(unittest.TestCase):
    """Test integration with different initial guess strategies"""
    
    def setUp(self):
        self.sensors = [
            Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            Sensor("s2", Position3D(50000, 0, 100), "ioo1")
        ]
        self.ioos = [
            IoO("ioo1", Position3D(25000, 0, 100))
        ]
        self.measurements = [
            Measurement(1234567890, "s1", 58.44, -25.45, 5.52),
            Measurement(1234567890, "s2", 62.31, -18.72, 6.12)
        ]
    
    def test_set_strategy_by_name(self):
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        # Set strategy by name
        geolocator.set_initial_guess_strategy('weighted', altitude_assumption=1000)
        
        # Verify it was set
        from initial_guess import WeightedEllipseCenterStrategy
        self.assertIsInstance(geolocator.initial_guess_strategy, WeightedEllipseCenterStrategy)
    
    def test_set_strategy_instance(self):
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        # Create custom strategy
        from initial_guess import IsodopplerIntersectionStrategy
        strategy = IsodopplerIntersectionStrategy(altitude_assumption=5000)
        
        # Set strategy instance
        geolocator.set_initial_guess_strategy(strategy)
        
        self.assertEqual(geolocator.initial_guess_strategy, strategy)
    
    def test_custom_initial_guess(self):
        geolocator = BistaticRadarGeolocator(self.sensors, self.ioos)
        
        # Provide custom initial guess directly
        custom_guess = np.array([30000, 20000, 5000])
        
        result = geolocator.geolocate(
            self.measurements,
            estimate_velocity=False,
            initial_guess=custom_guess
        )
        
        # Verify custom guess was used
        np.testing.assert_array_equal(result['initial_guess'], custom_guess)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_no_measurements(self):
        sensors = [Sensor("s1", Position3D(0, 0, 100), "ioo1")]
        ioos = [IoO("ioo1", Position3D(25000, 0, 100))]
        geolocator = BistaticRadarGeolocator(sensors, ioos)
        
        with self.assertRaises(ValueError):
            geolocator.geolocate([])
    
    def test_single_measurement(self):
        sensors = [Sensor("s1", Position3D(0, 0, 100), "ioo1")]
        ioos = [IoO("ioo1", Position3D(25000, 0, 100))]
        geolocator = BistaticRadarGeolocator(sensors, ioos)
        
        measurements = [Measurement(1234567890, "s1", 50.0, -20.0, 10.0)]
        
        # Should still work but solution is highly underdetermined
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        # May or may not succeed depending on initial guess
        self.assertIn('success', result)
        self.assertIn('position', result)
    
    def test_target_at_sensor(self):
        # Test when target is at sensor location
        sensors = [
            Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            Sensor("s2", Position3D(50000, 0, 100), "ioo1")
        ]
        ioos = [IoO("ioo1", Position3D(25000, 0, 100))]
        geolocator = BistaticRadarGeolocator(sensors, ioos)
        
        # Create measurements as if target is at sensor s1
        sensor_pos = sensors[0].position.to_array()
        measurements = []
        
        for sensor in sensors:
            ioo = ioos[0]
            tdoa = geolocator.predict_tdoa(sensor_pos, sensor, ioo)
            measurements.append(
                Measurement(1234567890, sensor.id, tdoa, 0.0, 10.0)
            )
        
        result = geolocator.geolocate(measurements, estimate_velocity=False)
        
        if result['success']:
            # Should converge close to sensor position
            error = np.linalg.norm(result['position'] - sensor_pos)
            self.assertLess(error, 1000)  # Within 1km


if __name__ == '__main__':
    unittest.main()

"""
Unit tests for initial guess strategies
"""

import unittest
import numpy as np
from initial_guess import (
    Position3D, Sensor, IoO, Measurement,
    EllipseCenterStrategy, WeightedEllipseCenterStrategy,
    IsodopplerIntersectionStrategy, GridSearchStrategy,
    create_initial_guess_strategy
)


class TestDataClasses(unittest.TestCase):
    """Test basic data classes"""
    
    def test_position3d(self):
        pos = Position3D(1000, 2000, 300)
        np.testing.assert_array_equal(pos.to_array(), np.array([1000, 2000, 300]))
        np.testing.assert_array_equal(pos.to_2d_array(), np.array([1000, 2000]))
    
    def test_sensor(self):
        pos = Position3D(1000, 2000, 300)
        sensor = Sensor("s1", pos, "ioo1")
        self.assertEqual(sensor.id, "s1")
        self.assertEqual(sensor.ioo_id, "ioo1")
        self.assertEqual(sensor.position.east, 1000)
    
    def test_measurement(self):
        meas = Measurement(
            timestamp=1234567890,
            sensor_id="s1",
            delay_km=50.5,
            doppler_hz=-25.3,
            snr=10.2
        )
        self.assertEqual(meas.timestamp, 1234567890)
        self.assertEqual(meas.delay_km, 50.5)


class TestEllipseCenterStrategy(unittest.TestCase):
    """Test ellipse center initial guess strategy"""
    
    def setUp(self):
        # Create test sensors and IoOs
        self.sensors = {
            "s1": Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            "s2": Sensor("s2", Position3D(50000, 0, 150), "ioo1")
        }
        self.ioos = {
            "ioo1": IoO("ioo1", Position3D(25000, 40000, 200))
        }
        self.measurements = [
            Measurement(1234567890, "s1", 50.0, -20.0, 10.0),
            Measurement(1234567890, "s2", 55.0, -15.0, 12.0)
        ]
    
    def test_2d_position_only(self):
        strategy = EllipseCenterStrategy(altitude_assumption=1000, 
                                         estimate_velocity=False, 
                                         is_3d=False)
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Should return 2D position
        self.assertEqual(len(guess), 2)
        
        # Check that it's the average of ellipse centers
        center1 = 0.5 * (np.array([0, 0]) + np.array([25000, 40000]))
        center2 = 0.5 * (np.array([50000, 0]) + np.array([25000, 40000]))
        expected = 0.5 * (center1 + center2)
        np.testing.assert_array_almost_equal(guess, expected)
    
    def test_3d_position_only(self):
        strategy = EllipseCenterStrategy(altitude_assumption=None,
                                         estimate_velocity=False,
                                         is_3d=True)
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Should return 3D position
        self.assertEqual(len(guess), 3)
        
        # Check altitude is average of all stations
        expected_alt = (100 + 150 + 200) / 3
        self.assertAlmostEqual(guess[2], expected_alt)
    
    def test_2d_with_velocity(self):
        strategy = EllipseCenterStrategy(altitude_assumption=1000,
                                         estimate_velocity=True,
                                         is_3d=False)
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Should return [x, y, vx, vy]
        self.assertEqual(len(guess), 4)
        
        # Velocity should be initialized to zero
        np.testing.assert_array_equal(guess[2:], np.zeros(2))
    
    def test_3d_with_velocity(self):
        strategy = EllipseCenterStrategy(altitude_assumption=None,
                                         estimate_velocity=True,
                                         is_3d=True)
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Should return [x, y, z, vx, vy, vz]
        self.assertEqual(len(guess), 6)
        
        # Velocity should be initialized to zero
        np.testing.assert_array_equal(guess[3:], np.zeros(3))
    
    def test_single_measurement(self):
        strategy = EllipseCenterStrategy()
        single_meas = [self.measurements[0]]
        guess = strategy.compute(single_meas, self.sensors, self.ioos)
        
        # Should still work with single measurement
        self.assertEqual(len(guess), 3)  # 3D position by default


class TestWeightedEllipseCenterStrategy(unittest.TestCase):
    """Test weighted ellipse center strategy"""
    
    def setUp(self):
        self.sensors = {
            "s1": Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            "s2": Sensor("s2", Position3D(50000, 0, 150), "ioo1")
        }
        self.ioos = {
            "ioo1": IoO("ioo1", Position3D(25000, 40000, 200))
        }
    
    def test_snr_weighting(self):
        # Create measurements with very different SNRs
        measurements = [
            Measurement(1234567890, "s1", 50.0, -20.0, 1.0),   # Low SNR
            Measurement(1234567890, "s2", 55.0, -15.0, 100.0)  # High SNR
        ]
        
        strategy = WeightedEllipseCenterStrategy(altitude_assumption=1000,
                                                 estimate_velocity=False,
                                                 is_3d=False)
        guess = strategy.compute(measurements, self.sensors, self.ioos)
        
        # Result should be closer to sensor 2's ellipse center
        center1 = 0.5 * (np.array([0, 0]) + np.array([25000, 40000]))
        center2 = 0.5 * (np.array([50000, 0]) + np.array([25000, 40000]))
        
        # Check that guess is closer to center2 than center1
        dist_to_center1 = np.linalg.norm(guess - center1)
        dist_to_center2 = np.linalg.norm(guess - center2)
        self.assertLess(dist_to_center2, dist_to_center1)
    
    def test_equal_snr(self):
        # With equal SNR, should match unweighted strategy
        measurements = [
            Measurement(1234567890, "s1", 50.0, -20.0, 10.0),
            Measurement(1234567890, "s2", 55.0, -15.0, 10.0)
        ]
        
        weighted_strategy = WeightedEllipseCenterStrategy(altitude_assumption=1000,
                                                          estimate_velocity=False,
                                                          is_3d=False)
        unweighted_strategy = EllipseCenterStrategy(altitude_assumption=1000,
                                                    estimate_velocity=False,
                                                    is_3d=False)
        
        weighted_guess = weighted_strategy.compute(measurements, self.sensors, self.ioos)
        unweighted_guess = unweighted_strategy.compute(measurements, self.sensors, self.ioos)
        
        np.testing.assert_array_almost_equal(weighted_guess, unweighted_guess)


class TestIsodopplerIntersectionStrategy(unittest.TestCase):
    """Test isodoppler intersection strategy"""
    
    def setUp(self):
        self.sensors = {
            "s1": Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            "s2": Sensor("s2", Position3D(50000, 0, 150), "ioo1")
        }
        self.ioos = {
            "ioo1": IoO("ioo1", Position3D(25000, 0, 200))  # Aligned for easier testing
        }
    
    def test_zero_doppler_target(self):
        # Target with zero Doppler should be on perpendicular bisector
        measurements = [
            Measurement(1234567890, "s1", 50.0, 0.0, 10.0),  # Zero Doppler
            Measurement(1234567890, "s2", 55.0, 0.1, 12.0)   # Near zero
        ]
        
        strategy = IsodopplerIntersectionStrategy(altitude_assumption=1000,
                                                  estimate_velocity=False,
                                                  is_3d=False)
        guess = strategy.compute(measurements, self.sensors, self.ioos, freq_hz=100e6)
        
        # For zero Doppler, target should be roughly perpendicular to baseline
        # With sensors at x=0 and x=50000, IoO at x=25000, all at y=0,
        # the perpendicular bisector passes through x=12500 or x=37500
        self.assertTrue(guess[1] != 0)  # Should have non-zero y coordinate
    
    def test_velocity_estimation(self):
        # Non-zero Doppler should estimate initial velocity
        measurements = [
            Measurement(1234567890, "s1", 50.0, -50.0, 10.0),  # Significant Doppler
            Measurement(1234567890, "s2", 55.0, -45.0, 12.0)
        ]
        
        strategy = IsodopplerIntersectionStrategy(altitude_assumption=1000,
                                                  estimate_velocity=True,
                                                  is_3d=False)
        guess = strategy.compute(measurements, self.sensors, self.ioos, freq_hz=100e6)
        
        # Should return position + velocity
        self.assertEqual(len(guess), 4)
        
        # Velocity should be non-zero for non-zero Doppler
        velocity_magnitude = np.linalg.norm(guess[2:])
        self.assertGreater(velocity_magnitude, 0)


class TestGridSearchStrategy(unittest.TestCase):
    """Test grid search strategy"""
    
    def setUp(self):
        self.sensors = {
            "s1": Sensor("s1", Position3D(0, 0, 100), "ioo1"),
            "s2": Sensor("s2", Position3D(50000, 0, 150), "ioo1")
        }
        self.ioos = {
            "ioo1": IoO("ioo1", Position3D(25000, 40000, 200))
        }
        self.measurements = [
            Measurement(1234567890, "s1", 58.44, -20.0, 10.0),
            Measurement(1234567890, "s2", 62.31, -15.0, 12.0)
        ]
    
    def test_grid_search_bounds(self):
        grid_bounds = {
            'x': (0, 50000),
            'y': (0, 50000),
            'z': (0, 10000)
        }
        
        strategy = GridSearchStrategy(grid_bounds=grid_bounds,
                                      grid_points=5,
                                      altitude_assumption=None,
                                      estimate_velocity=False,
                                      is_3d=True)
        
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Result should be within bounds
        self.assertGreaterEqual(guess[0], grid_bounds['x'][0])
        self.assertLessEqual(guess[0], grid_bounds['x'][1])
        self.assertGreaterEqual(guess[1], grid_bounds['y'][0])
        self.assertLessEqual(guess[1], grid_bounds['y'][1])
        self.assertGreaterEqual(guess[2], grid_bounds['z'][0])
        self.assertLessEqual(guess[2], grid_bounds['z'][1])
    
    def test_grid_search_2d(self):
        grid_bounds = {
            'x': (0, 50000),
            'y': (0, 50000)
        }
        
        strategy = GridSearchStrategy(grid_bounds=grid_bounds,
                                      grid_points=10,
                                      altitude_assumption=1000,
                                      estimate_velocity=False,
                                      is_3d=False)
        
        guess = strategy.compute(self.measurements, self.sensors, self.ioos)
        
        # Should return 2D position
        self.assertEqual(len(guess), 2)


class TestFactoryFunction(unittest.TestCase):
    """Test strategy factory function"""
    
    def test_create_strategies(self):
        # Test creating each strategy type
        strategies = ['ellipse_center', 'weighted', 'isodoppler', 'grid']
        
        for name in strategies[:3]:  # First three don't need special params
            strategy = create_initial_guess_strategy(name)
            self.assertIsNotNone(strategy)
        
        # Grid search needs bounds
        grid_strategy = create_initial_guess_strategy(
            'grid',
            grid_bounds={'x': (0, 100000), 'y': (0, 100000), 'z': (0, 20000)}
        )
        self.assertIsNotNone(grid_strategy)
    
    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            create_initial_guess_strategy('nonexistent_strategy')
    
    def test_strategy_with_kwargs(self):
        strategy = create_initial_guess_strategy(
            'ellipse_center',
            altitude_assumption=5000,
            estimate_velocity=True
        )
        self.assertEqual(strategy.altitude_assumption, 5000)
        self.assertEqual(strategy.estimate_velocity, True)


if __name__ == '__main__':
    unittest.main()

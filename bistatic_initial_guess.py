import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import warnings
from scipy.optimize import fsolve
from geopy.distance import distance
from geopy import Point

# Note: Install required packages with:
# pip install numpy scipy geopy


@dataclass
class GeoPosition:
    """Geographic position with latitude, longitude, and optional altitude."""
    latitude: float
    longitude: float
    altitude: float = 0.0  # meters above sea level
    
    def to_enu(self, origin: 'GeoPosition') -> 'ENUPosition':
        """Convert to ENU coordinates relative to origin."""
        # Using geopy for accurate distance calculations
        north_dist = distance((origin.latitude, origin.longitude), 
                            (self.latitude, origin.longitude)).meters
        east_dist = distance((origin.latitude, origin.longitude), 
                           (origin.latitude, self.longitude)).meters
        
        # Adjust signs based on hemisphere
        if self.latitude < origin.latitude:
            north_dist = -north_dist
        if self.longitude < origin.longitude:
            east_dist = -east_dist
            
        up_dist = self.altitude - origin.altitude
        
        return ENUPosition(east_dist, north_dist, up_dist)


@dataclass
class ENUPosition:
    """East-North-Up local coordinate system position."""
    east: float  # meters
    north: float  # meters
    up: float = 0.0  # meters
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.east, self.north, self.up])
    
    def to_2d(self) -> np.ndarray:
        """Get 2D projection (east, north)."""
        return np.array([self.east, self.north])
    
    def to_geo(self, origin: GeoPosition) -> GeoPosition:
        """Convert back to geographic coordinates."""
        # Approximate conversion (accurate for small distances)
        lat_offset = self.north / 111111.0  # meters to degrees
        lon_offset = self.east / (111111.0 * np.cos(np.radians(origin.latitude)))
        
        return GeoPosition(
            latitude=origin.latitude + lat_offset,
            longitude=origin.longitude + lon_offset,
            altitude=origin.altitude + self.up
        )


@dataclass
class BistaticMeasurement:
    """Bistatic radar measurement from a single sensor."""
    sensor_pos: GeoPosition
    ioo_pos: GeoPosition
    bistatic_range: float  # meters
    doppler: Optional[float] = None  # Hz
    snr: Optional[float] = None  # dB
    timestamp: Optional[float] = None


@dataclass
class EllipseParameters:
    """Parameters defining an ellipse in 2D."""
    center: np.ndarray  # (x, y) center position
    semi_major: float  # semi-major axis length
    semi_minor: float  # semi-minor axis length
    rotation: float  # rotation angle in radians
    foci: Tuple[np.ndarray, np.ndarray]  # focal points


@dataclass
class TargetSolution:
    """A potential target position with quality metrics."""
    position: GeoPosition
    quality: float  # 0-1, where 1 is perfect intersection
    gdop: Optional[float] = None
    crossing_angle: Optional[float] = None


class BistaticGeolocation:
    """Algebraic solver for bistatic radar geolocation."""
    
    def __init__(self, altitude_assumption: Optional[float] = None,
                 range_uncertainty: float = 50.0,
                 complex_tolerance_factor: float = 0.1):
        """
        Initialize the geolocation solver.
        
        Parameters:
        -----------
        altitude_assumption : float, optional
            If provided, assumes targets at this altitude (meters).
            If None, will attempt to estimate altitude.
        range_uncertainty : float
            Expected measurement uncertainty in bistatic range (meters).
            Default: 50m
        complex_tolerance_factor : float
            Factor to multiply range_uncertainty for accepting complex roots.
            Default: 0.1 (accept imaginary parts up to 10% of range uncertainty)
        """
        self.altitude_assumption = altitude_assumption
        self.range_uncertainty = range_uncertainty
        self.complex_tolerance_factor = complex_tolerance_factor
        self.origin = None  # Will be set based on sensor positions
        
    @dataclass
    class TargetSolution:
        """A potential target position with quality metrics."""
        position: GeoPosition
        quality: float  # 0-1, where 1 is perfect intersection
        gdop: Optional[float] = None
        crossing_angle: Optional[float] = None
        
    def find_initial_positions(self, 
                             measurement1: BistaticMeasurement,
                             measurement2: BistaticMeasurement) -> List[TargetSolution]:
        """
        Find initial target position estimates from two bistatic measurements.
        
        Returns:
        --------
        List of TargetSolution objects with positions and quality metrics.
        """
        # Set coordinate system origin (midpoint between sensors)
        self._set_origin(measurement1.sensor_pos, measurement2.sensor_pos)
        
        # Convert to ENU coordinates
        sensor1_enu = measurement1.sensor_pos.to_enu(self.origin)
        sensor2_enu = measurement2.sensor_pos.to_enu(self.origin)
        ioo1_enu = measurement1.ioo_pos.to_enu(self.origin)
        ioo2_enu = measurement2.ioo_pos.to_enu(self.origin)
        
        solutions = []
        
        if self.altitude_assumption is not None:
            # 2D problem with assumed altitude
            intersections = self._solve_2d_intersection(
                sensor1_enu, sensor2_enu, ioo1_enu, ioo2_enu,
                measurement1.bistatic_range, measurement2.bistatic_range
            )
            
            # Convert to geographic coordinates with quality metrics
            for pos_2d, quality in intersections:
                pos_3d = ENUPosition(pos_2d[0], pos_2d[1], self.altitude_assumption)
                geo_pos = pos_3d.to_geo(self.origin)
                
                # Calculate additional metrics
                metrics = self.calculate_confidence_metrics(measurement1, measurement2, geo_pos)
                
                solution = self.TargetSolution(
                    position=geo_pos,
                    quality=quality,
                    gdop=metrics['gdop'],
                    crossing_angle=metrics['crossing_angle']
                )
                solutions.append(solution)
                
        else:
            # 3D problem - find intersection curve and sample altitudes
            solutions = self._solve_3d_intersection(
                sensor1_enu, sensor2_enu, ioo1_enu, ioo2_enu,
                measurement1, measurement2
            )
        
        # Sort by quality
        solutions.sort(key=lambda s: s.quality, reverse=True)
        
        return solutions
    
    def _set_origin(self, pos1: GeoPosition, pos2: GeoPosition):
        """Set the origin for the ENU coordinate system."""
        self.origin = GeoPosition(
            latitude=(pos1.latitude + pos2.latitude) / 2,
            longitude=(pos1.longitude + pos2.longitude) / 2,
            altitude=(pos1.altitude + pos2.altitude) / 2
        )
    
    def _solve_2d_intersection(self, 
                              sensor1: ENUPosition, sensor2: ENUPosition,
                              ioo1: ENUPosition, ioo2: ENUPosition,
                              range1: float, range2: float) -> List[Tuple[np.ndarray, float]]:
        """
        Solve ellipse intersection in 2D using algebraic methods.
        
        Returns:
        --------
        List of tuples (position, quality) where position is [x, y] array
        and quality is 0-1.
        """
        
        # Calculate ellipse parameters
        ellipse1 = self._calculate_ellipse_params(
            sensor1.to_2d(), ioo1.to_2d(), range1
        )
        ellipse2 = self._calculate_ellipse_params(
            sensor2.to_2d(), ioo2.to_2d(), range2
        )
        
        # Find intersections using algebraic method with quality tracking
        intersections = self._algebraic_ellipse_intersection(ellipse1, ellipse2)
        
        return intersections
    
    def _calculate_ellipse_params(self, sensor: np.ndarray, ioo: np.ndarray, 
                                 bistatic_range: float) -> EllipseParameters:
        """Calculate ellipse parameters from sensor, IoO, and bistatic range."""
        
        # Baseline distance between sensor and IoO
        baseline = np.linalg.norm(ioo - sensor)
        
        # For a valid ellipse: bistatic_range > baseline
        if bistatic_range <= baseline:
            raise ValueError(f"Invalid geometry: bistatic range ({bistatic_range}m) "
                           f"must be greater than baseline ({baseline}m)")
        
        # Semi-major axis: sum of distances to foci = bistatic_range
        a = bistatic_range / 2.0
        
        # Distance between foci
        c = baseline / 2.0
        
        # Semi-minor axis
        b = np.sqrt(a**2 - c**2)
        
        # Center (midpoint between foci)
        center = (sensor + ioo) / 2.0
        
        # Rotation angle
        focal_vector = ioo - sensor
        rotation = np.arctan2(focal_vector[1], focal_vector[0])
        
        return EllipseParameters(
            center=center,
            semi_major=a,
            semi_minor=b,
            rotation=rotation,
            foci=(sensor, ioo)
        )
    
    def _algebraic_ellipse_intersection(self, ellipse1: EllipseParameters, 
                                      ellipse2: EllipseParameters) -> List[Tuple[np.ndarray, float]]:
        """
        Find intersection points of two ellipses using algebraic methods.
        
        Returns:
        --------
        List of tuples (position, quality) where position is [x, y] and
        quality is 0-1 (1 = perfect intersection, <1 = near miss).
        """
        
        # Convert ellipses to implicit form: Ax² + Bxy + Cy² + Dx + Ey + F = 0
        coeffs1 = self._ellipse_to_implicit(ellipse1)
        coeffs2 = self._ellipse_to_implicit(ellipse2)
        
        # Use Sylvester resultant to eliminate y
        x_polynomial = self._compute_resultant_x(coeffs1, coeffs2)
        
        # Find roots including those with small imaginary parts
        x_roots_with_quality = self._find_real_roots(x_polynomial)
        
        # For each x, find corresponding y values
        intersection_points = []
        for x_real, x_imag_mag in x_roots_with_quality:
            y_values = self._solve_for_y_with_quality(x_real, coeffs1, coeffs2)
            
            for y_real, y_imag_mag in y_values:
                # Calculate overall solution quality
                total_imag = np.sqrt(x_imag_mag**2 + y_imag_mag**2)
                quality = np.exp(-total_imag / self.range_uncertainty)  # Exponential decay
                
                # Verify the point lies approximately on both ellipses
                if self._verify_intersection(x_real, y_real, coeffs1, coeffs2):
                    intersection_points.append((np.array([x_real, y_real]), quality))
                elif quality > 0.5:  # Accept near-misses with good quality
                    intersection_points.append((np.array([x_real, y_real]), quality * 0.8))
        
        return intersection_points
    
    def _ellipse_to_implicit(self, ellipse: EllipseParameters) -> Dict[str, float]:
        """
        Convert ellipse parameters to implicit form coefficients.
        
        Returns coefficients for: Ax² + Bxy + Cy² + Dx + Ey + F = 0
        """
        h, k = ellipse.center
        a, b = ellipse.semi_major, ellipse.semi_minor
        theta = ellipse.rotation
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rotation matrix elements
        cos2_t = cos_t**2
        sin2_t = sin_t**2
        sin_cos_t = sin_t * cos_t
        
        # Coefficients
        A = cos2_t/a**2 + sin2_t/b**2
        B = 2*sin_cos_t*(1/a**2 - 1/b**2)
        C = sin2_t/a**2 + cos2_t/b**2
        D = -2*h*A - k*B
        E = -h*B - 2*k*C
        F = h**2*A + h*k*B + k**2*C - 1
        
        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F}
    
    def _compute_resultant_x(self, coeffs1: Dict[str, float], 
                           coeffs2: Dict[str, float]) -> np.ndarray:
        """
        Compute the resultant polynomial in x by eliminating y.
        
        This uses the Sylvester matrix method to eliminate y from the system
        of two conic equations.
        """
        # Extract coefficients
        A1, B1, C1 = coeffs1['A'], coeffs1['B'], coeffs1['C']
        D1, E1, F1 = coeffs1['D'], coeffs1['E'], coeffs1['F']
        A2, B2, C2 = coeffs2['A'], coeffs2['B'], coeffs2['C']
        D2, E2, F2 = coeffs2['D'], coeffs2['E'], coeffs2['F']
        
        # Build the resultant polynomial coefficients
        # This is a 4th degree polynomial in x
        p4 = (B1*E2 - B2*E1)**2 - 4*(A1*E2**2 - 2*A2*E1*E2 + A2*E1**2)*C1 + 4*A1*C2*E1**2
        
        p3 = 2*(B1*E2 - B2*E1)*(B1*F2 - B2*F1) - 4*(A1*E2**2 - 2*A2*E1*E2 + A2*E1**2)*D1 + \
             8*A1*C2*D2*E1 - 4*A2*C1*D1*E2 + 4*A2*C2*D1*E1
        
        p2 = (B1*F2 - B2*F1)**2 - 4*(A1*F2 - A2*F1)*(C1*F2 - C2*F1) + \
             4*A1*C2*D2**2 - 4*A2*C1*D1**2 + 8*A1*C2*F1 - 8*A2*C1*F2
        
        p1 = 4*(B1*F2 - B2*F1)*(D1*F2 - D2*F1) - 4*(A1*F2 - A2*F1)*(E1*F2 - E2*F1)
        
        p0 = 4*(D1*F2 - D2*F1)**2 - 4*(A1*F2 - A2*F1)*(C1*F2 - C2*F1)
        
        return np.array([p4, p3, p2, p1, p0])
    
    def _find_real_roots(self, polynomial: np.ndarray, tolerance: float = 1e-10) -> List[Tuple[float, float]]:
        """
        Find real roots of a polynomial, including those with small imaginary parts.
        
        Returns:
        --------
        List of tuples (real_value, imaginary_magnitude) for each acceptable root.
        """
        roots = np.roots(polynomial)
        acceptable_roots = []
        
        # Dynamic tolerance based on measurement uncertainty
        complex_tolerance = self.range_uncertainty * self.complex_tolerance_factor
        
        for root in roots:
            if abs(root.imag) < tolerance:
                # Pure real root - best quality
                acceptable_roots.append((root.real, 0.0))
            elif abs(root.imag) < complex_tolerance:
                # Small imaginary part - indicates near-miss intersection
                acceptable_roots.append((root.real, abs(root.imag)))
        
        return acceptable_roots
    
    def _solve_for_y_with_quality(self, x: float, coeffs1: Dict[str, float], 
                                  coeffs2: Dict[str, float]) -> List[Tuple[float, float]]:
        """
        Given x, solve for y values with quality metrics.
        
        Returns:
        --------
        List of tuples (y_real, imaginary_magnitude) for each solution.
        """
        y_values = []
        
        # Solve first ellipse: C1*y² + (B1*x + E1)*y + (A1*x² + D1*x + F1) = 0
        A1, B1, C1 = coeffs1['A'], coeffs1['B'], coeffs1['C']
        D1, E1, F1 = coeffs1['D'], coeffs1['E'], coeffs1['F']
        
        a = C1
        b = B1*x + E1
        c = A1*x**2 + D1*x + F1
        
        if abs(a) > 1e-12:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                y1 = (-b + sqrt_disc) / (2*a)
                y2 = (-b - sqrt_disc) / (2*a)
                y_values.extend([(y1, 0.0), (y2, 0.0)])
            else:
                # Complex roots - check if imaginary part is small enough
                sqrt_disc = np.sqrt(-discriminant)
                y_real = -b / (2*a)
                y_imag = sqrt_disc / (2*a)
                if y_imag < self.range_uncertainty * self.complex_tolerance_factor:
                    y_values.extend([(y_real, y_imag)])
        elif abs(b) > 1e-12:
            y_values.append((-c/b, 0.0))
        
        return y_values
    
    def _verify_intersection(self, x: float, y: float, 
                           coeffs1: Dict[str, float], 
                           coeffs2: Dict[str, float], 
                           tolerance: float = 1e-6) -> bool:
        """Verify that a point lies on both ellipses."""
        # Check first ellipse
        val1 = (coeffs1['A']*x**2 + coeffs1['B']*x*y + coeffs1['C']*y**2 + 
                coeffs1['D']*x + coeffs1['E']*y + coeffs1['F'])
        
        # Check second ellipse  
        val2 = (coeffs2['A']*x**2 + coeffs2['B']*x*y + coeffs2['C']*y**2 + 
                coeffs2['D']*x + coeffs2['E']*y + coeffs2['F'])
        
        return abs(val1) < tolerance and abs(val2) < tolerance
    
    def _solve_3d_intersection(self, 
                              sensor1: ENUPosition, sensor2: ENUPosition,
                              ioo1: ENUPosition, ioo2: ENUPosition,
                              measurement1: BistaticMeasurement,
                              measurement2: BistaticMeasurement) -> List[TargetSolution]:
        """
        Solve 3D ellipsoid intersection.
        
        Since altitude is unknown but positive, we sample different altitudes
        and find valid intersections.
        """
        solutions = []
        
        # Sample altitudes (0 to 15km in 500m steps)
        altitudes = np.arange(0, 15000, 500)
        
        for alt in altitudes:
            # Project to 2D at this altitude
            # Adjust ranges based on altitude difference
            sensor1_alt_diff = alt - sensor1.up
            sensor2_alt_diff = alt - sensor2.up
            ioo1_alt_diff = alt - ioo1.up
            ioo2_alt_diff = alt - ioo2.up
            
            # Check if geometry is valid at this altitude
            range1_sq = measurement1.bistatic_range**2
            range2_sq = measurement2.bistatic_range**2
            alt_contribution1_sq = (sensor1_alt_diff + ioo1_alt_diff)**2
            alt_contribution2_sq = (sensor2_alt_diff + ioo2_alt_diff)**2
            
            if range1_sq > alt_contribution1_sq and range2_sq > alt_contribution2_sq:
                # Corrected 2D ranges (using Pythagorean theorem)
                range1_2d = np.sqrt(range1_sq - alt_contribution1_sq)
                range2_2d = np.sqrt(range2_sq - alt_contribution2_sq)
                
                try:
                    intersections_2d = self._solve_2d_intersection(
                        sensor1, sensor2, ioo1, ioo2, range1_2d, range2_2d
                    )
                    
                    for pos_2d, quality in intersections_2d:
                        pos_3d = ENUPosition(pos_2d[0], pos_2d[1], alt)
                        geo_pos = pos_3d.to_geo(self.origin)
                        
                        # Calculate additional metrics
                        metrics = self.calculate_confidence_metrics(
                            measurement1, measurement2, geo_pos
                        )
                        
                        # Adjust quality based on altitude uncertainty
                        # Prefer solutions near typical aircraft altitudes
                        alt_quality = np.exp(-((alt - 8000)**2) / (5000**2))
                        combined_quality = quality * 0.8 + alt_quality * 0.2
                        
                        solution = TargetSolution(
                            position=geo_pos,
                            quality=combined_quality,
                            gdop=metrics['gdop'],
                            crossing_angle=metrics['crossing_angle']
                        )
                        solutions.append(solution)
                        
                except ValueError:
                    # Invalid geometry at this altitude
                    continue
        
        return solutions
    
    def calculate_confidence_metrics(self, 
                                   measurement1: BistaticMeasurement,
                                   measurement2: BistaticMeasurement,
                                   target_pos: GeoPosition) -> Dict[str, float]:
        """
        Calculate confidence metrics for a target position.
        
        Returns:
        --------
        Dictionary with confidence metrics:
        - gdop: Geometric dilution of precision
        - crossing_angle: Angle between ellipses at intersection (degrees)
        - snr_factor: Combined SNR factor if available
        """
        # Convert to ENU
        target_enu = target_pos.to_enu(self.origin)
        sensor1_enu = measurement1.sensor_pos.to_enu(self.origin)
        sensor2_enu = measurement2.sensor_pos.to_enu(self.origin)
        
        # Calculate crossing angle
        vec1 = target_enu.to_2d() - sensor1_enu.to_2d()
        vec2 = target_enu.to_2d() - sensor2_enu.to_2d()
        
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        crossing_angle = np.degrees(abs(angle1 - angle2))
        
        # Normalize to 0-90 degrees
        if crossing_angle > 90:
            crossing_angle = 180 - crossing_angle
        
        # Calculate GDOP (simplified)
        # Better crossing angles (near 90°) give lower GDOP
        gdop = 1.0 / np.sin(np.radians(crossing_angle))
        
        # SNR factor
        snr_factor = 1.0
        if measurement1.snr and measurement2.snr:
            snr_factor = np.sqrt(measurement1.snr * measurement2.snr)
        
        return {
            'gdop': gdop,
            'crossing_angle': crossing_angle,
            'snr_factor': snr_factor,
            'confidence': min(crossing_angle / 90.0, 1.0) * min(snr_factor / 20.0, 1.0)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create measurement data
    measurement1 = BistaticMeasurement(
        sensor_pos=GeoPosition(latitude=40.0, longitude=-74.0, altitude=100),
        ioo_pos=GeoPosition(latitude=40.1, longitude=-73.9, altitude=200),
        bistatic_range=15000,  # 15 km
        snr=20.0
    )
    
    measurement2 = BistaticMeasurement(
        sensor_pos=GeoPosition(latitude=40.05, longitude=-74.05, altitude=150),
        ioo_pos=GeoPosition(latitude=40.08, longitude=-73.95, altitude=180),
        bistatic_range=12000,  # 12 km
        snr=18.0
    )
    
    # Create solver (2D with assumed altitude)
    # Note: range_uncertainty=50m means we'll accept roots with imaginary parts up to 5m
    solver = BistaticGeolocation(
        altitude_assumption=5000,  # 5km altitude
        range_uncertainty=50.0,    # 50m measurement uncertainty
        complex_tolerance_factor=0.1  # Accept imaginary parts up to 10% of uncertainty
    )
    
    # Find initial positions
    try:
        solutions = solver.find_initial_positions(measurement1, measurement2)
        
        print(f"Found {len(solutions)} potential target positions:")
        for i, sol in enumerate(solutions):
            print(f"\nPosition {i+1}:")
            print(f"  Latitude: {sol.position.latitude:.6f}°")
            print(f"  Longitude: {sol.position.longitude:.6f}°")
            print(f"  Altitude: {sol.position.altitude:.1f} m")
            print(f"  Solution Quality: {sol.quality:.3f}")
            print(f"  GDOP: {sol.gdop:.2f}")
            print(f"  Crossing angle: {sol.crossing_angle:.1f}°")
            
            # Flag near-miss solutions
            if sol.quality < 0.95:
                print("  ⚠️  Near-miss solution (ellipses don't perfectly intersect)")
            
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test 3D solver (unknown altitude)
    print("\n\nTesting 3D solver with unknown altitude:")
    solver_3d = BistaticGeolocation(
        altitude_assumption=None,
        range_uncertainty=100.0  # Higher uncertainty for 3D case
    )
    
    solutions_3d = solver_3d.find_initial_positions(measurement1, measurement2)
    print(f"Found {len(solutions_3d)} potential positions at various altitudes")
    
    # Show top 5 solutions
    print("\nTop 5 solutions by quality:")
    for i, sol in enumerate(solutions_3d[:5]):
        print(f"{i+1}. Alt={sol.position.altitude:.0f}m, "
              f"Quality={sol.quality:.3f}, "
              f"GDOP={sol.gdop:.2f}")

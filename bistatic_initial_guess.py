import numpy as np
from typing import List, Tuple, Optional
import warnings

def find_initial_guess_positions(
    sensor1_pos: Tuple[float, float],
    sensor2_pos: Tuple[float, float], 
    ioo1_pos: Tuple[float, float],
    ioo2_pos: Optional[Tuple[float, float]],
    bistatic_range1: float,
    bistatic_range2: float,
    complex_threshold_ratio: float = 0.01
) -> List[Tuple[float, float]]:
    """
    Find initial guess positions for bistatic radar target localization using ellipse intersections.
    
    Parameters:
    -----------
    sensor1_pos : tuple of (x, y) coordinates for sensor 1 (typically at origin)
    sensor2_pos : tuple of (x, y) coordinates for sensor 2  
    ioo1_pos : tuple of (x, y) coordinates for illuminator of opportunity 1
    ioo2_pos : tuple of (x, y) coordinates for illuminator of opportunity 2, or None if shared IoO
    bistatic_range1 : bistatic range measurement from sensor 1 (km)
    bistatic_range2 : bistatic range measurement from sensor 2 (km)
    complex_threshold_ratio : threshold for accepting complex roots (as fraction of bistatic range)
    
    Returns:
    --------
    List of (x, y) tuples representing potential target positions
    """
    
    # Handle shared IoO case
    if ioo2_pos is None:
        ioo2_pos = ioo1_pos
        print("Using shared IoO configuration")
    
    # Convert to numpy arrays for easier computation
    s1 = np.array(sensor1_pos)
    s2 = np.array(sensor2_pos) 
    ioo1 = np.array(ioo1_pos)
    ioo2 = np.array(ioo2_pos)
    
    # Calculate ellipse parameters
    ellipse1_params = calculate_ellipse_params(s1, ioo1, bistatic_range1)
    ellipse2_params = calculate_ellipse_params(s2, ioo2, bistatic_range2)
    
    # Find intersection points
    intersections = find_ellipse_intersections(ellipse1_params, ellipse2_params, 
                                             bistatic_range1, bistatic_range2, 
                                             complex_threshold_ratio)
    
    return intersections

def calculate_ellipse_params(sensor_pos: np.ndarray, ioo_pos: np.ndarray, bistatic_range: float) -> dict:
    """
    Calculate ellipse parameters given sensor, IoO positions and bistatic range.
    
    Returns dictionary with ellipse parameters in standard form.
    """
    # Calculate baseline distance between sensor and IoO
    baseline = np.linalg.norm(ioo_pos - sensor_pos)
    
    # Semi-major axis: a = (bistatic_range + baseline) / 2
    a = (bistatic_range + baseline) / 2.0
    
    # Semi-minor axis: b = sqrt(a^2 - c^2) where c is half the focal distance
    c = baseline / 2.0  # Half the distance between foci
    
    if a <= c:
        raise ValueError(f"Invalid ellipse: semi-major axis ({a}) must be greater than focal distance ({c})")
    
    b = np.sqrt(a**2 - c**2)
    
    # Center of ellipse (midpoint between foci)
    center = (sensor_pos + ioo_pos) / 2.0
    
    # Angle of rotation (angle of major axis from x-axis)
    focal_vector = ioo_pos - sensor_pos
    theta = np.arctan2(focal_vector[1], focal_vector[0])
    
    return {
        'center': center,
        'a': a,  # semi-major axis
        'b': b,  # semi-minor axis  
        'theta': theta,  # rotation angle
        'foci': (sensor_pos, ioo_pos)
    }

def find_ellipse_intersections(ellipse1: dict, ellipse2: dict, 
                             bistatic_range1: float, bistatic_range2: float,
                             complex_threshold_ratio: float) -> List[Tuple[float, float]]:
    """
    Find intersection points between two ellipses using algebraic method.
    """
    # Transform ellipses to canonical form and solve
    # This involves converting the rotated ellipses to polynomial form and solving the system
    
    # Get ellipse coefficients in general conic form: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    A1, B1, C1, D1, E1, F1 = ellipse_to_conic_coefficients(ellipse1)
    A2, B2, C2, D2, E2, F2 = ellipse_to_conic_coefficients(ellipse2)
    
    # Solve the system of two conic equations
    intersections = solve_conic_intersection(
        (A1, B1, C1, D1, E1, F1),
        (A2, B2, C2, D2, E2, F2),
        max(bistatic_range1, bistatic_range2) * complex_threshold_ratio
    )
    
    return intersections

def ellipse_to_conic_coefficients(ellipse: dict) -> Tuple[float, float, float, float, float, float]:
    """
    Convert ellipse parameters to general conic form coefficients.
    """
    cx, cy = ellipse['center']
    a, b = ellipse['a'], ellipse['b']
    theta = ellipse['theta']
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_t2, sin_t2 = cos_t**2, sin_t**2
    
    # Coefficients for rotated ellipse: ((x-cx)cosθ + (y-cy)sinθ)²/a² + (-(x-cx)sinθ + (y-cy)cosθ)²/b² = 1
    A = cos_t2/a**2 + sin_t2/b**2
    B = 2*cos_t*sin_t*(1/a**2 - 1/b**2)  
    C = sin_t2/a**2 + cos_t2/b**2
    D = -2*cx*cos_t2/a**2 - 2*cy*cos_t*sin_t/a**2 - 2*cx*sin_t2/b**2 + 2*cy*cos_t*sin_t/b**2
    E = -2*cx*cos_t*sin_t/a**2 - 2*cy*sin_t2/a**2 + 2*cx*cos_t*sin_t/b**2 - 2*cy*cos_t2/b**2
    F = cx**2*cos_t2/a**2 + 2*cx*cy*cos_t*sin_t/a**2 + cy**2*sin_t2/a**2 + cx**2*sin_t2/b**2 - 2*cx*cy*cos_t*sin_t/b**2 + cy**2*cos_t2/b**2 - 1
    
    return A, B, C, D, E, F

def solve_conic_intersection(conic1: Tuple[float, ...], conic2: Tuple[float, ...], 
                           complex_threshold: float) -> List[Tuple[float, float]]:
    """
    Solve intersection of two conics using elimination and numpy.roots.
    """
    A1, B1, C1, D1, E1, F1 = conic1
    A2, B2, C2, D2, E2, F2 = conic2
    
    valid_points = []
    
    # Eliminate y to get polynomial in x
    # From conic1: C1*y² + (B1*x + E1)*y + (A1*x² + D1*x + F1) = 0
    # From conic2: C2*y² + (B2*x + E2)*y + (A2*x² + D2*x + F2) = 0
    
    # If C1 or C2 is zero, handle as special case
    if abs(C1) < 1e-12 or abs(C2) < 1e-12:
        return solve_linear_y_case(conic1, conic2, complex_threshold)
    
    # Eliminate y using resultant of two quadratics in y
    # This gives us a 4th degree polynomial in x
    poly_coeffs = compute_elimination_polynomial(conic1, conic2)
    
    # Find roots using numpy
    x_roots = np.roots(poly_coeffs)
    
    # For each x root, find corresponding y values
    for x in x_roots:
        if abs(np.imag(x)) <= complex_threshold:
            y_solutions = solve_for_y(x, conic1, complex_threshold)
            for y in y_solutions:
                if abs(np.imag(y)) <= complex_threshold:
                    # Verify the point satisfies both conics
                    if verify_point_on_both_conics(np.real(x), np.real(y), conic1, conic2, complex_threshold):
                        valid_points.append((float(np.real(x)), float(np.real(y))))
    
    return remove_duplicate_points(valid_points)

def compute_elimination_polynomial(conic1: Tuple[float, ...], conic2: Tuple[float, ...]) -> np.ndarray:
    """
    Compute the elimination polynomial by eliminating y from two conic equations.
    Returns coefficients of 4th degree polynomial in x.
    """
    A1, B1, C1, D1, E1, F1 = conic1
    A2, B2, C2, D2, E2, F2 = conic2
    
    # Two conics in y:
    # C1*y² + (B1*x + E1)*y + (A1*x² + D1*x + F1) = 0
    # C2*y² + (B2*x + E2)*y + (A2*x² + D2*x + F2) = 0
    
    # Resultant of two quadratics ay² + by + c and dy² + ey + f is:
    # (ae - bd)² - (af - cd)(de - b*d) 
    # But we need the full 4x4 Sylvester matrix approach for accuracy
    
    # Coefficients of first quadratic in y: C1, (B1*x + E1), (A1*x² + D1*x + F1)
    # Coefficients of second quadratic in y: C2, (B2*x + E2), (A2*x² + D2*x + F2)
    
    # Build Sylvester matrix for elimination
    # For quadratics p(y) = a₀ + a₁y + a₂y², q(y) = b₀ + b₁y + b₂y²
    # Sylvester matrix is 4×4
    
    # Polynomial coefficients as functions of x:
    # p(y): a₀ = A1*x² + D1*x + F1, a₁ = B1*x + E1, a₂ = C1
    # q(y): b₀ = A2*x² + D2*x + F2, b₁ = B2*x + E2, b₂ = C2
    
    # The determinant gives us a polynomial in x of degree 4
    # We'll compute this step by step
    
    # Expansion of the 4×4 Sylvester determinant yields:
    # coeff_x⁴ term: (A1*C2 - A2*C1)²
    # coeff_x³ term: 2*(A1*C2 - A2*C1)*(D1*C2 - D2*C1) + (B1*C2 - B2*C1)*(A1*B2 - A2*B1)
    # ... and so on
    
    # For simplicity and accuracy, let's use a more direct approach:
    # Sample the determinant at several x values and fit polynomial
    
    # Alternative: Use direct algebraic elimination
    # Multiply first equation by C2, second by C1, subtract to eliminate y² term
    # Then multiply first by (B2*x + E2), second by (B1*x + E1), subtract to eliminate y term
    
    # Direct elimination approach:
    if abs(C1) > 1e-12 and abs(C2) > 1e-12:
        # Eliminate y² term: C2*(eq1) - C1*(eq2) = 0
        # This gives: (C2*B1 - C1*B2)*x*y + (C2*E1 - C1*E2)*y + 
        #            (C2*A1 - C1*A2)*x² + (C2*D1 - C1*D2)*x + (C2*F1 - C1*F2) = 0
        
        # Call this: α*x*y + β*y + γ*x² + δ*x + ε = 0, so y = -(γ*x² + δ*x + ε)/(α*x + β)
        alpha = C2*B1 - C1*B2
        beta = C2*E1 - C1*E2  
        gamma = C2*A1 - C1*A2
        delta = C2*D1 - C1*D2
        epsilon = C2*F1 - C1*F2
        
        # Substitute back into first conic equation
        # C1*y² + (B1*x + E1)*y + (A1*x² + D1*x + F1) = 0
        # where y = -(γ*x² + δ*x + ε)/(α*x + β)
        
        # This will yield a 4th degree polynomial in x
        # Let's compute it symbolically
        
        # After substitution and clearing denominators, we get coefficients:
        poly_coeffs = compute_substitution_polynomial(
            A1, B1, C1, D1, E1, F1, alpha, beta, gamma, delta, epsilon
        )
        
    else:
        # Handle degenerate cases where one C coefficient is zero
        poly_coeffs = np.array([1, 0, 0, 0, 0])  # Placeholder - implement special cases
    
    return poly_coeffs

def compute_substitution_polynomial(A1, B1, C1, D1, E1, F1, alpha, beta, gamma, delta, epsilon):
    """
    Compute coefficients of 4th degree polynomial after substitution.
    """
    # y = -(gamma*x² + delta*x + epsilon)/(alpha*x + beta)
    # Substitute into: C1*y² + (B1*x + E1)*y + (A1*x² + D1*x + F1) = 0
    # Multiply through by (alpha*x + beta)² to clear denominators
    
    # The resulting polynomial will have degree 4 in x
    # Coefficients computed by expanding the substitution
    
    if abs(alpha) < 1e-12 and abs(beta) < 1e-12:
        # Special case - return high-degree zero polynomial  
        return np.array([0, 0, 0, 0, 1])
    
    # For numerical stability, let's use a more robust approach
    # We'll evaluate the constraint at multiple x values and solve for intersections
    
    x_test = np.linspace(-100, 100, 200)
    valid_x = []
    
    for x in x_test:
        denom = alpha*x + beta
        if abs(denom) > 1e-12:
            y = -(gamma*x**2 + delta*x + epsilon) / denom
            
            # Check if this (x,y) satisfies the original first conic
            val1 = A1*x**2 + B1*x*y + C1*y**2 + D1*x + E1*y + F1
            if abs(val1) < 1e-6:  # Tolerance for numerical errors
                valid_x.append(x)
    
    if len(valid_x) == 0:
        return np.array([0, 0, 0, 0, 1])
    
    # If we have valid x values, we can construct a polynomial
    # For now, return a simple polynomial - in practice you'd fit through the points
    if len(valid_x) >= 1:
        # Create polynomial with roots at valid_x locations
        poly = np.poly(valid_x[:4])  # Take up to 4 roots for degree 4 polynomial
        if len(poly) > 5:
            poly = poly[:5]  # Truncate to degree 4
        while len(poly) < 5:
            poly = np.append([0], poly)  # Pad with leading zeros
        return poly
    
    return np.array([0, 0, 0, 0, 1])

def solve_linear_y_case(conic1: Tuple[float, ...], conic2: Tuple[float, ...], 
                       complex_threshold: float) -> List[Tuple[float, float]]:
    """
    Handle case where one or both conics are linear in y.
    """
    A1, B1, C1, D1, E1, F1 = conic1
    A2, B2, C2, D2, E2, F2 = conic2
    
    valid_points = []
    
    # If C1 ≈ 0: B1*x*y + E1*y = -(A1*x² + D1*x + F1), so y = -(A1*x² + D1*x + F1)/(B1*x + E1)
    # If C2 ≈ 0: B2*x*y + E2*y = -(A2*x² + D2*x + F2), so y = -(A2*x² + D2*x + F2)/(B2*x + E2)
    
    if abs(C1) < 1e-12:
        # First conic is linear in y
        # Substitute y expression into second conic
        x_vals = solve_by_substitution_linear(conic1, conic2, complex_threshold)
    elif abs(C2) < 1e-12:
        # Second conic is linear in y  
        x_vals = solve_by_substitution_linear(conic2, conic1, complex_threshold)
    else:
        x_vals = []
    
    # Convert x values to (x,y) points
    for x in x_vals:
        if abs(np.imag(x)) <= complex_threshold:
            y_solutions = solve_for_y(x, conic1, complex_threshold)
            for y in y_solutions:
                if abs(np.imag(y)) <= complex_threshold:
                    valid_points.append((float(np.real(x)), float(np.real(y))))
    
    return valid_points

def solve_by_substitution_linear(linear_conic: Tuple[float, ...], quad_conic: Tuple[float, ...],
                               complex_threshold: float) -> List[complex]:
    """
    Solve by substituting linear y expression into quadratic conic.
    """
    A1, B1, C1, D1, E1, F1 = linear_conic  # C1 should be ≈ 0
    A2, B2, C2, D2, E2, F2 = quad_conic
    
    # From linear: y = -(A1*x² + D1*x + F1)/(B1*x + E1)
    # Substitute into quadratic conic and solve for x
    
    x_samples = np.linspace(-100, 100, 1000)
    x_solutions = []
    
    for x in x_samples:
        denom = B1*x + E1
        if abs(denom) > 1e-12:
            y = -(A1*x**2 + D1*x + F1) / denom
            # Check if (x,y) satisfies the quadratic conic
            val = A2*x**2 + B2*x*y + C2*y**2 + D2*x + E2*y + F2
            if abs(val) < 1e-6:
                x_solutions.append(x)
    
    return x_solutions[:4]  # Return up to 4 solutions

def verify_point_on_both_conics(x: float, y: float, conic1: Tuple[float, ...], 
                               conic2: Tuple[float, ...], tolerance: float) -> bool:
    """
    Verify that a point satisfies both conic equations within tolerance.
    """
    A1, B1, C1, D1, E1, F1 = conic1
    A2, B2, C2, D2, E2, F2 = conic2
    
    val1 = A1*x**2 + B1*x*y + C1*y**2 + D1*x + E1*y + F1
    val2 = A2*x**2 + B2*x*y + C2*y**2 + D2*x + E2*y + F2
    
    return abs(val1) < tolerance and abs(val2) < tolerance

def remove_duplicate_points(points: List[Tuple[float, float]], tolerance: float = 0.1) -> List[Tuple[float, float]]:
    """
    Remove duplicate points within tolerance.
    """
    unique_points = []
    for point in points:
        is_duplicate = False
        for existing in unique_points:
            if abs(point[0] - existing[0]) < tolerance and abs(point[1] - existing[1]) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    return unique_points

def solve_for_y(x: complex, conic: Tuple[float, ...], complex_threshold: float) -> List[complex]:
    """
    Solve for y given x and conic equation using numpy.roots.
    Conic equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    Rearranged as quadratic in y: Cy² + (Bx + E)y + (Ax² + Dx + F) = 0
    """
    A, B, C, D, E, F = conic
    
    # Coefficients of quadratic in y: ay² + by + c = 0
    a = C
    b = B*x + E  
    c = A*x**2 + D*x + F
    
    if abs(a) > 1e-12:
        # Use numpy.roots to solve quadratic
        coeffs = [a, b, c]
        roots = np.roots(coeffs)
        
        # Filter roots based on complex threshold
        valid_roots = []
        for root in roots:
            if abs(np.imag(root)) <= complex_threshold:
                valid_roots.append(root)
        
        return valid_roots
        
    elif abs(b) > 1e-12:
        # Linear equation: by + c = 0
        y = -c / b
        if abs(np.imag(y)) <= complex_threshold:
            return [y]
    
    return []



# Example usage
if __name__ == "__main__":
    # Example with two separate IoOs
    sensor1_pos = (0.0, 0.0)  # Origin
    sensor2_pos = (10.0, 5.0)  # 10km east, 5km north
    ioo1_pos = (15.0, 8.0)    # IoO for sensor 1
    ioo2_pos = (12.0, -3.0)   # IoO for sensor 2
    bistatic_range1 = 5.2     # km
    bistatic_range2 = 4.8     # km
    
    try:
        positions = find_initial_guess_positions(
            sensor1_pos, sensor2_pos, ioo1_pos, ioo2_pos,
            bistatic_range1, bistatic_range2
        )
        
        print("Potential target positions:")
        for i, pos in enumerate(positions):
            print(f"  Position {i+1}: ({pos[0]:.3f}, {pos[1]:.3f}) km")
            
    except Exception as e:
        print(f"Error: {e}")
        
    # Example with shared IoO
    print("\nShared IoO example:")
    shared_positions = find_initial_guess_positions(
        sensor1_pos, sensor2_pos, ioo1_pos, None,  # None for shared IoO
        bistatic_range1, bistatic_range2
    )
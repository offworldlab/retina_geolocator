"""
Test runner script with coverage reporting
"""

import unittest
import sys
import os
from io import StringIO

# Try to import coverage module
try:
    import coverage
    HAS_COVERAGE = True
except ImportError:
    HAS_COVERAGE = False
    print("Coverage module not installed. Install with: pip install coverage")
    print("Running tests without coverage...\n")


def run_tests_with_coverage():
    """Run all tests with code coverage analysis"""
    
    # Start coverage
    cov = coverage.Coverage(source=['initial_guess', 'least_squares_geolocator'])
    cov.start()
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Print coverage report
    print("\n" + "="*70)
    print("COVERAGE REPORT")
    print("="*70)
    
    # String buffer to capture report
    string_buffer = StringIO()
    cov.report(file=string_buffer)
    print(string_buffer.getvalue())
    
    # Generate HTML report
    print("\nGenerating HTML coverage report in 'htmlcov' directory...")
    cov.html_report(directory='htmlcov')
    
    return result


def run_tests_without_coverage():
    """Run all tests without coverage"""
    
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_specific_test_class(test_class_name):
    """Run a specific test class"""
    
    # Import test modules
    import test_initial_guess
    import test_least_squares_geolocator
    
    # Find the test class
    test_class = None
    for module in [test_initial_guess, test_least_squares_geolocator]:
        if hasattr(module, test_class_name):
            test_class = getattr(module, test_class_name)
            break
    
    if test_class is None:
        print(f"Test class '{test_class_name}' not found!")
        return None
    
    # Run the specific test class
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def print_test_summary(result):
    """Print a summary of test results"""
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")


def main():
    """Main test runner"""
    
    print("🧪 Bistatic Radar Geolocation Test Suite")
    print("="*70)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            print("Usage: python run_tests.py [options]")
            print("\nOptions:")
            print("  --help           Show this help message")
            print("  --no-coverage    Run tests without coverage analysis")
            print("  --class NAME     Run only the specified test class")
            print("\nExamples:")
            print("  python run_tests.py")
            print("  python run_tests.py --no-coverage")
            print("  python run_tests.py --class TestEllipseCenterStrategy")
            return
        
        elif sys.argv[1] == '--no-coverage':
            result = run_tests_without_coverage()
        
        elif sys.argv[1] == '--class' and len(sys.argv) > 2:
            test_class_name = sys.argv[2]
            print(f"Running specific test class: {test_class_name}")
            result = run_specific_test_class(test_class_name)
            if result is None:
                return
        
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    
    else:
        # Default: run with coverage if available
        if HAS_COVERAGE:
            result = run_tests_with_coverage()
        else:
            result = run_tests_without_coverage()
    
    # Print summary
    print_test_summary(result)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()

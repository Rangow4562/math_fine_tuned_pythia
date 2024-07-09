import unittest
from main import main
from io import StringIO
import sys

class TestFunctionalPipeline(unittest.TestCase):

    def setUp(self):
        self.saved_stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout to capture print statements

    def tearDown(self):
        sys.stdout = self.saved_stdout  # Restore stdout

    def test_main_pipeline(self):
        try:
            main()
            output = sys.stdout.getvalue().strip()
            expected_output = "Generated Answer:"
            self.assertIn(expected_output, output, f"Expected output '{expected_output}' not found in actual output:\n{output}")
        except Exception as e:
            self.fail(f"main() raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()

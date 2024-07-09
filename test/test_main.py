import unittest
from main import main

class TestMainPipeline(unittest.TestCase):

    def test_main_pipeline(self):
        try:
            main()  # Ensure no exceptions are raised during execution
        except Exception as e:
            self.fail(f"main() raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()

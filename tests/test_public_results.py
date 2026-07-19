import unittest
from pathlib import Path

from summarize_public_results import summarize, validate


class PublicResultsTest(unittest.TestCase):
    def test_public_results_match_locked_claims(self):
        data_dir = Path(__file__).resolve().parents[1] / "data"
        validate(summarize(data_dir))


if __name__ == "__main__":
    unittest.main()

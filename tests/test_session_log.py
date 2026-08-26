import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from session_log import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_writes_header_and_rate_limits_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            logger = SessionLogger(Path(directory), started_at=10.0, interval_seconds=0.2)
            self.assertTrue(logger.write(10.0, {"status": "MONITORING"}))
            self.assertFalse(logger.write(10.1, {"status": "DROWSY"}))
            self.assertTrue(logger.write(10.2, {"status": "DROWSY"}))
            path = logger.path
            logger.close()
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["status"] for row in rows], ["MONITORING", "DROWSY"])


if __name__ == "__main__":
    unittest.main()

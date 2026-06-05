import json
import os
import tempfile
import unittest

from scripts.kaggle_submission_ledger import check_duplicate, record_submission


class KaggleSubmissionLedgerTests(unittest.TestCase):
    def test_record_and_detect_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = os.path.join(tmpdir, "ledger.jsonl")
            csv_path = os.path.join(tmpdir, "cand.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("region_id,pred_week1\nR1,1.0\n")

            dup, _ = check_duplicate(csv_path, ledger)
            self.assertFalse(dup)

            record_submission(csv_path, "test", ledger)
            dup, info = check_duplicate(csv_path, ledger)
            self.assertTrue(dup)
            self.assertIn("md5", info)

            with open(ledger, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], "cand.csv")


if __name__ == "__main__":
    unittest.main()

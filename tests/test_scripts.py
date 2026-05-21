import os
import subprocess
import sys
import unittest


class ScriptTests(unittest.TestCase):
    def test_temporal_tree_script_help_imports_local_model_package(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_temporal_tree_submission.py", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generate temporal-block tree Kaggle submission", result.stdout)
        self.assertIn("--feature-set", result.stdout)
        self.assertIn("--gpu", result.stdout)

    def test_cnn_script_help_imports_local_model_package(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_cnn_submission.py", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Generate small 1D CNN Kaggle submission", result.stdout)
        self.assertIn("--model", result.stdout)
        self.assertIn("--seed", result.stdout)
        self.assertIn("--patience", result.stdout)
        self.assertIn("--dropout", result.stdout)
        self.assertIn("--weight-decay", result.stdout)
        self.assertIn("--scheduler", result.stdout)
        self.assertIn("--calendar", result.stdout)
        self.assertIn("cnn_gru", result.stdout)

    def test_temporal_backtest_script_help_imports_local_model_package(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_temporal_backtest.py", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Run temporal backtest validation", result.stdout)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--recent-cutoffs", result.stdout)
        self.assertIn("--epochs", result.stdout)
        self.assertIn("--calendar", result.stdout)


if __name__ == "__main__":
    unittest.main()

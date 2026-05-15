from setuptools import setup, find_packages
from setuptools import Command
import subprocess, sys, os


class DataCommand(Command):
    """Download Kaggle competition data using kagglehub."""
    description = "download Kaggle competition data (requires ~/.kaggle/kaggle.json)"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        os.makedirs("data", exist_ok=True)

        # Check if data already exists
        if os.path.exists("data/train.csv") and os.path.exists("data/test.csv"):
            print("[*] Data already exists in data/. Skipping download.")
            return

        print("[*] Attempting Kaggle download...")
        print("[*] Kaggle invite link: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32")
        print("[!] Requires: ~/.kaggle/kaggle.json (Kaggle API key)")
        print("[!] You must first accept the competition invitation (NYCU email login).")

        for handle in [
            "dm-2026-final-project",
            "natural-disaster-severity-prediction",
        ]:
            try:
                print(f"    Trying: {handle}")
                kagglehub.competition_download(handle, path="data")
                print(f"[*] Downloaded: {handle}")
                return
            except Exception:
                continue

        print("[!] Automatic download failed.")
        print("    1. Accept invite: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32")
        print("    2. Download train.csv, test.csv, sample_submission.csv")
        print("    3. Place all files in data/")


setup(
    name="dm2026_final_project",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
    ],
    cmdclass={
        "data": DataCommand,
    },
)

from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, sys, os


class InstallWithData(install):
    """Install package + dependencies, then download Kaggle data."""

    def run(self):
        install.run(self)
        self._download_data()

    def _download_data(self):
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        os.makedirs("data", exist_ok=True)

        if os.path.exists("data/train.csv") and os.path.exists("data/test.csv"):
            print("[*] Data already exists in data/. Skipping download.")
            return

        print("[*] Downloading Kaggle competition data...")
        print("[*] Invite: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32")
        print("[!] Requires ~/.kaggle/kaggle.json (Kaggle API key)")

        for handle in [
            "dm-2026-final-project",
            "natural-disaster-severity-prediction",
        ]:
            try:
                kagglehub.competition_download(handle, path="data")
                print(f"[*] Downloaded: {handle}")
                return
            except Exception:
                continue

        print("[!] Automatic download failed.")
        print("    1. Accept invite: https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32")
        print("    2. Download train.csv, test.csv, sample_submission.csv to data/")


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
        "install": InstallWithData,
    },
)

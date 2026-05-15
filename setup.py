from setuptools import setup, find_packages

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
)

from setuptools import setup, find_packages

setup(
    name="gfc_net",
    version="1.0.0",
    description="Graph Structural Calibration and Topological Attention for Robust Node Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.64.0",
        "tabulate>=0.8.9",
        "scipy>=1.7.0",
    ],
)

from setuptools import setup, find_packages

setup(
    name="gfc_net",
    version="1.0.0",
    description="Graph Feature Calibration and Confidence Alignment for Semi-Supervised Node Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
)

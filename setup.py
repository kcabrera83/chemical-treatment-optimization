from setuptools import setup, find_packages

setup(
    name="chemical-treatment-optimization",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "pandas>=2.0",
    ],
    author="Ing. Kelvin Cabrera",
    description="ML-based oilfield chemical treatment optimization system",
)

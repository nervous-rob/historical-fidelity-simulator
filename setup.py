"""Setup configuration for Historical Fidelity Simulator."""

from setuptools import setup, find_packages

# Read version from historical_fidelity_simulator/__init__.py
with open("historical_fidelity_simulator/__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="historical-fidelity-simulator",
    version=version,
    description="A simulator for studying historical fidelity through classical and quantum approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rob Browning",
    author_email="rob@nervouslabs.com",
    url="https://github.com/nervousrob/historical-fidelity-simulator",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "qutip",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "pylint",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
) 
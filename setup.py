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
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qutip>=4.6.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "numba>=0.54.0",  # Required for base functionality
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "pylint",
            "ipykernel",
            "jupyter",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",  # For CUDA 11.x
            # OR "cupy-cuda12x>=12.0.0",  # For CUDA 12.x
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
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",  # Add GPU support indicator
    ],
) 
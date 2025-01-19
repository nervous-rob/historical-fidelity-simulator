"""Setup configuration for Historical Fidelity Simulator."""

from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="historical-fidelity-simulator",
    use_scm_version=True,
    description="A simulator for studying historical fidelity through classical and quantum approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Historical Fidelity Simulator Contributors",
    url="https://github.com/historical-fidelity-simulator/historical-fidelity-simulator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "qutip>=4.6.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "numba>=0.54.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
            "pylint>=2.15.0",
            "ipykernel>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",  # For CUDA 11.x
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
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
) 
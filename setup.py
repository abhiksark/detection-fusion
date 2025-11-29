from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="detection-fusion",
    version="1.0.0",
    author="Abhik Sarkar",
    description="DetectionFusion: Python toolkit for fusing multiple object detection results with ground truth validation and error analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhiksark/detection-fusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "pydantic>=2.0.0",
        "scikit-learn>=0.23.0",
        "PyYAML>=5.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "torch": [
            "torch>=1.7.0",
            "torchvision>=0.8.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "cli": [
            "click>=8.0.0",
            "rich>=12.0.0",
        ],
        "full": [
            "torch>=1.7.0",
            "torchvision>=0.8.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "click>=8.0.0",
            "rich>=12.0.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "ruff>=0.0.200",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "detection-fusion=detection_fusion.cli.main:cli",
            "dfusion=detection_fusion.cli.main:cli",
        ],
    },
)
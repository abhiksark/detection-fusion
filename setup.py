from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="detection-fusion",
    version="0.2.0",
    author="Abhik Sarkar",
    description="DetectionFusion: Python toolkit for fusing multiple object detection results with ground truth validation and error analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhiksark/detection-fusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pycocotools>=2.0.2",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "viz": [
            "plotly>=5.0",
            "bokeh>=2.3",
        ],
    },
)
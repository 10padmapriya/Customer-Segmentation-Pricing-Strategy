from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="retail-segmentation",
    version="1.0.0",
    description=(
        "Customer segmentation + pricing strategy pipeline. "
        "KMeans · DBSCAN · GMM · GBR WTP model · SHAP. "
        "Built on UCI Online Retail II (1M+ real transactions)."
    ),
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "ucimlrepo>=0.0.3",
        "scikit-learn>=1.4.0",
        "shap>=0.44.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "jupyter>=1.0.0", "flake8", "black"]
    },
)

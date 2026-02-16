"""
Setup script for Updraft-LM
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="updraft-lm",
    version="1.0.0",
    author="Updraft-LM Team",
    author_email="",
    description="A powerful GPT-1 level transformer language model with modern TUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/desenyon/updraft-lm",
    packages=find_packages(exclude=["tests", "checkpoints", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "pylint>=3.0.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "updraft=main:main",
            "updraft-tui=tui.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

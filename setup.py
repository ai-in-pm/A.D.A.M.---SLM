"""
Setup script for ADAM SLM
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "ADAM SLM - Advanced Deep Attention Model Small Language Model"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [
        "torch>=2.3.0",
        "tiktoken>=0.5.1",
        "matplotlib>=3.7.1",
        "tqdm>=4.66.1",
        "numpy>=1.26",
        "pandas>=2.2.1",
        "transformers",
        "accelerate",
        "datasets",
        "wandb",
    ]

setup(
    name="adam-slm",
    version="1.0.0",
    author="AI in PM",
    author_email="ai.in.pm@example.com",
    description="Advanced Deep Attention Model Small Language Model",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ai-in-pm/adam-slm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "jupyter>=1.0",
        ],
        "training": [
            "wandb",
            "tensorboard",
            "deepspeed",
        ],
        "inference": [
            "onnx",
            "onnxruntime",
        ],
    },
    entry_points={
        "console_scripts": [
            "adam-slm-train=examples.train_adam_slm:main",
        ],
    },
    include_package_data=True,
    package_data={
        "adam_slm": ["*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "natural language processing",
        "transformer",
        "language model",
        "pytorch",
        "attention",
        "rope",
        "swiglu",
        "gqa",
    ],
    project_urls={
        "Bug Reports": "https://github.com/ai-in-pm/adam-slm/issues",
        "Source": "https://github.com/ai-in-pm/adam-slm",
        "Documentation": "https://github.com/ai-in-pm/adam-slm/wiki",
    },
)

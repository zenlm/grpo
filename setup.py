from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hanzo-grpo",
    version="0.1.0",
    author="Hanzo AI",
    author_email="dev@hanzo.ai",
    description="Guided Reinforcement Policy Optimization for LLM fine-tuning on Hanzo AI platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanzoai/grpo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.14.0",
        "transformers>=4.35.0",
        "trl>=0.7.0",
        "torch>=2.0.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "hanzoai>=0.1.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "typer>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "monitoring": [
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
            "loguru>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "grpo=grpo.cli:main",
        ],
    },
)
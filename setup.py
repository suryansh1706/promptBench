from setuptools import setup, find_packages

setup(
    name="promptbench",
    version="1.0.0",
    description="A Framework for Auditing Demographic Bias in Large Language Model Responses",
    author="Parth Mudgal, Suryansh Rastogi, Siddharth Sharma",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "streamlit>=1.30.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "promptbench=promptbench.cli:main",
        ],
    },
)

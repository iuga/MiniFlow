from setuptools import setup, find_packages

setup(
    name="MiniFlow",
    version="0.1.0",
    description="A lightweight deeplearning framework based on NumPy",
    author="@iugax",
    packages=find_packages(exclude=["tests", "tools", "docs", ".github"]),
    install_requires=[
        'click==6.6',
        'numpy==1.12.1',
    ]
)

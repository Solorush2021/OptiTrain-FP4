from setuptools import setup, find_packages

setup(
    name="optitrain-fp4",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "deepspeed>=0.10.0"
    ],
    author="Solorush2021",
    description="Optimized FP4 Mixed-Precision Training with Muon and Triton on Blackwell Architecture",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Solorush2021/OptiTrain-FP4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

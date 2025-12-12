from setuptools import setup, find_packages

setup(
    name="hamt",
    version="0.1.0",
    description="Holographic Associative Memory Transformers for Energy-Efficient LLMs",
    author="NeuroLab AI Syndicate",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "datasets>=2.15.0",
    ],
)

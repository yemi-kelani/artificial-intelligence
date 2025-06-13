from setuptools import setup, find_packages

setup(
    name="ai-models",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy<2.0.0",
        "torch",
        "matplotlib",
        "tqdm",
        "ipywidgets",
        "keybert",
        "Requests",
        "summa",
        "tokenizers",
        "torch",
        "transformers",
        "matplotlib",
        "ipykernel"
    ],
)
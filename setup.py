from setuptools import setup, find_packages

setup(
    
    name="artificial_intelligence",
    
    version="2.0.0",
    
    author="Yemi Kelani",
    
    author_email="aak.development@gmail.com",
    
    description="Artificial Intelligence related projects.",
    
    long_description=open("README.md").read(),
    
    long_description_content_type="text/markdown",
    
    url = "https://github.com/yemi-kelani/artificial-intelligence",
    
    packages = find_packages(),  # Automatically find packages in your project
    
    install_requires = [
        "keybert",
        "numpy",
        "Requests",
        "summa",
        "tokenizers",
        "torch",
        "transformers",
        "numpy",
        "tqdm"
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    python_requires = ">=3.11.3",
    
    entry_points = {
        'console_scripts': [
            'play   = models.ReinforcementLearning.DeepQ_TicTacToe_v2.play:play',
        ]
    },
    
)
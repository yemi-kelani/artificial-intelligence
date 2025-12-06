# Setup Development Environment

Set up and validate the development environment for this AI research project.

## Initial Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate virtual environment (Windows)
# .venv\Scripts\activate

# Install project in development mode
pip install -e .
```

## Validate Environment
```bash
# Check Python version
python --version

# Verify key packages
pip show torch torchvision
pip show numpy pandas matplotlib
pip show jupyter notebook

# Test imports
python -c "import torch; import numpy; import pandas; import matplotlib; print('All imports successful')"
```

## Quick Environment Check
```bash
# One-line validation
python -c "import torch, numpy, pandas, matplotlib; print(f'✓ Environment ready - PyTorch {torch.__version__}, NumPy {numpy.__version__}')"
```
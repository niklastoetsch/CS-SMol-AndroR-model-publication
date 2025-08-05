# Installation Guide

## System Requirements

- **Python**: 3.7 or higher (tested with Python 3.8-3.11)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: At least 1GB free space for dependencies

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository**:
```bash
git clone https://github.com/niklastoetsch/CS-SMol-AndroR-model-publication.git
cd CS-SMol-AndroR-model-publication
```

2. **Create a virtual environment** (recommended):
```bash
# Using venv
python -m venv andror_env
source andror_env/bin/activate  # On Windows: andror_env\Scripts\activate

# Or using conda
conda create -n andror_env python=3.9
conda activate andror_env
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation

If you prefer to install packages individually:

```bash
# Core scientific packages
pip install pandas numpy scipy matplotlib seaborn

# Machine learning
pip install scikit-learn

# Molecular informatics
pip install rdkit-pypi

# Jupyter notebooks
pip install jupyter ipykernel

# Model interpretation
pip install shap

# Progress bars
pip install tqdm
```

## RDKit Installation Notes

RDKit is the most complex dependency. If you encounter issues:

### Alternative RDKit Installation Methods

1. **Using conda** (often more reliable):
```bash
conda install -c conda-forge rdkit
```

2. **Using mamba** (faster conda alternative):
```bash
mamba install -c conda-forge rdkit
```

3. **Docker option** (if local installation fails):
```bash
docker pull rdkit/rdkit-python3-jupyter
```

### Common RDKit Issues

- **Linux**: May require system packages: `sudo apt-get install python3-dev`
- **macOS**: Ensure Xcode command line tools: `xcode-select --install`
- **Windows**: Use conda/mamba for most reliable installation

## Verification

Test your installation:

```python
# Test basic imports
import pandas as pd
import numpy as np
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

# Test RDKit functionality
mol = Chem.MolFromSmiles("CCO")
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
print("Installation successful!")
```

## Jupyter Notebook Setup

1. **Install Jupyter kernel**:
```bash
python -m ipykernel install --user --name andror_env --display-name "AndroR Models"
```

2. **Start Jupyter**:
```bash
jupyter notebook
```

3. **Select the correct kernel** in Jupyter: Kernel → Change Kernel → "AndroR Models"

## Data Setup

The analysis requires specific datasets that are not included in the repository:

1. **Download/obtain the following files**:
   - `andror_df_all_clusters.csv`
   - `AndroR_4_final_dataset_training_set.csv`

2. **Place data files** in the parent directory:
```
parent_directory/
├── andror_df_all_clusters.csv
├── AndroR_4_final_dataset_training_set.csv
└── CS-SMol-AndroR-model-publication/
    ├── README.md
    ├── ml.py
    └── ...
```

## Running the Code

### Quick Test

```python
from ml import create_pipeline
from utils import FP_COLUMNS
import pandas as pd

# Test pipeline creation
pipeline = create_pipeline()
print("Pipeline created successfully:", pipeline)

# Test utilities
print("Fingerprint columns:", len(FP_COLUMNS))
```

### Full Analysis

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Run notebooks in order**:
   - `unrestricted_dataset.ipynb` (main analysis)
   - `comparison_for_different_feature_spaces.ipynb`
   - `fingerprint_based_clustering.ipynb`
   - `SHAP_analysis.ipynb`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and packages installed
2. **RDKit import errors**: Try conda installation method
3. **Memory errors**: Reduce dataset size or increase system memory
4. **Jupyter kernel not found**: Re-install kernel with `ipykernel install`

### Performance Optimization

1. **Parallel processing**: Increase `n_jobs` parameter in scikit-learn functions
2. **Memory management**: Process data in chunks for large datasets
3. **Caching**: Use pickle files for intermediate results (already implemented)

### Getting Help

1. **Check dependencies**: Verify all packages installed correctly
2. **Update packages**: `pip install --upgrade -r requirements.txt`
3. **GitHub Issues**: Report problems on the repository issue tracker
4. **Documentation**: Refer to individual package documentation for specific issues

## Development Setup

For contributors or advanced users:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install jupyter black flake8 pytest

# Format code
black *.py

# Run linting
flake8 *.py

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Docker Alternative

If local installation is problematic:

```dockerfile
FROM continuumio/miniconda3

RUN conda install -c conda-forge rdkit pandas scikit-learn matplotlib jupyter
COPY . /app
WORKDIR /app
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
```

This installation guide should help users get the environment set up successfully for reproducing the analysis and building upon the models.
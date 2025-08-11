# Installation Guide

This installation guide should help users get the environment set up successfully for reproducing the analysis and building upon the models.

## Installation Methods

### Using pip (Recommended)

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

## RDKit Installation Notes

RDKit is the most complex dependency. If you encounter issues:

### Alternative RDKit Installation Methods

**Using conda** (often more reliable):
```bash
conda install -c conda-forge rdkit
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

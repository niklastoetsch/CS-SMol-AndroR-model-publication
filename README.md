# CS-SMol-AndroR-model-publication

## Overview

This repository contains the code and analysis for computational toxicology models targeting the androgen receptor (AndroR). The project implements machine learning approaches to predict chemical inhibitory activity against the androgen receptor, with applications in endocrine disruption screening and chemical safety assessment.

## Key Features

- **Machine Learning Pipeline**: Gradient boosting classifier with stratified group cross-validation
- **Multiple Feature Representations**: Support for molecular fingerprints and RDKit descriptors  
- **Comprehensive Analysis**: SHAP explainability, clustering analysis, and performance evaluation
- **Robust Evaluation**: Balanced accuracy, ROC analysis, calibration curves, and multiple metrics

## Repository Structure

```
├── ml.py                     # Core ML pipeline and cross-validation framework
├── analysis.py               # Analysis classes for model evaluation and visualization
├── utils.py                  # Utility functions for molecular feature generation
├── unrestricted_dataset.ipynb           # Main analysis workflow
├── comparison_for_different_feature_spaces.ipynb  # Feature space comparison
├── fingerprint_based_clustering.ipynb   # Molecular clustering analysis
└── SHAP_analysis.ipynb       # Model explainability analysis
```

## Installation

### Requirements

This project requires Python 3.7+ and the following packages:

```bash
pip install pandas numpy scikit-learn matplotlib tqdm rdkit-pypi
```

For Jupyter notebook execution:
```bash
pip install jupyter ipykernel
```

See `requirements.txt` for specific version requirements.

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/niklastoetsch/CS-SMol-AndroR-model-publication.git
cd CS-SMol-AndroR-model-publication
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

**Note**: The datasets referenced in the notebooks are not included in this repository. To reproduce the analysis, you will need:

- `andror_df_all_clusters.csv` - Complete dataset with cluster assignments
- `AndroR_4_final_dataset_training_set.csv` - Training dataset with molecular features

These files should be placed in the parent directory (`../`) relative to the notebook locations.

## Usage

### Running the Analysis

1. **Main workflow**: Execute `unrestricted_dataset.ipynb` for the primary analysis
2. **Feature comparison**: Run `comparison_for_different_feature_spaces.ipynb` 
3. **Clustering analysis**: Execute `fingerprint_based_clustering.ipynb`
4. **Model interpretation**: Run `SHAP_analysis.ipynb`

### Using the ML Pipeline

```python
from ml import run_cv
from utils import add_fingerprints_to_df
import pandas as pd

# Load your data
df = pd.read_csv("your_dataset.csv")

# Add molecular fingerprints
df = add_fingerprints_to_df(df)

# Prepare features and labels
X = df[['fp_0', 'fp_1', ...]]  # fingerprint columns
y = df['activity_label']       # target variable
groups = df['cluster_id']      # for grouped CV

# Run cross-validation
results = run_cv(X, y, groups=groups, n_splits=5, n_repetitions=5)
```

## Methodology

### Machine Learning Approach

- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
- **Cross-validation**: Stratified Group K-Fold (5 folds, 5 repetitions)
- **Class balancing**: Sample weighting based on class frequencies
- **Feature types**: Morgan fingerprints (ECFP6, 1024 bits) and RDKit molecular descriptors

### Evaluation Metrics

- Balanced Accuracy
- ROC-AUC
- Precision-Recall curves
- Calibration analysis
- Matthews Correlation Coefficient (MCC)
- Negative Predictive Value (NPV)

## Model Performance

The models achieve robust performance in androgen receptor inhibition prediction with proper cross-validation accounting for molecular similarity clusters. Detailed results and visualizations are available in the analysis notebooks.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss the proposed modifications.

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the corresponding author.

## Acknowledgments

This work contributes to the field of computational toxicology and endocrine disruption screening through machine learning approaches for chemical safety assessment.
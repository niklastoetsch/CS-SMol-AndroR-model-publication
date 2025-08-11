# CS-SMol-AndroR-model-publication

## Overview

This repository contains the code and analysis for computational toxicology models targeting the androgen receptor (AndroR). The project implements machine learning approaches to predict chemical inhibitory activity against the androgen receptor, with applications in endocrine disruption screening and chemical safety assessment.

## Key Features

- **Machine Learning Pipeline**: Gradient boosting classifier with stratified group cross-validation based on Tanimoto clusters
- **Multiple Feature Representations**: Support for molecular fingerprints and RDKit descriptors
- **Analysis**: SHAP explainability and performance evaluation

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

See `Installation.md` for details

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

## Reproducing the Results from the Paper

1. **Clustering analysis**: Execute `fingerprint_based_clustering.ipynb` to obtain the Tanimoto clusters 
2. **Feature comparison**: Run `comparison_for_different_feature_spaces.ipynb`
3. **Comparison between full and unrestricted datasets and predictions on CoMPARA dataset**: Execute `unrestricted_dataset.ipynb` for the primary analysis
4. **Model interpretation**: Run `SHAP_analysis.ipynb`

## Methodology

Refer to `METHODOLOGY.md` or the paper for details

### Machine Learning Approach

- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
- **Cross-validation**: Stratified Group K-Fold (5 folds, 5 repetitions)
- **Class balancing**: Sample weighting based on class frequencies
- **Feature types**: Morgan fingerprints (ECFP6, 1024 bits) and RDKit molecular descriptors

## Model Performance

The models achieve robust performance in androgen receptor inhibition prediction with proper cross-validation accounting for molecular similarity clusters. Detailed results and visualizations are available in the analysis notebooks or the paper.

## Citation

If you use our dataset and/or this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or problems, please open an issue on GitHub or contact the corresponding author.

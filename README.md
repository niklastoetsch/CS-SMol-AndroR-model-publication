# CS-SMol-AndroR-model-publication

## Overview

This repository contains the code and analysis for computational toxicology models targeting the androgen receptor (AndroR). The project implements machine learning approaches to predict chemical inhibitory activity against the androgen receptor, with applications in endocrine disruption screening and chemical safety assessment.

## Potential Applications
- **Chemical Screening**: Prioritization of chemicals for experimental testing
- **Risk Assessment**: Support for regulatory decision-making
- **Drug Development**: Early identification of endocrine disruption potential
- **Chemical Design**: Guide synthesis of compounds with lowered probability of endocrine disruption

## Key Features

- **Machine Learning Pipeline**: Gradient boosting classifier with stratified group cross-validation based on Tanimoto clusters
- **Multiple Feature Representations**: Support for molecular fingerprints and RDKit descriptors
- **Analysis**: Performance evaluation and SHAP explainability

## Repository Structure

```
├── ml.py                                                       # Core ML pipeline and cross-validation framework
├── analysis.py                                                 # Analysis classes for model evaluation and visualization
├── utils.py                                                    # Utility functions for molecular feature generation
├── fingerprint_based_clustering.ipynb                          # Molecular clustering for proper grouping during cross-validation
├── comparison_for_different_feature_spaces.ipynb               # Feature space comparison
├── compara_predictions_and_CV_for_unrestricted_dataset.ipynb   # Main analysis workflow
└── SHAP_analysis.ipynb                                         # Model explainability analysis
```

## Installation

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

See `requirements.txt` for specific version requirements.

### Jupyter Notebook Setup

1. **Install Jupyter kernel**:
```bash
python -m ipykernel install --user --name andror_env --display-name "AndroR Models"
```

2. **Start Jupyter**:
```bash
jupyter notebook
```

3. **Select the correct kernel** in Jupyter: Kernel → Change Kernel → "AndroR Models"

## Data Requirements

**Note**: The datasets referenced in the notebooks are not included in this repository. To reproduce the analysis, you will need:

- `andror_df_all_clusters.csv` - Complete dataset with cluster assignments
- `AndroR_4_final_dataset_training_set.csv` - Training dataset with molecular features

These files should be placed in the parent directory (`../`) relative to the notebook locations.
```
parent_directory/
├── andror_df_all_clusters.csv
├── AndroR_4_final_dataset_training_set.csv
└── CS-SMol-AndroR-model-publication/
    ├── README.md
    ├── ml.py
    └── ...
```

## Reproducing the Results from the Paper

1. **Clustering analysis**: Execute `fingerprint_based_clustering.ipynb` to obtain the Tanimoto clusters 
2. **Feature comparison**: Run `comparison_for_different_feature_spaces.ipynb`
3. **Comparison between full and unrestricted datasets and predictions on CoMPARA dataset**: Execute `compara_predictions_and_CV_for_unrestricted_dataset.ipynb` for the primary analysis
4. **Model interpretation**: Run `SHAP_analysis.ipynb`

## Methodology

Please also refer to the paper.

### Molecular Representation

1. **SMILES Processing**: Chemical structures are represented as SMILES (Simplified Molecular Input Line Entry System) strings
2. **Molecular Fingerprints**: Extended-Connectivity Fingerprints (ECFP6) with 1024 bits generated using RDKit
   - Radius: 3 (equivalent to ECFP6)
   - Bit vector length: 1024
   - Hashing function: Morgan algorithm
3. **Molecular Descriptors**: RDKit molecular descriptors calculated for alternative feature representation

### Data Splitting and Clustering

- **Molecular Clustering**: Chemicals are clustered based on Tanimoto similarity using molecular fingerprints
- **Cluster-based Splitting**: Cross-validation respects molecular clusters to prevent data leakage from similar compounds
- **Stratified Grouping**: Maintains class balance while respecting cluster membership


### Machine Learning Approach

- **Algorithm**: Gradient Boosting Classifier (scikit-learn). Handles non-linear relationships and feature interactions common in molecular data. Reasonably prone to overfitting, performed well when compared to other algorithms (see paper)
- **Default Hyperparameters**: No hyperparameter optimization to focus on methodology demonstration.
- **Cross-validation**: Stratified Group K-Fold (5 folds, 5 repetitions)
- **Sample Weighting**: Balanced class weights computed using scikit-learn's `compute_class_weight`. Applied during model training to handle class imbalance. Alternative to oversampling/undersampling to preserve original data distribution
- **Class balancing**: Sample weighting based on class frequencies
- **Feature types**: Morgan fingerprints (ECFP6, 1024 bits) and RDKit molecular descriptors

## Model Performance

The models achieve robust performance in androgen receptor inhibition prediction with proper cross-validation accounting for molecular similarity clusters. Detailed results and visualizations are available in the analysis notebooks or the paper.
Performance uncertainties are determined using Bayesian inference for predictions via [Bayesian inference](https://peerj.com/articles/cs-398/).
Feature importance is determined via SHapley Additive exPlanations (SHAP).

## Citation

If you use our dataset and/or this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or problems, please open an issue on GitHub or contact the corresponding author.

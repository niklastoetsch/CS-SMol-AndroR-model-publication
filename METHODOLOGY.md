# Methodology

## Overview

This document provides detailed methodology for the androgen receptor (AndroR) inhibition prediction models implemented in this repository.

## Problem Statement

The androgen receptor is a critical nuclear hormone receptor involved in male reproductive development and various physiological processes. Chemicals that interfere with androgen receptor function can act as endocrine disruptors, potentially causing adverse health effects. This project develops computational models to predict chemical inhibitory activity against the androgen receptor for screening and prioritization in chemical safety assessment.

## Dataset and Data Preprocessing

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

## Machine Learning Methodology

### Algorithm Selection

**Gradient Boosting Classifier** (scikit-learn implementation)
- **Rationale**: Handles non-linear relationships and feature interactions common in molecular data
- **Robustness**: Less prone to overfitting compared to deep learning approaches on small datasets
- **Interpretability**: Supports feature importance analysis and SHAP explanations

### Cross-Validation Strategy

**Stratified Group K-Fold Cross-Validation**
- **K-folds**: 5 folds
- **Repetitions**: 5 repetitions with different random seeds
- **Stratification**: Maintains class balance across folds
- **Grouping**: Ensures compounds from the same molecular cluster are not split across train/test sets
- **Total models**: 25 models (5 folds × 5 repetitions)

### Class Imbalance Handling

**Sample Weighting**
- Balanced class weights computed using scikit-learn's `compute_class_weight`
- Applied during model training to handle class imbalance
- Alternative to oversampling/undersampling to preserve original data distribution

### Hyperparameter Configuration

**Default Gradient Boosting Parameters**:
- Using scikit-learn defaults for reproducibility
- No hyperparameter optimization to focus on methodology demonstration
- Can be extended with grid search or Bayesian optimization for production use

## Feature Engineering

### Molecular Fingerprints (Primary)
- **Type**: Morgan fingerprints (ECFP6)
- **Implementation**: RDKit `GetMorganFingerprintAsBitVect`
- **Parameters**: radius=3, nBits=1024
- **Advantages**: Captures local molecular environments, standard in cheminformatics

### Molecular Descriptors (Alternative)
- **Source**: RDKit descriptor calculation
- **Coverage**: Physicochemical properties, topological indices, structural features
- **Usage**: Comparative analysis with fingerprint-based models

## Model Evaluation

### Performance Metrics

1. **Balanced Accuracy**: Primary metric accounting for class imbalance
2. **ROC-AUC**: Area under receiver operating characteristic curve
3. **Precision-Recall**: Particularly important for imbalanced datasets
4. **Matthews Correlation Coefficient (MCC)**: Robust measure for binary classification
5. **Negative Predictive Value (NPV)**: Important for screening applications

### Calibration Analysis
- **Reliability Diagrams**: Assess probability calibration quality
- **Brier Score**: Quantitative measure of calibration performance
- **Critical for Risk Assessment**: Well-calibrated probabilities essential for regulatory applications

### Model Interpretation

**SHAP (SHapley Additive exPlanations)**
- **Global Importance**: Feature importance across all predictions
- **Local Explanations**: Individual prediction explanations
- **Chemical Interpretation**: Identify structural alerts and important molecular features

## Clustering Analysis

### Purpose
- **Chemical Space Exploration**: Understand dataset diversity
- **Similarity Analysis**: Identify structurally related compounds
- **Validation Strategy**: Inform cross-validation grouping

### Methodology
- **Tanimoto Similarity**: Standard metric for molecular similarity
- **Leader Clustering**: Efficient clustering algorithm for large molecular datasets
- **Threshold**: Configurable similarity threshold (default: 0.65)

## Reproducibility Considerations

### Random Seed Management
- **Cross-validation**: Different random seeds for each repetition
- **Model Training**: Deterministic training when possible
- **Sampling**: Consistent random states for reproducible results

### Version Control
- **Package Versions**: Pinned in requirements.txt
- **Data Provenance**: Clear documentation of data sources and preprocessing
- **Code Organization**: Modular design for reusability

## Limitations and Future Directions

### Current Limitations
1. **Dataset Size**: Limited by available experimental data
2. **Feature Representation**: Single fingerprint type in primary analysis
3. **Algorithm Scope**: Single ML algorithm demonstration
4. **Validation**: Internal cross-validation only

### Future Enhancements
1. **External Validation**: Independent test sets
2. **Ensemble Methods**: Combining multiple algorithms
3. **Deep Learning**: Graph neural networks for molecular representation
4. **Multi-task Learning**: Simultaneous prediction of multiple endpoints
5. **Uncertainty Quantification**: Conformal prediction or Bayesian approaches

## Regulatory and Safety Context

### Applications
- **Chemical Screening**: Prioritization of chemicals for experimental testing
- **Risk Assessment**: Support for regulatory decision-making
- **Drug Development**: Early identification of endocrine disruption potential
- **Chemical Design**: Guide synthesis of safer alternatives

### Validation Requirements
- **Performance Thresholds**: Balanced accuracy, sensitivity, specificity criteria
- **Applicability Domain**: Clear definition of model's valid application space
- **Uncertainty Estimation**: Confidence measures for predictions
- **External Validation**: Independent datasets for model validation

This methodology follows best practices in computational toxicology and regulatory science for developing reliable and interpretable models for chemical safety assessment.
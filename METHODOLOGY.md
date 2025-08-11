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
- **Rationale**: Handles non-linear relationships and feature interactions common in molecular data. Reasonably prone to overfitting, performed well when compared to other algorithms (see paper)
- **Robustness**: 

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

### Model Interpretation

**SHAP (SHapley Additive exPlanations)**
- **Global Importance**: Feature importance across all predictions
- **Local Explanations**: Individual prediction explanations
- **Chemical Interpretation**: Identify structural alerts and important molecular features

## Tanimoto Based Clustering Analysis

### Purpose
- **Chemical Space Exploration**: Understand dataset diversity
- **Similarity Analysis**: Identify structurally related compounds
- **Validation Strategy**: Inform cross-validation grouping

### Methodology
- **Tanimoto Similarity**: Standard metric for molecular similarity
- **Leader Clustering**: Efficient clustering algorithm for large molecular datasets
- **Threshold**: Configurable similarity threshold (default: 0.65)

## Regulatory and Safety Context

### Applications
- **Chemical Screening**: Prioritization of chemicals for experimental testing
- **Risk Assessment**: Support for regulatory decision-making
- **Drug Development**: Early identification of endocrine disruption potential
- **Chemical Design**: Guide synthesis of safer alternatives
- **Uncertainty Estimation**: Confidence measures for predictions via Bayesian inference
- **External Validation**: Model validation on CoMPARA dataset

### Validation Requirements
- **Applicability Domain**: Clear definition of model's valid application space

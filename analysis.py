"""
Analysis and Evaluation Tools for Androgen Receptor Prediction Models

This module provides classes and functions for evaluating binary classification
models in the context of chemical toxicity prediction, with emphasis on metrics
relevant to regulatory and screening applications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report
from sklearn.calibration import calibration_curve


def calc_npv(tpr: float, tnr: float, prevalence: float) -> float:
    """
    Calculate Negative Predictive Value from sensitivity, specificity, and prevalence.
    
    Parameters
    ----------
    tpr : float
        True Positive Rate (sensitivity)
    tnr : float  
        True Negative Rate (specificity)
    prevalence : float
        Prevalence of positive class in population
        
    Returns
    -------
    float
        Negative Predictive Value
    """
    return tnr * (1-prevalence) / ((tnr * (1-prevalence)) + (1 - tpr) * prevalence)


def calc_precision(tpr: float, tnr: float, prevalence: float) -> float:
    """
    Calculate Precision (PPV) from sensitivity, specificity, and prevalence.
    
    Parameters
    ----------
    tpr : float
        True Positive Rate (sensitivity)
    tnr : float
        True Negative Rate (specificity)  
    prevalence : float
        Prevalence of positive class in population
        
    Returns
    -------
    float
        Precision (Positive Predictive Value)
    """
    return tpr * (prevalence) / ((tpr * (prevalence)) + (1 - tnr) * (1-prevalence))


def calc_mcc(tpr: float, tnr: float, prevalence: float) -> float:
    """
    Calculate Matthews Correlation Coefficient from sensitivity, specificity, and prevalence.
    
    The MCC is a robust metric for binary classification that accounts for class
    imbalance and all elements of the confusion matrix.
    
    Parameters
    ----------
    tpr : float
        True Positive Rate (sensitivity)
    tnr : float
        True Negative Rate (specificity)
    prevalence : float
        Prevalence of positive class in population
        
    Returns
    -------
    float
        Matthews Correlation Coefficient (-1 to 1)
    """
    ppv = calc_precision(tpr, tnr, prevalence)
    npv = calc_npv(tpr, tnr, prevalence)
    fdr = 1 - ppv
    fnr = 1 - tpr
    fpr = 1 - tnr
    FOR = 1 - npv
    return np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fdr * fnr * fpr * FOR)


class Predictions:
    """
    Analysis class for binary classification predictions.
    
    Provides comprehensive evaluation metrics and visualizations for binary
    classification models, particularly suited for chemical toxicity prediction.
    
    Parameters
    ----------
    y : array-like
        True labels
    y_hat : array-like  
        Predicted labels
    y_hat_proba : array-like, shape (n_samples, 2)
        Predicted probabilities for each class
    **kwargs
        Additional keyword arguments
        
    Attributes
    ----------
    y : array-like
        True labels
    y_hat : array-like
        Predicted labels  
    y_hat_proba : array-like
        Predicted probabilities
    """

    def __init__(self, y, y_hat, y_hat_proba, **kwargs):
        self.y = y
        self.y_hat = y_hat
        self.y_hat_proba = y_hat_proba

    @property
    def prevalence(self) -> float:
        """
        Calculate prevalence of positive class ('inhibitor').
        
        Returns
        -------
        float
            Fraction of samples that are inhibitors
        """
        return (self.y == "inhibitor").mean()

    def plot_all_metrics(self):
        
        fpr, tpr, thresholds = self.get_roc_curve()

        ba_by_threshold = pd.Series((tpr + (1 - fpr)) / 2, index=thresholds)
        ba_by_threshold.plot(label="balanced accuracy")

        ppv_by_threshold = pd.Series(calc_precision(tpr, (1-fpr), self.prevalence), index=thresholds)
        ppv_by_threshold.plot(label="PPV")

        npv_by_threshold = pd.Series(calc_npv(tpr, (1-fpr), self.prevalence), index=thresholds)
        npv_by_threshold.plot(label="NPV")

        mcc_by_threshold = pd.Series((calc_mcc(tpr, (1-fpr), self.prevalence) + 1) / 2, index=thresholds)
        mcc_by_threshold.plot(label="(MCC + 1) / 2")


        plt.plot(thresholds, tpr, label='tpr')
        plt.plot(thresholds, (1 - fpr), label='tnr')
        plt.xlabel('Threshold')
        plt.ylabel('Metric')
        plt.legend()
        plt.axvline(self.prevalence, linestyle="--", c="k")

    def get_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y, self.y_hat_proba[:,1], pos_label="inhibitor")
        return fpr, tpr, thresholds

    def _plot_roc_curve(self):
        fpr, tpr, _ = self.get_roc_curve()
        plt.plot(fpr, tpr)

    def plot_roc_curve(self, plot_thresholds=[]):
        self._plot_roc_curve()

        # fpr_list = np.array([0, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
        # tpr_list = np.array([0, 0.2, 0.4, 0.7, 0.85, 0.93, 0.99, 1.0])

        # plt.plot(fpr_list, tpr_list, marker="o", c="k")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    def _plot_calibration_curve(self):
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y, self.y_hat_proba[:, 1], 
                                    n_bins=10, strategy='uniform', pos_label="inhibitor")
        plt.plot(mean_predicted_value, fraction_of_positives, '.--')

    def plot_calibration_curve(self):
        self._plot_calibration_curve()
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean predicted value")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve")

    def _plot_npv_tnr_curve(self):
        npv, tnr, _ = precision_recall_curve(self.y, self.y_hat_proba[:,0], pos_label="inactive")
        plt.plot(tnr, npv)

    def plot_npv_tnr_curve(self):
        self._plot_npv_tnr_curve()

        plt.xlabel('True Negative Rate')
        plt.ylabel('Negative Predictive Value')
        plt.axhline(1 - self.prevalence, linestyle="--", c="k")

    def _plot_BA(self):
        fpr, tpr, thresholds = roc_curve(self.y, self.y_hat_proba[:,1], pos_label="inhibitor")
        ba_by_threshold = pd.Series((tpr + (1 - fpr)) / 2, index=thresholds)
        ba_by_threshold.plot(label="balanced accuracy")

    def plot_BA(self):
        self._plot_BA()
        plt.xlabel('Threshold')
        plt.ylabel('Balanced Accuracy')
        # plt.legend()
        plt.axvline(self.prevalence, linestyle="--", c="k")
        plt.ylim(0.45, 1.05)

    def _plot_precision_recall_curve(self):
        ppv, tpr, _ = precision_recall_curve(self.y, self.y_hat_proba[:, 1], pos_label="inhibitor")
        plt.plot(tpr, ppv)

    def plot_precision_recall_curve(self):
        self._plot_precision_recall_curve()
        plt.xlabel('True Positve Rate')
        plt.ylabel('Precision')

        plt.axhline(self.prevalence, linestyle="--", c="k")


class CV(Predictions):

    def __init__(self, splits, plot_thresholds=[]):
        self.splits = splits
        self.folds = [Predictions(**x) for x in self.splits]
        self.plot_thresholds = plot_thresholds

    @property
    def prevalence(self):
        return np.mean([f.prevalence for f in self.folds])

    def _plot_roc_curve(self):
        for f in self.folds:
            f._plot_roc_curve()

        fpr_list = []
        tpr_list = []

        for t in self.plot_thresholds:
            all_y = np.concatenate([x.y for x in self.folds]) 
            all_y_hat_proba = np.concatenate([x.y_hat_proba for x in self.folds])
            cr = classification_report(
                all_y == "inhibitor", 
                all_y_hat_proba[:, 1] > t, 
                output_dict=True)
            fpr = 1 - cr['False']["recall"]
            tpr = cr['True']["recall"]
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        plt.plot(fpr_list, tpr_list, marker="o", c="k")

    def _plot_calibration_curve(self):
        for f in self.folds:
            f._plot_calibration_curve()

    def _plot_npv_tnr_curve(self):
        for f in self.folds:
            f._plot_npv_tnr_curve()

    def _plot_BA(self):
        for f in self.folds:
            f._plot_BA()

    def _plot_precision_recall_curve(self):
        for f in self.folds:
            f._plot_precision_recall_curve()

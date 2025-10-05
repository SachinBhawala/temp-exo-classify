import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class ModelCalibration:
    def __init__(self, model, X_test, y_test, class_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_proba = model.predict_proba(X_test)
        
    def plot_calibration_curve(self, class_idx=0, n_bins=10):
        """Plot calibration curve for a specific class"""
        # Get probabilities for the target class
        y_prob = self.y_proba[:, class_idx]
        
        # Binarize the labels for this class
        y_binary = (self.y_test == class_idx).astype(int)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_binary, y_prob)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
               label=f'{self.class_names[class_idx]} (Brier: {brier_score:.3f})')
        
        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {self.class_names[class_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, brier_score
    
    def plot_all_calibration_curves(self, n_bins=10):
        """Plot calibration curves for all classes"""
        n_classes = len(self.class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
        
        if n_classes == 1:
            axes = [axes]
        
        brier_scores = {}
        
        for class_idx, ax in enumerate(axes):
            y_prob = self.y_proba[:, class_idx]
            y_binary = (self.y_test == class_idx).astype(int)
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob, n_bins=n_bins, strategy='uniform'
            )
            
            brier_score = brier_score_loss(y_binary, y_prob)
            brier_scores[self.class_names[class_idx]] = brier_score
            
            ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                   label=f'Brier: {brier_score:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'{self.class_names[class_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, brier_scores
    
    def find_optimal_threshold(self, class_idx=0, metric='f1'):
        """Find optimal probability threshold for a specific class"""
        y_prob = self.y_proba[:, class_idx]
        y_binary = (self.y_test == class_idx).astype(int)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_binary, y_prob)
        
        # Calculate F1 score for each threshold
        f1_scores = []
        for i in range(len(thresholds)):
            y_pred_thresh = (y_prob >= thresholds[i]).astype(int)
            if metric == 'f1':
                score = f1_score(y_binary, y_pred_thresh, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_binary, y_pred_thresh, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_binary, y_pred_thresh, zero_division=0)
            f1_scores.append(score)
        
        # Find optimal threshold
        if len(f1_scores) > 0:
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_score = f1_scores[optimal_idx]
        else:
            optimal_threshold = 0.5
            optimal_score = 0.0
        
        return {
            'threshold': optimal_threshold,
            'score': optimal_score,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'scores': f1_scores
        }
    
    def plot_threshold_analysis(self, class_idx=0):
        """Plot threshold analysis for precision, recall, and F1"""
        y_prob = self.y_proba[:, class_idx]
        y_binary = (self.y_test == class_idx).astype(int)
        
        # Calculate metrics for different thresholds
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores_list = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            prec = precision_score(y_binary, y_pred, zero_division=0)
            rec = recall_score(y_binary, y_pred, zero_division=0)
            f1 = f1_score(y_binary, y_pred, zero_division=0)
            
            precisions.append(prec)
            recalls.append(rec)
            f1_scores_list.append(f1)
        
        # Find optimal F1 threshold
        optimal_idx = np.argmax(f1_scores_list)
        optimal_threshold = thresholds[optimal_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores_list, label='F1-Score', linewidth=2)
        ax.axvline(optimal_threshold, color='red', linestyle='--', 
                  label=f'Optimal threshold: {optimal_threshold:.3f}', linewidth=2)
        ax.axvline(0.5, color='gray', linestyle=':', 
                  label='Default threshold: 0.5', linewidth=1)
        
        ax.set_xlabel('Probability Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Threshold Analysis - {self.class_names[class_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        return fig, optimal_threshold
    
    def get_threshold_recommendations(self):
        """Get threshold recommendations for all classes"""
        recommendations = {}
        
        for class_idx in range(len(self.class_names)):
            class_name = self.class_names[class_idx]
            
            # Find optimal thresholds for different metrics
            f1_result = self.find_optimal_threshold(class_idx, metric='f1')
            prec_result = self.find_optimal_threshold(class_idx, metric='precision')
            rec_result = self.find_optimal_threshold(class_idx, metric='recall')
            
            recommendations[class_name] = {
                'f1_threshold': f1_result['threshold'],
                'f1_score': f1_result['score'],
                'precision_threshold': prec_result['threshold'],
                'recall_threshold': rec_result['threshold']
            }
        
        return recommendations

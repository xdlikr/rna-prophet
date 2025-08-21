import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class RegressionEvaluator:
    """Comprehensive evaluation of regression models."""
    
    def __init__(self, target_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            target_names: Names of target variables
        """
        self.target_names = target_names
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of targets (overrides instance names)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Handle single target case
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        target_names = target_names or self.target_names or [f'target_{i}' for i in range(y_true.shape[1])]
        
        results = {
            'overall': self._calculate_overall_metrics(y_true, y_pred),
            'per_target': {},
            'target_names': target_names,
            'n_samples': y_true.shape[0],
            'n_targets': y_true.shape[1]
        }
        
        # Calculate metrics for each target
        for i, target_name in enumerate(target_names):
            if i < y_true.shape[1]:
                results['per_target'][target_name] = self._calculate_target_metrics(
                    y_true[:, i], y_pred[:, i]
                )
        
        return results
    
    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate overall metrics across all targets."""
        metrics = {}
        
        # Mean metrics across all targets
        r2_scores = []
        mse_scores = []
        mae_scores = []
        
        for i in range(y_true.shape[1]):
            r2_scores.append(r2_score(y_true[:, i], y_pred[:, i]))
            mse_scores.append(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae_scores.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        
        metrics['mean_r2'] = np.mean(r2_scores)
        metrics['std_r2'] = np.std(r2_scores)
        metrics['mean_mse'] = np.mean(mse_scores)
        metrics['mean_mae'] = np.mean(mae_scores)
        
        # Overall metrics treating all predictions as one vector
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        metrics['overall_r2'] = r2_score(y_true_flat, y_pred_flat)
        metrics['overall_mse'] = mean_squared_error(y_true_flat, y_pred_flat)
        metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        metrics['overall_mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
        
        return metrics
    
    def _calculate_target_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single target."""
        metrics = {}
        
        # Basic regression metrics
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Correlation metrics
        try:
            metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true, y_pred)
            metrics['spearman_r'], metrics['spearman_p'] = spearmanr(y_true, y_pred)
        except:
            metrics['pearson_r'] = metrics['pearson_p'] = np.nan
            metrics['spearman_r'] = metrics['spearman_p'] = np.nan
        
        # Mean absolute percentage error (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape if np.isfinite(mape) else np.nan
        
        # Relative metrics
        metrics['mean_true'] = np.mean(y_true)
        metrics['std_true'] = np.std(y_true)
        metrics['mean_pred'] = np.mean(y_pred)
        metrics['std_pred'] = np.std(y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_residual'] = np.max(np.abs(residuals))
        
        return metrics
    
    def create_evaluation_report(self, 
                               evaluation_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """
        Create a human-readable evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_predictions
            save_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("RNA PROPERTY PREDICTION - EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Samples: {evaluation_results['n_samples']}")
        report_lines.append(f"Targets: {evaluation_results['n_targets']}")
        report_lines.append(f"Target names: {', '.join(evaluation_results['target_names'])}")
        report_lines.append("")
        
        # Overall metrics
        overall = evaluation_results['overall']
        report_lines.append("OVERALL PERFORMANCE")
        report_lines.append("-" * 30)
        report_lines.append(f"Mean R²:       {overall['mean_r2']:.4f} (±{overall['std_r2']:.4f})")
        report_lines.append(f"Overall R²:    {overall['overall_r2']:.4f}")
        report_lines.append(f"Overall RMSE:  {overall['overall_rmse']:.4f}")
        report_lines.append(f"Overall MAE:   {overall['overall_mae']:.4f}")
        report_lines.append("")
        
        # Per-target metrics
        report_lines.append("PER-TARGET PERFORMANCE")
        report_lines.append("-" * 30)
        
        for target_name, metrics in evaluation_results['per_target'].items():
            report_lines.append(f"\n{target_name.upper()}:")
            report_lines.append(f"  R²:               {metrics['r2']:.4f}")
            report_lines.append(f"  RMSE:             {metrics['rmse']:.4f}")
            report_lines.append(f"  MAE:              {metrics['mae']:.4f}")
            report_lines.append(f"  Pearson r:        {metrics['pearson_r']:.4f}")
            report_lines.append(f"  Spearman r:       {metrics['spearman_r']:.4f}")
            if not np.isnan(metrics['mape']):
                report_lines.append(f"  MAPE:             {metrics['mape']:.2f}%")
        
        # Performance assessment
        report_lines.append("\n" + "=" * 60)
        report_lines.append("PERFORMANCE ASSESSMENT")
        report_lines.append("=" * 60)
        
        mean_r2 = overall['mean_r2']
        if mean_r2 > 0.8:
            assessment = "EXCELLENT - Very strong predictive performance"
        elif mean_r2 > 0.6:
            assessment = "GOOD - Strong predictive performance"
        elif mean_r2 > 0.4:
            assessment = "MODERATE - Reasonable predictive performance"
        elif mean_r2 > 0.2:
            assessment = "POOR - Weak predictive performance"
        else:
            assessment = "VERY POOR - Little to no predictive power"
        
        report_lines.append(f"Overall Assessment: {assessment}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDations:")
        if mean_r2 < 0.5:
            report_lines.append("- Consider feature engineering or additional data")
            report_lines.append("- Check for data quality issues")
            report_lines.append("- Try different model architectures")
        
        if overall['std_r2'] > 0.2:
            report_lines.append("- High variance across targets - consider target-specific models")
        
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_predictions(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        target_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create prediction plots for visualization.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of targets
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        target_names = target_names or self.target_names or [f'Target {i+1}' for i in range(y_true.shape[1])]
        n_targets = y_true.shape[1]
        
        # Create subplots
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_targets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_targets == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i in range(n_targets):
            ax = axes[i]
            
            # Scatter plot of true vs predicted
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Calculate R²
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{target_names[i]}\nR² = {r2:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_targets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      target_names: Optional[List[str]] = None,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create residual plots for model diagnostics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_names: Names of targets
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        target_names = target_names or self.target_names or [f'Target {i+1}' for i in range(y_true.shape[1])]
        n_targets = y_true.shape[1]
        
        # Create subplots
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_targets == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_targets == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i in range(n_targets):
            ax = axes[i]
            
            residuals = y_true[:, i] - y_pred[:, i]
            
            # Residual plot
            ax.scatter(y_pred[:, i], residuals, alpha=0.6, s=30)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{target_names[i]} - Residuals')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_targets, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
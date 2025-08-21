import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from ..models.pipeline import RNAPredictionPipeline
from ..evaluation.metrics import RegressionEvaluator


class RNAPropertyPredictor:
    """High-level predictor for RNA property prediction with sequence-specific insights."""
    
    def __init__(self, pipeline: Optional[RNAPredictionPipeline] = None):
        """
        Initialize RNA property predictor.
        
        Args:
            pipeline: Fitted RNAPredictionPipeline or None to load later
        """
        self.pipeline = pipeline
        self.evaluator = RegressionEvaluator()
        self.prediction_results = None
        self.evaluation_results = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger(__name__)
    
    def load_pipeline(self, pipeline_path: str) -> 'RNAPropertyPredictor':
        """Load a fitted pipeline from file."""
        self.pipeline = RNAPredictionPipeline()
        self.pipeline.load_pipeline(pipeline_path)
        self.logger.info(f"Pipeline loaded from {pipeline_path}")
        return self
    
    def predict_sequences(self, 
                         X: pd.DataFrame,
                         include_features: bool = True,
                         include_probabilities: bool = False) -> pd.DataFrame:
        """
        Predict properties for RNA sequences.
        
        Args:
            X: DataFrame with sequences and covariates
            include_features: Whether to include input features in output
            include_probabilities: Whether to include prediction confidence (not implemented)
            
        Returns:
            DataFrame with predictions and optional features
        """
        if self.pipeline is None or not self.pipeline.is_fitted_:
            raise ValueError("Pipeline must be loaded and fitted before prediction")
        
        self.logger.info(f"Predicting properties for {len(X)} sequences")
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        
        # Create output DataFrame
        result_df = X.copy() if include_features else pd.DataFrame(index=X.index)
        
        # Add predictions with clear naming
        for col in predictions.columns:
            result_df[f'predicted_{col}'] = predictions[col]
        
        # Add sequence-specific insights
        result_df = self._add_sequence_insights(result_df, X)
        
        self.prediction_results = result_df
        self.logger.info("Prediction completed")
        
        return result_df
    
    def _add_sequence_insights(self, result_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Add sequence-specific insights to predictions."""
        if self.pipeline.sequence_column not in X.columns:
            return result_df
        
        sequences = X[self.pipeline.sequence_column]
        
        # Add basic sequence properties
        result_df['sequence_length'] = sequences.str.len()
        result_df['gc_content'] = sequences.apply(self._calculate_gc_content)
        
        # Add prediction confidence categories
        if hasattr(self, 'prediction_results') and self.prediction_results is not None:
            # This would be enhanced with actual uncertainty estimation
            result_df['prediction_confidence'] = 'medium'  # Placeholder
        
        return result_df
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence."""
        if not sequence:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def evaluate_predictions(self, 
                           X: pd.DataFrame, 
                           y_true: pd.DataFrame,
                           save_plots: bool = True,
                           plot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate predictions against true values.
        
        Args:
            X: Input features
            y_true: True target values
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
            
        Returns:
            Evaluation results dictionary
        """
        if self.pipeline is None or not self.pipeline.is_fitted_:
            raise ValueError("Pipeline must be loaded and fitted before evaluation")
        
        self.logger.info("Evaluating model predictions")
        
        # Make predictions
        y_pred = self.pipeline.predict(X)
        
        # Ensure consistent column order
        target_columns = self.pipeline.target_names_
        y_true_aligned = y_true[target_columns]
        y_pred_aligned = y_pred[target_columns]
        
        # Evaluate predictions
        self.evaluation_results = self.evaluator.evaluate_predictions(
            y_true_aligned.values, 
            y_pred_aligned.values,
            target_names=target_columns
        )
        
        # Save plots if requested
        if save_plots:
            plot_dir = plot_dir or "outputs/plots"
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
            
            # Prediction plots
            pred_plot_path = Path(plot_dir) / "predictions_vs_true.png"
            self.evaluator.plot_predictions(
                y_true_aligned.values,
                y_pred_aligned.values,
                target_names=target_columns,
                save_path=str(pred_plot_path)
            )
            
            # Residual plots
            residual_plot_path = Path(plot_dir) / "residual_plots.png"
            self.evaluator.plot_residuals(
                y_true_aligned.values,
                y_pred_aligned.values,
                target_names=target_columns,
                save_path=str(residual_plot_path)
            )
            
            self.logger.info(f"Evaluation plots saved to {plot_dir}")
        
        return self.evaluation_results
    
    def generate_sequence_report(self, 
                               X: pd.DataFrame,
                               top_n: int = 10,
                               sort_by: str = 'predicted_yield') -> pd.DataFrame:
        """
        Generate a sequence-specific report with rankings and insights.
        
        Args:
            X: Input sequences and features
            top_n: Number of top sequences to highlight
            sort_by: Column to sort by for rankings
            
        Returns:
            DataFrame with sequence report
        """
        if self.prediction_results is None:
            self.predict_sequences(X)
        
        report_df = self.prediction_results.copy()
        
        # Add rankings
        if sort_by in report_df.columns:
            report_df['rank'] = report_df[sort_by].rank(ascending=False, method='min')
            report_df = report_df.sort_values(sort_by, ascending=False)
        
        # Add insights for top sequences
        report_df['insights'] = self._generate_sequence_insights(report_df)
        
        # Highlight top performers
        report_df['top_performer'] = report_df['rank'] <= top_n if 'rank' in report_df.columns else False
        
        self.logger.info(f"Generated sequence report for {len(report_df)} sequences")
        
        return report_df
    
    def _generate_sequence_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate textual insights for each sequence."""
        insights = []
        
        for idx, row in df.iterrows():
            insight_parts = []
            
            # Length insights
            if 'sequence_length' in row:
                length = row['sequence_length']
                if length > 2000:
                    insight_parts.append("long sequence")
                elif length < 500:
                    insight_parts.append("short sequence")
            
            # GC content insights
            if 'gc_content' in row:
                gc = row['gc_content']
                if gc > 0.6:
                    insight_parts.append("high GC content")
                elif gc < 0.3:
                    insight_parts.append("low GC content")
            
            # Prediction insights
            pred_cols = [col for col in row.index if col.startswith('predicted_')]
            if pred_cols:
                for col in pred_cols:
                    value = row[col]
                    property_name = col.replace('predicted_', '')
                    
                    if property_name == 'yield' and value > 0.8:
                        insight_parts.append("high yield predicted")
                    elif property_name == 'dsRNA_percent' and value > 10:
                        insight_parts.append("high dsRNA formation risk")
                    elif property_name == 'expression' and value > 1.5:
                        insight_parts.append("strong expression expected")
            
            insights.append("; ".join(insight_parts) if insight_parts else "standard profile")
        
        return insights
    
    def get_feature_importance_insights(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get feature importance insights from the fitted model.
        
        Args:
            top_n: Number of top features to highlight
            
        Returns:
            Dictionary with feature importance insights
        """
        if self.pipeline is None or not self.pipeline.is_fitted_:
            raise ValueError("Pipeline must be loaded and fitted")
        
        # Get feature importance
        importance_df = self.pipeline.get_feature_importance()
        
        insights = {
            'overall_top_features': {},
            'per_target_insights': {},
            'feature_categories': self._categorize_features(importance_df.index.tolist())
        }
        
        # Overall top features (mean importance across targets)
        mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        insights['overall_top_features'] = mean_importance.head(top_n).to_dict()
        
        # Per-target insights
        for target in importance_df.columns:
            target_importance = importance_df[target].sort_values(ascending=False)
            insights['per_target_insights'][target] = {
                'top_features': target_importance.head(top_n).to_dict(),
                'top_feature_types': self._analyze_feature_types(target_importance.head(top_n).index.tolist())
            }
        
        return insights
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'embedding_features': [],
            'structure_features': [],
            'covariate_features': []
        }
        
        for feature in feature_names:
            if feature.startswith('PC') or 'embedding' in feature.lower():
                categories['embedding_features'].append(feature)
            elif any(struct_term in feature.lower() for struct_term in ['mfe', 'hairpin', 'stem', 'gc', 'bulge', 'loop']):
                categories['structure_features'].append(feature)
            else:
                categories['covariate_features'].append(feature)
        
        return categories
    
    def _analyze_feature_types(self, top_features: List[str]) -> Dict[str, int]:
        """Analyze what types of features are most important."""
        feature_types = {'embedding': 0, 'structure': 0, 'covariate': 0}
        
        for feature in top_features:
            if feature.startswith('PC') or 'embedding' in feature.lower():
                feature_types['embedding'] += 1
            elif any(struct_term in feature.lower() for struct_term in ['mfe', 'hairpin', 'stem', 'gc', 'bulge', 'loop']):
                feature_types['structure'] += 1
            else:
                feature_types['covariate'] += 1
        
        return feature_types
    
    def save_predictions(self, 
                        filepath: str,
                        include_evaluation: bool = True) -> None:
        """
        Save prediction results to file.
        
        Args:
            filepath: Path to save results
            include_evaluation: Whether to include evaluation metrics
        """
        if self.prediction_results is None:
            raise ValueError("No predictions available to save")
        
        # Save predictions
        self.prediction_results.to_csv(filepath, index=False)
        
        # Save evaluation results if available
        if include_evaluation and self.evaluation_results is not None:
            eval_filepath = filepath.replace('.csv', '_evaluation.json')
            import json
            with open(eval_filepath, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                eval_data = self._make_json_serializable(self.evaluation_results)
                json.dump(eval_data, f, indent=2)
        
        self.logger.info(f"Predictions saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
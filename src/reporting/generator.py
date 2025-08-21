import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from ..prediction.predictor import RNAPropertyPredictor
from ..analysis.importance import FeatureImportanceAnalyzer
from ..evaluation.metrics import RegressionEvaluator


class ReportGenerator:
    """Generates comprehensive reports for RNA property prediction results."""
    
    def __init__(self):
        """Initialize report generator."""
        self.predictor = None
        self.importance_analyzer = FeatureImportanceAnalyzer()
        self.evaluator = RegressionEvaluator()
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self,
                                    predictor: RNAPropertyPredictor,
                                    X_test: pd.DataFrame,
                                    y_test: Optional[pd.DataFrame] = None,
                                    output_dir: str = "outputs/reports",
                                    report_name: str = "rna_prediction_report") -> Dict[str, str]:
        """
        Generate a comprehensive report including predictions, evaluation, and insights.
        
        Args:
            predictor: Fitted RNAPropertyPredictor
            X_test: Test features
            y_test: Test targets (optional, for evaluation)
            output_dir: Directory to save reports
            report_name: Base name for report files
            
        Returns:
            Dictionary with paths to generated report files
        """
        self.predictor = predictor
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating comprehensive report in {output_dir}")
        
        report_files = {}
        
        # Generate predictions
        predictions_df = predictor.predict_sequences(X_test)
        
        # Save prediction results
        pred_file = output_path / f"{report_name}_predictions.csv"
        predictions_df.to_csv(pred_file, index=False)
        report_files['predictions'] = str(pred_file)
        
        # Generate sequence-specific report
        sequence_report = predictor.generate_sequence_report(X_test)
        seq_report_file = output_path / f"{report_name}_sequence_analysis.csv"
        sequence_report.to_csv(seq_report_file, index=False)
        report_files['sequence_analysis'] = str(seq_report_file)
        
        # Feature importance analysis
        importance_df = predictor.pipeline.get_feature_importance()
        importance_analysis = self.importance_analyzer.analyze_importance(importance_df)
        
        # Save feature importance data
        importance_file = output_path / f"{report_name}_feature_importance.csv"
        importance_df.to_csv(importance_file)
        report_files['feature_importance'] = str(importance_file)
        
        # Generate feature importance report
        importance_report = self.importance_analyzer.create_importance_report(importance_analysis)
        importance_report_file = output_path / f"{report_name}_importance_analysis.txt"
        with open(importance_report_file, 'w') as f:
            f.write(importance_report)
        report_files['importance_report'] = str(importance_report_file)
        
        # Generate visualizations
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Feature importance heatmap
        heatmap_file = plots_dir / f"{report_name}_importance_heatmap.png"
        self.importance_analyzer.plot_importance_heatmap(save_path=str(heatmap_file))
        report_files['importance_heatmap'] = str(heatmap_file)
        
        # Category importance plot
        category_plot_file = plots_dir / f"{report_name}_category_importance.png"
        self.importance_analyzer.plot_category_importance(save_path=str(category_plot_file))
        report_files['category_plot'] = str(category_plot_file)
        
        # Evaluation if ground truth is provided
        if y_test is not None:
            evaluation_results = predictor.evaluate_predictions(X_test, y_test, 
                                                               save_plots=True, 
                                                               plot_dir=str(plots_dir))
            
            # Save evaluation results
            eval_file = output_path / f"{report_name}_evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(predictor._make_json_serializable(evaluation_results), f, indent=2)
            report_files['evaluation'] = str(eval_file)
            
            # Generate evaluation report
            eval_report = self.evaluator.create_evaluation_report(evaluation_results)
            eval_report_file = output_path / f"{report_name}_evaluation_report.txt"
            with open(eval_report_file, 'w') as f:
                f.write(eval_report)
            report_files['evaluation_report'] = str(eval_report_file)
        
        # Generate executive summary
        summary_file = output_path / f"{report_name}_executive_summary.html"
        self._generate_executive_summary(
            predictions_df, importance_analysis, 
            evaluation_results if y_test is not None else None,
            str(summary_file)
        )
        report_files['executive_summary'] = str(summary_file)
        
        # Generate metadata
        metadata = self._create_report_metadata(X_test, y_test, report_files)
        metadata_file = output_path / f"{report_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        report_files['metadata'] = str(metadata_file)
        
        self.logger.info(f"Comprehensive report generated with {len(report_files)} files")
        return report_files
    
    def _generate_executive_summary(self,
                                  predictions_df: pd.DataFrame,
                                  importance_analysis: Dict[str, Any],
                                  evaluation_results: Optional[Dict[str, Any]],
                                  save_path: str) -> None:
        """Generate HTML executive summary."""
        html_content = self._create_html_template()
        
        # Overview section
        overview = f"""
        <h2>Executive Summary</h2>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Number of Sequences Analyzed:</strong> {len(predictions_df)}</p>
        <p><strong>Targets Predicted:</strong> {', '.join([col.replace('predicted_', '') for col in predictions_df.columns if col.startswith('predicted_')])}</p>
        """
        
        # Performance summary if evaluation available
        performance_section = ""
        if evaluation_results:
            overall_r2 = evaluation_results['overall']['mean_r2']
            performance_rating = self._get_performance_rating(overall_r2)
            
            performance_section = f"""
            <h3>Model Performance</h3>
            <p><strong>Overall R² Score:</strong> {overall_r2:.3f} ({performance_rating})</p>
            <div class="performance-bar">
                <div class="performance-fill" style="width: {overall_r2 * 100}%"></div>
            </div>
            """
        
        # Top insights
        insights_section = self._generate_insights_section(predictions_df, importance_analysis)
        
        # Key findings
        findings_section = self._generate_findings_section(importance_analysis)
        
        # Combine all sections
        full_content = html_content.format(
            overview=overview,
            performance=performance_section,
            insights=insights_section,
            findings=findings_section
        )
        
        with open(save_path, 'w') as f:
            f.write(full_content)
    
    def _create_html_template(self) -> str:
        """Create HTML template for executive summary."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RNA Property Prediction - Executive Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .performance-bar {{ width: 100%; height: 20px; background-color: #ecf0f1; border-radius: 10px; }}
                .performance-fill {{ height: 100%; background-color: #27ae60; border-radius: 10px; }}
                .insight-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
                .finding-box {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>RNA Property Prediction Report</h1>
            {overview}
            {performance}
            {insights}
            {findings}
        </body>
        </html>
        """
    
    def _get_performance_rating(self, r2_score: float) -> str:
        """Get performance rating based on R² score."""
        if r2_score > 0.8:
            return "Excellent"
        elif r2_score > 0.6:
            return "Good" 
        elif r2_score > 0.4:
            return "Fair"
        elif r2_score > 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_insights_section(self, 
                                 predictions_df: pd.DataFrame,
                                 importance_analysis: Dict[str, Any]) -> str:
        """Generate insights section for HTML report."""
        insights = []
        
        # Prediction insights
        pred_columns = [col for col in predictions_df.columns if col.startswith('predicted_')]
        
        for col in pred_columns:
            if col in predictions_df.columns:
                target_name = col.replace('predicted_', '')
                values = predictions_df[col]
                
                insights.append(f"""
                <div class="insight-box">
                    <strong>{target_name.title()} Predictions:</strong>
                    Mean: {values.mean():.3f}, 
                    Range: {values.min():.3f} - {values.max():.3f},
                    Top 10% average: {values.quantile(0.9):.3f}
                </div>
                """)
        
        # Feature importance insights
        if 'top_features' in importance_analysis:
            top_features = importance_analysis['top_features'].get('overall', {}).get('features', [])[:5]
            if top_features:
                insights.append(f"""
                <div class="insight-box">
                    <strong>Most Important Features:</strong>
                    {', '.join(top_features)}
                </div>
                """)
        
        return f"<h3>Key Insights</h3>{''.join(insights)}"
    
    def _generate_findings_section(self, importance_analysis: Dict[str, Any]) -> str:
        """Generate findings section for HTML report."""
        findings = []
        
        # Category analysis findings
        if 'category_analysis' in importance_analysis:
            cat_analysis = importance_analysis['category_analysis']
            
            for category, data in cat_analysis.items():
                avg_contribution = np.mean(list(data.get('contribution_percentage', {}).values()))
                findings.append(f"""
                <div class="finding-box">
                    <strong>{category.replace('_', ' ').title()}:</strong>
                    Average contribution of {avg_contribution:.1f}% across targets
                </div>
                """)
        
        # Stability findings
        if 'stability_analysis' in importance_analysis:
            stability = importance_analysis['stability_analysis']
            overall_stability = stability.get('overall_stability', 0)
            
            stability_interpretation = "high" if overall_stability < 0.5 else "moderate" if overall_stability < 1.0 else "low"
            findings.append(f"""
            <div class="finding-box">
                <strong>Feature Stability:</strong>
                {stability_interpretation.title()} stability across targets (CV: {overall_stability:.2f})
            </div>
            """)
        
        return f"<h3>Key Findings</h3>{''.join(findings)}"
    
    def _create_report_metadata(self,
                               X_test: pd.DataFrame,
                               y_test: Optional[pd.DataFrame],
                               report_files: Dict[str, str]) -> Dict[str, Any]:
        """Create metadata for the report."""
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'data_info': {
                'n_sequences': len(X_test),
                'n_features': len(X_test.columns),
                'sequence_column': self.predictor.pipeline.sequence_column if self.predictor else 'sequence',
                'has_ground_truth': y_test is not None
            },
            'model_info': {
                'targets': self.predictor.pipeline.target_names_ if self.predictor else [],
                'pipeline_fitted': self.predictor.pipeline.is_fitted_ if self.predictor else False
            },
            'report_files': report_files,
            'file_descriptions': {
                'predictions': 'Individual sequence predictions with features',
                'sequence_analysis': 'Sequence-specific insights and rankings',
                'feature_importance': 'Feature importance scores across targets',
                'importance_report': 'Detailed feature importance analysis',
                'evaluation': 'Model performance metrics (if ground truth available)',
                'executive_summary': 'HTML summary report for stakeholders',
                'metadata': 'Report generation metadata'
            }
        }
        
        return metadata
    
    def generate_sequence_insights_report(self,
                                        sequence_report: pd.DataFrame,
                                        save_path: str) -> None:
        """Generate detailed sequence insights report."""
        insights_lines = []
        
        insights_lines.append("SEQUENCE-SPECIFIC INSIGHTS REPORT")
        insights_lines.append("=" * 50)
        insights_lines.append(f"Total sequences analyzed: {len(sequence_report)}")
        insights_lines.append("")
        
        # Top performers analysis
        if 'rank' in sequence_report.columns:
            top_10 = sequence_report.head(10)
            insights_lines.append("TOP 10 PERFORMING SEQUENCES")
            insights_lines.append("-" * 30)
            
            for idx, row in top_10.iterrows():
                seq_id = f"Sequence_{idx}"
                rank = row.get('rank', 'N/A')
                insights = row.get('insights', 'No insights available')
                
                insights_lines.append(f"\n{seq_id} (Rank #{rank}):")
                insights_lines.append(f"  Insights: {insights}")
                
                # Add prediction values
                pred_cols = [col for col in row.index if col.startswith('predicted_')]
                for col in pred_cols:
                    target_name = col.replace('predicted_', '')
                    value = row[col]
                    insights_lines.append(f"  {target_name}: {value:.3f}")
        
        # Insights distribution
        if 'insights' in sequence_report.columns:
            insights_lines.append("\n\nINSIGHTS DISTRIBUTION")
            insights_lines.append("-" * 30)
            
            insight_counts = sequence_report['insights'].value_counts()
            for insight, count in insight_counts.head(10).items():
                percentage = count / len(sequence_report) * 100
                insights_lines.append(f"{insight}: {count} sequences ({percentage:.1f}%)")
        
        # Sequence characteristics
        if 'sequence_length' in sequence_report.columns:
            lengths = sequence_report['sequence_length']
            insights_lines.append(f"\n\nSEQUENCE CHARACTERISTICS")
            insights_lines.append("-" * 30)
            insights_lines.append(f"Length range: {lengths.min()} - {lengths.max()} nucleotides")
            insights_lines.append(f"Average length: {lengths.mean():.0f} nucleotides")
        
        if 'gc_content' in sequence_report.columns:
            gc_content = sequence_report['gc_content']
            insights_lines.append(f"GC content range: {gc_content.min():.3f} - {gc_content.max():.3f}")
            insights_lines.append(f"Average GC content: {gc_content.mean():.3f}")
        
        report_text = "\n".join(insights_lines)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Sequence insights report saved to {save_path}")
    
    def create_model_summary_card(self, 
                                evaluation_results: Dict[str, Any],
                                save_path: str) -> None:
        """Create a concise model summary card."""
        summary_lines = []
        
        summary_lines.append("╔══════════════════════════════════════╗")
        summary_lines.append("║        MODEL PERFORMANCE CARD       ║")
        summary_lines.append("╠══════════════════════════════════════╣")
        
        overall = evaluation_results.get('overall', {})
        mean_r2 = overall.get('mean_r2', 0)
        overall_rmse = overall.get('overall_rmse', 0)
        
        summary_lines.append(f"║ Overall R² Score:        {mean_r2:8.3f} ║")
        summary_lines.append(f"║ Overall RMSE:            {overall_rmse:8.3f} ║")
        summary_lines.append(f"║ Performance Rating:   {self._get_performance_rating(mean_r2):>10} ║")
        summary_lines.append("╠══════════════════════════════════════╣")
        
        # Per-target performance
        per_target = evaluation_results.get('per_target', {})
        for target, metrics in per_target.items():
            r2 = metrics.get('r2', 0)
            summary_lines.append(f"║ {target:<12} R²:     {r2:8.3f} ║")
        
        summary_lines.append("╚══════════════════════════════════════╝")
        
        card_text = "\n".join(summary_lines)
        
        with open(save_path, 'w') as f:
            f.write(card_text)
        
        self.logger.info(f"Model summary card saved to {save_path}")
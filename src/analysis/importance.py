import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path


class FeatureImportanceAnalyzer:
    """Analyzes and visualizes feature importance from RNA prediction models."""
    
    def __init__(self):
        """Initialize feature importance analyzer."""
        self.importance_data = None
        self.feature_categories = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_importance(self, 
                          importance_df: pd.DataFrame,
                          feature_categories: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of feature importance.
        
        Args:
            importance_df: DataFrame with features as index and targets as columns
            feature_categories: Optional categorization of features
            
        Returns:
            Dictionary containing analysis results
        """
        self.importance_data = importance_df.copy()
        self.feature_categories = feature_categories or self._auto_categorize_features(importance_df.index.tolist())
        
        self.logger.info(f"Analyzing importance for {len(importance_df)} features across {len(importance_df.columns)} targets")
        
        analysis = {
            'summary_stats': self._calculate_summary_stats(),
            'top_features': self._identify_top_features(),
            'category_analysis': self._analyze_feature_categories(),
            'target_specific': self._analyze_target_specific_importance(),
            'correlation_analysis': self._analyze_feature_correlations(),
            'stability_analysis': self._analyze_importance_stability()
        }
        
        return analysis
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for feature importance."""
        stats = {}
        
        # Overall statistics
        all_importance = self.importance_data.values.flatten()
        stats['overall'] = {
            'mean': np.mean(all_importance),
            'std': np.std(all_importance),
            'min': np.min(all_importance),
            'max': np.max(all_importance),
            'median': np.median(all_importance),
            'non_zero_features': np.sum(all_importance > 0),
            'total_features': len(all_importance)
        }
        
        # Per-target statistics
        stats['per_target'] = {}
        for target in self.importance_data.columns:
            target_importance = self.importance_data[target]
            stats['per_target'][target] = {
                'mean': target_importance.mean(),
                'std': target_importance.std(),
                'top_feature': target_importance.idxmax(),
                'top_importance': target_importance.max(),
                'effective_features': np.sum(target_importance > target_importance.mean())
            }
        
        return stats
    
    def _identify_top_features(self, top_n: int = 20) -> Dict[str, Any]:
        """Identify top features overall and per target."""
        top_features = {}
        
        # Overall top features (mean importance across targets)
        mean_importance = self.importance_data.mean(axis=1)
        top_features['overall'] = {
            'features': mean_importance.nlargest(top_n).index.tolist(),
            'importances': mean_importance.nlargest(top_n).values.tolist()
        }
        
        # Per-target top features
        top_features['per_target'] = {}
        for target in self.importance_data.columns:
            target_importance = self.importance_data[target]
            top_features['per_target'][target] = {
                'features': target_importance.nlargest(top_n).index.tolist(),
                'importances': target_importance.nlargest(top_n).values.tolist()
            }
        
        # Common top features across targets
        top_features['common_features'] = self._find_common_top_features(top_n)
        
        return top_features
    
    def _find_common_top_features(self, top_n: int = 10) -> Dict[str, Any]:
        """Find features that are consistently important across targets."""
        # Get top features for each target
        top_per_target = {}
        for target in self.importance_data.columns:
            top_features = self.importance_data[target].nlargest(top_n).index.tolist()
            top_per_target[target] = set(top_features)
        
        # Find intersection
        if len(top_per_target) > 1:
            common = set.intersection(*top_per_target.values())
        else:
            common = list(top_per_target.values())[0] if top_per_target else set()
        
        return {
            'features': list(common),
            'count': len(common),
            'percentage': len(common) / top_n * 100 if top_n > 0 else 0
        }
    
    def _analyze_feature_categories(self) -> Dict[str, Any]:
        """Analyze importance by feature categories."""
        if not self.feature_categories:
            return {}
        
        category_analysis = {}
        
        for category, features in self.feature_categories.items():
            # Filter features that exist in importance data
            existing_features = [f for f in features if f in self.importance_data.index]
            
            if not existing_features:
                continue
            
            category_importance = self.importance_data.loc[existing_features]
            
            category_analysis[category] = {
                'feature_count': len(existing_features),
                'mean_importance': category_importance.mean().to_dict(),
                'total_importance': category_importance.sum().to_dict(),
                'top_feature_per_target': {},
                'contribution_percentage': {}
            }
            
            # Find top feature in category for each target
            for target in self.importance_data.columns:
                target_cat_importance = category_importance[target]
                if len(target_cat_importance) > 0:
                    top_idx = target_cat_importance.idxmax()
                    category_analysis[category]['top_feature_per_target'][target] = {
                        'feature': top_idx,
                        'importance': target_cat_importance[top_idx]
                    }
                    
                    # Calculate contribution percentage
                    total_importance = self.importance_data[target].sum()
                    cat_contribution = target_cat_importance.sum()
                    contribution_pct = (cat_contribution / total_importance * 100) if total_importance > 0 else 0
                    category_analysis[category]['contribution_percentage'][target] = contribution_pct
        
        return category_analysis
    
    def _analyze_target_specific_importance(self) -> Dict[str, Any]:
        """Analyze which features are specific to certain targets."""
        target_analysis = {}
        
        for target in self.importance_data.columns:
            target_importance = self.importance_data[target]
            
            # Features that are much more important for this target than others
            other_targets = [col for col in self.importance_data.columns if col != target]
            if other_targets:
                other_mean = self.importance_data[other_targets].mean(axis=1)
                specificity_ratio = target_importance / (other_mean + 1e-10)  # Avoid division by zero
                
                specific_features = specificity_ratio.nlargest(10)
                
                target_analysis[target] = {
                    'specific_features': specific_features.index.tolist(),
                    'specificity_ratios': specific_features.values.tolist(),
                    'unique_contribution': self._calculate_unique_contribution(target)
                }
        
        return target_analysis
    
    def _calculate_unique_contribution(self, target: str) -> float:
        """Calculate how much of target's top features are unique to it."""
        target_top_10 = set(self.importance_data[target].nlargest(10).index)
        
        other_targets = [col for col in self.importance_data.columns if col != target]
        other_top_features = set()
        
        for other_target in other_targets:
            other_top_10 = set(self.importance_data[other_target].nlargest(10).index)
            other_top_features.update(other_top_10)
        
        unique_features = target_top_10 - other_top_features
        return len(unique_features) / len(target_top_10) * 100 if target_top_10 else 0
    
    def _analyze_feature_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between feature importance across targets."""
        if len(self.importance_data.columns) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = self.importance_data.T.corr()
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'mean_correlation': corr_matrix.mean().mean(),
            'high_correlation_pairs': self._find_high_correlation_pairs(corr_matrix)
        }
    
    def _find_high_correlation_pairs(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of targets with high importance correlation."""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                target1 = corr_matrix.columns[i]
                target2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) >= threshold:
                    high_corr_pairs.append({
                        'target1': target1,
                        'target2': target2,
                        'correlation': correlation,
                        'interpretation': 'similar importance patterns' if correlation > 0 else 'opposite importance patterns'
                    })
        
        return high_corr_pairs
    
    def _analyze_importance_stability(self) -> Dict[str, Any]:
        """Analyze stability of feature importance across targets."""
        # Calculate coefficient of variation for each feature
        feature_cv = {}
        for feature in self.importance_data.index:
            importance_values = self.importance_data.loc[feature]
            mean_val = importance_values.mean()
            std_val = importance_values.std()
            cv = (std_val / mean_val) if mean_val > 0 else float('inf')
            feature_cv[feature] = cv
        
        # Sort by stability (low CV = stable)
        stable_features = sorted(feature_cv.items(), key=lambda x: x[1])[:10]
        unstable_features = sorted(feature_cv.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'most_stable_features': [{'feature': f, 'cv': cv} for f, cv in stable_features],
            'most_unstable_features': [{'feature': f, 'cv': cv} for f, cv in unstable_features if cv != float('inf')],
            'overall_stability': np.mean(list(feature_cv.values()))
        }
    
    def _auto_categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Automatically categorize features based on naming patterns."""
        categories = {
            'embedding_features': [],
            'structure_features': [],
            'covariate_features': []
        }
        
        structure_keywords = ['mfe', 'hairpin', 'stem', 'gc', 'bulge', 'loop', 'energy', 'fold']
        embedding_keywords = ['pc', 'component', 'embedding', 'tsne']
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in embedding_keywords):
                categories['embedding_features'].append(feature)
            elif any(keyword in feature_lower for keyword in structure_keywords):
                categories['structure_features'].append(feature)
            else:
                categories['covariate_features'].append(feature)
        
        return categories
    
    def plot_importance_heatmap(self, 
                               top_n: int = 20,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create heatmap of top feature importance across targets."""
        if self.importance_data is None:
            raise ValueError("No importance data available. Run analyze_importance first.")
        
        # Get top features by mean importance
        mean_importance = self.importance_data.mean(axis=1)
        top_features = mean_importance.nlargest(top_n).index
        
        # Create heatmap data
        heatmap_data = self.importance_data.loc[top_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   cmap='YlOrRd', 
                   fmt='.3f',
                   cbar_kws={'label': 'Feature Importance'},
                   ax=ax)
        
        ax.set_title(f'Top {top_n} Feature Importance Across Targets')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Importance heatmap saved to {save_path}")
        
        return fig
    
    def plot_category_importance(self,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot importance by feature categories."""
        if self.importance_data is None or self.feature_categories is None:
            raise ValueError("No importance data or categories available")
        
        # Calculate category importance
        category_importance = {}
        for category, features in self.feature_categories.items():
            existing_features = [f for f in features if f in self.importance_data.index]
            if existing_features:
                cat_importance = self.importance_data.loc[existing_features].sum()
                category_importance[category] = cat_importance
        
        if not category_importance:
            self.logger.warning("No valid categories found for plotting")
            return plt.figure()
        
        # Create plot
        category_df = pd.DataFrame(category_importance)
        
        fig, ax = plt.subplots(figsize=figsize)
        category_df.plot(kind='bar', ax=ax)
        
        ax.set_title('Feature Importance by Category')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Total Importance')
        ax.legend(title='Feature Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Category importance plot saved to {save_path}")
        
        return fig
    
    def create_importance_report(self, 
                               analysis_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """Create a comprehensive text report of feature importance analysis."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 70)
        report_lines.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Summary statistics
        if 'summary_stats' in analysis_results:
            summary = analysis_results['summary_stats']
            report_lines.append("SUMMARY STATISTICS")
            report_lines.append("-" * 30)
            
            overall = summary.get('overall', {})
            report_lines.append(f"Total features: {overall.get('total_features', 'N/A')}")
            report_lines.append(f"Features with importance > 0: {overall.get('non_zero_features', 'N/A')}")
            report_lines.append(f"Mean importance: {overall.get('mean', 0):.4f}")
            report_lines.append(f"Max importance: {overall.get('max', 0):.4f}")
            report_lines.append("")
        
        # Top features
        if 'top_features' in analysis_results:
            top_features = analysis_results['top_features']
            report_lines.append("TOP FEATURES (Overall)")
            report_lines.append("-" * 30)
            
            if 'overall' in top_features:
                for i, (feature, importance) in enumerate(zip(
                    top_features['overall']['features'][:10],
                    top_features['overall']['importances'][:10]
                )):
                    report_lines.append(f"{i+1:2d}. {feature:<30} {importance:.4f}")
            report_lines.append("")
        
        # Category analysis
        if 'category_analysis' in analysis_results:
            cat_analysis = analysis_results['category_analysis']
            report_lines.append("FEATURE CATEGORY ANALYSIS")
            report_lines.append("-" * 30)
            
            for category, data in cat_analysis.items():
                report_lines.append(f"\n{category.upper()}:")
                report_lines.append(f"  Feature count: {data.get('feature_count', 0)}")
                
                if 'contribution_percentage' in data:
                    for target, percentage in data['contribution_percentage'].items():
                        report_lines.append(f"  {target} contribution: {percentage:.1f}%")
        
        # Target-specific analysis
        if 'target_specific' in analysis_results:
            target_analysis = analysis_results['target_specific']
            report_lines.append("\n\nTARGET-SPECIFIC FEATURES")
            report_lines.append("-" * 30)
            
            for target, data in target_analysis.items():
                report_lines.append(f"\n{target.upper()}:")
                report_lines.append(f"  Unique contribution: {data.get('unique_contribution', 0):.1f}%")
                
                specific_features = data.get('specific_features', [])[:5]
                for feature in specific_features:
                    report_lines.append(f"    - {feature}")
        
        # Stability analysis
        if 'stability_analysis' in analysis_results:
            stability = analysis_results['stability_analysis']
            report_lines.append("\n\nFEATURE STABILITY")
            report_lines.append("-" * 30)
            
            stable_features = stability.get('most_stable_features', [])[:5]
            report_lines.append("Most stable features:")
            for feature_data in stable_features:
                feature = feature_data['feature']
                cv = feature_data['cv']
                report_lines.append(f"  - {feature} (CV: {cv:.3f})")
        
        report_lines.append("\n" + "=" * 70)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Importance report saved to {save_path}")
        
        return report_text
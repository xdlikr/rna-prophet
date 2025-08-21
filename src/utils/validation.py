import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import warnings


class DataValidator:
    """Validates data quality and provides diagnostic information."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, 
                        sequence_col: str = 'sequence',
                        target_cols: List[str] = None) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        results = {
            'dataset_size': len(df),
            'sequence_validation': self._validate_sequences(df[sequence_col]),
            'target_validation': self._validate_targets(df, target_cols or []),
            'missing_data': self._check_missing_data(df),
            'data_quality': self._assess_data_quality(df),
            'recommendations': []
        }
        
        self._generate_recommendations(results)
        self.validation_results = results
        return results
    
    def _validate_sequences(self, sequences: pd.Series) -> Dict[str, Any]:
        """Validate RNA sequences."""
        valid_chars = set('ACGU')
        
        results = {
            'total_sequences': len(sequences),
            'valid_sequences': 0,
            'invalid_sequences': [],
            'length_stats': {},
            'composition_stats': {}
        }
        
        lengths = []
        compositions = {'A': [], 'C': [], 'G': [], 'U': []}
        
        for idx, seq in enumerate(sequences):
            if pd.isna(seq):
                results['invalid_sequences'].append((idx, 'null_sequence'))
                continue
            
            seq = str(seq).upper()
            if not all(c in valid_chars for c in seq):
                invalid_chars = set(seq) - valid_chars
                results['invalid_sequences'].append((idx, f'invalid_chars: {invalid_chars}'))
                continue
            
            results['valid_sequences'] += 1
            lengths.append(len(seq))
            
            for base in 'ACGU':
                compositions[base].append(seq.count(base) / len(seq))
        
        if lengths:
            results['length_stats'] = {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            }
        
        for base in 'ACGU':
            if compositions[base]:
                results['composition_stats'][base] = {
                    'mean': np.mean(compositions[base]),
                    'std': np.std(compositions[base])
                }
        
        return results
    
    def _validate_targets(self, df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
        """Validate target variables."""
        results = {}
        
        for col in target_cols:
            if col not in df.columns:
                results[col] = {'status': 'missing'}
                continue
            
            values = df[col].dropna()
            results[col] = {
                'status': 'present',
                'missing_count': df[col].isna().sum(),
                'missing_percentage': df[col].isna().sum() / len(df) * 100,
                'stats': {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median()
                },
                'outliers': self._detect_outliers(values),
                'distribution': self._assess_distribution(values)
            }
        
        return results
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data patterns."""
        missing_counts = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_counts.sum()
        
        return {
            'total_missing_cells': total_missing,
            'missing_percentage': total_missing / total_cells * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'complete_rows': len(df) - df.isna().any(axis=1).sum()
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        return {
            'duplicate_rows': df.duplicated().sum(),
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
            'high_cardinality_columns': [
                col for col in df.columns 
                if df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.5
            ]
        }
    
    def _detect_outliers(self, values: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(values) * 100,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'extreme_values': {
                'min_outlier': outliers.min() if len(outliers) > 0 else None,
                'max_outlier': outliers.max() if len(outliers) > 0 else None
            }
        }
    
    def _assess_distribution(self, values: pd.Series) -> Dict[str, Any]:
        """Assess distribution characteristics."""
        from scipy import stats
        
        return {
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'normality_test': {
                'statistic': stats.shapiro(values.sample(min(5000, len(values))))[0],
                'p_value': stats.shapiro(values.sample(min(5000, len(values))))[1]
            }
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Dataset size recommendations
        dataset_size = results['dataset_size']
        if dataset_size < 1000:
            recommendations.append(
                f"Small dataset ({dataset_size} samples). Consider collecting more data or using regularization."
            )
        
        # Sequence validation recommendations
        seq_validation = results['sequence_validation']
        if seq_validation['invalid_sequences']:
            recommendations.append(
                f"Found {len(seq_validation['invalid_sequences'])} invalid sequences. Clean before processing."
            )
        
        # Missing data recommendations
        missing_data = results['missing_data']
        if missing_data['missing_percentage'] > 10:
            recommendations.append(
                f"High missing data ({missing_data['missing_percentage']:.1f}%). Consider imputation strategies."
            )
        
        # Target validation recommendations
        for col, target_info in results['target_validation'].items():
            if target_info.get('missing_percentage', 0) > 20:
                recommendations.append(
                    f"High missing rate in {col} ({target_info['missing_percentage']:.1f}%). May impact model performance."
                )
            
            if target_info.get('outliers', {}).get('percentage', 0) > 10:
                recommendations.append(
                    f"Many outliers in {col} ({target_info['outliers']['percentage']:.1f}%). Consider robust scaling."
                )
        
        results['recommendations'] = recommendations
    
    def print_summary(self) -> None:
        """Print validation summary."""
        if not self.validation_results:
            print("No validation results available. Run validate_dataset() first.")
            return
        
        results = self.validation_results
        print("=== Data Validation Summary ===")
        print(f"Dataset size: {results['dataset_size']} samples")
        print(f"Valid sequences: {results['sequence_validation']['valid_sequences']}")
        print(f"Missing data: {results['missing_data']['missing_percentage']:.1f}%")
        
        print("\n=== Recommendations ===")
        for rec in results['recommendations']:
            print(f"- {rec}")
        
        if not results['recommendations']:
            print("- Data quality looks good!")
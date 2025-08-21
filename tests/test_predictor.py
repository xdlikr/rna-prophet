import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.prediction.predictor import RNAPropertyPredictor


class TestRNAPropertyPredictor:
    
    def setup_method(self):
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.is_fitted_ = True
        mock_pipeline.target_names_ = ['yield', 'dsRNA_percent', 'expression']
        mock_pipeline.sequence_column = 'sequence'
        
        self.predictor = RNAPropertyPredictor(pipeline=mock_pipeline)
        
        # Create test data
        self.test_X = pd.DataFrame({
            'sequence': ['ACGU' * 20, 'UGCA' * 20, 'AAUU' * 20],
            'enzyme_type': ['T7', 'SP6', 'T7'],
            'temperature': [37.0, 42.0, 37.0]
        })
        
        self.test_y = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8]
        })
    
    def test_init(self):
        predictor = RNAPropertyPredictor()
        assert predictor.pipeline is None
        assert predictor.prediction_results is None
        assert predictor.evaluation_results is None
    
    def test_init_with_pipeline(self):
        mock_pipeline = Mock()
        predictor = RNAPropertyPredictor(pipeline=mock_pipeline)
        assert predictor.pipeline is mock_pipeline
    
    def test_load_pipeline(self, tmp_path):
        # Create a mock pipeline file
        pipeline_path = tmp_path / "test_pipeline.joblib"
        
        with patch('src.prediction.predictor.RNAPredictionPipeline') as mock_pipeline_class:
            mock_pipeline_instance = Mock()
            mock_pipeline_class.return_value = mock_pipeline_instance
            
            predictor = RNAPropertyPredictor()
            result = predictor.load_pipeline(str(pipeline_path))
            
            assert result is predictor  # Should return self
            assert predictor.pipeline is mock_pipeline_instance
            mock_pipeline_instance.load_pipeline.assert_called_once_with(str(pipeline_path))
    
    def test_predict_sequences(self):
        # Mock pipeline predictions
        mock_predictions = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8]
        })
        self.predictor.pipeline.predict.return_value = mock_predictions
        
        # Test prediction
        result = self.predictor.predict_sequences(self.test_X)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Check that predictions are added with 'predicted_' prefix
        assert 'predicted_yield' in result.columns
        assert 'predicted_dsRNA_percent' in result.columns
        assert 'predicted_expression' in result.columns
        
        # Check that original features are included by default
        assert 'sequence' in result.columns
        assert 'enzyme_type' in result.columns
    
    def test_predict_sequences_without_features(self):
        mock_predictions = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8]
        })
        self.predictor.pipeline.predict.return_value = mock_predictions
        
        result = self.predictor.predict_sequences(self.test_X, include_features=False)
        
        # Should not include original features
        assert 'sequence' not in result.columns
        assert 'enzyme_type' not in result.columns
        
        # But should include predictions and insights
        assert 'predicted_yield' in result.columns
        assert 'sequence_length' in result.columns
        assert 'gc_content' in result.columns
    
    def test_predict_sequences_not_fitted(self):
        self.predictor.pipeline.is_fitted_ = False
        
        with pytest.raises(ValueError, match="Pipeline must be loaded and fitted"):
            self.predictor.predict_sequences(self.test_X)
    
    def test_predict_sequences_no_pipeline(self):
        predictor = RNAPropertyPredictor()
        
        with pytest.raises(ValueError, match="Pipeline must be loaded and fitted"):
            predictor.predict_sequences(self.test_X)
    
    def test_calculate_gc_content(self):
        # Test GC content calculation
        assert self.predictor._calculate_gc_content('ACGT') == 0.5  # 2 out of 4
        assert self.predictor._calculate_gc_content('AAUU') == 0.0  # 0 out of 4
        assert self.predictor._calculate_gc_content('GGCC') == 1.0  # 4 out of 4
        assert self.predictor._calculate_gc_content('') == 0.0     # Empty string
    
    def test_evaluate_predictions(self):
        # Mock pipeline predictions
        mock_predictions = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8]
        })
        self.predictor.pipeline.predict.return_value = mock_predictions
        
        # Mock evaluator
        mock_evaluation = {
            'overall': {'mean_r2': 0.85, 'overall_rmse': 0.12},
            'per_target': {
                'yield': {'r2': 0.9, 'mse': 0.01},
                'dsRNA_percent': {'r2': 0.8, 'mse': 0.05},
                'expression': {'r2': 0.85, 'mse': 0.03}
            },
            'target_names': ['yield', 'dsRNA_percent', 'expression']
        }
        
        with patch.object(self.predictor.evaluator, 'evaluate_predictions') as mock_eval, \
             patch.object(self.predictor.evaluator, 'plot_predictions') as mock_plot_pred, \
             patch.object(self.predictor.evaluator, 'plot_residuals') as mock_plot_resid:
            
            mock_eval.return_value = mock_evaluation
            
            result = self.predictor.evaluate_predictions(self.test_X, self.test_y)
            
            assert result == mock_evaluation
            assert self.predictor.evaluation_results == mock_evaluation
            
            # Check that evaluation was called correctly
            mock_eval.assert_called_once()
            
            # Check that plots were generated
            mock_plot_pred.assert_called_once()
            mock_plot_resid.assert_called_once()
    
    def test_generate_sequence_report(self):
        # First call predict_sequences to set prediction_results
        mock_predictions = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8]
        })
        self.predictor.pipeline.predict.return_value = mock_predictions
        
        # Generate predictions first
        self.predictor.predict_sequences(self.test_X)
        
        # Generate report
        report = self.predictor.generate_sequence_report(self.test_X, top_n=2)
        
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3
        assert 'rank' in report.columns
        assert 'insights' in report.columns
        assert 'top_performer' in report.columns
        
        # Check that top performers are marked
        top_performers = report[report['top_performer']].shape[0]
        assert top_performers == 2  # top_n = 2
    
    def test_get_feature_importance_insights(self):
        # Mock feature importance
        mock_importance_df = pd.DataFrame({
            'yield': [0.3, 0.2, 0.1, 0.05, 0.05],
            'dsRNA_percent': [0.25, 0.15, 0.2, 0.1, 0.1],
            'expression': [0.2, 0.3, 0.15, 0.1, 0.05]
        }, index=['PC1', 'mfe', 'gc_content', 'enzyme_type_T7', 'temperature'])
        
        self.predictor.pipeline.get_feature_importance.return_value = mock_importance_df
        
        insights = self.predictor.get_feature_importance_insights(top_n=3)
        
        assert isinstance(insights, dict)
        assert 'overall_top_features' in insights
        assert 'per_target_insights' in insights
        assert 'feature_categories' in insights
        
        # Check that we got insights for each target
        for target in self.predictor.pipeline.target_names_:
            assert target in insights['per_target_insights']
            assert 'top_features' in insights['per_target_insights'][target]
    
    def test_save_predictions(self, tmp_path):
        # Setup prediction results
        self.predictor.prediction_results = pd.DataFrame({
            'sequence': ['ACGU', 'UGCA'],
            'predicted_yield': [0.8, 0.6]
        })
        
        self.predictor.evaluation_results = {
            'overall': {'mean_r2': 0.85},
            'target_names': ['yield']
        }
        
        # Test saving
        save_path = tmp_path / "test_predictions.csv"
        self.predictor.save_predictions(str(save_path), include_evaluation=True)
        
        # Check that files were created
        assert save_path.exists()
        
        # Check evaluation file
        eval_path = tmp_path / "test_predictions_evaluation.json"
        assert eval_path.exists()
        
        # Load and verify CSV content
        loaded_df = pd.read_csv(save_path)
        assert len(loaded_df) == 2
        assert 'predicted_yield' in loaded_df.columns
    
    def test_save_predictions_no_results(self, tmp_path):
        save_path = tmp_path / "test_predictions.csv"
        
        with pytest.raises(ValueError, match="No predictions available to save"):
            self.predictor.save_predictions(str(save_path))
    
    def test_make_json_serializable(self):
        # Test with various numpy types
        test_data = {
            'array': np.array([1, 2, 3]),
            'int': np.int64(42),
            'float': np.float64(3.14),
            'nested': {
                'list': [np.int32(1), np.float32(2.5)]
            }
        }
        
        result = self.predictor._make_json_serializable(test_data)
        
        # Should convert numpy types to native Python types
        assert isinstance(result['array'], list)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['nested']['list'][0], int)
        assert isinstance(result['nested']['list'][1], float)
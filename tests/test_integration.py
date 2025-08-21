import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Integration tests for the complete pipeline


class TestIntegration:
    """Integration tests for the complete RNA prediction pipeline."""
    
    def setup_method(self):
        """Setup test data and temporary directories."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'sequence': [
                'ACGU' * 100,  # 400 nucleotides
                'UGCA' * 100,  # 400 nucleotides  
                'AAUU' * 100,  # 400 nucleotides
                'GCAU' * 100,  # 400 nucleotides
                'AGUC' * 100   # 400 nucleotides
            ],
            'yield': [0.8, 0.6, 0.9, 0.7, 0.85],
            'dsRNA_percent': [3.2, 8.1, 2.5, 6.8, 4.1],
            'expression': [1.4, 0.9, 1.8, 1.1, 1.6],
            'enzyme_type': ['T7', 'SP6', 'T7', 'SP6', 'T7'],
            'temperature': [37.0, 42.0, 37.0, 42.0, 37.0]
        })
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample data file
        self.data_file = self.temp_path / "test_data.csv"
        self.sample_data.to_csv(self.data_file, index=False)
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.features.embeddings.SequenceEmbedder.extract_embeddings')
    @patch('src.features.structure.RNAStructureFeatures.extract_features')
    def test_data_loading_and_preprocessing(self, mock_structure, mock_embeddings):
        """Test data loading and preprocessing components."""
        from src.data.loader import RNADataLoader
        from src.features.preprocessing import CovariatePreprocessor
        
        # Mock structure features
        mock_structure_df = pd.DataFrame({
            'mfe': [-10.5, -8.2, -12.1, -9.5, -11.0],
            'mfe_normalized': [-0.026, -0.021, -0.030, -0.024, -0.028],
            'hairpin_count': [2, 1, 3, 2, 2],
            'stem_count': [3, 2, 4, 3, 3],
            'gc_content': [0.5, 0.4, 0.6, 0.45, 0.55],
            'sequence_length': [400, 400, 400, 400, 400],
            'bulge_count': [1, 0, 2, 1, 1],
            'loop_count': [2, 1, 3, 2, 2]
        })
        mock_structure.return_value = mock_structure_df
        
        # Mock embeddings
        mock_embeddings.return_value = np.random.randn(5, 64)  # 5 sequences, 64 dimensions
        
        # Test data loading
        loader = RNADataLoader()
        df = loader.load_data(str(self.data_file))
        
        assert len(df) == 5
        assert 'sequence' in df.columns
        assert 'yield' in df.columns
        
        # Test data splitting
        train_df, test_df = loader.split_data(df, test_size=0.4)
        assert len(train_df) == 3
        assert len(test_df) == 2
        
        # Test covariate preprocessing
        covariate_columns = loader.get_covariate_columns(df)
        preprocessor = CovariatePreprocessor()
        processed_covariates = preprocessor.fit_transform(df, covariate_columns)
        
        assert len(processed_covariates) == len(df)
        assert len(processed_covariates.columns) > 0  # Should have processed features
    
    @patch('src.features.embeddings.SequenceEmbedder.load_model')
    @patch('src.features.embeddings.SequenceEmbedder.extract_embeddings')
    @patch('src.features.structure.RNAStructureFeatures._check_vienna_rna')
    def test_feature_extraction_pipeline(self, mock_vienna_check, mock_embeddings, mock_load_model):
        """Test the complete feature extraction pipeline."""
        from src.models.pipeline import SequenceFeatureExtractor
        
        # Mock ViennaRNA check
        mock_vienna_check.return_value = None
        
        # Mock model loading
        mock_load_model.return_value = None
        
        # Mock embeddings
        mock_embeddings.return_value = np.random.randn(5, 128)
        
        # Create feature extractor
        extractor = SequenceFeatureExtractor(
            embedding_config={'model_name': 'test_model'},
            dimensionality_config={'method': 'pca', 'n_components': 10}
        )
        
        # Test fit and transform
        extractor.fit(self.sample_data)
        features = extractor.transform(self.sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(self.sample_data)
        assert len(features.columns) > 0
    
    @patch('src.features.embeddings.SequenceEmbedder.load_model')
    @patch('src.features.embeddings.SequenceEmbedder.extract_embeddings')
    @patch('src.features.structure.RNAStructureFeatures._check_vienna_rna')
    def test_model_training_and_prediction(self, mock_vienna_check, mock_embeddings, mock_load_model):
        """Test model training and prediction."""
        from src.models.pipeline import RNAPredictionPipeline
        from src.data.loader import RNADataLoader
        
        # Setup mocks
        mock_vienna_check.return_value = None
        mock_load_model.return_value = None
        mock_embeddings.return_value = np.random.randn(5, 64)
        
        # Load data
        loader = RNADataLoader()
        df = loader.load_data(str(self.data_file))
        
        target_columns = loader.get_target_columns(df)
        covariate_columns = loader.get_covariate_columns(df)
        
        # Create pipeline with minimal configuration
        pipeline = RNAPredictionPipeline(
            embedding_config={'model_name': 'test_model'},
            dimensionality_config={'method': 'pca', 'n_components': 5},
            model_config={'n_estimators': 10, 'random_state': 42}  # Small for fast testing
        )
        
        # Split data
        train_df, test_df = loader.split_data(df, test_size=0.4)
        
        # Train model
        y_train = train_df[target_columns]
        pipeline.fit(train_df, y_train, 
                    covariate_columns=covariate_columns,
                    target_columns=target_columns)
        
        assert pipeline.is_fitted_
        
        # Make predictions
        predictions = pipeline.predict(test_df)
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(test_df)
        assert list(predictions.columns) == target_columns
        
        # Test scoring
        y_test = test_df[target_columns]
        score = pipeline.score(test_df, y_test)
        assert isinstance(score, float)
    
    @patch('src.features.embeddings.SequenceEmbedder.load_model')
    @patch('src.features.embeddings.SequenceEmbedder.extract_embeddings')
    @patch('src.features.structure.RNAStructureFeatures._check_vienna_rna')
    def test_prediction_and_reporting(self, mock_vienna_check, mock_embeddings, mock_load_model):
        """Test prediction and reporting functionality."""
        from src.models.pipeline import RNAPredictionPipeline
        from src.prediction.predictor import RNAPropertyPredictor
        from src.data.loader import RNADataLoader
        
        # Setup mocks
        mock_vienna_check.return_value = None
        mock_load_model.return_value = None
        mock_embeddings.return_value = np.random.randn(5, 32)
        
        # Create and train a simple pipeline
        loader = RNADataLoader()
        df = loader.load_data(str(self.data_file))
        
        pipeline = RNAPredictionPipeline(
            embedding_config={'model_name': 'test_model'},
            dimensionality_config={'method': 'pca', 'n_components': 3},
            model_config={'n_estimators': 5, 'random_state': 42}
        )
        
        target_columns = loader.get_target_columns(df)
        covariate_columns = loader.get_covariate_columns(df)
        
        # Use all data for training (integration test)
        y_all = df[target_columns]
        pipeline.fit(df, y_all, 
                    covariate_columns=covariate_columns,
                    target_columns=target_columns)
        
        # Create predictor
        predictor = RNAPropertyPredictor(pipeline)
        
        # Test predictions
        predictions_df = predictor.predict_sequences(df)
        
        assert isinstance(predictions_df, pd.DataFrame)
        assert len(predictions_df) == len(df)
        
        # Check that predicted columns are present
        pred_cols = [col for col in predictions_df.columns if col.startswith('predicted_')]
        assert len(pred_cols) == len(target_columns)
        
        # Test sequence report generation
        sequence_report = predictor.generate_sequence_report(df, top_n=3)
        
        assert isinstance(sequence_report, pd.DataFrame)
        assert 'rank' in sequence_report.columns
        assert 'insights' in sequence_report.columns
        assert 'top_performer' in sequence_report.columns
        
        # Test feature importance
        importance_insights = predictor.get_feature_importance_insights(top_n=5)
        
        assert isinstance(importance_insights, dict)
        assert 'overall_top_features' in importance_insights
        assert 'per_target_insights' in importance_insights
    
    def test_data_validation(self):
        """Test data validation functionality."""
        from src.utils.validation import DataValidator
        from src.data.loader import RNADataLoader
        
        # Test with valid data
        loader = RNADataLoader()
        df = loader.load_data(str(self.data_file))
        
        validator = DataValidator()
        results = validator.validate_dataset(df)
        
        assert isinstance(results, dict)
        assert 'dataset_size' in results
        assert 'sequence_validation' in results
        assert 'target_validation' in results
        
        # Check that no invalid sequences are found
        assert len(results['sequence_validation']['invalid_sequences']) == 0
        
        # Test with invalid data
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'sequence'] = 'ACGTX'  # Invalid character
        
        invalid_file = self.temp_path / "invalid_data.csv"
        invalid_data.to_csv(invalid_file, index=False)
        
        invalid_df = loader.load_data(str(invalid_file))
        
        # This should raise an error during validation in the loader
        with pytest.raises(ValueError, match="Invalid RNA sequences"):
            loader._validate_data(invalid_df)
    
    @patch('src.features.embeddings.SequenceEmbedder.load_model')
    @patch('src.features.embeddings.SequenceEmbedder.extract_embeddings')
    @patch('src.features.structure.RNAStructureFeatures._check_vienna_rna')
    def test_save_load_pipeline(self, mock_vienna_check, mock_embeddings, mock_load_model):
        """Test saving and loading of complete pipeline."""
        from src.models.pipeline import RNAPredictionPipeline
        from src.data.loader import RNADataLoader
        
        # Setup mocks
        mock_vienna_check.return_value = None
        mock_load_model.return_value = None
        mock_embeddings.return_value = np.random.randn(5, 16)
        
        # Create and train pipeline
        loader = RNADataLoader()
        df = loader.load_data(str(self.data_file))
        
        pipeline = RNAPredictionPipeline(
            embedding_config={'model_name': 'test_model'},
            dimensionality_config={'method': 'pca', 'n_components': 2},
            model_config={'n_estimators': 3, 'random_state': 42}
        )
        
        target_columns = loader.get_target_columns(df)
        covariate_columns = loader.get_covariate_columns(df)
        
        y_all = df[target_columns]
        pipeline.fit(df, y_all,
                    covariate_columns=covariate_columns,
                    target_columns=target_columns)
        
        # Make predictions with original pipeline
        original_predictions = pipeline.predict(df)
        
        # Save pipeline
        model_path = self.temp_path / "test_pipeline.joblib"
        pipeline.save_pipeline(str(model_path))
        
        # Load pipeline
        new_pipeline = RNAPredictionPipeline()
        new_pipeline.load_pipeline(str(model_path))
        
        # Make predictions with loaded pipeline
        loaded_predictions = new_pipeline.predict(df)
        
        # Compare predictions
        pd.testing.assert_frame_equal(original_predictions, loaded_predictions, rtol=1e-10)
        
        assert new_pipeline.is_fitted_
        assert new_pipeline.target_names_ == pipeline.target_names_
    
    def test_cli_integration(self):
        """Test CLI command structure (without actual execution)."""
        from main import cli
        
        # Test that CLI is properly structured
        assert hasattr(cli, 'commands')
        
        expected_commands = ['train', 'predict', 'evaluate', 'validate', 'info', 'init']
        for cmd in expected_commands:
            assert cmd in cli.commands
        
        # Test that commands have proper parameters
        train_cmd = cli.commands['train']
        assert 'input_file' in [p.name for p in train_cmd.params]
        assert 'output_dir' in [p.name for p in train_cmd.params]
        
        predict_cmd = cli.commands['predict']
        assert 'model_path' in [p.name for p in predict_cmd.params]
        assert 'input_file' in [p.name for p in predict_cmd.params]
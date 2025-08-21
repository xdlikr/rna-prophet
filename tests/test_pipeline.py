import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.models.pipeline import SequenceFeatureExtractor, RNAPredictionPipeline


class TestSequenceFeatureExtractor:
    
    def setup_method(self):
        # Mock dependencies to avoid loading actual models
        with patch('src.models.pipeline.SequenceEmbedder'), \
             patch('src.models.pipeline.DimensionalityReducer'), \
             patch('src.models.pipeline.RNAStructureFeatures'):
            
            self.extractor = SequenceFeatureExtractor(
                embedding_config={'model_name': 'test_model'},
                dimensionality_config={'method': 'pca', 'n_components': 5},
                structure_config={}
            )
        
        # Create test data
        self.test_data = pd.DataFrame({
            'sequence': ['ACGU' * 20, 'UGCA' * 20, 'AAUU' * 20],
            'other_col': [1, 2, 3]
        })
    
    def test_init(self):
        assert self.extractor.sequence_column == 'sequence'
        assert self.extractor.embedding_config['model_name'] == 'test_model'
        assert self.extractor.dimensionality_config['method'] == 'pca'
        assert not self.extractor.is_fitted_
    
    @patch('src.models.pipeline.SequenceEmbedder')
    @patch('src.models.pipeline.DimensionalityReducer')
    @patch('src.models.pipeline.RNAStructureFeatures')
    def test_fit(self, mock_structure, mock_reducer, mock_embedder):
        # Setup mocks
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.extract_embeddings.return_value = np.random.randn(3, 100)
        
        mock_reducer_instance = Mock()
        mock_reducer.return_value = mock_reducer_instance
        mock_reducer_instance.get_feature_names.return_value = ['PC1', 'PC2', 'PC3']
        
        mock_structure_instance = Mock()
        mock_structure.return_value = mock_structure_instance
        
        # Create extractor and fit
        extractor = SequenceFeatureExtractor(
            embedding_config={'model_name': 'test'},
            dimensionality_config={'method': 'pca', 'n_components': 3}
        )
        
        extractor.fit(self.test_data)
        
        assert extractor.is_fitted_
        assert extractor.embedder is not None
        assert extractor.reducer is not None
        assert extractor.structure_extractor is not None
    
    def test_fit_missing_sequence_column(self):
        data_no_seq = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Sequence column 'sequence' not found"):
            self.extractor.fit(data_no_seq)
    
    @patch('src.models.pipeline.SequenceEmbedder')
    @patch('src.models.pipeline.DimensionalityReducer')
    @patch('src.models.pipeline.RNAStructureFeatures')
    def test_transform(self, mock_structure, mock_reducer, mock_embedder):
        # Setup mocks for fitted extractor
        self.extractor.is_fitted_ = True
        
        # Mock embedder
        mock_embedder_instance = Mock()
        mock_embedder_instance.extract_embeddings.return_value = np.random.randn(3, 100)
        self.extractor.embedder = mock_embedder_instance
        
        # Mock reducer
        mock_reducer_instance = Mock()
        mock_reducer_instance.transform.return_value = np.random.randn(3, 5)
        mock_reducer_instance.get_feature_names.return_value = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        self.extractor.reducer = mock_reducer_instance
        
        # Mock structure extractor
        mock_structure_instance = Mock()
        structure_features = pd.DataFrame({
            'mfe': [-10.5, -8.2, -12.1],
            'gc_content': [0.5, 0.4, 0.6],
            'hairpin_count': [2, 1, 3]
        })
        mock_structure_instance.extract_features.return_value = structure_features
        self.extractor.structure_extractor = mock_structure_instance
        
        # Transform
        result = self.extractor.transform(self.test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'mfe' in result.columns
        assert 'PC1' in result.columns
    
    def test_transform_not_fitted(self):
        with pytest.raises(ValueError, match="SequenceFeatureExtractor must be fitted"):
            self.extractor.transform(self.test_data)


class TestRNAPredictionPipeline:
    
    def setup_method(self):
        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'sequence': ['ACGU' * 20, 'UGCA' * 20, 'AAUU' * 20, 'GCAU' * 20],
            'enzyme_type': ['T7', 'SP6', 'T7', 'SP6'],
            'temperature': [37.0, 42.0, 37.0, 42.0]
        })
        
        self.y = pd.DataFrame({
            'yield': [0.8, 0.6, 0.9, 0.7],
            'dsRNA_percent': [3.2, 8.1, 2.5, 6.8],
            'expression': [1.4, 0.9, 1.8, 1.1]
        })
        
        self.covariate_columns = ['enzyme_type', 'temperature']
        self.target_columns = ['yield', 'dsRNA_percent', 'expression']
    
    def test_init(self):
        pipeline = RNAPredictionPipeline()
        
        assert pipeline.sequence_column == 'sequence'
        assert not pipeline.is_fitted_
        assert pipeline.pipeline is None
    
    @patch('src.models.pipeline.SequenceFeatureExtractor')
    @patch('src.models.pipeline.CovariatePreprocessor')
    @patch('src.models.pipeline.MultiTaskXGBoost')
    def test_build_pipeline(self, mock_model, mock_cov_proc, mock_seq_ext):
        pipeline = RNAPredictionPipeline()
        
        # Test with covariates
        built_pipeline = pipeline._build_pipeline(['enzyme_type', 'temperature'])
        
        assert built_pipeline is not None
        assert 'features' in built_pipeline.named_steps
        assert 'model' in built_pipeline.named_steps
    
    @patch('src.models.pipeline.SequenceFeatureExtractor')
    @patch('src.models.pipeline.CovariatePreprocessor')
    @patch('src.models.pipeline.MultiTaskXGBoost')
    @patch('sklearn.pipeline.Pipeline.fit')
    def test_fit(self, mock_pipeline_fit, mock_model, mock_cov_proc, mock_seq_ext):
        pipeline = RNAPredictionPipeline()
        
        # Mock pipeline fit
        mock_pipeline_fit.return_value = None
        
        result = pipeline.fit(self.X, self.y, 
                            covariate_columns=self.covariate_columns,
                            target_columns=self.target_columns)
        
        assert result is pipeline  # Should return self
        assert pipeline.is_fitted_
        assert pipeline.target_names_ == self.target_columns
        assert pipeline.pipeline is not None
    
    def test_fit_missing_sequence_column(self):
        pipeline = RNAPredictionPipeline(sequence_column='missing_col')
        
        with pytest.raises(ValueError, match="Sequence column 'missing_col' not found"):
            pipeline.fit(self.X, self.y)
    
    @patch('src.models.pipeline.SequenceFeatureExtractor')
    @patch('src.models.pipeline.CovariatePreprocessor')
    @patch('src.models.pipeline.MultiTaskXGBoost')
    def test_predict(self, mock_model, mock_cov_proc, mock_seq_ext):
        pipeline = RNAPredictionPipeline()
        pipeline.is_fitted_ = True
        pipeline.target_names_ = self.target_columns
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.predict.return_value = np.random.randn(4, 3)
        pipeline.pipeline = mock_pipeline
        
        predictions = pipeline.predict(self.X)
        
        assert isinstance(predictions, pd.DataFrame)
        assert list(predictions.columns) == self.target_columns
        assert len(predictions) == len(self.X)
    
    def test_predict_not_fitted(self):
        pipeline = RNAPredictionPipeline()
        
        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.predict(self.X)
    
    @patch('src.models.pipeline.SequenceFeatureExtractor')
    @patch('src.models.pipeline.CovariatePreprocessor')
    @patch('src.models.pipeline.MultiTaskXGBoost')
    def test_score(self, mock_model, mock_cov_proc, mock_seq_ext):
        pipeline = RNAPredictionPipeline()
        pipeline.is_fitted_ = True
        pipeline.target_names_ = self.target_columns
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.score.return_value = 0.75
        pipeline.pipeline = mock_pipeline
        
        score = pipeline.score(self.X, self.y)
        
        assert score == 0.75
    
    @patch('src.models.pipeline.SequenceFeatureExtractor')
    @patch('src.models.pipeline.CovariatePreprocessor')
    @patch('src.models.pipeline.MultiTaskXGBoost')
    def test_save_load_pipeline(self, mock_model, mock_cov_proc, mock_seq_ext, tmp_path):
        # Create and "fit" pipeline
        pipeline = RNAPredictionPipeline()
        pipeline.is_fitted_ = True
        pipeline.target_names_ = self.target_columns
        pipeline.feature_names_ = ['feature_1', 'feature_2']
        pipeline.pipeline = Mock()  # Mock fitted pipeline
        
        # Save
        save_path = tmp_path / "test_pipeline.joblib"
        pipeline.save_pipeline(str(save_path))
        
        # Load
        new_pipeline = RNAPredictionPipeline()
        new_pipeline.load_pipeline(str(save_path))
        
        assert new_pipeline.is_fitted_
        assert new_pipeline.target_names_ == self.target_columns
        assert new_pipeline.feature_names_ == ['feature_1', 'feature_2']
    
    def test_save_unfitted_pipeline(self, tmp_path):
        pipeline = RNAPredictionPipeline()
        save_path = tmp_path / "test_pipeline.joblib"
        
        with pytest.raises(ValueError, match="Cannot save unfitted pipeline"):
            pipeline.save_pipeline(str(save_path))
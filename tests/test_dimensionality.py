import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.features.dimensionality import DimensionalityReducer, EmbeddingProcessor


class TestDimensionalityReducer:
    
    def setup_method(self):
        self.reducer = DimensionalityReducer(method='pca', n_components=10)
        self.test_embeddings = np.random.randn(50, 128)  # 50 samples, 128 features
    
    def test_init(self):
        assert self.reducer.method == 'pca'
        assert self.reducer.n_components == 10
        assert self.reducer.random_state == 42
    
    def test_fit_pca(self):
        self.reducer.fit(self.test_embeddings)
        
        assert self.reducer.reducer is not None
        assert self.reducer.scaler is not None
        assert self.reducer.explained_variance_ratio_ is not None
        assert len(self.reducer.explained_variance_ratio_) == 10
    
    def test_fit_adjusts_components(self):
        # Test with small dataset where n_components > n_samples
        small_embeddings = np.random.randn(5, 128)
        reducer = DimensionalityReducer(method='pca', n_components=50)
        
        reducer.fit(small_embeddings)
        assert reducer.n_components == 5  # Should be adjusted to min(n_samples, n_features)
    
    def test_transform_pca(self):
        self.reducer.fit(self.test_embeddings)
        reduced = self.reducer.transform(self.test_embeddings)
        
        assert reduced.shape == (50, 10)
        assert not np.allclose(reduced, 0)  # Should not be all zeros
    
    def test_fit_transform_pca(self):
        reduced = self.reducer.fit_transform(self.test_embeddings)
        
        assert reduced.shape == (50, 10)
        assert self.reducer.reducer is not None
        assert self.reducer.scaler is not None
    
    def test_transform_without_fit_raises_error(self):
        with pytest.raises(ValueError, match="Must fit reducer before transforming"):
            self.reducer.transform(self.test_embeddings)
    
    def test_get_feature_names_pca(self):
        self.reducer.fit(self.test_embeddings)
        names = self.reducer.get_feature_names()
        
        assert len(names) == 10
        assert names[0] == 'PC1'
        assert names[9] == 'PC10'
    
    def test_get_feature_names_tsne(self):
        reducer = DimensionalityReducer(method='tsne', n_components=2)
        names = reducer.get_feature_names()
        
        assert len(names) == 2
        assert names[0] == 'tSNE1'
        assert names[1] == 'tSNE2'
    
    def test_invalid_method_raises_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            DimensionalityReducer(method='invalid_method')
    
    def test_save_load(self, tmp_path):
        self.reducer.fit(self.test_embeddings)
        filepath = tmp_path / "test_reducer.joblib"
        
        # Save
        self.reducer.save(str(filepath))
        
        # Create new reducer and load
        new_reducer = DimensionalityReducer()
        new_reducer.load(str(filepath))
        
        assert new_reducer.method == 'pca'
        assert new_reducer.n_components == 10
        assert new_reducer.reducer is not None
        assert new_reducer.scaler is not None
        
        # Test that transform produces same results
        original_reduced = self.reducer.transform(self.test_embeddings)
        loaded_reduced = new_reducer.transform(self.test_embeddings)
        
        np.testing.assert_array_almost_equal(original_reduced, loaded_reduced)


class TestEmbeddingProcessor:
    
    def setup_method(self):
        embedder_config = {
            'model_name': 'test_model',
            'max_length': 128,
            'batch_size': 2
        }
        reducer_config = {
            'method': 'pca',
            'n_components': 10
        }
        
        with patch('src.features.embeddings.SequenceEmbedder'):
            self.processor = EmbeddingProcessor(embedder_config, reducer_config)
    
    @patch.object(EmbeddingProcessor, '__init__', lambda x, y, z: None)
    def test_process_sequences(self):
        # Initialize processor manually to avoid import issues in test
        self.processor.embedder = Mock()
        self.processor.reducer = Mock()
        self.processor.logger = Mock()
        
        # Mock embedder
        mock_embeddings = np.random.randn(3, 128)
        self.processor.embedder.extract_embeddings.return_value = mock_embeddings
        
        # Mock reducer
        mock_reduced = np.random.randn(3, 10)
        self.processor.reducer.fit_transform.return_value = mock_reduced
        self.processor.reducer.get_feature_names.return_value = [f'PC{i+1}' for i in range(10)]
        
        sequences = ['ACGT', 'UGCA', 'AAUU']
        result_df = self.processor.process_sequences(sequences)
        
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == (3, 10)
        assert list(result_df.columns) == [f'PC{i+1}' for i in range(10)]
        
        # Verify methods were called
        self.processor.embedder.extract_embeddings.assert_called_once_with(sequences)
        self.processor.reducer.fit_transform.assert_called_once()
    
    @patch.object(EmbeddingProcessor, '__init__', lambda x, y, z: None)
    def test_get_processing_stats(self):
        # Setup processor manually
        self.processor.raw_embeddings = np.random.randn(5, 128)
        self.processor.reduced_embeddings = np.random.randn(5, 10)
        self.processor.reducer = Mock()
        self.processor.reducer.method = 'pca'
        self.processor.reducer.n_components = 10
        self.processor.reducer.explained_variance_ratio_ = np.array([0.3, 0.2, 0.1, 0.05, 0.05])
        
        stats = self.processor.get_processing_stats()
        
        assert stats['raw_embedding_shape'] == (5, 128)
        assert stats['reduced_embedding_shape'] == (5, 10)
        assert stats['reduction_method'] == 'pca'
        assert stats['n_components'] == 10
        assert 'explained_variance_ratio' in stats
        assert 'total_explained_variance' in stats
    
    @patch.object(EmbeddingProcessor, '__init__', lambda x, y, z: None)
    def test_save_all(self, tmp_path):
        # Setup processor manually
        self.processor.reducer = Mock()
        self.processor.logger = Mock()
        self.processor.raw_embeddings = np.random.randn(5, 128)
        self.processor.reduced_embeddings = np.random.randn(5, 10)
        
        base_path = str(tmp_path / "test_processor")
        self.processor.save_all(base_path)
        
        # Verify reducer.save was called
        self.processor.reducer.save.assert_called_once_with(f"{base_path}_reducer.joblib")
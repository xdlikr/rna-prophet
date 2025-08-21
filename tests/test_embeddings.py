import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from src.features.embeddings import SequenceEmbedder, EmbeddingConfig


class TestSequenceEmbedder:
    
    def setup_method(self):
        # Mock the model loading to avoid downloading
        with patch.object(SequenceEmbedder, 'load_model'):
            self.embedder = SequenceEmbedder(
                model_name="test_model",
                max_length=128,
                batch_size=2
            )
    
    def test_init(self):
        assert self.embedder.model_name == "test_model"
        assert self.embedder.max_length == 128
        assert self.embedder.batch_size == 2
        assert self.embedder.device in ['cpu', 'cuda']
    
    @patch('src.features.embeddings.AutoTokenizer')
    @patch('src.features.embeddings.AutoModel')
    def test_load_model_success(self, mock_model, mock_tokenizer):
        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value.pad_token = None
        mock_tokenizer.from_pretrained.return_value.eos_token = '[EOS]'
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        embedder = SequenceEmbedder()
        embedder.load_model()
        
        assert embedder.tokenizer is not None
        assert embedder.model is not None
        mock_model_instance.to.assert_called_once()
        mock_model_instance.eval.assert_called_once()
    
    @patch('src.features.embeddings.AutoTokenizer')
    def test_load_model_failure(self, mock_tokenizer):
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        embedder = SequenceEmbedder()
        with pytest.raises(RuntimeError, match="Model loading failed"):
            embedder.load_model()
    
    def test_extract_embeddings_calls_load_model(self):
        sequences = ['ACGT', 'UGCA']
        
        with patch.object(self.embedder, 'load_model') as mock_load, \
             patch.object(self.embedder, '_process_batch') as mock_process:
            
            mock_process.return_value = np.random.randn(2, 128)
            
            embeddings = self.embedder.extract_embeddings(sequences)
            
            mock_load.assert_called_once()
            assert embeddings.shape == (2, 128)
    
    def test_process_batch_dna_conversion(self):
        # Mock tokenizer and model
        self.embedder.tokenizer = Mock()
        self.embedder.model = Mock()
        self.embedder.model_name = "DNABERT-2"
        
        # Mock tokenizer output
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        self.embedder.tokenizer.return_value = mock_encoded
        
        # Mock model output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 3, 128)
        self.embedder.model.return_value = mock_output
        
        sequences = ['ACGU', 'UGCA']  # RNA sequences
        embeddings = self.embedder._process_batch(sequences)
        
        # Check that RNA sequences were converted to DNA
        call_args = self.embedder.tokenizer.call_args[0][0]
        assert 'U' not in str(call_args)  # U should be converted to T
        
        assert embeddings.shape == (2, 128)
    
    def test_get_embedding_dim(self):
        # Mock the model and tokenizer
        self.embedder.tokenizer = Mock()
        self.embedder.model = Mock()
        
        mock_encoded = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        self.embedder.tokenizer.return_value = mock_encoded
        
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 3, 256)
        self.embedder.model.return_value = mock_output
        
        dim = self.embedder.get_embedding_dim()
        assert dim == 256
    
    def test_save_load_embeddings(self, tmp_path):
        embeddings = np.random.randn(10, 128)
        filepath = tmp_path / "test_embeddings.npy"
        
        self.embedder.save_embeddings(embeddings, str(filepath))
        loaded = self.embedder.load_embeddings(str(filepath))
        
        np.testing.assert_array_equal(embeddings, loaded)


class TestEmbeddingConfig:
    
    def test_get_model_config(self):
        config = EmbeddingConfig.get_model_config('dnabert2')
        
        assert 'name' in config
        assert 'max_length' in config
        assert 'description' in config
        assert config['max_length'] == 2048
    
    def test_get_model_config_invalid(self):
        with pytest.raises(ValueError, match="Unknown model key"):
            EmbeddingConfig.get_model_config('invalid_model')
    
    def test_list_available_models(self):
        models = EmbeddingConfig.list_available_models()
        
        assert isinstance(models, dict)
        assert 'dnabert2' in models
        assert 'evo' in models
        assert all(isinstance(desc, str) for desc in models.values())
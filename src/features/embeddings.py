import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple, Dict, Any
import logging
import warnings


class SequenceEmbedder:
    """Extracts frozen embeddings from pretrained DNA/RNA models."""
    
    def __init__(self, 
                 model_name: str = "zhihan1996/DNABERT-2-117M",
                 max_length: int = 2048,
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize sequence embedder.
        
        Args:
            model_name: HuggingFace model name (DNABERT-2 or Evo)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for embedding extraction
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.model = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the embedder."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Suppress some transformers warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    def load_model(self) -> None:
        """Load pretrained model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def extract_embeddings(self, sequences: List[str]) -> np.ndarray:
        """
        Extract embeddings for a list of sequences.
        
        Args:
            sequences: List of RNA/DNA sequences
            
        Returns:
            Array of embeddings with shape (n_sequences, embedding_dim)
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        self.logger.info(f"Extracting embeddings for {len(sequences)} sequences")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_embeddings = self._process_batch(batch_sequences)
            all_embeddings.append(batch_embeddings)
            
            if (i // self.batch_size + 1) % 10 == 0:
                self.logger.info(f"Processed {i + len(batch_sequences)}/{len(sequences)} sequences")
        
        embeddings = np.vstack(all_embeddings)
        self.logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def _process_batch(self, sequences: List[str]) -> np.ndarray:
        """Process a batch of sequences."""
        # Handle sequence conversion based on model type
        if "DNABERT" in self.model_name:
            # DNABERT expects DNA sequences (T instead of U)
            sequences = [seq.replace('U', 'T') for seq in sequences]
        elif "evo" in self.model_name.lower():
            # Evo models expect RNA sequences (keep U)
            sequences = [seq.replace('T', 'U') for seq in sequences]
        
        # Tokenize sequences
        try:
            encoded = self.tokenizer(
                sequences,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(sequences), 768))  # Default BERT embedding size
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Extract embeddings
        with torch.no_grad():
            try:
                outputs = self.model(**encoded)
                # Use mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()
                
            except Exception as e:
                self.logger.error(f"Model forward pass failed: {e}")
                # Return zero embeddings as fallback
                return np.zeros((len(sequences), 768))
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if self.model is None:
            self.load_model()
        
        # Test with a dummy sequence to get embedding dimension
        dummy_seq = "ACGUACGU"  # Start with RNA
        
        # Convert based on model type
        if "DNABERT" in self.model_name:
            dummy_seq = dummy_seq.replace('U', 'T')
        elif "evo" in self.model_name.lower():
            dummy_seq = dummy_seq.replace('T', 'U')
        
        encoded = self.tokenizer(
            [dummy_seq],
            truncation=True,
            padding=True,
            max_length=min(50, self.max_length),
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embedding_dim = outputs.last_hidden_state.size(-1)
        
        return embedding_dim
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Save embeddings to file."""
        np.save(filepath, embeddings)
        self.logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        embeddings = np.load(filepath)
        self.logger.info(f"Embeddings loaded from {filepath}, shape: {embeddings.shape}")
        return embeddings


class EmbeddingConfig:
    """Configuration class for different embedding models."""
    
    MODELS = {
        'dnabert2': {
            'name': 'zhihan1996/DNABERT-2-117M',
            'max_length': 2048,
            'description': 'DNABERT-2 117M parameters, good for DNA sequences',
            'sequence_type': 'dna'
        },
        'dnabert2_large': {
            'name': 'zhihan1996/DNABERT-2-512M', 
            'max_length': 2048,
            'description': 'DNABERT-2 512M parameters, better performance but slower',
            'sequence_type': 'dna'
        },
        'evo': {
            'name': 'togethercomputer/evo-1-8k-base',
            'max_length': 8192,
            'description': 'Evo model, supports longer sequences up to 8k',
            'sequence_type': 'rna'
        },
        'evo2_7b': {
            'name': 'arcinstitute/evo2_7b',
            'max_length': 1048576,  # 1M context
            'description': 'Evo-2 7B parameter model with 1M context (official Arc Institute)',
            'sequence_type': 'dna'
        },
        'evo2_40b': {
            'name': 'arcinstitute/evo2_40b',
            'max_length': 1048576,  # 1M context
            'description': 'Evo-2 40B parameter model with 1M context (requires multiple GPUs)',
            'sequence_type': 'dna'
        },
        'evo2_7b_base': {
            'name': 'arcinstitute/evo2_7b_base',
            'max_length': 8192,
            'description': 'Evo-2 7B base model with 8K context',
            'sequence_type': 'dna'
        },
        'evo2_40b_base': {
            'name': 'arcinstitute/evo2_40b_base',
            'max_length': 8192,
            'description': 'Evo-2 40B base model with 8K context',
            'sequence_type': 'dna'
        },
        'evo2_1b_base': {
            'name': 'arcinstitute/evo2_1b_base',
            'max_length': 8192,
            'description': 'Evo-2 1B base model with 8K context (lightweight)',
            'sequence_type': 'dna'
        }
    }
    
    @classmethod
    def get_model_config(cls, model_key: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if model_key not in cls.MODELS:
            available = list(cls.MODELS.keys())
            raise ValueError(f"Unknown model key '{model_key}'. Available: {available}")
        
        return cls.MODELS[model_key].copy()
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """List all available models with descriptions."""
        return {k: v['description'] for k, v in cls.MODELS.items()}
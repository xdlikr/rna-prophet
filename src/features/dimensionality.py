import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Tuple, Optional, Dict, Any
import joblib
import logging


class DimensionalityReducer:
    """Handles dimensionality reduction for sequence embeddings."""
    
    def __init__(self, 
                 method: str = 'pca',
                 n_components: int = 128,
                 random_state: int = 42):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('pca', 'tsne')
            n_components: Number of components to keep
            random_state: Random state for reproducibility
        """
        if method.lower() not in ['pca', 'tsne']:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'")
            
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        
        self.reducer = None
        self.scaler = None
        self.explained_variance_ratio_ = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger(__name__)
    
    def fit(self, embeddings: np.ndarray) -> 'DimensionalityReducer':
        """
        Fit the dimensionality reduction method.
        
        Args:
            embeddings: Input embeddings with shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting {self.method} with {self.n_components} components")
        self.logger.info(f"Input shape: {embeddings.shape}")
        
        # Standardize embeddings first
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Adjust n_components if necessary
        max_components = min(embeddings.shape[0], embeddings.shape[1])
        if self.n_components > max_components:
            self.logger.warning(f"Reducing n_components from {self.n_components} to {max_components}")
            self.n_components = max_components
        
        # Fit the reducer
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
            self.reducer.fit(embeddings_scaled)
            self.explained_variance_ratio_ = self.reducer.explained_variance_ratio_
            
            self.logger.info(f"PCA explained variance ratio (first 5): {self.explained_variance_ratio_[:5]}")
            self.logger.info(f"Total explained variance: {self.explained_variance_ratio_.sum():.3f}")
            
        elif self.method == 'tsne':
            # t-SNE doesn't have a separate fit method, will fit_transform directly
            self.reducer = TSNE(
                n_components=min(self.n_components, 3),  # t-SNE typically used for 2D/3D
                random_state=self.random_state,
                perplexity=min(30, embeddings.shape[0] // 4)  # Adjust perplexity for small datasets
            )
            self.logger.info("t-SNE will be fit during transform")
            
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'pca' or 'tsne'")
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted reducer.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Reduced embeddings
        """
        if self.scaler is None:
            raise ValueError("Must fit reducer before transforming")
        
        # Scale embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        if self.method == 'pca':
            if self.reducer is None:
                raise ValueError("PCA reducer not fitted")
            reduced = self.reducer.transform(embeddings_scaled)
            
        elif self.method == 'tsne':
            # For t-SNE, we need to fit_transform on all data at once
            self.logger.info("Performing t-SNE fit_transform")
            reduced = self.reducer.fit_transform(embeddings_scaled)
            
        self.logger.info(f"Reduced shape: {reduced.shape}")
        return reduced
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform embeddings in one step."""
        return self.fit(embeddings).transform(embeddings)
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio for PCA."""
        return self.explained_variance_ratio_
    
    def get_feature_names(self) -> list:
        """Generate feature names for reduced dimensions."""
        if self.method == 'pca':
            return [f'PC{i+1}' for i in range(self.n_components)]
        elif self.method == 'tsne':
            return [f'tSNE{i+1}' for i in range(self.n_components)]
        else:
            return [f'{self.method}_{i+1}' for i in range(self.n_components)]
    
    def save(self, filepath: str) -> None:
        """Save fitted reducer."""
        save_dict = {
            'method': self.method,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'reducer': self.reducer,
            'scaler': self.scaler,
            'explained_variance_ratio_': self.explained_variance_ratio_
        }
        joblib.dump(save_dict, filepath)
        self.logger.info(f"Reducer saved to {filepath}")
    
    def load(self, filepath: str) -> 'DimensionalityReducer':
        """Load fitted reducer."""
        save_dict = joblib.load(filepath)
        
        self.method = save_dict['method']
        self.n_components = save_dict['n_components']
        self.random_state = save_dict['random_state']
        self.reducer = save_dict['reducer']
        self.scaler = save_dict['scaler']
        self.explained_variance_ratio_ = save_dict['explained_variance_ratio_']
        
        self.logger.info(f"Reducer loaded from {filepath}")
        return self


class EmbeddingProcessor:
    """High-level processor that combines embedding extraction and dimensionality reduction."""
    
    def __init__(self,
                 embedder_config: Dict[str, Any],
                 reducer_config: Dict[str, Any]):
        """
        Initialize embedding processor.
        
        Args:
            embedder_config: Configuration for SequenceEmbedder
            reducer_config: Configuration for DimensionalityReducer
        """
        from .embeddings import SequenceEmbedder
        
        self.embedder = SequenceEmbedder(**embedder_config)
        self.reducer = DimensionalityReducer(**reducer_config)
        self.raw_embeddings = None
        self.reduced_embeddings = None
        
        self.logger = logging.getLogger(__name__)
    
    def process_sequences(self, sequences: list) -> pd.DataFrame:
        """
        Process sequences end-to-end: extract embeddings and reduce dimensions.
        
        Args:
            sequences: List of RNA/DNA sequences
            
        Returns:
            DataFrame with reduced embeddings
        """
        self.logger.info(f"Processing {len(sequences)} sequences end-to-end")
        
        # Extract embeddings
        self.raw_embeddings = self.embedder.extract_embeddings(sequences)
        
        # Reduce dimensions
        self.reduced_embeddings = self.reducer.fit_transform(self.raw_embeddings)
        
        # Create DataFrame
        feature_names = self.reducer.get_feature_names()
        df = pd.DataFrame(self.reduced_embeddings, columns=feature_names)
        
        self.logger.info(f"Final processed embeddings shape: {df.shape}")
        return df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing."""
        stats = {
            'raw_embedding_shape': self.raw_embeddings.shape if self.raw_embeddings is not None else None,
            'reduced_embedding_shape': self.reduced_embeddings.shape if self.reduced_embeddings is not None else None,
            'reduction_method': self.reducer.method,
            'n_components': self.reducer.n_components
        }
        
        if self.reducer.explained_variance_ratio_ is not None:
            stats['explained_variance_ratio'] = self.reducer.explained_variance_ratio_
            stats['total_explained_variance'] = self.reducer.explained_variance_ratio_.sum()
        
        return stats
    
    def save_all(self, base_filepath: str) -> None:
        """Save all components."""
        self.reducer.save(f"{base_filepath}_reducer.joblib")
        
        if self.raw_embeddings is not None:
            np.save(f"{base_filepath}_raw_embeddings.npy", self.raw_embeddings)
        
        if self.reduced_embeddings is not None:
            np.save(f"{base_filepath}_reduced_embeddings.npy", self.reduced_embeddings)
        
        self.logger.info(f"All components saved with base path: {base_filepath}")
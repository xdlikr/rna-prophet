#!/usr/bin/env python3
"""
Script to extract embeddings from RNA sequences using pretrained models.
This script can be used standalone for embedding extraction.
"""

import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.embeddings import SequenceEmbedder, EmbeddingConfig
from src.features.dimensionality import DimensionalityReducer, EmbeddingProcessor
from src.data.loader import RNADataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from RNA sequences')
    parser.add_argument('input_file', help='CSV file with RNA sequences')
    parser.add_argument('--output_dir', '-o', default='outputs/embeddings', 
                       help='Output directory for embeddings')
    parser.add_argument('--config', '-c', default='config/embedding_config.yaml',
                       help='Configuration file')
    parser.add_argument('--model', '-m', default='dnabert2',
                       help='Model to use (dnabert2, dnabert2_large, evo, evo2, evo2_large)')
    parser.add_argument('--n_components', '-n', type=int, default=128,
                       help='Number of PCA components')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                       help='Batch size for processing')
    parser.add_argument('--sequence_col', default='sequence',
                       help='Name of sequence column in CSV')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        config = {}
    
    # Load data
    print(f"Loading data from {args.input_file}")
    loader = RNADataLoader()
    df = loader.load_data(args.input_file)
    
    if args.sequence_col not in df.columns:
        raise ValueError(f"Column '{args.sequence_col}' not found in input file")
    
    sequences = df[args.sequence_col].tolist()
    print(f"Loaded {len(sequences)} sequences")
    
    # Get model configuration
    try:
        model_config = EmbeddingConfig.get_model_config(args.model)
    except ValueError:
        print(f"Unknown model: {args.model}")
        print("Available models:", list(EmbeddingConfig.list_available_models().keys()))
        return 1
    
    # Override batch size if provided
    if args.batch_size:
        model_config['batch_size'] = args.batch_size
    
    # Configure embedder
    embedder_config = {
        'model_name': model_config['name'],
        'max_length': model_config['max_length'],
        'batch_size': model_config['batch_size']
    }
    
    # Configure reducer
    reducer_config = {
        'method': 'pca',
        'n_components': args.n_components,
        'random_state': 42
    }
    
    print(f"Using model: {model_config['name']}")
    print(f"Reducing to {args.n_components} components")
    
    # Process sequences
    processor = EmbeddingProcessor(embedder_config, reducer_config)
    
    try:
        reduced_df = processor.process_sequences(sequences)
        
        # Add sequence information to output
        output_df = df.copy()
        for col in reduced_df.columns:
            output_df[col] = reduced_df[col]
        
        # Save results
        output_file = Path(args.output_dir) / f"embeddings_{args.model}_{args.n_components}d.csv"
        output_df.to_csv(output_file, index=False)
        print(f"Embeddings saved to {output_file}")
        
        # Save processing statistics
        stats = processor.get_processing_stats()
        stats_file = Path(args.output_dir) / f"embedding_stats_{args.model}.yaml"
        with open(stats_file, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        print(f"Statistics saved to {stats_file}")
        
        # Save models
        model_base = Path(args.output_dir) / f"models_{args.model}"
        processor.save_all(str(model_base))
        print(f"Models saved with base name {model_base}")
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
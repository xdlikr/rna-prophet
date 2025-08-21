#!/usr/bin/env python3
"""
Model Testing Script for RNA-Prophet

This script helps you test if embedding models (DNABERT-2, Evo, Evo2) work properly
on your system. It provides detailed feedback about what's working and what needs fixing.
"""

import sys
import os
import traceback
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test if basic dependencies are available."""
    print("🔍 Testing basic imports...")
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('torch', 'pytorch'),
        ('transformers', 'transformers')
    ]
    
    missing_packages = []
    
    for package, install_name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (install with: pip install {install_name})")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n🚨 Missing packages. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All basic imports successful!")
    return True


def test_rna_prophet_imports():
    """Test if RNA-Prophet modules import correctly."""
    print("\n🔍 Testing RNA-Prophet imports...")
    
    modules = [
        ('src.features.embeddings', 'SequenceEmbedder'),
        ('src.features.dimensionality', 'DimensionalityReducer'),
        ('src.data.loader', 'RNADataLoader'),
        ('src.models.pipeline', 'RNAPredictionPipeline')
    ]
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
            return False
    
    print("✅ All RNA-Prophet imports successful!")
    return True


def test_model_availability(model_key):
    """Test if a specific model can be loaded."""
    print(f"\n🤖 Testing {model_key} model availability...")
    
    try:
        from src.features.embeddings import SequenceEmbedder, EmbeddingConfig
        
        # Get model config
        try:
            config = EmbeddingConfig.get_model_config(model_key)
            print(f"  ✅ Model config found: {config['name']}")
        except ValueError as e:
            print(f"  ❌ Model config error: {e}")
            return False
        
        # Try to create embedder (don't load model yet)
        embedder = SequenceEmbedder(
            model_name=config['name'],
            max_length=min(config['max_length'], 512),  # Use shorter length for testing
            batch_size=2
        )
        print(f"  ✅ SequenceEmbedder created")
        
        # Try to load the model
        print(f"  🔄 Attempting to load model (this may take a while)...")
        start_time = time.time()
        
        try:
            embedder.load_model()
            load_time = time.time() - start_time
            print(f"  ✅ Model loaded successfully in {load_time:.1f}s")
            
            # Test embedding extraction
            test_sequences = ['ACGUACGUACGU', 'UGCAUGCAUGCA']
            print(f"  🔄 Testing embedding extraction...")
            
            embeddings = embedder.extract_embeddings(test_sequences)
            print(f"  ✅ Embeddings extracted: shape {embeddings.shape}")
            
            # Test embedding dimension
            dim = embedder.get_embedding_dim()
            print(f"  ✅ Embedding dimension: {dim}")
            
            return True
            
        except Exception as e:
            load_time = time.time() - start_time
            print(f"  ❌ Model loading failed after {load_time:.1f}s")
            print(f"     Error: {str(e)}")
            
            # Check if it's a network/download issue
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                print(f"  💡 This looks like a network issue. Try again with stable internet.")
            elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
                print(f"  💡 Model not found. This model might not be publicly available yet.")
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"  💡 Memory issue. Try reducing batch_size or use CPU.")
            
            return False
            
    except Exception as e:
        print(f"  ❌ Setup error: {str(e)}")
        traceback.print_exc()
        return False


def test_model_with_pipeline(model_key):
    """Test model integration with the full pipeline."""
    print(f"\n🔧 Testing {model_key} with pipeline integration...")
    
    try:
        from src.models.pipeline import RNAPredictionPipeline
        from src.data.loader import RNADataLoader
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'sequence': ['ACGU' * 50, 'UGCA' * 50, 'AAUU' * 50],  # 200 nt sequences
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [3.2, 8.1, 2.5],
            'expression': [1.4, 0.9, 1.8],
            'enzyme_type': ['T7', 'SP6', 'T7'],
            'temperature': [37.0, 42.0, 37.0]
        })
        
        print(f"  ✅ Test data created: {len(test_data)} sequences")
        
        # Get model config
        from src.features.embeddings import EmbeddingConfig
        config = EmbeddingConfig.get_model_config(model_key)
        
        # Create pipeline
        pipeline = RNAPredictionPipeline(
            embedding_config={
                'model_name': config['name'],
                'max_length': min(config['max_length'], 1024),
                'batch_size': 2
            },
            dimensionality_config={'method': 'pca', 'n_components': 5},
            model_config={'n_estimators': 3, 'random_state': 42}  # Very small for testing
        )
        print(f"  ✅ Pipeline created")
        
        # Test fitting
        print(f"  🔄 Testing pipeline fit (this may take a while)...")
        start_time = time.time()
        
        loader = RNADataLoader()
        target_columns = loader.get_target_columns(test_data)
        covariate_columns = loader.get_covariate_columns(test_data)
        
        y_data = test_data[target_columns]
        
        pipeline.fit(test_data, y_data,
                    covariate_columns=covariate_columns,
                    target_columns=target_columns)
        
        fit_time = time.time() - start_time
        print(f"  ✅ Pipeline fit successful in {fit_time:.1f}s")
        
        # Test prediction
        predictions = pipeline.predict(test_data)
        print(f"  ✅ Predictions generated: shape {predictions.shape}")
        
        # Test scoring
        score = pipeline.score(test_data, y_data)
        print(f"  ✅ Pipeline score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False


def suggest_alternatives(failed_models, working_models):
    """Suggest alternatives based on test results."""
    print(f"\n💡 Recommendations:")
    
    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
        print(f"   You can use any of these for RNA-Prophet")
        
        if 'dnabert2' in working_models:
            print(f"   🚀 Quick start: python main.py train data.csv --embedding-model dnabert2")
    
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
        
        for model in failed_models:
            if model.startswith('evo2'):
                print(f"   💡 {model}: May not be publicly available yet. Use 'evo' or 'dnabert2' instead")
            elif model.startswith('evo'):
                print(f"   💡 {model}: Try with better internet connection or use 'dnabert2'")
            elif model.startswith('dnabert2'):
                print(f"   💡 {model}: Check internet connection and try again")
    
    if not working_models:
        print(f"🚨 No models working. Check:")
        print(f"   1. Internet connection")
        print(f"   2. pip install transformers torch")
        print(f"   3. Available disk space (models are large)")
        print(f"   4. GPU memory if using CUDA")


def main():
    """Main testing function."""
    print("🧬 RNA-Prophet Model Testing Script")
    print("=" * 50)
    
    # Test basic setup
    if not test_basic_imports():
        print("\n🚨 Basic imports failed. Fix dependencies first.")
        return
    
    if not test_rna_prophet_imports():
        print("\n🚨 RNA-Prophet imports failed. Check installation.")
        return
    
    # Define models to test
    models_to_test = ['dnabert2', 'dnabert2_large', 'evo', 'evo2', 'evo2_large']
    
    print(f"\n🧪 Testing {len(models_to_test)} models...")
    print("Note: This may take several minutes as models are downloaded")
    
    working_models = []
    failed_models = []
    
    for model in models_to_test:
        print(f"\n{'='*20} Testing {model} {'='*20}")
        
        # Test basic model availability
        if test_model_availability(model):
            print(f"✅ {model}: Basic functionality works")
            
            # Test with pipeline (only for working models to save time)
            if test_model_with_pipeline(model):
                print(f"✅ {model}: Full pipeline integration works")
                working_models.append(model)
            else:
                print(f"⚠️  {model}: Basic works but pipeline integration failed")
                failed_models.append(model)
        else:
            print(f"❌ {model}: Not working")
            failed_models.append(model)
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"🎯 TESTING SUMMARY")
    print(f"{'='*50}")
    
    suggest_alternatives(failed_models, working_models)
    
    if working_models:
        print(f"\n🎉 Success! You can use RNA-Prophet with: {', '.join(working_models)}")
        print(f"\nNext steps:")
        print(f"1. Prepare your data in CSV format")
        print(f"2. Run: python main.py validate your_data.csv")
        print(f"3. Run: python main.py train your_data.csv --embedding-model {working_models[0]}")
    else:
        print(f"\n😞 No models are working. Please check the recommendations above.")


if __name__ == '__main__':
    main()
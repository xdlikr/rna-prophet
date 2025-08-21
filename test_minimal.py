#!/usr/bin/env python3
"""
Minimal Model Test Script

Quick test to check if DNABERT-2 works without downloading large models.
Use this first to verify basic setup.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_minimal():
    """Test basic functionality without downloading models."""
    print("🧬 RNA-Prophet Minimal Test")
    print("=" * 30)
    
    # Test 1: Basic imports
    print("1. Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        from src.features.embeddings import SequenceEmbedder, EmbeddingConfig
        from src.data.loader import RNADataLoader
        print("   ✅ All imports successful")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Model configurations
    print("2. Testing model configurations...")
    try:
        models = EmbeddingConfig.list_available_models()
        print(f"   ✅ Found {len(models)} models:")
        for model, desc in models.items():
            print(f"      - {model}: {desc}")
    except Exception as e:
        print(f"   ❌ Config failed: {e}")
        return False
    
    # Test 3: Data loading
    print("3. Testing data loading...")
    try:
        loader = RNADataLoader()
        
        # Test with sample data
        test_data = pd.DataFrame({
            'sequence': ['ACGUACGU', 'UGCAUGCA'],
            'yield': [0.8, 0.6],
            'dsRNA_percent': [3.2, 8.1],
            'expression': [1.4, 0.9]
        })
        
        loader._validate_data(test_data)
        targets = loader.get_target_columns(test_data)
        print(f"   ✅ Data validation works, targets: {targets}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # Test 4: CLI availability
    print("4. Testing CLI...")
    try:
        import main
        print("   ✅ CLI script accessible")
        print(f"   💡 Try: python main.py info --models")
    except Exception as e:
        print(f"   ❌ CLI failed: {e}")
        return False
    
    print("\n🎉 Minimal test passed!")
    print("\nNext steps:")
    print("1. Run: python test_models.py")
    print("2. Or try: python main.py info --models")
    print("3. Test with real data: python main.py validate config/sample_data.csv")
    
    return True

if __name__ == '__main__':
    test_minimal()
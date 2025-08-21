# 🧪 RNA-Prophet Model Testing Guide

This guide helps you test if embedding models work properly on your system.

## Quick Test (Recommended First)

Start with the minimal test to verify basic setup:

```bash
python test_minimal.py
```

This checks:
- ✅ All dependencies are installed
- ✅ RNA-Prophet modules import correctly  
- ✅ Model configurations are valid
- ✅ Data loading works
- ✅ CLI is accessible

## Full Model Test

Test actual model loading and embedding extraction:

```bash
python test_models.py
```

This comprehensive test:
- 🤖 Tests all 8 embedding models (DNABERT-2, Evo, Evo-2)
- 📥 Downloads models (may take time)
- 🧬 Extracts embeddings from test sequences
- 🔧 Tests full pipeline integration
- 💡 Provides specific recommendations

## Expected Results

### ✅ DNABERT-2 Models
These should work on most systems:
- `dnabert2` - Fast, reliable, 2k context
- `dnabert2_large` - Better accuracy, 2k context

### ⚠️ Evo Models  
May require good internet and more memory:
- `evo` - 8k context, RNA-native

### 🆕 Evo-2 Models (Official Arc Institute)
Now publicly available but with specific requirements:
- `evo2_7b` - 7B parameters, 1M context (recommended)
- `evo2_40b` - 40B parameters, 1M context (requires multiple GPUs)
- `evo2_7b_base` - 7B parameters, 8K context
- `evo2_40b_base` - 40B parameters, 8K context
- `evo2_1b_base` - 1B parameters, 8K context (lightweight)

**Requirements**: NVIDIA GPU with Compute Capability 8.9+, CUDA 12.1+, Python 3.12

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. Model Download Fails**
```bash
# Check internet connection
# Try again later (HuggingFace servers may be busy)
# Use working models (usually DNABERT-2)
```

**3. Out of Memory**
```bash
# Reduce batch size or use CPU
python main.py train data.csv --embedding-model dnabert2 --batch-size 4
```

**4. Evo-2 Models Fail to Load**
```bash
# Check requirements: NVIDIA GPU Compute Capability 8.9+, CUDA 12.1+, Python 3.12
# Install prerequisites:
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2

# Use alternatives if hardware incompatible:
python main.py train data.csv --embedding-model evo
python main.py train data.csv --embedding-model dnabert2
```

### Test Outputs

**✅ Success Example:**
```
🤖 Testing dnabert2 model availability...
  ✅ Model config found: zhihan1996/DNABERT-2-117M
  ✅ SequenceEmbedder created
  ✅ Model loaded successfully in 45.2s
  ✅ Embeddings extracted: shape (2, 768)
  ✅ Embedding dimension: 768
```

**❌ Failure Example:**
```
🤖 Testing evo2_7b model availability...
  ✅ Model config found: arcinstitute/evo2_7b
  ✅ SequenceEmbedder created
  ❌ Model loading failed after 30.1s
     Error: CUDA compute capability 8.9 required
  💡 Requires NVIDIA GPU with Compute Capability 8.9+, CUDA 12.1+, Python 3.12
     Install with: pip install evo2
```

## What to Do Next

### If All Models Work ✅
```bash
# You're all set! Use the best available model
python main.py train data.csv --embedding-model evo2_7b
```

### If Only DNABERT-2 Works ✅
```bash
# Use DNABERT-2 (perfectly fine for most use cases)
python main.py train data.csv --embedding-model dnabert2
```

### If No Models Work ❌
Check:
1. Internet connection
2. Available disk space (models are 1-10GB)
3. Python environment
4. Dependencies: `pip install -r requirements.txt`

## Model Recommendations

### For Testing/Development
```bash
--embedding-model dnabert2 --n-components 64
```

### For Production
```bash
--embedding-model dnabert2_large --n-components 128
```

### For Long Sequences (>2kb)
```bash
--embedding-model evo --n-components 256
```

### For Maximum Performance (if available)
```bash
--embedding-model evo2_large --n-components 512
```

## Performance Expectations

| Model | Download Time | Memory Usage | Accuracy |
|-------|---------------|--------------|----------|
| `dnabert2` | 2-5 min | 4GB | Good |
| `dnabert2_large` | 5-10 min | 8GB | Better |
| `evo` | 10-15 min | 12GB | Excellent |
| `evo2_7b` | 15-25 min | 24GB+ | Outstanding |
| `evo2_40b` | 45+ min | 100GB+ | Best |
| `evo2_1b_base` | 5-10 min | 8GB | Very Good |

**Note**: Evo-2 models require specific hardware (GPU Compute Capability 8.9+)

## Getting Help

If tests fail:
1. Run `python test_minimal.py` first
2. Check error messages carefully
3. Try different models
4. Reduce batch sizes for memory issues
5. Ensure stable internet for downloads

Remember: Even if only DNABERT-2 works, RNA-Prophet will still provide excellent results for most RNA property prediction tasks! 🧬
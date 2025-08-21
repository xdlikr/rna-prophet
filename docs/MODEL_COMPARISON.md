# RNA-Prophet Model Comparison Guide

This guide helps you choose the optimal embedding model for your RNA property prediction tasks.

## Available Models

### DNABERT-2 Models
| Model | Parameters | Max Length | Sequence Type | Best For |
|-------|------------|------------|---------------|----------|
| `dnabert2` | 117M | 2,048 | DNA | General RNA sequences, fast inference |
| `dnabert2_large` | 512M | 2,048 | DNA | Higher accuracy, moderate speed |

### Evo Models (RNA-Native)
| Model | Parameters | Max Length | Sequence Type | Best For |
|-------|------------|------------|---------------|----------|
| `evo` | ~7B | 8,192 | RNA | Long RNA sequences, RNA-specific features |
| `evo2` | ~7B | 131,072 | RNA | Very long sequences, genomic context |
| `evo2_large` | 650B | 131,072 | RNA | State-of-the-art performance, maximum accuracy |

## Model Selection Guide

### By Sequence Length
- **≤ 2kb sequences**: `dnabert2` or `dnabert2_large`
- **2-8kb sequences**: `evo` or `dnabert2_large`
- **8kb+ sequences**: `evo2` or `evo2_large`
- **Very long (>50kb)**: `evo2_large` only

### By Use Case
- **Fast prototyping**: `dnabert2`
- **Production accuracy**: `dnabert2_large` or `evo`
- **RNA-specific features**: `evo`, `evo2`, or `evo2_large`
- **Maximum performance**: `evo2_large`
- **Limited compute**: `dnabert2`

### By Dataset Size
- **Small (<1k sequences)**: `dnabert2` (faster training)
- **Medium (1k-10k)**: `dnabert2_large` or `evo`
- **Large (>10k)**: `evo2` or `evo2_large`

## Performance Characteristics

### Computational Requirements
| Model | GPU Memory | Batch Size | Speed | Training Time |
|-------|------------|------------|-------|---------------|
| `dnabert2` | 4-8GB | 32 | Fast | ~30min/1k seqs |
| `dnabert2_large` | 8-16GB | 16 | Medium | ~1h/1k seqs |
| `evo` | 8-16GB | 8 | Medium | ~1h/1k seqs |
| `evo2` | 16-24GB | 4 | Slow | ~2h/1k seqs |
| `evo2_large` | 40GB+ | 2 | Very Slow | ~4h/1k seqs |

### Expected Performance (RNA Property Prediction)
| Model | Typical R² | Strengths | Limitations |
|-------|------------|-----------|-------------|
| `dnabert2` | 0.6-0.75 | Fast, reliable, well-tested | DNA-centric, shorter context |
| `dnabert2_large` | 0.65-0.8 | Good balance of speed/accuracy | DNA-centric, shorter context |
| `evo` | 0.7-0.85 | RNA-native, longer context | Slower inference |
| `evo2` | 0.75-0.9 | Very long context, RNA-native | High memory requirements |
| `evo2_large` | 0.8-0.95 | State-of-the-art performance | Extreme compute requirements |

## Usage Examples

### Standard RNA Sequences (~2kb)
```bash
# Fast and reliable
python main.py train data.csv --embedding-model dnabert2

# Better accuracy
python main.py train data.csv --embedding-model dnabert2_large
```

### Long RNA Sequences (4-8kb)
```bash
# RNA-optimized
python main.py train data.csv --embedding-model evo --n-components 256

# Maximum context
python main.py train data.csv --embedding-model evo2 --n-components 512
```

### Very Long Sequences (>8kb)
```bash
# Only option for very long sequences
python main.py train data.csv --embedding-model evo2 --n-components 1024
```

### Production Deployment
```bash
# Best performance regardless of cost
python main.py train data.csv --embedding-model evo2_large --n-components 512
```

## Technical Differences

### Sequence Processing
- **DNABERT-2**: Converts RNA (U) → DNA (T) automatically
- **Evo models**: Native RNA processing, maintains U nucleotides
- **Context**: Evo models understand RNA secondary structure better

### Architecture
- **DNABERT-2**: BERT-based transformer, bidirectional attention
- **Evo**: Causal transformer optimized for genomic sequences
- **Evo-2**: Enhanced architecture with longer context capabilities

### Training Data
- **DNABERT-2**: Trained on DNA sequences from multiple species
- **Evo**: Trained on genomic sequences including RNA
- **Evo-2**: Trained on diverse genomic data with improved RNA representation

## Recommendations by Scenario

### Research & Development
```bash
# Quick experiments
--embedding-model dnabert2 --n-components 64

# Thorough analysis
--embedding-model evo --n-components 256
```

### Production Pipeline
```bash
# Balanced production
--embedding-model dnabert2_large --n-components 128

# High-end production
--embedding-model evo2 --n-components 512
```

### Limited Resources
```bash
# Minimal setup
--embedding-model dnabert2 --n-components 32 --batch-size 8
```

### Maximum Performance
```bash
# No resource constraints
--embedding-model evo2_large --n-components 1024 --batch-size 2
```

## Troubleshooting

### Out of Memory Errors
1. Reduce batch size: `--batch-size 4`
2. Reduce PCA components: `--n-components 64`
3. Use smaller model: `dnabert2` instead of `evo2`

### Slow Training
1. Use GPU if available
2. Increase batch size if memory allows
3. Consider smaller model for prototyping

### Poor Performance
1. Try RNA-native models (`evo`, `evo2`) for RNA sequences
2. Increase model size (`dnabert2_large`, `evo2_large`)
3. Increase PCA components for more sequence information
4. Check sequence length compatibility

## Future Models

RNA-Prophet is designed to easily accommodate new models. To add support for new embedding models:

1. Update `EmbeddingConfig.MODELS` in `src/features/embeddings.py`
2. Add configuration in `config/embedding_config.yaml`
3. Test with `python main.py info --models`

Stay tuned for updates as new genomic language models become available!
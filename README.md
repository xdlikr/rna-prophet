# 🧬 RNA-Prophet

AI-powered prediction of RNA production properties (IVT yield, dsRNA%, expression) with sequence-specific insights for biotech applications.

## Features

- **Multi-target Prediction**: Simultaneously predicts yield, dsRNA%, and expression
- **Advanced Models**: DNABERT-2, Evo, and Evo-2 for sequence embeddings
- **Long Sequence Support**: Up to 131k nucleotides with Evo-2
- **Sequence Insights**: Actionable recommendations for each RNA
- **Production Ready**: CLI interface with comprehensive reporting

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install ViennaRNA
conda install -c bioconda viennarna  # or brew install viennarna

# Initialize project
python main.py init my_project
cd my_project

# Train model
python main.py train data/your_data.csv

# Make predictions
python main.py predict models/rna_prediction_model.joblib data/new_sequences.csv
```

## Data Format

CSV with columns: `sequence`, `yield`, `dsRNA_percent`, `expression`, plus optional covariates.

```csv
sequence,yield,dsRNA_percent,expression,enzyme_type,temperature
ACGUACGU...,0.85,3.2,1.4,T7,37.0
```

## Model Options

| Model | Context | Best For |
|-------|---------|----------|
| `dnabert2` | 2k | Fast, general purpose |
| `evo` | 8k | Long RNA sequences |
| `evo2` | 131k | Very long sequences |
| `evo2_large` | 131k | Maximum accuracy |

```bash
# Use different models
python main.py train data.csv --embedding-model evo2
python main.py info --models  # see all options
```

## Commands

- `train` - Train prediction model
- `predict` - Make predictions on new sequences  
- `evaluate` - Evaluate model performance
- `validate` - Check data quality
- `info` - Show available models
- `init` - Create new project

## Architecture

1. **Sequence Embeddings**: Pretrained language models (DNABERT-2/Evo/Evo-2)
2. **Structure Features**: RNA folding analysis via ViennaRNA
3. **Multi-task XGBoost**: Predicts all targets simultaneously
4. **Sequence Insights**: Automated analysis and recommendations

## Performance

Typical R² scores: 0.6-0.9 depending on model and data quality.

## Requirements

- Python 3.8+
- 8-16GB RAM (more for large models)
- Optional: CUDA GPU for faster embedding extraction

## Citation

```bibtex
@software{rna_prophet,
  title={RNA-Prophet: AI-powered RNA property prediction},
  author={RNA-Prophet Contributors},
  year={2024},
  url={https://github.com/xdlikr/rna-prophet}
}
```

## License

MIT License
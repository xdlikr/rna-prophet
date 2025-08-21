#!/usr/bin/env python3
"""
Main CLI interface for RNA property prediction pipeline.

This script provides a command-line interface for training models, making predictions,
and generating reports for RNA sequence property prediction.
"""

import click
import pandas as pd
import yaml
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import RNADataLoader
from src.models.pipeline import RNAPredictionPipeline
from src.prediction.predictor import RNAPropertyPredictor
from src.reporting.generator import ReportGenerator
from src.utils.validation import DataValidator


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        click.echo(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', default='config/model_config.yaml', 
              help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose, config):
    """RNA Property Prediction Pipeline - Predict IVT yield, dsRNA%, and expression."""
    setup_logging(verbose)
    
    # Load configuration
    config_data = load_config(config)
    
    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config'] = config_data
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='outputs/models', 
              help='Output directory for trained model')
@click.option('--model-name', '-m', default='rna_prediction_model',
              help='Name for the saved model')
@click.option('--test-size', '-t', type=float, default=0.2,
              help='Fraction of data to use for testing')
@click.option('--embedding-model', default='dnabert2',
              help='Embedding model to use (dnabert2, dnabert2_large, evo, evo2, evo2_large)')
@click.option('--n-components', '-n', type=int, default=128,
              help='Number of PCA components for embeddings')
@click.option('--validate-data', is_flag=True,
              help='Run data validation before training')
@click.pass_context
def train(ctx, input_file, output_dir, model_name, test_size, 
          embedding_model, n_components, validate_data):
    """Train RNA property prediction model."""
    click.echo("🧬 Starting RNA Property Prediction Model Training")
    click.echo(f"📁 Input file: {input_file}")
    click.echo(f"💾 Output directory: {output_dir}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        click.echo("📊 Loading data...")
        loader = RNADataLoader()
        df = loader.load_data(input_file)
        click.echo(f"✅ Loaded {len(df)} sequences")
        
        # Data validation
        if validate_data:
            click.echo("🔍 Validating data...")
            validator = DataValidator()
            validation_results = validator.validate_dataset(df)
            validator.print_summary()
            
            # Check for critical issues
            if len(validation_results['sequence_validation']['invalid_sequences']) > 0:
                click.echo("❌ Found invalid sequences. Please fix data before training.")
                return
        
        # Split data
        click.echo(f"🔄 Splitting data (test_size={test_size})...")
        train_df, test_df = loader.split_data(df, test_size=test_size)
        click.echo(f"📈 Training samples: {len(train_df)}")
        click.echo(f"📉 Test samples: {len(test_df)}")
        
        # Get column information
        target_columns = loader.get_target_columns(df)
        covariate_columns = loader.get_covariate_columns(df)
        
        click.echo(f"🎯 Target columns: {target_columns}")
        click.echo(f"🔧 Covariate columns: {covariate_columns}")
        
        # Configure pipeline
        config = ctx.obj.get('config', {})
        
        # Embedding configuration
        from src.features.embeddings import EmbeddingConfig
        try:
            model_config = EmbeddingConfig.get_model_config(embedding_model)
            embedding_config = {
                'model_name': model_config['name'],
                'max_length': model_config['max_length'],
                'batch_size': model_config.get('batch_size', 32)
            }
        except ValueError:
            click.echo(f"❌ Unknown embedding model: {embedding_model}")
            click.echo(f"Available models: {list(EmbeddingConfig.list_available_models().keys())}")
            return
        
        # Dimensionality reduction config
        dimensionality_config = {
            'method': 'pca',
            'n_components': n_components,
            'random_state': 42
        }
        
        # Model configuration
        model_config = config.get('xgboost', {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        })
        
        # Create and train pipeline
        click.echo("🤖 Creating prediction pipeline...")
        pipeline = RNAPredictionPipeline(
            embedding_config=embedding_config,
            dimensionality_config=dimensionality_config,
            model_config=model_config
        )
        
        click.echo("🏋️ Training model...")
        X_train = train_df
        y_train = train_df[target_columns]
        
        with click.progressbar(length=1, label='Training model') as bar:
            pipeline.fit(X_train, y_train, 
                        covariate_columns=covariate_columns,
                        target_columns=target_columns)
            bar.update(1)
        
        click.echo("✅ Model training completed!")
        
        # Evaluate on test set
        click.echo("📊 Evaluating model...")
        X_test = test_df
        y_test = test_df[target_columns]
        
        score = pipeline.score(X_test, y_test)
        click.echo(f"🎯 Test R² Score: {score:.4f}")
        
        evaluation_results = pipeline.evaluate(X_test, y_test)
        for target, metrics in evaluation_results.items():
            r2 = metrics['r2']
            rmse = metrics['rmse']
            click.echo(f"   {target}: R²={r2:.4f}, RMSE={rmse:.4f}")
        
        # Save model
        model_path = Path(output_dir) / f"{model_name}.joblib"
        pipeline.save_pipeline(str(model_path))
        click.echo(f"💾 Model saved to {model_path}")
        
        # Save test predictions for analysis
        predictions = pipeline.predict(X_test)
        test_results = test_df.copy()
        for col in predictions.columns:
            test_results[f'predicted_{col}'] = predictions[col]
        
        results_path = Path(output_dir) / f"{model_name}_test_results.csv"
        test_results.to_csv(results_path, index=False)
        click.echo(f"📈 Test results saved to {results_path}")
        
        click.echo("🎉 Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='outputs/predictions',
              help='Output directory for predictions')
@click.option('--report-name', '-r', default='predictions',
              help='Base name for output files')
@click.option('--generate-report', is_flag=True,
              help='Generate comprehensive HTML report')
@click.pass_context
def predict(ctx, model_path, input_file, output_dir, report_name, generate_report):
    """Make predictions using a trained model."""
    click.echo("🔮 Starting RNA Property Prediction")
    click.echo(f"🤖 Model: {model_path}")
    click.echo(f"📁 Input file: {input_file}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        click.echo("📥 Loading model...")
        predictor = RNAPropertyPredictor()
        predictor.load_pipeline(model_path)
        click.echo("✅ Model loaded successfully")
        
        # Load input data
        click.echo("📊 Loading input data...")
        loader = RNADataLoader(required_columns=['sequence'])  # Only sequence required for prediction
        df = loader.load_data(input_file)
        click.echo(f"✅ Loaded {len(df)} sequences")
        
        # Make predictions
        click.echo("🔮 Generating predictions...")
        with click.progressbar(length=1, label='Making predictions') as bar:
            predictions_df = predictor.predict_sequences(df)
            bar.update(1)
        
        # Save predictions
        pred_file = Path(output_dir) / f"{report_name}_predictions.csv"
        predictions_df.to_csv(pred_file, index=False)
        click.echo(f"💾 Predictions saved to {pred_file}")
        
        # Generate sequence report
        click.echo("📋 Generating sequence analysis...")
        sequence_report = predictor.generate_sequence_report(df, top_n=min(10, len(df)))
        
        report_file = Path(output_dir) / f"{report_name}_sequence_analysis.csv"
        sequence_report.to_csv(report_file, index=False)
        click.echo(f"📊 Sequence analysis saved to {report_file}")
        
        # Print summary statistics
        pred_cols = [col for col in predictions_df.columns if col.startswith('predicted_')]
        click.echo("\n📈 Prediction Summary:")
        for col in pred_cols:
            target_name = col.replace('predicted_', '')
            values = predictions_df[col]
            click.echo(f"   {target_name}: mean={values.mean():.3f}, "
                      f"range=[{values.min():.3f}, {values.max():.3f}]")
        
        # Generate comprehensive report if requested
        if generate_report:
            click.echo("📊 Generating comprehensive report...")
            report_generator = ReportGenerator()
            
            report_files = report_generator.generate_comprehensive_report(
                predictor=predictor,
                X_test=df,
                y_test=None,  # No ground truth for pure prediction
                output_dir=str(Path(output_dir) / "reports"),
                report_name=report_name
            )
            
            click.echo("📋 Generated report files:")
            for report_type, file_path in report_files.items():
                click.echo(f"   {report_type}: {file_path}")
        
        click.echo("🎉 Prediction completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Prediction failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='outputs/evaluation',
              help='Output directory for evaluation results')
@click.option('--report-name', '-r', default='evaluation',
              help='Base name for output files')
@click.pass_context
def evaluate(ctx, model_path, test_file, output_dir, report_name):
    """Evaluate model performance on test data with ground truth."""
    click.echo("📊 Starting Model Evaluation")
    click.echo(f"🤖 Model: {model_path}")
    click.echo(f"📁 Test file: {test_file}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        click.echo("📥 Loading model...")
        predictor = RNAPropertyPredictor()
        predictor.load_pipeline(model_path)
        
        # Load test data
        click.echo("📊 Loading test data...")
        loader = RNADataLoader()
        df = loader.load_data(test_file)
        
        target_columns = loader.get_target_columns(df)
        click.echo(f"✅ Loaded {len(df)} sequences with targets: {target_columns}")
        
        # Prepare data
        X_test = df
        y_test = df[target_columns]
        
        # Evaluate model
        click.echo("📈 Evaluating model performance...")
        with click.progressbar(length=1, label='Running evaluation') as bar:
            evaluation_results = predictor.evaluate_predictions(
                X_test, y_test, save_plots=True, 
                plot_dir=str(Path(output_dir) / "plots")
            )
            bar.update(1)
        
        # Print results
        click.echo("\n🎯 Evaluation Results:")
        overall = evaluation_results['overall']
        click.echo(f"   Overall R²: {overall['mean_r2']:.4f} (±{overall['std_r2']:.4f})")
        click.echo(f"   Overall RMSE: {overall['overall_rmse']:.4f}")
        
        click.echo("\n📊 Per-Target Performance:")
        for target, metrics in evaluation_results['per_target'].items():
            click.echo(f"   {target}:")
            click.echo(f"     R²: {metrics['r2']:.4f}")
            click.echo(f"     RMSE: {metrics['rmse']:.4f}")
            click.echo(f"     MAE: {metrics['mae']:.4f}")
            click.echo(f"     Pearson r: {metrics['pearson_r']:.4f}")
        
        # Generate comprehensive report
        click.echo("📋 Generating evaluation report...")
        report_generator = ReportGenerator()
        
        report_files = report_generator.generate_comprehensive_report(
            predictor=predictor,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
            report_name=report_name
        )
        
        click.echo("\n📄 Generated report files:")
        for report_type, file_path in report_files.items():
            click.echo(f"   {report_type}: {file_path}")
        
        # Create performance summary card
        summary_card_path = Path(output_dir) / f"{report_name}_performance_card.txt"
        report_generator.create_model_summary_card(evaluation_results, str(summary_card_path))
        click.echo(f"🏆 Performance card: {summary_card_path}")
        
        click.echo("🎉 Evaluation completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-file', '-o', default='validation_report.txt',
              help='Output file for validation report')
@click.option('--show-details', is_flag=True,
              help='Show detailed validation information')
@click.pass_context
def validate(ctx, input_file, output_file, show_details):
    """Validate input data quality and format."""
    click.echo("🔍 Starting Data Validation")
    click.echo(f"📁 Input file: {input_file}")
    
    try:
        # Load data
        click.echo("📊 Loading data...")
        loader = RNADataLoader()
        df = loader.load_data(input_file)
        click.echo(f"✅ Loaded {len(df)} sequences")
        
        # Run validation
        click.echo("🔍 Running validation checks...")
        validator = DataValidator()
        validation_results = validator.validate_dataset(df)
        
        # Print summary
        validator.print_summary()
        
        # Save detailed report if requested
        if show_details or output_file != 'validation_report.txt':
            click.echo(f"💾 Saving detailed report to {output_file}")
            
            report_lines = []
            report_lines.append("DATA VALIDATION DETAILED REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {pd.Timestamp.now()}")
            report_lines.append(f"Input file: {input_file}")
            report_lines.append("")
            
            # Add validation results
            import json
            report_lines.append(json.dumps(validation_results, indent=2, default=str))
            
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
        
        # Check for critical issues
        critical_issues = []
        if validation_results['sequence_validation']['invalid_sequences']:
            critical_issues.append("Invalid sequences found")
        
        if validation_results['missing_data']['missing_percentage'] > 50:
            critical_issues.append("High missing data percentage")
        
        if critical_issues:
            click.echo("\n⚠️  Critical Issues Found:")
            for issue in critical_issues:
                click.echo(f"   - {issue}")
            click.echo("Please address these issues before training.")
        else:
            click.echo("\n✅ Data validation passed!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--models', is_flag=True, help='List available embedding models')
@click.option('--config', is_flag=True, help='Show current configuration')
@click.pass_context
def info(ctx, models, config):
    """Show information about available models and configuration."""
    
    if models:
        click.echo("🤖 Available Embedding Models:")
        from src.features.embeddings import EmbeddingConfig
        
        available_models = EmbeddingConfig.list_available_models()
        for model_key, description in available_models.items():
            model_config = EmbeddingConfig.get_model_config(model_key)
            click.echo(f"\n   {model_key}:")
            click.echo(f"     Description: {description}")
            click.echo(f"     Model: {model_config['name']}")
            click.echo(f"     Max length: {model_config['max_length']}")
    
    if config:
        click.echo("\n⚙️  Current Configuration:")
        config_data = ctx.obj.get('config', {})
        if config_data:
            import json
            click.echo(json.dumps(config_data, indent=2))
        else:
            click.echo("   No configuration loaded (using defaults)")


@cli.command()
@click.argument('output_dir', type=click.Path())
def init(output_dir):
    """Initialize a new project directory with example data and configuration."""
    click.echo(f"🚀 Initializing new RNA prediction project in {output_dir}")
    
    try:
        project_path = Path(output_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        directories = ['data', 'models', 'outputs', 'config']
        for dir_name in directories:
            (project_path / dir_name).mkdir(exist_ok=True)
        
        # Copy example data
        example_data_source = Path(__file__).parent / 'config' / 'sample_data.csv'
        example_data_dest = project_path / 'data' / 'example_data.csv'
        
        if example_data_source.exists():
            import shutil
            shutil.copy2(example_data_source, example_data_dest)
            click.echo(f"📊 Example data copied to {example_data_dest}")
        
        # Copy configuration files
        config_files = ['model_config.yaml', 'embedding_config.yaml']
        for config_file in config_files:
            config_source = Path(__file__).parent / 'config' / config_file
            config_dest = project_path / 'config' / config_file
            
            if config_source.exists():
                import shutil
                shutil.copy2(config_source, config_dest)
                click.echo(f"⚙️  Configuration copied to {config_dest}")
        
        # Create README
        readme_content = f"""# RNA Property Prediction Project

This project uses the RNA Property Prediction Pipeline to predict IVT yield, dsRNA%, and expression.

## Getting Started

1. Place your RNA sequence data in the `data/` directory
2. Configure model parameters in `config/model_config.yaml`
3. Train a model: `python main.py train data/your_data.csv`
4. Make predictions: `python main.py predict models/your_model.joblib data/new_sequences.csv`

## Directory Structure

- `data/`: Input data files
- `models/`: Trained model files
- `outputs/`: Prediction results and reports
- `config/`: Configuration files

## Example Data

Example data is provided in `data/example_data.csv` to test the pipeline.
"""
        
        readme_path = project_path / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        click.echo(f"📝 README created at {readme_path}")
        click.echo("🎉 Project initialization completed!")
        click.echo(f"\nNext steps:")
        click.echo(f"  cd {output_dir}")
        click.echo(f"  python main.py validate data/example_data.csv")
        click.echo(f"  python main.py train data/example_data.csv")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
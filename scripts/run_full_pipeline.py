#!/usr/bin/env python3
"""
Complete end-to-end pipeline script for RNA property prediction.

This script demonstrates the full workflow from data loading to final reporting.
"""

import pandas as pd
import yaml
import logging
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.loader import RNADataLoader
from src.models.pipeline import RNAPredictionPipeline
from src.prediction.predictor import RNAPropertyPredictor
from src.reporting.generator import ReportGenerator
from src.utils.validation import DataValidator
from src.features.embeddings import EmbeddingConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_full_pipeline(data_file: str,
                     output_dir: str = "outputs",
                     embedding_model: str = "dnabert2",
                     n_components: int = 128,
                     test_size: float = 0.2,
                     config_path: str = "config/model_config.yaml"):
    """
    Run the complete RNA property prediction pipeline.
    
    Args:
        data_file: Path to input CSV file
        output_dir: Output directory for all results
        embedding_model: Embedding model to use
        n_components: Number of PCA components
        test_size: Test split fraction
        config_path: Path to configuration file
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🧬 Starting Complete RNA Property Prediction Pipeline")
    logger.info(f"📁 Input data: {data_file}")
    logger.info(f"💾 Output directory: {output_dir}")
    
    # Create output directories
    output_path = Path(output_dir)
    for subdir in ['models', 'predictions', 'reports', 'plots']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load and validate data
        logger.info("📊 Step 1: Loading and validating data...")
        loader = RNADataLoader()
        df = loader.load_data(data_file)
        logger.info(f"✅ Loaded {len(df)} sequences")
        
        # Data validation
        validator = DataValidator()
        validation_results = validator.validate_dataset(df)
        validator.print_summary()
        
        # Check for critical issues
        invalid_sequences = validation_results['sequence_validation']['invalid_sequences']
        if invalid_sequences:
            logger.error(f"❌ Found {len(invalid_sequences)} invalid sequences")
            raise ValueError("Data validation failed - fix invalid sequences")
        
        # Step 2: Split data
        logger.info(f"🔄 Step 2: Splitting data (test_size={test_size})...")
        train_df, test_df = loader.split_data(df, test_size=test_size)
        logger.info(f"📈 Training samples: {len(train_df)}")
        logger.info(f"📉 Test samples: {len(test_df)}")
        
        # Get column information
        target_columns = loader.get_target_columns(df)
        covariate_columns = loader.get_covariate_columns(df)
        logger.info(f"🎯 Targets: {target_columns}")
        logger.info(f"🔧 Covariates: {covariate_columns}")
        
        # Step 3: Configure pipeline
        logger.info("⚙️  Step 3: Configuring pipeline...")
        
        # Load configuration
        config = load_config(config_path)
        
        # Embedding configuration
        try:
            model_config = EmbeddingConfig.get_model_config(embedding_model)
            embedding_config = {
                'model_name': model_config['name'],
                'max_length': model_config['max_length'],
                'batch_size': model_config.get('batch_size', 32)
            }
        except ValueError:
            logger.error(f"❌ Unknown embedding model: {embedding_model}")
            available = list(EmbeddingConfig.list_available_models().keys())
            logger.error(f"Available models: {available}")
            raise
        
        # Other configurations
        dimensionality_config = {
            'method': 'pca',
            'n_components': n_components,
            'random_state': 42
        }
        
        model_config = config.get('xgboost', {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        })
        
        # Step 4: Train model
        logger.info("🏋️ Step 4: Training model...")
        pipeline = RNAPredictionPipeline(
            embedding_config=embedding_config,
            dimensionality_config=dimensionality_config,
            model_config=model_config
        )
        
        X_train = train_df
        y_train = train_df[target_columns]
        
        pipeline.fit(X_train, y_train, 
                    covariate_columns=covariate_columns,
                    target_columns=target_columns)
        
        logger.info("✅ Model training completed")
        
        # Step 5: Evaluate model
        logger.info("📊 Step 5: Evaluating model...")
        X_test = test_df
        y_test = test_df[target_columns]
        
        test_score = pipeline.score(X_test, y_test)
        logger.info(f"🎯 Test R² Score: {test_score:.4f}")
        
        evaluation_results = pipeline.evaluate(X_test, y_test)
        for target, metrics in evaluation_results.items():
            logger.info(f"   {target}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        # Step 6: Save model
        logger.info("💾 Step 6: Saving model...")
        model_path = output_path / 'models' / 'rna_prediction_model.joblib'
        pipeline.save_pipeline(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Step 7: Generate predictions and reports
        logger.info("📋 Step 7: Generating comprehensive reports...")
        
        # Create predictor
        predictor = RNAPropertyPredictor(pipeline)
        
        # Generate comprehensive report
        report_generator = ReportGenerator()
        report_files = report_generator.generate_comprehensive_report(
            predictor=predictor,
            X_test=X_test,
            y_test=y_test,
            output_dir=str(output_path / 'reports'),
            report_name='rna_prediction_pipeline'
        )
        
        logger.info("📄 Generated report files:")
        for report_type, file_path in report_files.items():
            logger.info(f"   {report_type}: {file_path}")
        
        # Step 8: Create performance summary
        logger.info("🏆 Step 8: Creating performance summary...")
        
        summary_card_path = output_path / 'reports' / 'performance_summary.txt'
        report_generator.create_model_summary_card(
            predictor.evaluation_results, str(summary_card_path)
        )
        
        # Save pipeline metadata
        metadata = {
            'input_file': data_file,
            'output_directory': str(output_path),
            'model_configuration': {
                'embedding_model': embedding_model,
                'n_components': n_components,
                'test_size': test_size
            },
            'data_statistics': {
                'total_sequences': len(df),
                'training_sequences': len(train_df),
                'test_sequences': len(test_df),
                'target_columns': target_columns,
                'covariate_columns': covariate_columns
            },
            'performance_metrics': {
                'test_r2_score': test_score,
                'per_target_metrics': evaluation_results
            },
            'output_files': report_files
        }
        
        import json
        metadata_path = output_path / 'pipeline_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"📋 Pipeline metadata saved to {metadata_path}")
        
        # Final summary
        logger.info("\n🎉 Pipeline completed successfully!")
        logger.info("📊 Summary:")
        logger.info(f"   Model performance (R²): {test_score:.4f}")
        logger.info(f"   Number of sequences processed: {len(df)}")
        logger.info(f"   Output directory: {output_path}")
        logger.info(f"   Key files:")
        logger.info(f"     - Model: {model_path}")
        logger.info(f"     - Executive summary: {report_files.get('executive_summary', 'N/A')}")
        logger.info(f"     - Performance card: {summary_card_path}")
        
        return {
            'success': True,
            'model_path': str(model_path),
            'test_score': test_score,
            'output_directory': str(output_path),
            'report_files': report_files,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'output_directory': str(output_path)
        }


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete RNA property prediction pipeline')
    parser.add_argument('data_file', help='Path to input CSV file')
    parser.add_argument('--output-dir', '-o', default='outputs', 
                       help='Output directory (default: outputs)')
    parser.add_argument('--embedding-model', '-m', default='dnabert2',
                       help='Embedding model to use: dnabert2, dnabert2_large, evo, evo2, evo2_large (default: dnabert2)')
    parser.add_argument('--n-components', '-n', type=int, default=128,
                       help='Number of PCA components (default: 128)')
    parser.add_argument('--test-size', '-t', type=float, default=0.2,
                       help='Test split fraction (default: 0.2)')
    parser.add_argument('--config', '-c', default='config/model_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Run pipeline
    results = run_full_pipeline(
        data_file=args.data_file,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        n_components=args.n_components,
        test_size=args.test_size,
        config_path=args.config
    )
    
    if results['success']:
        print("\n✅ Pipeline completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
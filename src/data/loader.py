import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path


class RNADataLoader:
    """Loads and validates RNA sequence data from CSV files."""
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        self.required_columns = required_columns or [
            'sequence', 'yield', 'dsRNA_percent', 'expression'
        ]
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load RNA data from CSV file with validation."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        self._validate_data(df)
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist and data is clean."""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df['sequence'].isnull().any():
            raise ValueError("Found null sequences")
        
        if not all(df['sequence'].str.match(r'^[ACGU]+$', na=False)):
            raise ValueError("Invalid RNA sequences found (must contain only A,C,G,U)")
        
        target_cols = ['yield', 'dsRNA_percent', 'expression']
        for col in target_cols:
            if col in df.columns and df[col].isnull().any():
                print(f"Warning: Found null values in {col}")
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple train-test split without stratification."""
        n_test = max(1, int(len(df) * test_size))  # Ensure at least 1 test sample
        np.random.seed(random_state)
        test_indices = np.random.choice(len(df), n_test, replace=False)
        
        test_df = df.iloc[test_indices].reset_index(drop=True)
        train_df = df.drop(test_indices).reset_index(drop=True)
        
        return train_df, test_df
    
    def get_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract target column names from dataframe."""
        target_cols = ['yield', 'dsRNA_percent', 'expression']
        return [col for col in target_cols if col in df.columns]
    
    def get_covariate_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract covariate column names (excluding sequence and targets)."""
        exclude_cols = {'sequence'} | set(self.get_target_columns(df))
        return [col for col in df.columns if col not in exclude_cols]
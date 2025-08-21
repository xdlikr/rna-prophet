import pytest
import pandas as pd
import tempfile
import os
from src.data.loader import RNADataLoader


class TestRNADataLoader:
    
    def setup_method(self):
        self.loader = RNADataLoader()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'sequence': ['ACGU' * 10, 'UGCA' * 15, 'AAUU' * 8],
            'yield': [0.8, 0.6, 0.9],
            'dsRNA_percent': [5.2, 8.1, 3.4],
            'expression': [1.2, 0.8, 1.5],
            'enzyme_type': ['T7', 'SP6', 'T7'],
            'temperature': [37.0, 42.0, 37.0]
        })
    
    def test_load_valid_data(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            df = self.loader.load_data(temp_file)
            assert len(df) == 3
            assert 'sequence' in df.columns
            assert 'yield' in df.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load_data('nonexistent.csv')
    
    def test_validate_missing_columns(self):
        invalid_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            self.loader._validate_data(invalid_data)
    
    def test_validate_invalid_sequences(self):
        invalid_data = self.test_data.copy()
        invalid_data.loc[0, 'sequence'] = 'ACGTX'  # Invalid character
        with pytest.raises(ValueError, match="Invalid RNA sequences"):
            self.loader._validate_data(invalid_data)
    
    def test_split_data(self):
        train_df, test_df = self.loader.split_data(self.test_data, test_size=0.33)
        assert len(train_df) + len(test_df) == len(self.test_data)
        assert len(test_df) == 1  # 33% of 3 rounded down
    
    def test_get_target_columns(self):
        targets = self.loader.get_target_columns(self.test_data)
        expected = ['yield', 'dsRNA_percent', 'expression']
        assert set(targets) == set(expected)
    
    def test_get_covariate_columns(self):
        covariates = self.loader.get_covariate_columns(self.test_data)
        expected = ['enzyme_type', 'temperature']
        assert set(covariates) == set(expected)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple, Dict, Any
import joblib


class CovariatePreprocessor:
    """Handles encoding and standardization of covariates."""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.categorical_cols = []
        self.numerical_cols = []
    
    def fit(self, df: pd.DataFrame, covariate_columns: List[str]) -> 'CovariatePreprocessor':
        """Fit preprocessor on covariate columns."""
        if not covariate_columns:
            return self
        
        self._identify_column_types(df, covariate_columns)
        
        transformers = []
        if self.categorical_cols:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_cols)
            )
        if self.numerical_cols:
            transformers.append(
                ('num', StandardScaler(), self.numerical_cols)
            )
        
        if transformers:
            self.preprocessor = ColumnTransformer(transformers, remainder='drop')
            self.preprocessor.fit(df[covariate_columns])
            self._generate_feature_names()
        
        return self
    
    def transform(self, df: pd.DataFrame, covariate_columns: List[str]) -> pd.DataFrame:
        """Transform covariate columns."""
        if not covariate_columns or self.preprocessor is None:
            return pd.DataFrame()
        
        transformed = self.preprocessor.transform(df[covariate_columns])
        return pd.DataFrame(transformed, columns=self.feature_names, index=df.index)
    
    def fit_transform(self, df: pd.DataFrame, covariate_columns: List[str]) -> pd.DataFrame:
        """Fit and transform covariate columns."""
        return self.fit(df, covariate_columns).transform(df, covariate_columns)
    
    def _identify_column_types(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Identify categorical vs numerical columns."""
        for col in columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_cols.append(col)
            else:
                unique_vals = df[col].nunique()
                if unique_vals <= 10:  # Treat as categorical if few unique values
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
    
    def _generate_feature_names(self) -> None:
        """Generate feature names after transformation."""
        feature_names = []
        
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_names)
        
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        
        self.feature_names = feature_names
    
    def save(self, filepath: str) -> None:
        """Save fitted preprocessor."""
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols
        }, filepath)
    
    def load(self, filepath: str) -> 'CovariatePreprocessor':
        """Load fitted preprocessor."""
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.feature_names = data['feature_names']
        self.categorical_cols = data['categorical_cols']
        self.numerical_cols = data['numerical_cols']
        return self


class TargetProcessor:
    """Handles target variable processing."""
    
    def __init__(self):
        self.scalers = {}
        self.target_columns = []
    
    def fit(self, df: pd.DataFrame, target_columns: List[str]) -> 'TargetProcessor':
        """Fit scalers on target columns."""
        self.target_columns = target_columns
        for col in target_columns:
            scaler = StandardScaler()
            scaler.fit(df[[col]])
            self.scalers[col] = scaler
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform target columns."""
        result = df.copy()
        for col in self.target_columns:
            if col in df.columns:
                result[col] = self.scalers[col].transform(df[[col]]).flatten()
        return result[self.target_columns]
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform target columns."""
        result = df.copy()
        for col in self.target_columns:
            if col in df.columns:
                result[col] = self.scalers[col].inverse_transform(df[[col]]).flatten()
        return result
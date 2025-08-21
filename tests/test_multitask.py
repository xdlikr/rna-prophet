import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from src.models.multitask import MultiTaskXGBoost


class TestMultiTaskXGBoost:
    
    def setup_method(self):
        self.model = MultiTaskXGBoost(
            n_estimators=10,  # Small for faster tests
            random_state=42
        )
        
        # Create test data
        np.random.seed(42)
        self.X = np.random.randn(50, 10)  # 50 samples, 10 features
        self.y = np.random.randn(50, 3)   # 50 samples, 3 targets
        
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_df = pd.DataFrame(self.y, columns=['yield', 'dsRNA_percent', 'expression'])
    
    def test_init(self):
        assert self.model.n_estimators == 10
        assert self.model.learning_rate == 0.1
        assert self.model.random_state == 42
        assert not self.model.is_fitted_
    
    def test_get_xgb_params(self):
        params = self.model._get_xgb_params()
        
        assert params['n_estimators'] == 10
        assert params['learning_rate'] == 0.1
        assert params['random_state'] == 42
        assert params['n_jobs'] == -1
    
    def test_validate_input(self):
        # Test numpy array
        X_validated = self.model._validate_input(self.X)
        np.testing.assert_array_equal(X_validated, self.X)
        
        # Test DataFrame
        X_validated = self.model._validate_input(self.X_df)
        np.testing.assert_array_equal(X_validated, self.X_df.values)
    
    def test_validate_target(self):
        # Test 2D array
        y_validated = self.model._validate_target(self.y)
        np.testing.assert_array_equal(y_validated, self.y)
        
        # Test 1D array
        y_1d = self.y[:, 0]
        y_validated = self.model._validate_target(y_1d)
        assert y_validated.shape == (50, 1)
        
        # Test DataFrame
        y_validated = self.model._validate_target(self.y_df)
        np.testing.assert_array_equal(y_validated, self.y_df.values)
    
    def test_fit_and_predict(self):
        # Fit model
        self.model.fit(self.X, self.y)
        
        assert self.model.is_fitted_
        assert self.model.model is not None
        assert len(self.model.feature_names_) == 10
        assert len(self.model.target_names_) == 3
        
        # Make predictions
        predictions = self.model.predict(self.X)
        assert predictions.shape == (50, 3)
    
    def test_fit_with_feature_and_target_names(self):
        feature_names = [f'feat_{i}' for i in range(10)]
        target_names = ['target_1', 'target_2', 'target_3']
        
        self.model.fit(self.X, self.y, 
                      feature_names=feature_names,
                      target_names=target_names)
        
        assert self.model.feature_names_ == feature_names
        assert self.model.target_names_ == target_names
    
    def test_predict_before_fit_raises_error(self):
        with pytest.raises(ValueError, match="Model must be fitted"):
            self.model.predict(self.X)
    
    def test_score(self):
        self.model.fit(self.X, self.y)
        score = self.model.score(self.X, self.y)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1  # R² can be negative but usually positive for training data
    
    def test_get_feature_importance(self):
        self.model.fit(self.X, self.y)
        importance = self.model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 3  # Number of targets
        
        for target_name, imp_array in importance.items():
            assert len(imp_array) == 10  # Number of features
            assert all(imp >= 0 for imp in imp_array)  # Importance should be non-negative
    
    def test_get_feature_importance_df(self):
        self.model.fit(self.X, self.y)
        importance_df = self.model.get_feature_importance_df()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert importance_df.shape == (10, 3)  # 10 features, 3 targets
        assert list(importance_df.columns) == self.model.target_names_
    
    def test_evaluate_targets(self):
        self.model.fit(self.X, self.y)
        evaluation = self.model.evaluate_targets(self.X, self.y)
        
        assert isinstance(evaluation, dict)
        assert len(evaluation) == 3  # Number of targets
        
        for target_name, metrics in evaluation.items():
            assert 'r2' in metrics
            assert 'mse' in metrics
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_cross_validate(self):
        cv_results = self.model.cross_validate(self.X, self.y, cv=3)
        
        assert isinstance(cv_results, dict)
        assert 'scores' in cv_results
        assert 'mean' in cv_results
        assert 'std' in cv_results
        assert len(cv_results['scores']) == 3  # 3-fold CV
    
    def test_save_load_model(self, tmp_path):
        # Fit and save model
        self.model.fit(self.X, self.y)
        model_path = tmp_path / "test_model.joblib"
        self.model.save_model(str(model_path))
        
        # Create new model and load
        new_model = MultiTaskXGBoost()
        new_model.load_model(str(model_path))
        
        assert new_model.is_fitted_
        assert new_model.feature_names_ == self.model.feature_names_
        assert new_model.target_names_ == self.model.target_names_
        
        # Test that predictions are the same
        original_pred = self.model.predict(self.X)
        loaded_pred = new_model.predict(self.X)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_save_unfitted_model_raises_error(self, tmp_path):
        model_path = tmp_path / "test_model.joblib"
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            self.model.save_model(str(model_path))
    
    def test_get_set_params(self):
        # Test get_params
        params = self.model.get_params()
        assert params['n_estimators'] == 10
        assert params['learning_rate'] == 0.1
        
        # Test set_params
        self.model.set_params(n_estimators=20, learning_rate=0.2)
        assert self.model.n_estimators == 20
        assert self.model.learning_rate == 0.2
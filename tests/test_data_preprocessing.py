"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample metrics data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='1H'
        )
        
        data = []
        for i, timestamp in enumerate(timestamps):
            data.append({
                'timestamp': timestamp,
                'cpu_usage': 200 + np.random.normal(0, 50),
                'memory_usage': 512 + np.random.normal(0, 100),
                'cpu_requests': 500,
                'memory_requests': 1024,
                'cpu_limits': 1000,
                'memory_limits': 2048,
                'pod_restarts': np.random.poisson(0.1),
                'deployment_replicas': 3,
                'namespace': 'default',
                'deployment_name': 'test-app'
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config_file(self):
        """Create a temporary config file for testing."""
        config_content = """
prometheus:
  url: "http://prometheus:9090"
  timeout: 30
  retry_attempts: 3
  metrics:
    cpu_usage: "container_cpu_usage_seconds_total"
    memory_usage: "container_memory_usage_bytes"

kubernetes:
  config_path: "~/.kube/config"
  context: null
  namespace: "default"
  api_timeout: 30

model:
  type: "regression"
  architecture:
    layers: [128, 64, 32]
    dropout: 0.2
    activation: "relu"
    output_activation: "linear"
  
  training:
    epochs: 100
    batch_size: 32
    validation_split: 0.2
    early_stopping_patience: 10
    learning_rate: 0.001
    
  features:
    - "cpu_usage_p95"
    - "cpu_usage_p99"
    - "memory_usage_p95"
    - "memory_usage_p99"
    - "pod_restarts"
    - "current_cpu_requests"
    - "current_memory_requests"
    - "current_cpu_limits"
    - "current_memory_limits"
    - "deployment_replicas"
    - "pod_age_hours"

data:
  collection_period_days: 30
  scrape_interval_minutes: 5
  retention_days: 90
  min_data_points: 100
  
  preprocessing:
    outlier_threshold: 3.0
    smoothing_window: 5
    fill_missing: "interpolate"

recommendations:
  safety_margin_cpu: 1.2
  safety_margin_memory: 1.3
  min_cpu_request: 10
  min_memory_request: 64
  max_cpu_limit_multiplier: 4.0
  max_memory_limit_multiplier: 3.0
  
  confidence_threshold: 0.8
  min_recommendation_confidence: 0.7
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def preprocessor(self, config_file):
        """Create a DataPreprocessor instance."""
        return DataPreprocessor(config_file)
    
    def test_initialization(self, preprocessor):
        """Test DataPreprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.preprocessing_config is not None
        assert preprocessor.features is not None
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling."""
        # Add some missing values
        sample_data.loc[0, 'cpu_usage'] = np.nan
        sample_data.loc[1, 'memory_usage'] = np.nan
        
        result = preprocessor._handle_missing_values(sample_data.copy())
        
        # Check that missing values are filled
        assert not result['cpu_usage'].isna().any()
        assert not result['memory_usage'].isna().any()
    
    def test_remove_outliers(self, preprocessor, sample_data):
        """Test outlier removal."""
        # Add some outliers
        sample_data.loc[0, 'cpu_usage'] = 10000  # Extreme outlier
        sample_data.loc[1, 'memory_usage'] = 50000  # Extreme outlier
        
        original_len = len(sample_data)
        result = preprocessor._remove_outliers(sample_data.copy())
        
        # Check that outliers are removed
        assert len(result) < original_len
    
    def test_engineer_features(self, preprocessor, sample_data):
        """Test feature engineering."""
        result = preprocessor._engineer_features(sample_data.copy())
        
        # Check that new features are added
        expected_features = ['hour_of_day', 'day_of_week', 'is_weekend']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_targets(self, preprocessor, sample_data):
        """Test target variable creation."""
        # Add required features for target creation
        sample_data['cpu_usage_p95'] = sample_data['cpu_usage'].rolling(window=12).quantile(0.95)
        sample_data['memory_usage_p95'] = sample_data['memory_usage'].rolling(window=12).quantile(0.95)
        
        result = preprocessor._create_targets(sample_data.copy())
        
        # Check that target variables are created
        expected_targets = ['target_cpu_request', 'target_memory_request', 
                          'target_cpu_limit', 'target_memory_limit']
        for target in expected_targets:
            assert target in result.columns
    
    def test_prepare_training_data(self, preprocessor, sample_data):
        """Test training data preparation."""
        # Add required features and targets
        sample_data['cpu_usage_p95'] = sample_data['cpu_usage'].rolling(window=12).quantile(0.95)
        sample_data['memory_usage_p95'] = sample_data['memory_usage'].rolling(window=12).quantile(0.95)
        sample_data = preprocessor._create_targets(sample_data)
        
        X, y, feature_names = preprocessor.prepare_training_data(sample_data)
        
        # Check that X and y have correct shapes
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feature_names)
        assert y.shape[1] == 4  # 4 target variables
    
    def test_preprocess_data_integration(self, preprocessor, sample_data):
        """Test the complete preprocessing pipeline."""
        processed_df, metadata = preprocessor.preprocess_data(sample_data)
        
        # Check that preprocessing was successful
        assert not processed_df.empty
        assert 'original_shape' in metadata
        assert 'processed_shape' in metadata
        assert 'features_used' in metadata
    
    def test_empty_dataframe(self, preprocessor):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        processed_df, metadata = preprocessor.preprocess_data(empty_df)
        
        assert processed_df.empty
        assert metadata == {}
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling."""
        # Add some features to scale
        sample_data['cpu_usage_p95'] = sample_data['cpu_usage'].rolling(window=12).quantile(0.95)
        sample_data['memory_usage_p95'] = sample_data['memory_usage'].rolling(window=12).quantile(0.95)
        
        result, scalers = preprocessor._scale_features(sample_data.copy())
        
        # Check that scalers are created
        assert len(scalers) > 0
        
        # Check that scaled features have reasonable values
        for col in scalers.keys():
            assert result[col].std() > 0  # Should have some variance after scaling 
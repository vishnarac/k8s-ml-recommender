"""
Data preprocessing module for cleaning and preparing Kubernetes metrics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import yaml

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses Kubernetes metrics data for ML training."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the data preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['data']['preprocessing']
        self.features = self.config['model']['features']
        
        # Initialize scalers
        self.scalers = {}
        self.imputer = None
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess the raw metrics data.
        
        Args:
            df: Raw metrics DataFrame
            
        Returns:
            Tuple of (processed DataFrame, preprocessing metadata)
        """
        logger.info(f"Preprocessing {len(df)} data points")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for preprocessing")
            return df, {}
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Step 1: Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Step 2: Remove outliers
        processed_df = self._remove_outliers(processed_df)
        
        # Step 3: Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # Step 4: Select and validate features
        processed_df = self._select_features(processed_df)
        
        # Step 5: Scale features
        processed_df, scalers = self._scale_features(processed_df)
        
        # Step 6: Create target variables
        processed_df = self._create_targets(processed_df)
        
        # Store preprocessing metadata
        metadata = {
            'original_shape': df.shape,
            'processed_shape': processed_df.shape,
            'features_used': list(processed_df.columns),
            'scalers': scalers,
            'outliers_removed': len(df) - len(processed_df)
        }
        
        logger.info(f"Preprocessing complete. Final shape: {processed_df.shape}")
        return processed_df, metadata
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        
        # Check missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Initialize imputer if not already done
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # For non-numeric columns, use forward fill then backward fill
        if len(non_numeric_cols) > 0:
            df[non_numeric_cols] = df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using statistical methods."""
        logger.info("Removing outliers")
        
        outlier_threshold = self.preprocessing_config['outlier_threshold']
        original_len = len(df)
        
        # Remove outliers for numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['timestamp', 'hour_of_day', 'day_of_week', 'is_weekend']:
                continue  # Skip time-based features
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            # Mark outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            if outliers.sum() > 0:
                logger.info(f"Removing {outliers.sum()} outliers from {col}")
                df = df[~outliers]
        
        removed_count = original_len - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier rows")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better model performance."""
        logger.info("Engineering features")
        
        # Time-based features (if not already present)
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        # Resource efficiency features
        if all(col in df.columns for col in ['cpu_usage', 'cpu_requests']):
            df['cpu_efficiency'] = df['cpu_usage'] / (df['cpu_requests'] * 1000 + 1e-6)
        
        if all(col in df.columns for col in ['memory_usage', 'memory_requests']):
            df['memory_efficiency'] = df['memory_usage'] / (df['memory_requests'] + 1e-6)
        
        # Resource pressure features
        if all(col in df.columns for col in ['cpu_usage', 'cpu_limits']):
            df['cpu_pressure'] = df['cpu_usage'] / (df['cpu_limits'] * 1000 + 1e-6)
        
        if all(col in df.columns for col in ['memory_usage', 'memory_limits']):
            df['memory_pressure'] = df['memory_usage'] / (df['memory_limits'] + 1e-6)
        
        # Rolling statistics
        window_size = self.preprocessing_config['smoothing_window']
        
        for col in ['cpu_usage', 'memory_usage']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                df[f'{col}_rolling_max'] = df[col].rolling(window=window_size, min_periods=1).max()
        
        # Lag features for time series
        for col in ['cpu_usage', 'memory_usage']:
            if col in df.columns:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_6'] = df[col].shift(6)  # 30 minutes if 5-min intervals
        
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate required features."""
        logger.info("Selecting features")
        
        # Check which features are available
        available_features = []
        missing_features = []
        
        for feature in self.features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Add timestamp and deployment info if available
        additional_features = ['timestamp', 'namespace', 'deployment_name']
        for feature in additional_features:
            if feature in df.columns and feature not in available_features:
                available_features.append(feature)
        
        # Select only available features
        selected_df = df[available_features].copy()
        
        logger.info(f"Selected {len(available_features)} features: {available_features}")
        return selected_df
    
    def _scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale features for ML training."""
        logger.info("Scaling features")
        
        # Identify features to scale (exclude non-numeric and target variables)
        exclude_cols = ['timestamp', 'namespace', 'deployment_name']
        scale_cols = [col for col in df.columns if col not in exclude_cols]
        
        scalers = {}
        
        for col in scale_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                # Use RobustScaler for better outlier handling
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                scalers[col] = scaler
        
        logger.info(f"Scaled {len(scalers)} features")
        return df, scalers
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML training."""
        logger.info("Creating target variables")
        
        # Calculate optimal resource requests based on usage patterns
        if 'cpu_usage_p95' in df.columns:
            # CPU request = 95th percentile + safety margin
            safety_margin = self.config['recommendations']['safety_margin_cpu']
            df['target_cpu_request'] = df['cpu_usage_p95'] * safety_margin
            
            # Ensure minimum CPU request
            min_cpu = self.config['recommendations']['min_cpu_request']
            df['target_cpu_request'] = df['target_cpu_request'].clip(lower=min_cpu)
        
        if 'memory_usage_p95' in df.columns:
            # Memory request = 95th percentile + safety margin
            safety_margin = self.config['recommendations']['safety_margin_memory']
            df['target_memory_request'] = df['memory_usage_p95'] * safety_margin
            
            # Ensure minimum memory request
            min_memory = self.config['recommendations']['min_memory_request']
            df['target_memory_request'] = df['target_memory_request'].clip(lower=min_memory)
        
        # Calculate optimal resource limits
        if 'target_cpu_request' in df.columns:
            max_multiplier = self.config['recommendations']['max_cpu_limit_multiplier']
            df['target_cpu_limit'] = df['target_cpu_request'] * max_multiplier
        
        if 'target_memory_request' in df.columns:
            max_multiplier = self.config['recommendations']['max_memory_limit_multiplier']
            df['target_memory_limit'] = df['target_memory_request'] * max_multiplier
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for ML training.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing training data")
        
        # Define target columns
        target_cols = ['target_cpu_request', 'target_memory_request', 
                      'target_cpu_limit', 'target_memory_limit']
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = target_cols + ['timestamp', 'namespace', 'deployment_name']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with missing targets
        df_clean = df.dropna(subset=target_cols)
        
        if len(df_clean) == 0:
            raise ValueError("No valid training data after removing missing targets")
        
        # Prepare X and y
        X = df_clean[feature_cols].values
        y = df_clean[target_cols].values
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Targets: {target_cols}")
        
        return X, y, feature_cols
    
    def inverse_transform_targets(self, predictions: np.ndarray, scalers: Dict) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        if not scalers:
            return predictions
        
        # Apply inverse transformation for each target
        transformed_predictions = predictions.copy()
        
        target_cols = ['target_cpu_request', 'target_memory_request', 
                      'target_cpu_limit', 'target_memory_limit']
        
        for i, col in enumerate(target_cols):
            if col in scalers:
                transformed_predictions[:, i] = scalers[col].inverse_transform(
                    predictions[:, i].reshape(-1, 1)
                ).flatten()
        
        return transformed_predictions 
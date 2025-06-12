"""
TensorFlow model for Kubernetes resource recommendations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
import yaml
import json
import os

logger = logging.getLogger(__name__)


class ResourceRecommendationModel:
    """TensorFlow model for predicting optimal Kubernetes resource requests and limits."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the resource recommendation model.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.training_config = self.model_config['training']
        
        # Model components
        self.model = None
        self.history = None
        self.feature_names = None
        self.target_names = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def build_model(self, input_dim: int, output_dim: int = 4) -> keras.Model:
        """Build the TensorFlow model architecture.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets (default: 4 for cpu/memory requests/limits)
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with {input_dim} inputs and {output_dim} outputs")
        
        model_type = self.model_config['type']
        architecture = self.model_config['architecture']
        
        if model_type == "regression":
            model = self._build_regression_model(input_dim, output_dim, architecture)
        elif model_type == "lstm":
            model = self._build_lstm_model(input_dim, output_dim, architecture)
        elif model_type == "ensemble":
            model = self._build_ensemble_model(input_dim, output_dim, architecture)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Model built and compiled successfully")
        return model
    
    def _build_regression_model(self, input_dim: int, output_dim: int, architecture: Dict) -> keras.Model:
        """Build a feedforward neural network for regression."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in architecture['layers']:
            model.add(layers.Dense(units, activation=architecture['activation']))
            model.add(layers.Dropout(architecture['dropout']))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(output_dim, activation=architecture['output_activation']))
        
        return model
    
    def _build_lstm_model(self, input_dim: int, output_dim: int, architecture: Dict) -> keras.Model:
        """Build an LSTM model for time series prediction."""
        model = keras.Sequential()
        
        # Reshape input for LSTM (samples, timesteps, features)
        # For now, we'll use a single timestep, but this can be extended
        model.add(layers.Input(shape=(1, input_dim)))
        
        # LSTM layers
        for units in architecture['layers']:
            model.add(layers.LSTM(units, return_sequences=True, dropout=architecture['dropout']))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=architecture['activation']))
        model.add(layers.Dropout(architecture['dropout']))
        
        # Output layer
        model.add(layers.Dense(output_dim, activation=architecture['output_activation']))
        
        return model
    
    def _build_ensemble_model(self, input_dim: int, output_dim: int, architecture: Dict) -> keras.Model:
        """Build an ensemble model combining multiple architectures."""
        # This is a simplified ensemble - in production, you might want to train
        # multiple models and average their predictions
        
        # Main branch
        main_input = layers.Input(shape=(input_dim,))
        
        # Dense branch
        dense_branch = layers.Dense(128, activation=architecture['activation'])(main_input)
        dense_branch = layers.Dropout(architecture['dropout'])(dense_branch)
        dense_branch = layers.Dense(64, activation=architecture['activation'])(dense_branch)
        
        # Residual branch
        residual_branch = layers.Dense(64, activation=architecture['activation'])(main_input)
        residual_branch = layers.Dropout(architecture['dropout'])(residual_branch)
        
        # Combine branches
        combined = layers.Concatenate()([dense_branch, residual_branch])
        combined = layers.Dense(32, activation=architecture['activation'])(combined)
        combined = layers.Dropout(architecture['dropout'])(combined)
        
        # Output
        output = layers.Dense(output_dim, activation=architecture['output_activation'])(combined)
        
        model = keras.Model(inputs=main_input, outputs=output)
        return model
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        target_names: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            feature_names: List of feature names
            target_names: List of target names
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Training model with {len(X)} samples")
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Build model
        self.model = self.build_model(X.shape[1], y.shape[1])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Define callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        train_metrics = self.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = self.model.evaluate(X_val, y_val, verbose=0)
        
        metrics = {
            'train_loss': train_metrics[0],
            'train_mae': train_metrics[1],
            'train_mape': train_metrics[2],
            'val_loss': val_metrics[0],
            'val_mae': val_metrics[1],
            'val_mape': val_metrics[2],
            'feature_names': feature_names,
            'target_names': target_names
        }
        
        logger.info(f"Training completed. Final validation loss: {val_metrics[0]:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray, n_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals using Monte Carlo dropout.
        
        Args:
            X: Input features
            n_iterations: Number of forward passes for uncertainty estimation
            
        Returns:
            Tuple of (mean_predictions, confidence_intervals)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Enable dropout during inference
        predictions = []
        for _ in range(n_iterations):
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        confidence_intervals = np.std(predictions, axis=0)
        
        return mean_predictions, confidence_intervals
    
    def save_model(self, model_path: str):
        """Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_config': self.model_config,
            'training_history': self.history.history if self.history else None
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.target_names = metadata['target_names']
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using permutation importance.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained before calculating feature importance")
        
        # Baseline performance
        baseline_score = self.model.evaluate(X, np.zeros((X.shape[0], len(self.target_names))), verbose=0)[0]
        
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Permute the feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate performance with permuted feature
            permuted_score = self.model.evaluate(X_permuted, np.zeros((X.shape[0], len(self.target_names))), verbose=0)[0]
            
            # Importance is the increase in loss
            importance_scores[feature_name] = permuted_score - baseline_score
        
        # Normalize importance scores
        max_importance = max(importance_scores.values())
        if max_importance > 0:
            importance_scores = {k: v / max_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using feature contributions.
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Get the sample
        sample = X[sample_idx:sample_idx+1]
        
        # Make prediction
        prediction = self.predict(sample)[0]
        
        # Calculate feature contributions using integrated gradients approximation
        contributions = []
        baseline = np.zeros_like(sample)
        
        for i in range(50):  # Number of steps for approximation
            interpolated = baseline + (sample - baseline) * i / 50
            grad = tf.gradient(self.model(interpolated), interpolated)[0]
            contributions.append(grad.numpy())
        
        # Average contributions
        avg_contributions = np.mean(contributions, axis=0)[0]
        
        # Create explanation
        explanation = {
            'prediction': dict(zip(self.target_names, prediction)),
            'feature_contributions': dict(zip(self.feature_names, avg_contributions)),
            'sample_idx': sample_idx
        }
        
        return explanation 
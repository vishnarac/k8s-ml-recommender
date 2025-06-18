"""
Model management utilities for Kubernetes resource recommendation engine.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml

from .resource_model import ResourceRecommendationModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle operations."""
    
    def __init__(self, models_dir: str = "./models", config_path: str = "config.yaml"):
        """Initialize model manager.
        
        Args:
            models_dir: Directory to store models
            config_path: Path to configuration file
        """
        self.models_dir = models_dir
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_models_directory()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _ensure_models_directory(self):
        """Ensure models directory exists."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def get_model_path(self, namespace: str, deployment: str) -> str:
        """Get model file path for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            Path to model file
        """
        model_key = f"{namespace}_{deployment}"
        return os.path.join(self.models_dir, f"{model_key}.h5")
    
    def model_exists(self, namespace: str, deployment: str) -> bool:
        """Check if model exists for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            True if model exists
        """
        model_path = self.get_model_path(namespace, deployment)
        return os.path.exists(model_path)
    
    def load_model(self, namespace: str, deployment: str) -> Optional[ResourceRecommendationModel]:
        """Load model for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            Loaded model or None if not found
        """
        model_path = self.get_model_path(namespace, deployment)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            model = ResourceRecommendationModel()
            model.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return None
    
    def save_model(self, model: ResourceRecommendationModel, namespace: str, deployment: str) -> bool:
        """Save model for a deployment.
        
        Args:
            model: Model to save
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            True if saved successfully
        """
        model_path = self.get_model_path(namespace, deployment)
        
        try:
            model.save_model(model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")
            return False
    
    def delete_model(self, namespace: str, deployment: str) -> bool:
        """Delete model for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            True if deleted successfully
        """
        model_path = self.get_model_path(namespace, deployment)
        
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Deleted model: {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error deleting model {model_path}: {e}")
                return False
        else:
            logger.warning(f"Model not found: {model_path}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.h5'):
                filepath = os.path.join(self.models_dir, filename)
                stat = os.stat(filepath)
                
                # Parse namespace and deployment from filename
                model_key = filename.replace('.h5', '')
                parts = model_key.split('_', 1)
                
                if len(parts) == 2:
                    namespace, deployment = parts
                else:
                    namespace = "unknown"
                    deployment = model_key
                
                model_info = {
                    "filename": filename,
                    "filepath": filepath,
                    "namespace": namespace,
                    "deployment": deployment,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                }
                
                models.append(model_info)
        
        return models
    
    def get_model_info(self, namespace: str, deployment: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            Model information dictionary or None if not found
        """
        model_path = self.get_model_path(namespace, deployment)
        
        if not os.path.exists(model_path):
            return None
        
        stat = os.stat(model_path)
        
        return {
            "namespace": namespace,
            "deployment": deployment,
            "filepath": model_path,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "exists": True
        }
    
    def create_model(self, namespace: str, deployment: str, model_config: Optional[Dict[str, Any]] = None) -> ResourceRecommendationModel:
        """Create a new model for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            model_config: Model configuration
            
        Returns:
            New model instance
        """
        if model_config is None:
            model_config = self.config.get('model', {})
        
        model = ResourceRecommendationModel()
        
        # Apply configuration if provided
        if model_config:
            model.configure(model_config)
        
        logger.info(f"Created new model for {namespace}/{deployment}")
        return model
    
    def train_model(self, namespace: str, deployment: str, X, y, feature_names: List[str], target_names: List[str]) -> Dict[str, Any]:
        """Train a model for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            X: Training features
            y: Training targets
            feature_names: List of feature names
            target_names: List of target names
            
        Returns:
            Training metrics
        """
        # Create or load existing model
        if self.model_exists(namespace, deployment):
            model = self.load_model(namespace, deployment)
            if model is None:
                model = self.create_model(namespace, deployment)
        else:
            model = self.create_model(namespace, deployment)
        
        # Train the model
        training_metrics = model.train(X, y, feature_names, target_names)
        
        # Save the trained model
        if self.save_model(model, namespace, deployment):
            training_metrics['model_saved'] = True
            training_metrics['model_path'] = self.get_model_path(namespace, deployment)
        else:
            training_metrics['model_saved'] = False
        
        return training_metrics
    
    def predict(self, namespace: str, deployment: str, X) -> Optional[Any]:
        """Make predictions using a model.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            X: Input features
            
        Returns:
            Predictions or None if model not found
        """
        model = self.load_model(namespace, deployment)
        
        if model is None:
            logger.warning(f"No model found for {namespace}/{deployment}")
            return None
        
        try:
            predictions = model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def get_model_metadata(self, namespace: str, deployment: str) -> Optional[Dict[str, Any]]:
        """Get model metadata.
        
        Args:
            namespace: Kubernetes namespace
            deployment: Deployment name
            
        Returns:
            Model metadata or None if not found
        """
        model = self.load_model(namespace, deployment)
        
        if model is None:
            return None
        
        return {
            "namespace": namespace,
            "deployment": deployment,
            "model_type": getattr(model, 'model_type', 'unknown'),
            "feature_names": getattr(model, 'feature_names', []),
            "target_names": getattr(model, 'target_names', []),
            "training_history": getattr(model, 'training_history', {}),
            "model_config": getattr(model, 'config', {})
        }
    
    def cleanup_old_models(self, max_age_days: int = 30) -> int:
        """Clean up old models.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of models deleted
        """
        deleted_count = 0
        current_time = datetime.now()
        
        for model_info in self.list_models():
            age_days = (current_time - model_info['created']).days
            
            if age_days > max_age_days:
                if self.delete_model(model_info['namespace'], model_info['deployment']):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count 
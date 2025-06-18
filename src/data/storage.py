"""
Data storage utilities for Kubernetes resource recommendation engine.
"""

import os
import json
import pickle
import pandas as pd
from typing import Dict, Any, Optional, Union
import yaml
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles data persistence and retrieval operations."""
    
    def __init__(self, base_path: str = "./data"):
        """Initialize storage with base path.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.base_path,
            f"{self.base_path}/datasets",
            f"{self.base_path}/models",
            f"{self.base_path}/cache"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = "parquet") -> str:
        """Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            format: File format (parquet, csv, json)
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.base_path, "datasets", filename)
        
        if format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved DataFrame to {filepath}")
        return filepath
    
    def load_dataframe(self, filename: str, format: str = "parquet") -> pd.DataFrame:
        """Load DataFrame from file.
        
        Args:
            filename: Name of the file
            format: File format (parquet, csv, json)
            
        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.base_path, "datasets", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "parquet":
            df = pd.read_parquet(filepath)
        elif format == "csv":
            df = pd.read_csv(filepath)
        elif format == "json":
            df = pd.read_json(filepath, orient="records")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    
    def save_model(self, model: Any, filename: str) -> str:
        """Save model to file.
        
        Args:
            model: Model object to save
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.base_path, "models", filename)
        
        if hasattr(model, 'save'):
            # For TensorFlow/Keras models
            model.save(filepath)
        else:
            # For other models, use pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Saved model to {filepath}")
        return filepath
    
    def load_model(self, filename: str) -> Any:
        """Load model from file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Loaded model
        """
        filepath = os.path.join(self.base_path, "models", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Try to load as TensorFlow model first
        if filename.endswith('.h5') or os.path.isdir(filepath):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(filepath)
                logger.info(f"Loaded TensorFlow model from {filepath}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load as TensorFlow model: {e}")
        
        # Fall back to pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model from {filepath}")
        return model
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str) -> str:
        """Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.base_path, "cache", filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {filepath}")
        return filepath
    
    def load_metadata(self, filename: str) -> Dict[str, Any]:
        """Load metadata from JSON file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Loaded metadata
        """
        filepath = os.path.join(self.base_path, "cache", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata from {filepath}")
        return metadata
    
    def list_datasets(self) -> list:
        """List available datasets.
        
        Returns:
            List of dataset filenames
        """
        dataset_path = os.path.join(self.base_path, "datasets")
        if not os.path.exists(dataset_path):
            return []
        
        return [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    
    def list_models(self) -> list:
        """List available models.
        
        Returns:
            List of model filenames
        """
        model_path = os.path.join(self.base_path, "models")
        if not os.path.exists(model_path):
            return []
        
        return [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    
    def delete_file(self, filename: str, file_type: str = "datasets") -> bool:
        """Delete a file.
        
        Args:
            filename: Name of the file to delete
            file_type: Type of file (datasets, models, cache)
            
        Returns:
            True if deleted successfully
        """
        filepath = os.path.join(self.base_path, file_type, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted file: {filepath}")
            return True
        else:
            logger.warning(f"File not found: {filepath}")
            return False
    
    def get_file_info(self, filename: str, file_type: str = "datasets") -> Dict[str, Any]:
        """Get information about a file.
        
        Args:
            filename: Name of the file
            file_type: Type of file (datasets, models, cache)
            
        Returns:
            Dictionary with file information
        """
        filepath = os.path.join(self.base_path, file_type, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = os.stat(filepath)
        
        info = {
            "filename": filename,
            "filepath": filepath,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "exists": True
        }
        
        return info 
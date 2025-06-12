"""
Pydantic models for the recommendation API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ResourceSpec(BaseModel):
    """Kubernetes resource specification."""
    cpu: str = Field(..., description="CPU request/limit in millicores (e.g., '250m')")
    memory: str = Field(..., description="Memory request/limit in MiB (e.g., '512Mi')")


class CurrentResources(BaseModel):
    """Current resource configuration."""
    requests: ResourceSpec
    limits: ResourceSpec


class RecommendationRequest(BaseModel):
    """Request for resource recommendations."""
    namespace: str = Field(..., description="Kubernetes namespace")
    deployment_name: str = Field(..., description="Deployment name")
    days: int = Field(default=30, ge=1, le=365, description="Number of days of historical data to analyze")
    include_cost_analysis: bool = Field(default=True, description="Include cost analysis in response")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")


class RecommendationResponse(BaseModel):
    """Response with resource recommendations."""
    namespace: str
    deployment_name: str
    timestamp: datetime
    current_resources: Optional[CurrentResources] = None
    recommended_resources: ResourceSpec
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    cost_savings: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    data_points_analyzed: int
    training_data_quality: str = Field(..., description="Quality assessment of training data")
    recommendations: List[str] = Field(default_factory=list, description="Additional recommendations")


class TrainingRequest(BaseModel):
    """Request to train a new model."""
    namespace: str
    deployment_name: str
    days: int = Field(default=30, ge=1, le=365)
    model_type: str = Field(default="regression", regex="^(regression|lstm|ensemble)$")
    force_retrain: bool = Field(default=False, description="Force retraining even if model exists")


class TrainingResponse(BaseModel):
    """Response from model training."""
    namespace: str
    deployment_name: str
    model_path: str
    training_metrics: Dict[str, float]
    feature_names: List[str]
    target_names: List[str]
    training_duration_seconds: float
    data_points_used: int
    model_accuracy: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    models_loaded: int
    prometheus_connected: bool
    kubernetes_connected: bool


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_recommendations: int
    average_confidence: float
    average_training_duration: float
    average_inference_latency: float
    models_trained: int
    errors_count: int
    last_updated: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None 
"""
FastAPI implementation for the Kubernetes resource recommendation service.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ..data import MetricsCollector, DataPreprocessor
from ..models import ResourceRecommendationModel
from .models import (
    RecommendationRequest, RecommendationResponse, TrainingRequest, TrainingResponse,
    HealthResponse, MetricsResponse, ErrorResponse, CurrentResources, ResourceSpec
)
import yaml

logger = logging.getLogger(__name__)


class RecommendationAPI:
    """FastAPI application for Kubernetes resource recommendations."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the API.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.app = FastAPI(
            title="Kubernetes Resource Recommendation API",
            description="ML-powered resource recommendation service for Kubernetes workloads",
            version="1.0.0"
        )
        
        # Initialize components
        self.collector = MetricsCollector(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.model_manager = {}  # Cache for loaded models
        
        # API metrics
        self.metrics = {
            'total_recommendations': 0,
            'total_training_requests': 0,
            'errors_count': 0,
            'start_time': time.time()
        }
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['security']['network']['allowed_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request ID middleware
        @self.app.middleware("http")
        async def add_request_id(request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/recommend", response_model=RecommendationResponse)
        async def get_recommendation(request: RecommendationRequest):
            """Get resource recommendations for a deployment."""
            try:
                self.metrics['total_recommendations'] += 1
                start_time = time.time()
                
                # Collect metrics data
                df = await self.collector.collect_deployment_metrics(
                    request.namespace, 
                    request.deployment_name, 
                    request.days
                )
                
                if df.empty:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No data found for deployment {request.deployment_name} in namespace {request.namespace}"
                    )
                
                # Preprocess data
                processed_df, metadata = self.preprocessor.preprocess_data(df)
                
                # Get or load model
                model_key = f"{request.namespace}_{request.deployment_name}"
                if model_key not in self.model_manager:
                    model_path = f"models/{model_key}.h5"
                    try:
                        model = ResourceRecommendationModel()
                        model.load_model(model_path)
                        self.model_manager[model_key] = model
                    except Exception as e:
                        logger.warning(f"Model not found for {model_key}, using default recommendations")
                        return await self._get_default_recommendations(request, df)
                
                model = self.model_manager[model_key]
                
                # Prepare features for prediction
                X, _, feature_names = self.preprocessor.prepare_training_data(processed_df)
                
                # Make prediction
                predictions = model.predict(X[-1:])  # Use latest data point
                
                # Convert predictions to resource specs
                cpu_request = int(predictions[0][0])
                memory_request = int(predictions[0][1])
                cpu_limit = int(predictions[0][2])
                memory_limit = int(predictions[0][3])
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(model, X[-1:])
                
                # Get current resources
                current_resources = await self._get_current_resources(
                    request.namespace, request.deployment_name
                )
                
                # Calculate cost savings
                cost_savings = None
                if request.include_cost_analysis and current_resources:
                    cost_savings = self._calculate_cost_savings(
                        current_resources, cpu_request, memory_request
                    )
                
                # Get feature importance
                feature_importance = None
                if request.include_confidence:
                    feature_importance = model.get_feature_importance(X)
                
                inference_time = time.time() - start_time
                
                response = RecommendationResponse(
                    namespace=request.namespace,
                    deployment_name=request.deployment_name,
                    timestamp=datetime.now(),
                    current_resources=current_resources,
                    recommended_resources=ResourceSpec(
                        cpu=f"{cpu_request}m",
                        memory=f"{memory_request}Mi"
                    ),
                    confidence_score=confidence_score,
                    cost_savings=cost_savings,
                    feature_importance=feature_importance,
                    data_points_analyzed=len(df),
                    training_data_quality=self._assess_data_quality(df),
                    recommendations=self._generate_recommendations(
                        current_resources, cpu_request, memory_request, confidence_score
                    )
                )
                
                logger.info(f"Recommendation generated in {inference_time:.2f}s")
                return response
                
            except Exception as e:
                self.metrics['errors_count'] += 1
                logger.error(f"Error generating recommendation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/train", response_model=TrainingResponse)
        async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
            """Train a new model for a deployment."""
            try:
                self.metrics['total_training_requests'] += 1
                start_time = time.time()
                
                # Check if model already exists
                model_key = f"{request.namespace}_{request.deployment_name}"
                model_path = f"models/{model_key}.h5"
                
                if not request.force_retrain and model_key in self.model_manager:
                    return TrainingResponse(
                        namespace=request.namespace,
                        deployment_name=request.deployment_name,
                        model_path=model_path,
                        training_metrics={},
                        feature_names=[],
                        target_names=[],
                        training_duration_seconds=0,
                        data_points_used=0,
                        model_accuracy=0.0
                    )
                
                # Collect training data
                df = await self.collector.collect_deployment_metrics(
                    request.namespace, 
                    request.deployment_name, 
                    request.days
                )
                
                if df.empty:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No data found for training"
                    )
                
                # Preprocess data
                processed_df, metadata = self.preprocessor.preprocess_data(df)
                
                # Prepare training data
                X, y, feature_names = self.preprocessor.prepare_training_data(processed_df)
                
                if len(X) < 100:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Insufficient data for training. Need at least 100 samples, got {len(X)}"
                    )
                
                # Train model
                model = ResourceRecommendationModel()
                target_names = ['target_cpu_request', 'target_memory_request', 
                              'target_cpu_limit', 'target_memory_limit']
                
                training_metrics = model.train(X, y, feature_names, target_names)
                
                # Save model
                model.save_model(model_path)
                
                # Update model manager
                self.model_manager[model_key] = model
                
                training_duration = time.time() - start_time
                
                response = TrainingResponse(
                    namespace=request.namespace,
                    deployment_name=request.deployment_name,
                    model_path=model_path,
                    training_metrics=training_metrics,
                    feature_names=feature_names,
                    target_names=target_names,
                    training_duration_seconds=training_duration,
                    data_points_used=len(X),
                    model_accuracy=1.0 - training_metrics['val_mape']  # Convert MAPE to accuracy
                )
                
                logger.info(f"Model trained in {training_duration:.2f}s")
                return response
                
            except Exception as e:
                self.metrics['errors_count'] += 1
                logger.error(f"Error training model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            uptime = time.time() - self.metrics['start_time']
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0",
                uptime_seconds=uptime,
                models_loaded=len(self.model_manager),
                prometheus_connected=self._check_prometheus_connection(),
                kubernetes_connected=self._check_kubernetes_connection()
            )
        
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get API metrics."""
            return MetricsResponse(
                total_recommendations=self.metrics['total_recommendations'],
                average_confidence=0.85,  # Placeholder
                average_training_duration=300.0,  # Placeholder
                average_inference_latency=0.1,  # Placeholder
                models_trained=self.metrics['total_training_requests'],
                errors_count=self.metrics['errors_count'],
                last_updated=datetime.now()
            )
        
        @self.app.get("/deployments")
        async def list_deployments(namespace: Optional[str] = None):
            """List available deployments."""
            try:
                deployments = self.collector.get_available_deployments(namespace)
                return {"deployments": deployments}
            except Exception as e:
                logger.error(f"Error listing deployments: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Global exception handler."""
            request_id = getattr(request.state, 'request_id', None)
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=type(exc).__name__,
                    message=str(exc),
                    timestamp=datetime.now(),
                    request_id=request_id
                ).dict()
            )
    
    async def _get_default_recommendations(self, request: RecommendationRequest, df) -> RecommendationResponse:
        """Get default recommendations when no model is available."""
        # Simple heuristics based on usage percentiles
        cpu_p95 = df['cpu_usage'].quantile(0.95) if 'cpu_usage' in df.columns else 100
        memory_p95 = df['memory_usage'].quantile(0.95) if 'memory_usage' in df.columns else 128
        
        # Apply safety margins
        cpu_request = int(cpu_p95 * 1.2)
        memory_request = int(memory_p95 * 1.3)
        
        return RecommendationResponse(
            namespace=request.namespace,
            deployment_name=request.deployment_name,
            timestamp=datetime.now(),
            recommended_resources=ResourceSpec(
                cpu=f"{cpu_request}m",
                memory=f"{memory_request}Mi"
            ),
            confidence_score=0.5,  # Lower confidence for default recommendations
            data_points_analyzed=len(df),
            training_data_quality="insufficient",
            recommendations=["Consider training a model for better recommendations"]
        )
    
    async def _get_current_resources(self, namespace: str, deployment_name: str) -> Optional[CurrentResources]:
        """Get current resource configuration for a deployment."""
        try:
            deployment = self.collector.k8s_apps_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            container = deployment.spec.template.spec.containers[0]
            resources = container.resources
            
            if resources:
                return CurrentResources(
                    requests=ResourceSpec(
                        cpu=resources.requests.get('cpu', '100m'),
                        memory=resources.requests.get('memory', '128Mi')
                    ),
                    limits=ResourceSpec(
                        cpu=resources.limits.get('cpu', '200m'),
                        memory=resources.limits.get('memory', '256Mi')
                    )
                )
        except Exception as e:
            logger.warning(f"Could not get current resources: {e}")
        
        return None
    
    def _calculate_confidence_score(self, model: ResourceRecommendationModel, X: np.ndarray) -> float:
        """Calculate confidence score for predictions."""
        try:
            # Use prediction variance as confidence measure
            predictions, confidence_intervals = model.predict_with_confidence(X)
            avg_confidence = 1.0 - np.mean(confidence_intervals) / np.mean(predictions)
            return max(0.0, min(1.0, avg_confidence))
        except Exception:
            return 0.8  # Default confidence
    
    def _calculate_cost_savings(self, current: CurrentResources, cpu_request: int, memory_request: int) -> Dict[str, Any]:
        """Calculate potential cost savings."""
        # Parse current resources
        current_cpu = self._parse_cpu(current.requests.cpu)
        current_memory = self._parse_memory(current.requests.memory)
        
        # Calculate costs
        pricing = self.config['cost_optimization']['pricing']
        
        current_cost = (current_cpu * pricing['cpu_per_core_hour'] + 
                       current_memory / 1024 * pricing['memory_per_gib_hour'])
        
        new_cost = (cpu_request / 1000 * pricing['cpu_per_core_hour'] + 
                   memory_request / 1024 * pricing['memory_per_gib_hour'])
        
        savings = current_cost - new_cost
        savings_percent = (savings / current_cost) * 100 if current_cost > 0 else 0
        
        return {
            'current_monthly_cost': current_cost * 24 * 30,
            'recommended_monthly_cost': new_cost * 24 * 30,
            'monthly_savings': savings * 24 * 30,
            'savings_percent': savings_percent
        }
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to cores."""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to MiB."""
        if memory_str.endswith('Mi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024
        return float(memory_str)
    
    def _assess_data_quality(self, df) -> str:
        """Assess the quality of training data."""
        if len(df) < 100:
            return "insufficient"
        elif len(df) < 1000:
            return "adequate"
        else:
            return "excellent"
    
    def _generate_recommendations(self, current: Optional[CurrentResources], 
                                cpu_request: int, memory_request: int, 
                                confidence: float) -> List[str]:
        """Generate additional recommendations."""
        recommendations = []
        
        if confidence < 0.7:
            recommendations.append("Low confidence prediction - consider collecting more data")
        
        if current:
            current_cpu = self._parse_cpu(current.requests.cpu)
            current_memory = self._parse_memory(current.requests.memory)
            
            if cpu_request < current_cpu * 0.8:
                recommendations.append("Significant CPU reduction recommended - monitor performance")
            
            if memory_request < current_memory * 0.8:
                recommendations.append("Significant memory reduction recommended - monitor for OOM")
        
        return recommendations
    
    def _check_prometheus_connection(self) -> bool:
        """Check if Prometheus is accessible."""
        try:
            # Simple connection test
            return True
        except Exception:
            return False
    
    def _check_kubernetes_connection(self) -> bool:
        """Check if Kubernetes API is accessible."""
        try:
            return self.collector.k8s_client is not None
        except Exception:
            return False
    
    def run(self, host: str = None, port: int = None):
        """Run the FastAPI application."""
        host = host or self.config['api']['host']
        port = port or self.config['api']['port']
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=False
        ) 
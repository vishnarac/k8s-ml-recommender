#!/usr/bin/env python3
"""
Test script for the k8s-ml-recommender API using sample data.
"""

import requests
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    console.print(Panel.fit("🧪 Testing K8s ML Recommender API", style="bold blue"))
    
    # Test health endpoint
    console.print("\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            console.print(f"✅ Health check passed: {health_data['status']}")
            console.print(f"   Version: {health_data['version']}")
            console.print(f"   Uptime: {health_data['uptime_seconds']:.1f}s")
        else:
            console.print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        console.print(f"❌ Health check error: {e}")
        return
    
    # Test metrics endpoint
    console.print("\n2. Testing Metrics Endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            console.print(f"✅ Metrics retrieved successfully")
            console.print(f"   Total recommendations: {metrics_data['total_recommendations']}")
            console.print(f"   Models trained: {metrics_data['models_trained']}")
        else:
            console.print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        console.print(f"❌ Metrics error: {e}")
    
    # Test recommendation endpoint (will fail without real data, but shows the API structure)
    console.print("\n3. Testing Recommendation Endpoint...")
    try:
        payload = {
            "namespace": "default",
            "deployment_name": "web-app",
            "days": 30,
            "include_cost_analysis": True,
            "include_confidence": True
        }
        
        response = requests.post(f"{base_url}/recommend", json=payload)
        if response.status_code == 200:
            rec_data = response.json()
            console.print(f"✅ Recommendation generated successfully")
            console.print(f"   Confidence: {rec_data['confidence_score']:.2f}")
            console.print(f"   Data points: {rec_data['data_points_analyzed']}")
        else:
            error_data = response.json()
            console.print(f"⚠️  Recommendation failed (expected without real data): {error_data['detail']}")
    except Exception as e:
        console.print(f"❌ Recommendation error: {e}")
    
    # Test training endpoint
    console.print("\n4. Testing Training Endpoint...")
    try:
        payload = {
            "namespace": "default",
            "deployment_name": "web-app",
            "days": 30,
            "model_type": "regression",
            "force_retrain": False
        }
        
        response = requests.post(f"{base_url}/train", json=payload)
        if response.status_code == 200:
            train_data = response.json()
            console.print(f"✅ Training completed successfully")
            console.print(f"   Model path: {train_data['model_path']}")
            console.print(f"   Accuracy: {train_data['model_accuracy']:.2f}")
        else:
            error_data = response.json()
            console.print(f"⚠️  Training failed (expected without real data): {error_data['detail']}")
    except Exception as e:
        console.print(f"❌ Training error: {e}")
    
    # Display API documentation
    console.print("\n" + "="*50)
    console.print(Panel.fit(
        "📚 API Documentation Available at:\n"
        "   http://localhost:8000/docs\n"
        "   http://localhost:8000/redoc\n\n"
        "🔧 Available Endpoints:\n"
        "   GET  /health - Health check\n"
        "   GET  /metrics - API metrics\n"
        "   POST /recommend - Get recommendations\n"
        "   POST /train - Train model\n"
        "   GET  /deployments - List deployments",
        title="API Information",
        border_style="green"
    ))


if __name__ == "__main__":
    test_api() 
# Kubernetes Resource Recommendation Engine - Usage Examples

This document provides comprehensive examples of how to use the Kubernetes Resource Recommendation Engine.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd k8s-ml-recommender

# Install dependencies
pip install -r requirements.txt

# Build Docker image (optional)
docker build -t k8s-resource-recommender .
```

### 2. Generate Sample Data

```bash
# Generate sample data for testing
python data/sample_dataset.py
```

### 3. Basic Usage

```bash
# Get recommendations for a deployment
python cli.py recommend --namespace default --deployment my-app

# Train a model for a deployment
python cli.py train --namespace default --deployment my-app --days 30

# Start the API server
python cli.py serve --host 0.0.0.0 --port 8000
```

## 📊 CLI Examples

### Getting Recommendations

```bash
# Basic recommendation
python cli.py recommend --namespace default --deployment web-app

# With custom time range
python cli.py recommend --namespace default --deployment web-app --days 60

# Output in different formats
python cli.py recommend --namespace default --deployment web-app --output json
python cli.py recommend --namespace default --deployment web-app --output yaml

# Include cost analysis
python cli.py recommend --namespace default --deployment web-app --include-costs
```

### Training Models

```bash
# Train a basic regression model
python cli.py train --namespace default --deployment web-app --days 30

# Train with different model types
python cli.py train --namespace default --deployment web-app --model-type lstm
python cli.py train --namespace default --deployment web-app --model-type ensemble

# Force retraining
python cli.py train --namespace default --deployment web-app --force
```

### Analyzing Deployments

```bash
# List all deployments
python cli.py list-deployments

# List deployments in specific namespace
python cli.py list-deployments --namespace production

# Analyze resource usage patterns
python cli.py analyze --namespace default --deployment web-app --days 30
```

## 🌐 API Examples

### Starting the API Server

```bash
# Start with default settings
python api.py

# Start with custom settings
python cli.py serve --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "deployment_name": "web-app",
    "days": 30,
    "include_cost_analysis": true,
    "include_confidence": true
  }'
```

Response:
```json
{
  "namespace": "default",
  "deployment_name": "web-app",
  "timestamp": "2024-01-15T10:30:00Z",
  "current_resources": {
    "requests": {
      "cpu": "500m",
      "memory": "1024Mi"
    },
    "limits": {
      "cpu": "1000m",
      "memory": "2048Mi"
    }
  },
  "recommended_resources": {
    "cpu": "250m",
    "memory": "512Mi"
  },
  "confidence_score": 0.85,
  "cost_savings": {
    "current_monthly_cost": 45.60,
    "recommended_monthly_cost": 22.80,
    "monthly_savings": 22.80,
    "savings_percent": 50.0
  },
  "feature_importance": {
    "cpu_usage_p95": 0.8,
    "memory_usage_p95": 0.7,
    "pod_restarts": 0.3
  },
  "data_points_analyzed": 8640,
  "training_data_quality": "excellent",
  "recommendations": [
    "Significant CPU reduction recommended - monitor performance",
    "Significant memory reduction recommended - monitor for OOM"
  ]
}
```

#### Train a Model

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "deployment_name": "web-app",
    "days": 30,
    "model_type": "regression",
    "force_retrain": false
  }'
```

#### Health Check

```bash
curl "http://localhost:8000/health"
```

#### Get Metrics

```bash
curl "http://localhost:8000/metrics"
```

#### List Deployments

```bash
curl "http://localhost:8000/deployments?namespace=default"
```

## 🐳 Kubernetes Deployment

### Deploy to Kubernetes

```bash
# Create namespace and RBAC
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/service-account.yaml

# Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# Deploy the application
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -n resource-recommender
kubectl logs -f deployment/k8s-resource-recommender -n resource-recommender
```

### Access the Service

```bash
# Port forward to access the API
kubectl port-forward service/k8s-resource-recommender 8000:8000 -n resource-recommender

# Access the API
curl http://localhost:8000/health
```

## 🔧 Configuration

### Custom Configuration

Create a custom `config.yaml`:

```yaml
# Prometheus Configuration
prometheus:
  url: "http://your-prometheus:9090"
  timeout: 30
  retry_attempts: 3

# Model Configuration
model:
  type: "regression"
  architecture:
    layers: [256, 128, 64]
    dropout: 0.3
    activation: "relu"
  
  training:
    epochs: 150
    batch_size: 64
    learning_rate: 0.0005

# Recommendations
recommendations:
  safety_margin_cpu: 1.3  # 30% buffer
  safety_margin_memory: 1.4  # 40% buffer
  min_cpu_request: 25  # millicores
  min_memory_request: 128  # MiB

# Cost Optimization
cost_optimization:
  enabled: true
  cloud_provider: "aws"
  pricing:
    cpu_per_core_hour: 0.024
    memory_per_gib_hour: 0.003
```

Use custom config:

```bash
python cli.py recommend --config custom-config.yaml --namespace default --deployment web-app
```

## 📈 Advanced Usage

### Batch Processing

```bash
#!/bin/bash
# Process multiple deployments

deployments=("web-app" "api-server" "database" "cache")

for deployment in "${deployments[@]}"; do
    echo "Processing $deployment..."
    
    # Train model
    python cli.py train --namespace default --deployment $deployment --days 30
    
    # Get recommendations
    python cli.py recommend --namespace default --deployment $deployment --output json > "recommendations_$deployment.json"
done
```

### Integration with CI/CD

```yaml
# GitHub Actions example
name: Resource Optimization

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  optimize-resources:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Get resource recommendations
      run: |
        python cli.py recommend --namespace production --deployment web-app --output json > recommendations.json
    
    - name: Update deployment if savings > 20%
      run: |
        python scripts/update_deployment.py recommendations.json
```

### Monitoring and Alerting

```python
# Example monitoring script
import requests
import json
from datetime import datetime

def check_recommendations():
    """Check for significant resource optimization opportunities."""
    
    # Get recommendations for all deployments
    response = requests.get("http://localhost:8000/deployments")
    deployments = response.json()["deployments"]
    
    for deployment in deployments:
        # Get recommendations
        rec_response = requests.post("http://localhost:8000/recommend", json={
            "namespace": deployment["namespace"],
            "deployment_name": deployment["name"],
            "include_cost_analysis": True
        })
        
        if rec_response.status_code == 200:
            recommendation = rec_response.json()
            
            # Alert if significant savings possible
            if recommendation.get("cost_savings", {}).get("savings_percent", 0) > 20:
                print(f"🚨 High savings opportunity for {deployment['name']}: "
                      f"{recommendation['cost_savings']['savings_percent']:.1f}% savings")
                
                # Send alert (e.g., Slack, email)
                send_alert(recommendation)

if __name__ == "__main__":
    check_recommendations()
```

## 🔍 Troubleshooting

### Common Issues

1. **No data found for deployment**
   ```bash
   # Check if Prometheus is accessible
   curl http://prometheus:9090/api/v1/query?query=up
   
   # Verify deployment exists
   kubectl get deployments -n default
   ```

2. **Insufficient data for training**
   ```bash
   # Collect more data
   python cli.py train --namespace default --deployment web-app --days 60
   
   # Check data quality
   python cli.py analyze --namespace default --deployment web-app --days 30
   ```

3. **Model training fails**
   ```bash
   # Check logs
   python cli.py train --namespace default --deployment web-app --verbose
   
   # Try different model type
   python cli.py train --namespace default --deployment web-app --model-type lstm
   ```

### Debug Mode

```bash
# Enable verbose logging
python cli.py --verbose recommend --namespace default --deployment web-app

# Check API health
curl -v http://localhost:8000/health

# View detailed logs
kubectl logs -f deployment/k8s-resource-recommender -n resource-recommender
```

## 📚 Best Practices

1. **Data Collection**
   - Collect at least 30 days of data for reliable recommendations
   - Ensure Prometheus is properly configured and scraping metrics
   - Monitor data quality and completeness

2. **Model Training**
   - Train models during stable periods (avoid maintenance windows)
   - Use appropriate model types for your workload patterns
   - Regularly retrain models as workloads evolve

3. **Recommendations**
   - Start with conservative safety margins
   - Monitor application performance after applying recommendations
   - Use confidence scores to assess recommendation reliability

4. **Cost Optimization**
   - Set up alerts for high savings opportunities
   - Regularly review and update pricing information
   - Consider workload patterns when interpreting recommendations

5. **Production Deployment**
   - Use proper RBAC permissions
   - Implement monitoring and alerting
   - Set up regular backups of trained models
   - Use resource limits to prevent resource exhaustion 
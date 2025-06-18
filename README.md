# 🤖 K8s ML Recommender

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.20+-326CE5.svg)](https://kubernetes.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **TensorFlow-based ML-powered tool** that recommends optimal CPU and memory requests/limits for Kubernetes workloads, helping teams right-size their containers and reduce costs by **20-40%**.

## 🎯 Problem Solved

- **Eliminates guesswork** in setting CPU/memory requests and limits
- **Prevents OOM kills** and application crashes
- **Reduces cloud costs** by avoiding over-provisioning
- **Improves cluster stability** with better resource allocation
- **ML-powered insights** beyond simple percentile-based recommendations

## 📊 Key Features

- **🧠 ML-powered recommendations** using TensorFlow/Keras
- **📈 Historical data analysis** from Prometheus or Kubernetes Metrics API
- **🌐 REST API** for integration with CI/CD pipelines
- **🖥️ CLI tool** for manual recommendations
- **🐳 Kubernetes-native** deployment
- **💰 Cost optimization** insights with cloud pricing
- **🔄 Feedback loop** for continuous improvement
- **📊 Multiple model architectures** (regression, LSTM, ensemble)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │───▶│  Data Pipeline  │───▶│ TensorFlow Model│
│   Metrics API   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Kubernetes    │◀───│ Recommendation  │◀────────────┘
│   Workloads     │    │     Service     │
└─────────────────┘    └─────────────────┘
```

### Core Components

1. **Data Collection Layer**
   - Metrics collection from Prometheus or Kubernetes Metrics API
   - Data preprocessing and feature engineering
   - Storage and persistence

2. **ML Model Layer**
   - TensorFlow-based models (Regression, LSTM, Ensemble)
   - Model training, validation, and inference
   - Model caching and versioning

3. **API & Service Layer**
   - FastAPI-based REST API
   - Async request handling
   - Health checks and monitoring

4. **Client Interface Layer**
   - CLI tool for terminal usage
   - API clients for integration
   - Optional web UI (planned)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Kubernetes cluster with Prometheus or Metrics Server
- kubectl configured
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.ibm.com/vishnuc/k8s-ml-recommender.git
cd k8s-ml-recommender

# Install dependencies
pip install -r requirements.txt

# Build Docker image (optional)
docker build -t k8s-ml-recommender .
```

### Basic Usage

```bash
# Run the demo with sample data
python demo.py

# Get recommendations for a real deployment
python cli.py recommend --namespace default --deployment my-app

# Start the API server
python api.py
```

## 🔧 Usage

### CLI Examples

```bash
# Get recommendations
python cli.py recommend --namespace default --deployment web-app

# Train a model
python cli.py train --namespace default --deployment web-app --days 30

# List deployments
python cli.py list-deployments

# Analyze resource usage
python cli.py analyze --namespace default --deployment web-app
```

### API Examples

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "deployment_name": "web-app",
    "days": 30,
    "include_cost_analysis": true
  }'

# Train a model
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "default",
    "deployment_name": "web-app",
    "days": 30,
    "model_type": "regression"
  }'
```

## 🧠 Model Architecture

The system supports multiple ML model architectures:

1. **Regression Model** (Default)
   - Feedforward neural network
   - Dense layers with dropout and batch normalization
   - Best for stable workloads

2. **LSTM Model**
   - Time series prediction
   - Captures temporal patterns
   - Best for workloads with time-based patterns

3. **Ensemble Model**
   - Combines multiple model types
   - More robust predictions
   - Best for complex workloads

### Model Features

- CPU/Memory usage (current, p95, p99)
- Resource requests/limits
- Pod restart counts
- Time-based features (hour, day, weekend)
- Pod age and stability metrics

## 🐳 Kubernetes Deployment

### Deploy to Kubernetes

```bash
# Deploy using kubectl
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/service-account.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -n resource-recommender
```

### Production Considerations

- **RBAC**: Use appropriate service account with minimal permissions
- **Networking**: Apply network policies to restrict access
- **Monitoring**: Set up Prometheus rules and Grafana dashboards
- **Resources**: Start with 500m CPU, 1Gi memory; scale as needed
- **Scaling**: Use HPA for horizontal scaling based on load

## 📈 Performance

- **Training time**: ~5-10 minutes for 30 days of data
- **Inference latency**: <100ms per recommendation
- **Accuracy**: 85%+ resource utilization prediction
- **Cost savings**: 20-40% reduction in resource waste

## 📁 Project Structure

```
├── src/
│   ├── data/           # Data collection and preprocessing
│   ├── models/         # TensorFlow model definitions
│   ├── api/            # REST API implementation
│   └── cli/            # Command-line interface
├── k8s/                # Kubernetes manifests
├── helm/               # Helm charts
├── tests/              # Unit and integration tests
├── data/               # Sample datasets
└── docs/               # Documentation
```

## 🔧 Configuration

Configuration is handled via `config.yaml` and `config-local.yaml` files:

- **Prometheus/Metrics**: Connection settings and query parameters
- **Model**: Architecture, training parameters, and features
- **API**: Server settings, security, and rate limiting
- **Recommendations**: Safety margins and thresholds
- **Cost Optimization**: Cloud pricing and savings calculations

## 📝 Best Practices

1. **Data Collection**
   - Collect at least 30 days of data for reliable recommendations
   - Ensure proper metrics collection setup

2. **Model Training**
   - Train models during stable periods
   - Regularly retrain as workloads evolve

3. **Recommendations**
   - Start with conservative safety margins
   - Monitor application performance after applying recommendations

4. **Cost Optimization**
   - Set up alerts for high savings opportunities
   - Regularly review and update pricing information

## 📚 Documentation

Additional documentation:

- **Architecture**: Detailed system design and components
- **Model Details**: ML model architectures and training
- **Production Deployment**: Kubernetes setup and monitoring
- **Usage Examples**: CLI and API usage patterns
- **Local Setup**: Development environment setup

## 🔄 Changelog

## 🔍 Troubleshooting

Common issues:

1. **No data found for deployment**
   - Check Prometheus/Metrics Server connectivity
   - Verify deployment exists in the cluster

2. **Insufficient data for training**
   - Collect more data (30+ days recommended)
   - Check data quality and completeness

3. **Model training fails**
   - Check logs with `--verbose` flag
   - Try different model types

## 📈 Demo Results

The demo analyzes sample deployments and finds significant cost savings:

| Deployment | CPU Savings | Memory Savings | Key Insights |
|------------|-------------|----------------|--------------|
| **web-app** | 5.9% | 18.9% | Memory optimization opportunity |
| **api-server** | 41.8% | 42.6% | Major over-provisioning detected |
| **database** | 9.5% | -0.8% | CPU optimization, memory well-sized |
| **cache** | 55.2% | 54.5% | Significant over-provisioning |
| **batch-processor** | 15.7% | 10.9% | Moderate optimization opportunity |

## 🌟 Contributors

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

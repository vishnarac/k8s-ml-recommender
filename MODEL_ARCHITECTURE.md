# 🤖 ML Model Architecture & Execution Guide

## 📍 **Where Are the Models Located?**

### **1. Model Definition Location**
```
src/models/
├── __init__.py              # Model imports and exports
├── resource_model.py        # Main TensorFlow model implementation
└── model_manager.py         # Model lifecycle management (planned)
```

### **2. Model Storage Location**
```
models/                      # Trained model files (.h5, .pkl)
├── default_web-app.h5       # Saved TensorFlow models
├── default_api-server.h5
├── default_database.h5
└── ...
```

### **3. Model Configuration**
```
config.yaml                  # Model hyperparameters and settings
├── model:
│   ├── type: "regression"   # Model architecture type
│   ├── architecture: {...}  # Layer configurations
│   └── training: {...}      # Training parameters
```

## 🔄 **How Are Models Being Run?**

### **1. Model Training Flow**

```python
# Training happens in multiple places:

# A. CLI Training (src/cli/cli.py)
model = ResourceRecommendationModel()
training_metrics = model.train(X, y, feature_names, target_names)
model.save_model(model_path)

# B. API Training (src/api/recommendation_api.py)
model = ResourceRecommendationModel()
training_metrics = model.train(X, y, feature_names, target_names)
model.save_model(model_path)

# C. Demo Training (full_demo.py)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
```

### **2. Model Inference Flow**

```python
# Inference happens in multiple places:

# A. CLI Inference (src/cli/cli.py)
model = ResourceRecommendationModel()
model.load_model(model_path)
predictions = model.predict(X[-1:])

# B. API Inference (src/api/recommendation_api.py)
model = ResourceRecommendationModel()
model.load_model(model_path)
predictions = model.predict(X[-1:])

# C. Demo Inference (full_demo.py)
prediction = model_info['model'].predict(X_scaled)[0]
```

## 🏃‍♂️ **Where Are Models Being Run?**

### **1. Execution Environments**

#### **A. CLI Tool (Command Line)**
```bash
# Run from terminal
python cli.py train --namespace default --deployment web-app --days 30
python cli.py recommend --namespace default --deployment web-app
```
- **Location**: Local machine or container
- **Model**: TensorFlow/Keras models
- **Storage**: Local filesystem (`models/` directory)

#### **B. API Server (Web Service)**
```bash
# Run as FastAPI server
python api.py
# or
uvicorn src.api.recommendation_api:app --host 0.0.0.0 --port 8000
```
- **Location**: FastAPI server (localhost:8000)
- **Model**: TensorFlow/Keras models
- **Storage**: In-memory + persistent storage

### **2. Production Deployment**

#### **A. Kubernetes Pod**
```yaml
# k8s/deployment.yaml
containers:
- name: ml-recommender
  image: k8s-ml-recommender:latest
  ports:
  - containerPort: 8000
  volumeMounts:
  - name: models-storage
    mountPath: /app/models
```
- **Location**: Kubernetes cluster
- **Model**: TensorFlow/Keras models
- **Storage**: PersistentVolumeClaim

#### **B. Docker Container**
```dockerfile
# Dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "api.py"]
```
- **Location**: Docker container
- **Model**: TensorFlow/Keras models
- **Storage**: Container filesystem or mounted volumes

## 🧠 **Model Types & Architectures**

### **1. TensorFlow/Keras Models (Production)**

```python
# src/models/resource_model.py
class ResourceRecommendationModel:
    def _build_regression_model(self, input_dim, output_dim, architecture):
        """Feedforward neural network"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for units in architecture['layers']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.2))
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(output_dim, activation='linear'))
        return model
    
    def _build_lstm_model(self, input_dim, output_dim, architecture):
        """LSTM for time series prediction"""
        model = keras.Sequential()
        model.add(layers.Input(shape=(1, input_dim)))
        for units in architecture['layers']:
            model.add(layers.LSTM(units, return_sequences=True))
        model.add(layers.Flatten())
        model.add(layers.Dense(output_dim, activation='linear'))
        return model
```

### **2. Scikit-learn Models (Demo)**

```python
# full_demo.py (for demo speed)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

model = RandomForestRegressor(n_estimators=100, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)
```

## 📊 **Model Input/Output**

### **Input Features**
```python
feature_columns = [
    'cpu_usage',              # Current CPU usage
    'memory_usage',           # Current memory usage
    'cpu_usage_p95',          # 95th percentile CPU usage
    'cpu_usage_p99',          # 99th percentile CPU usage
    'memory_usage_p95',       # 95th percentile memory usage
    'memory_usage_p99',       # 99th percentile memory usage
    'pod_restarts',           # Number of pod restarts
    'hour_of_day',            # Time-based feature
    'day_of_week',            # Time-based feature
    'is_weekend'              # Time-based feature
]
```

### **Output Targets**
```python
target_names = [
    'target_cpu_request',     # Recommended CPU request
    'target_memory_request',  # Recommended memory request
    'target_cpu_limit',       # Recommended CPU limit
    'target_memory_limit'     # Recommended memory limit
]
```

## 💾 **Model Persistence**

### **1. Model Saving**
```python
# Save TensorFlow model
model.save_model("models/default_web-app.h5")

# Save Scikit-learn model
import joblib
joblib.dump(model, "models/default_web-app.pkl")
```

### **2. Model Loading**
```python
# Load TensorFlow model
model = ResourceRecommendationModel()
model.load_model("models/default_web-app.h5")

# Load Scikit-learn model
import joblib
model = joblib.load("models/default_web-app.pkl")
```

## 🔧 **Model Configuration**

### **config.yaml**
```yaml
model:
  type: "regression"  # regression, lstm, ensemble
  architecture:
    layers: [128, 64, 32]
    activation: "relu"
    dropout: 0.2
    output_activation: "linear"
  training:
    learning_rate: 0.001
    batch_size: 32
    epochs: 100
    early_stopping_patience: 10
```

## 🚀 **Execution Summary**

### **Development/Demo**
- **Where**: Local machine
- **How**: Python scripts (CLI, API, Demo)
- **Model**: Scikit-learn (fast) or TensorFlow (full)
- **Storage**: Local filesystem

### **Production**
- **Where**: Kubernetes cluster
- **How**: FastAPI server in container
- **Model**: TensorFlow/Keras (production-ready)
- **Storage**: PersistentVolumeClaim

### **Key Points**
1. **Models are defined** in `src/models/resource_model.py`
2. **Models are trained** via CLI, API, or demo scripts
3. **Models are stored** in `models/` directory as `.h5` files
4. **Models are executed** in the same environment they're deployed to
5. **Production models** run in Kubernetes pods with persistent storage
6. **Demo models** run locally with in-memory storage for speed

## 🔍 **Quick Commands**

```bash
# Train a model
python cli.py train --namespace default --deployment web-app --days 30

# Get recommendations
python cli.py recommend --namespace default --deployment web-app

# Start API server
python api.py


# Check model files
ls -la models/

# View model logs
kubectl logs -f deployment/k8s-ml-recommender -n resource-recommender
``` 
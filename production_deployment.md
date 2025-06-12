# 🏗️ Production Deployment Guide

## Deployment Patterns

### Pattern 1: In-Cluster Deployment (Recommended)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: k8s-ml-recommender
  namespace: resource-recommender
spec:
  replicas: 2
  selector:
    matchLabels:
      app: k8s-ml-recommender
  template:
    metadata:
      labels:
        app: k8s-ml-recommender
    spec:
      serviceAccountName: ml-recommender-sa
      containers:
      - name: ml-recommender
        image: k8s-ml-recommender:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        env:
        - name: PROMETHEUS_URL
          value: "http://prometheus-service.monitoring:9090"
        - name: KUBERNETES_SERVICE_HOST
          value: "kubernetes.default.svc"
        volumeMounts:
        - name: models-storage
          mountPath: /app/models
        - name: config
          mountPath: /app/config
      volumes:
      - name: models-storage
        persistentVolumeClaim:
          claimName: ml-models-pvc
      - name: config
        configMap:
          name: ml-recommender-config
```

### Pattern 2: External Service (Enterprise)

```yaml
# External cluster deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: k8s-ml-recommender
  namespace: ml-platform
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-recommender
        image: k8s-ml-recommender:latest
        env:
        - name: TARGET_CLUSTERS
          value: "cluster1,cluster2,cluster3"
        - name: KUBECONFIG_SECRET
          value: "multi-cluster-kubeconfig"
```

### Pattern 3: Hybrid Architecture

```yaml
# Lightweight agent in each cluster
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-recommender-agent
  namespace: resource-recommender
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: agent
        image: k8s-ml-agent:latest
        env:
        - name: CENTRAL_API_URL
          value: "https://ml-platform.company.com/api"
```

## 🔐 Security Considerations

### RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-recommender-sa
  namespace: resource-recommender
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-recommender-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "deployments", "namespaces"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ml-recommender-binding
subjects:
- kind: ServiceAccount
  name: ml-recommender-sa
  namespace: resource-recommender
roleRef:
  kind: ClusterRole
  name: ml-recommender-role
  apiGroup: rbac.authorization.k8s.io
```

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-recommender-policy
  namespace: resource-recommender
spec:
  podSelector:
    matchLabels:
      app: k8s-ml-recommender
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 443
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus/rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-recommender-rules
  namespace: monitoring
spec:
  groups:
  - name: ml-recommender
    rules:
    - alert: MLRecommenderDown
      expr: up{job="k8s-ml-recommender"} == 0
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "ML Recommender service is down"
    - alert: HighRecommendationLatency
      expr: histogram_quantile(0.95, http_request_duration_seconds{job="k8s-ml-recommender"}) > 1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "ML recommendations are slow"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ML Resource Recommender",
    "panels": [
      {
        "title": "Recommendations Generated",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(ml_recommendations_total[1h])",
            "legendFormat": "Recommendations/hour"
          }
        ]
      },
      {
        "title": "Average Cost Savings",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(ml_cost_savings_percent)",
            "legendFormat": "Savings %"
          }
        ]
      }
    ]
  }
}
```

## 🚀 Production Deployment Steps

### 1. Prepare the Cluster

```bash
# Create namespace
kubectl create namespace resource-recommender

# Apply RBAC
kubectl apply -f k8s/rbac.yaml

# Create storage
kubectl apply -f k8s/storage.yaml

# Apply network policies
kubectl apply -f k8s/network-policy.yaml
```

### 2. Deploy the Service

```bash
# Deploy the ML recommender
kubectl apply -f k8s/deployment.yaml

# Deploy the service
kubectl apply -f k8s/service.yaml

# Deploy ingress (if needed)
kubectl apply -f k8s/ingress.yaml
```

### 3. Configure Monitoring

```bash
# Apply Prometheus rules
kubectl apply -f prometheus/rules.yaml

# Deploy Grafana dashboard
kubectl apply -f grafana/dashboard.yaml
```

### 4. Verify Deployment

```bash
# Check deployment status
kubectl get pods -n resource-recommender

# Test API
kubectl port-forward svc/k8s-ml-recommender 8000:8000 -n resource-recommender
curl http://localhost:8000/health

# Check logs
kubectl logs -f deployment/k8s-ml-recommender -n resource-recommender
```

## 💰 Resource Requirements

### Minimal Production Setup

```yaml
resources:
  requests:
    cpu: 500m      # 0.5 CPU cores
    memory: 1Gi    # 1 GB RAM
  limits:
    cpu: 2         # 2 CPU cores
    memory: 4Gi    # 4 GB RAM
```

### Enterprise Setup

```yaml
resources:
  requests:
    cpu: 2         # 2 CPU cores
    memory: 4Gi    # 4 GB RAM
  limits:
    cpu: 8         # 8 CPU cores
    memory: 16Gi   # 16 GB RAM
```

## 🔄 Scaling Considerations

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-recommender-hpa
  namespace: resource-recommender
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: k8s-ml-recommender
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🛡️ Security Best Practices

1. **Service Account**: Use dedicated service account with minimal permissions
2. **Network Policies**: Restrict network access to only required services
3. **Secrets Management**: Use Kubernetes secrets for sensitive configuration
4. **Pod Security**: Enable Pod Security Standards
5. **Audit Logging**: Enable audit logs for all API calls
6. **Image Scanning**: Scan container images for vulnerabilities
7. **Regular Updates**: Keep the service updated with security patches

## 📈 Performance Optimization

1. **Caching**: Cache model predictions and frequently accessed data
2. **Connection Pooling**: Reuse connections to Prometheus and Kubernetes API
3. **Batch Processing**: Process recommendations in batches
4. **Resource Limits**: Set appropriate CPU and memory limits
5. **Monitoring**: Monitor performance metrics and optimize bottlenecks

## 🔧 Troubleshooting

### Common Issues

1. **Permission Denied**: Check RBAC configuration
2. **High Latency**: Check network policies and resource limits
3. **Memory Issues**: Increase memory limits or optimize ML models
4. **API Timeouts**: Check Prometheus and Kubernetes API connectivity

### Debug Commands

```bash
# Check service status
kubectl get pods -n resource-recommender

# Check logs
kubectl logs -f deployment/k8s-ml-recommender -n resource-recommender

# Check events
kubectl get events -n resource-recommender

# Test API connectivity
kubectl exec -it deployment/k8s-ml-recommender -n resource-recommender -- curl http://localhost:8000/health

# Check resource usage
kubectl top pods -n resource-recommender
``` 
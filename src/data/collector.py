"""
Metrics collector for gathering Kubernetes and Prometheus data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
import yaml
import os

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics from Prometheus and Kubernetes API."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the metrics collector.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Kubernetes client
        try:
            # Check if running in cluster
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount'):
                # Load in-cluster config
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            else:
                # Load local kubeconfig
                config_file = self.config['kubernetes'].get('config_path')
                if config_file and config_file != '/var/run/secrets/kubernetes.io/serviceaccount':
                    config.load_kube_config(config_file=config_file)
                else:
                    config.load_kube_config()
                logger.info("Loaded local Kubernetes configuration")
                
            self.k8s_client = client.CoreV1Api()
            self.k8s_apps_client = client.AppsV1Api()
            logger.info("Kubernetes clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Kubernetes config: {e}")
            self.k8s_client = None
            self.k8s_apps_client = None
        
        # Initialize Prometheus client (optional)
        try:
            if self.config.get('prometheus', {}).get('enabled', True):
                self.prometheus = PrometheusConnect(
                    url=self.config['prometheus']['url'],
                    disable_ssl=True
                )
                logger.info("Prometheus client initialized")
            else:
                self.prometheus = None
                logger.info("Prometheus disabled in configuration")
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus client: {e}")
            self.prometheus = None
        
        self.metrics_config = self.config['prometheus']['metrics']
        
    async def collect_deployment_metrics(
        self, 
        namespace: str, 
        deployment_name: str, 
        days: int = 30
    ) -> pd.DataFrame:
        """Collect comprehensive metrics for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment_name: Name of the deployment
            days: Number of days to collect data for
            
        Returns:
            DataFrame with collected metrics
        """
        logger.info(f"Collecting metrics for {namespace}/{deployment_name} for {days} days")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get deployment info
        deployment_info = await self._get_deployment_info(namespace, deployment_name)
        
        # Collect Prometheus metrics
        metrics_data = await self._collect_prometheus_metrics(
            namespace, deployment_name, start_time, end_time
        )
        
        # Collect Kubernetes API data
        k8s_data = await self._collect_kubernetes_data(
            namespace, deployment_name, start_time, end_time
        )
        
        # Merge and process data
        combined_data = self._merge_metrics_data(metrics_data, k8s_data, deployment_info)
        
        logger.info(f"Collected {len(combined_data)} data points")
        return combined_data
    
    async def _get_deployment_info(self, namespace: str, deployment_name: str) -> Dict:
        """Get deployment information from Kubernetes API."""
        try:
            deployment = self.k8s_apps_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            return {
                'replicas': deployment.spec.replicas,
                'selector': deployment.spec.selector.match_labels,
                'template': deployment.spec.template
            }
        except Exception as e:
            logger.error(f"Failed to get deployment info: {e}")
            return {}
    
    async def _collect_prometheus_metrics(
        self, 
        namespace: str, 
        deployment_name: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, List]:
        """Collect metrics from Prometheus."""
        metrics_data = {}
        
        # Prometheus query templates
        queries = {
            'cpu_usage': f"""
                rate(container_cpu_usage_seconds_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}[5m]) * 1000
            """,
            'memory_usage': f"""
                container_memory_usage_bytes{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """,
            'cpu_requests': f"""
                kube_pod_container_resource_requests_cpu_cores{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """,
            'memory_requests': f"""
                kube_pod_container_resource_requests_memory_bytes{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """,
            'cpu_limits': f"""
                kube_pod_container_resource_limits_cpu_cores{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """,
            'memory_limits': f"""
                kube_pod_container_resource_limits_memory_bytes{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """,
            'pod_restarts': f"""
                kube_pod_container_status_restarts_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-[a-z0-9]+-[a-z0-9]+"
                }}
            """
        }
        
        for metric_name, query in queries.items():
            try:
                result = self.prometheus.custom_query_range(
                    query=query,
                    start_time=start_time,
                    end_time=end_time,
                    step=self.config['data']['scrape_interval_minutes'] * 60
                )
                
                if result and 'result' in result:
                    metrics_data[metric_name] = result['result']
                else:
                    metrics_data[metric_name] = []
                    
            except Exception as e:
                logger.error(f"Failed to collect {metric_name}: {e}")
                metrics_data[metric_name] = []
        
        return metrics_data
    
    async def _collect_kubernetes_data(
        self, 
        namespace: str, 
        deployment_name: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict:
        """Collect additional data from Kubernetes API."""
        try:
            pods = self.k8s_client.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={deployment_name}"
            )
            
            pod_data = []
            for pod in pods.items:
                pod_info = {
                    'name': pod.metadata.name,
                    'creation_timestamp': pod.metadata.creation_timestamp,
                    'status': pod.status.phase,
                    'restart_count': sum(
                        container.restart_count for container in pod.status.container_statuses or []
                    ),
                    'node_name': pod.spec.node_name
                }
                pod_data.append(pod_info)
            
            return {'pods': pod_data}
            
        except Exception as e:
            logger.error(f"Failed to collect Kubernetes data: {e}")
            return {'pods': []}
    
    def _merge_metrics_data(
        self, 
        metrics_data: Dict, 
        k8s_data: Dict, 
        deployment_info: Dict
    ) -> pd.DataFrame:
        """Merge and process collected metrics data."""
        # Convert Prometheus data to DataFrame
        data_points = []
        
        # Process time series data
        for metric_name, time_series in metrics_data.items():
            for series in time_series:
                if 'values' in series:
                    for timestamp, value in series['values']:
                        data_points.append({
                            'timestamp': datetime.fromtimestamp(timestamp),
                            'metric': metric_name,
                            'value': float(value),
                            'pod': series.get('metric', {}).get('pod', 'unknown')
                        })
        
        if not data_points:
            return pd.DataFrame()
        
        # Create DataFrame and pivot
        df = pd.DataFrame(data_points)
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='metric',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Add deployment info
        df_pivot['deployment_replicas'] = deployment_info.get('replicas', 1)
        df_pivot['namespace'] = namespace
        df_pivot['deployment_name'] = deployment_name
        
        # Calculate derived features
        df_pivot = self._calculate_derived_features(df_pivot)
        
        return df_pivot
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for ML model."""
        # CPU and memory percentiles
        if 'cpu_usage' in df.columns:
            df['cpu_usage_p95'] = df['cpu_usage'].rolling(window=12).quantile(0.95)
            df['cpu_usage_p99'] = df['cpu_usage'].rolling(window=12).quantile(0.99)
            df['cpu_usage_avg'] = df['cpu_usage'].rolling(window=12).mean()
        
        if 'memory_usage' in df.columns:
            df['memory_usage_p95'] = df['memory_usage'].rolling(window=12).quantile(0.95)
            df['memory_usage_p99'] = df['memory_usage'].rolling(window=12).quantile(0.99)
            df['memory_usage_avg'] = df['memory_usage'].rolling(window=12).mean()
        
        # Resource utilization ratios
        if all(col in df.columns for col in ['cpu_usage', 'cpu_requests']):
            df['cpu_utilization'] = df['cpu_usage'] / (df['cpu_requests'] * 1000)
        
        if all(col in df.columns for col in ['memory_usage', 'memory_requests']):
            df['memory_utilization'] = df['memory_usage'] / df['memory_requests']
        
        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def get_available_deployments(self, namespace: str = None) -> List[Dict]:
        """Get list of available deployments."""
        try:
            if namespace:
                deployments = self.k8s_apps_client.list_namespaced_deployment(namespace=namespace)
            else:
                deployments = self.k8s_apps_client.list_deployment_for_all_namespaces()
            
            return [
                {
                    'name': dep.metadata.name,
                    'namespace': dep.metadata.namespace,
                    'replicas': dep.spec.replicas,
                    'available_replicas': dep.status.available_replicas or 0
                }
                for dep in deployments.items
            ]
        except Exception as e:
            logger.error(f"Failed to get deployments: {e}")
            return []

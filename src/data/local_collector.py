"""
Local data collector for k8s-ml-recommender that uses Kubernetes metrics API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Any
import yaml

from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class LocalMetricsCollector:
    """Collect metrics from Kubernetes using metrics API."""
    
    def __init__(self, config_path: str = "config-local.yaml"):
        """Initialize the collector.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load Kubernetes config
        try:
            config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
            self.k8s_apps_client = client.AppsV1Api()
            self.k8s_metrics_client = client.CustomObjectsApi()
            logger.info("Successfully connected to Kubernetes cluster")
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            raise
    
    def get_pod_metrics(self, namespace: str, deployment_name: str) -> List[Dict[str, Any]]:
        """Get pod metrics for a deployment.
        
        Args:
            namespace: Kubernetes namespace
            deployment_name: Deployment name
            
        Returns:
            List of metric data points
        """
        try:
            # Get pods for the deployment
            pods = self.k8s_client.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"app={deployment_name}"
            )
            
            if not pods.items:
                logger.warning(f"No pods found for deployment {deployment_name}")
                return []
            
            metrics_data = []
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                
                try:
                    # Get pod metrics
                    metrics = self.k8s_metrics_client.get_namespaced_custom_object(
                        group="metrics.k8s.io",
                        version="v1beta1",
                        namespace=namespace,
                        plural="pods",
                        name=pod_name
                    )
                    
                    if 'containers' in metrics:
                        for container in metrics['containers']:
                            if container['name'] == deployment_name:
                                # Parse CPU and memory usage
                                cpu_usage = self._parse_cpu(container['usage']['cpu'])
                                memory_usage = self._parse_memory(container['usage']['memory'])
                                
                                # Get current resource requests/limits
                                pod_spec = pod.spec.containers[0]
                                resources = pod_spec.resources
                                
                                cpu_requests = self._parse_cpu(resources.requests.get('cpu', '100m'))
                                memory_requests = self._parse_memory(resources.requests.get('memory', '128Mi'))
                                cpu_limits = self._parse_cpu(resources.limits.get('cpu', '200m'))
                                memory_limits = self._parse_memory(resources.limits.get('memory', '256Mi'))
                                
                                # Get pod age
                                pod_age = (datetime.now() - pod.metadata.creation_timestamp.replace(tzinfo=None)).total_seconds() / 3600
                                
                                # Get restart count
                                restart_count = pod.status.container_statuses[0].restart_count if pod.status.container_statuses else 0
                                
                                metrics_data.append({
                                    'timestamp': datetime.now(),
                                    'pod_name': pod_name,
                                    'cpu_usage': cpu_usage,
                                    'memory_usage': memory_usage,
                                    'cpu_requests': cpu_requests,
                                    'memory_requests': memory_requests,
                                    'cpu_limits': cpu_limits,
                                    'memory_limits': memory_limits,
                                    'pod_restarts': restart_count,
                                    'pod_age_hours': pod_age,
                                    'hour_of_day': datetime.now().hour,
                                    'day_of_week': datetime.now().weekday(),
                                    'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
                                    'namespace': namespace,
                                    'deployment_name': deployment_name
                                })
                                
                except ApiException as e:
                    if e.status == 404:
                        logger.warning(f"Metrics not available for pod {pod_name}")
                    else:
                        logger.error(f"Error getting metrics for pod {pod_name}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error getting metrics for pod {pod_name}: {e}")
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Error getting pod metrics: {e}")
            return []
    
    def collect_deployment_metrics(self, namespace: str, deployment_name: str, hours: int = 1) -> pd.DataFrame:
        """Collect metrics for a deployment over time.
        
        Args:
            namespace: Kubernetes namespace
            deployment_name: Deployment name
            hours: Number of hours to collect data
            
        Returns:
            DataFrame with collected metrics
        """
        logger.info(f"Collecting metrics for {namespace}/{deployment_name} over {hours} hours")
        
        all_metrics = []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)
        
        while datetime.now() < end_time:
            try:
                # Get current metrics
                current_metrics = self.get_pod_metrics(namespace, deployment_name)
                all_metrics.extend(current_metrics)
                
                # Wait before next collection
                time.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error during metrics collection: {e}")
                time.sleep(60)
        
        if not all_metrics:
            logger.warning("No metrics collected")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Calculate percentiles
        if len(df) > 0:
            df['cpu_usage_p95'] = df['cpu_usage'].quantile(0.95)
            df['cpu_usage_p99'] = df['cpu_usage'].quantile(0.99)
            df['memory_usage_p95'] = df['memory_usage'].quantile(0.95)
            df['memory_usage_p99'] = df['memory_usage'].quantile(0.99)
        
        logger.info(f"Collected {len(df)} data points")
        return df
    
    def get_available_deployments(self, namespace: Optional[str] = None) -> List[Dict[str, str]]:
        """Get list of available deployments.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            List of deployment information
        """
        try:
            if namespace:
                deployments = self.k8s_apps_client.list_namespaced_deployment(namespace=namespace)
            else:
                deployments = self.k8s_apps_client.list_deployment_for_all_namespaces()
            
            deployment_list = []
            for deployment in deployments.items:
                deployment_list.append({
                    'name': deployment.metadata.name,
                    'namespace': deployment.metadata.namespace,
                    'replicas': deployment.spec.replicas,
                    'available_replicas': deployment.status.available_replicas
                })
            
            return deployment_list
            
        except Exception as e:
            logger.error(f"Error getting deployments: {e}")
            return []
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU string to millicores."""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1])
        elif cpu_str.endswith('n'):
            return float(cpu_str[:-1]) / 1000000
        else:
            return float(cpu_str) * 1000
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory string to MiB."""
        if memory_str.endswith('Ki'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('Mi'):
            return float(memory_str[:-2])
        elif memory_str.endswith('Gi'):
            return float(memory_str[:-2]) * 1024
        else:
            return float(memory_str) / (1024 * 1024)  # Assume bytes 
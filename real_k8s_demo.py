#!/usr/bin/env python3
"""
Real Kubernetes Demo for k8s-ml-recommender

This script demonstrates the k8s-ml-recommender with real Kubernetes cluster data.
"""

import sys
import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.local_collector import LocalMetricsCollector
from models.resource_model import ResourceRecommendationModel
from data.preprocessor import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    print("🚀 Real Kubernetes k8s-ml-recommender Demo")
    print("=" * 50)
    
    try:
        # Initialize the local metrics collector
        print("📊 Initializing Kubernetes metrics collector...")
        collector = LocalMetricsCollector("config-local.yaml")
        
        # Get available deployments
        print("🔍 Discovering deployments in cluster...")
        deployments = collector.get_available_deployments("demo-apps")
        
        if not deployments:
            print("❌ No deployments found in demo-apps namespace")
            print("   Make sure the demo workloads are running:")
            print("   kubectl get pods -n demo-apps")
            return
        
        print(f"✅ Found {len(deployments)} deployments:")
        for dep in deployments:
            print(f"   - {dep['namespace']}/{dep['name']} ({dep['available_replicas']}/{dep['replicas']} ready)")
        
        # Collect metrics for each deployment
        print("\n📈 Collecting metrics from Kubernetes cluster...")
        print("   This will take a few minutes to gather enough data...")
        
        all_metrics = []
        
        for deployment in deployments:
            if deployment['available_replicas'] > 0:
                print(f"   Collecting metrics for {deployment['name']}...")
                
                # Collect metrics for a very short time (just for testing)
                metrics_df = collector.collect_deployment_metrics(
                    namespace=deployment['namespace'],
                    deployment_name=deployment['name'],
                    hours=0.01  # 36 seconds
                )
                
                if not metrics_df.empty:
                    all_metrics.append(metrics_df)
                    print(f"     ✅ Collected {len(metrics_df)} data points")
                else:
                    print(f"     ⚠️  No metrics collected for {deployment['name']}")
        
        if not all_metrics:
            print("❌ No metrics collected from any deployment")
            print("   This might be because:")
            print("   - Metrics server is not installed")
            print("   - Pods are not generating enough load")
            print("   - Network connectivity issues")
            return
        
        # Combine all metrics
        combined_df = pd.concat(all_metrics, ignore_index=True)
        print(f"\n✅ Total metrics collected: {len(combined_df)} data points")
        
        # Show sample of collected data
        print("\n📋 Sample of collected metrics:")
        print(combined_df[['deployment_name', 'cpu_usage', 'memory_usage', 'pod_restarts']].head())
        
        # Preprocess data
        print("\n🔧 Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_df, _ = preprocessor.preprocess_data(combined_df)
        
        if processed_df.empty:
            print("❌ No data after preprocessing")
            return
        
        print(f"✅ Preprocessed data: {len(processed_df)} samples")
        
        # Initialize recommendation model
        print("\n🤖 Initializing recommendation model...")
        model = ResourceRecommendationModel()
        
        # Train model (if enough data)
        if len(processed_df) >= 10:
            print("🎯 Training recommendation model...")
            model.train(processed_df)
            print("✅ Model training completed")
        else:
            print("⚠️  Not enough data for training (need at least 10 samples)")
            print("   Using statistical analysis instead...")
        
        # Generate recommendations
        print("\n💡 Generating resource recommendations...")
        
        recommendations = []
        for deployment in deployments:
            if deployment['available_replicas'] > 0:
                deployment_metrics = processed_df[
                    processed_df['deployment_name'] == deployment['name']
                ]
                
                if not deployment_metrics.empty:
                    # Print column names for debugging
                    print(f"   Available columns for {deployment['name']}: {deployment_metrics.columns.tolist()}")
                    
                    # Get current resource usage - use appropriate column names
                    if 'cpu_usage' in deployment_metrics.columns:
                        current_cpu = deployment_metrics['cpu_usage'].mean()
                        current_cpu_p95 = deployment_metrics['cpu_usage'].quantile(0.95)
                    elif 'cpu_usage_p95' in deployment_metrics.columns:
                        current_cpu = deployment_metrics['cpu_usage_p95'].mean()
                        current_cpu_p95 = deployment_metrics['cpu_usage_p95'].mean()  # Use mean of p95 values
                    else:
                        print(f"   ⚠️ No CPU usage data found for {deployment['name']}")
                        continue
                        
                    if 'memory_usage' in deployment_metrics.columns:
                        current_memory = deployment_metrics['memory_usage'].mean()
                        current_memory_p95 = deployment_metrics['memory_usage'].quantile(0.95)
                    elif 'memory_usage_p95' in deployment_metrics.columns:
                        current_memory = deployment_metrics['memory_usage_p95'].mean()
                        current_memory_p95 = deployment_metrics['memory_usage_p95'].mean()  # Use mean of p95 values
                    else:
                        print(f"   ⚠️ No memory usage data found for {deployment['name']}")
                        continue
                    
                    # Generate recommendations
                    if len(processed_df) >= 10:
                        # Use ML model
                        recommended_cpu = model.predict_cpu_request(deployment_metrics.iloc[0:1])
                        recommended_memory = model.predict_memory_request(deployment_metrics.iloc[0:1])
                    else:
                        # Use statistical analysis
                        recommended_cpu = max(current_cpu_p95 * 1.2, 50)  # 20% safety margin, min 50m
                        recommended_memory = max(current_memory_p95 * 1.3, 64)  # 30% safety margin, min 64Mi
                    
                    # Get current resource requests
                    if 'cpu_requests' in deployment_metrics.columns:
                        current_cpu_request = deployment_metrics['cpu_requests'].iloc[0]
                    else:
                        # Get from the original deployment specification
                        current_cpu_request = deployment.get('cpu_request', 100)  # Default to 100m if not found
                        print(f"   ⚠️ Using default cpu_request value: {current_cpu_request}m")
                    
                    if 'memory_requests' in deployment_metrics.columns:
                        current_memory_request = deployment_metrics['memory_requests'].iloc[0]
                    else:
                        # Get from the original deployment specification
                        current_memory_request = deployment.get('memory_request', 128)  # Default to 128Mi if not found
                        print(f"   ⚠️ Using default memory_request value: {current_memory_request}Mi")
                    
                    cpu_savings = (current_cpu_request - recommended_cpu) / current_cpu_request * 100
                    memory_savings = (current_memory_request - recommended_memory) / current_memory_request * 100
                    
                    recommendations.append({
                        'deployment': deployment['name'],
                        'namespace': deployment['namespace'],
                        'current_cpu_avg': round(current_cpu, 1),
                        'current_cpu_p95': round(current_cpu_p95, 1),
                        'current_memory_avg': round(current_memory, 1),
                        'current_memory_p95': round(current_memory_p95, 1),
                        'current_cpu_request': current_cpu_request,
                        'current_memory_request': current_memory_request,
                        'recommended_cpu': round(recommended_cpu, 1),
                        'recommended_memory': round(recommended_memory, 1),
                        'cpu_savings_pct': round(cpu_savings, 1),
                        'memory_savings_pct': round(memory_savings, 1),
                        'confidence': 'High' if len(processed_df) >= 10 else 'Medium'
                    })
        
        # Display recommendations
        print("\n" + "=" * 80)
        print("🎯 RESOURCE RECOMMENDATIONS")
        print("=" * 80)
        
        if not recommendations:
            print("❌ No recommendations generated")
            return
        
        total_cpu_savings = 0
        total_memory_savings = 0
        
        for rec in recommendations:
            print(f"\n📦 Deployment: {rec['namespace']}/{rec['deployment']}")
            print(f"   Confidence: {rec['confidence']}")
            print(f"   Current Usage (avg): CPU {rec['current_cpu_avg']}m, Memory {rec['current_memory_avg']}Mi")
            print(f"   Current Usage (p95): CPU {rec['current_cpu_p95']}m, Memory {rec['current_memory_p95']}Mi")
            print(f"   Current Requests: CPU {rec['current_cpu_request']}m, Memory {rec['current_memory_request']}Mi")
            print(f"   Recommended: CPU {rec['recommended_cpu']}m, Memory {rec['recommended_memory']}Mi")
            
            if rec['cpu_savings_pct'] > 0:
                print(f"   💰 CPU Savings: {rec['cpu_savings_pct']}%")
                total_cpu_savings += rec['cpu_savings_pct']
            
            if rec['memory_savings_pct'] > 0:
                print(f"   💰 Memory Savings: {rec['memory_savings_pct']}%")
                total_memory_savings += rec['memory_savings_pct']
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"Total deployments analyzed: {len(recommendations)}")
        print(f"Data points collected: {len(combined_df)}")
        print(f"Average CPU savings: {total_cpu_savings / len(recommendations):.1f}%")
        print(f"Average memory savings: {total_memory_savings / len(recommendations):.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"real_k8s_results_{timestamp}.csv"
        
        results_df = pd.DataFrame(recommendations)
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Kubernetes cluster is running")
        print("2. Check that the demo workloads are deployed: kubectl get pods -n demo-apps")
        print("3. Verify metrics server is installed: kubectl top pods -n demo-apps")
        print("4. Check cluster resources: kubectl describe node")


if __name__ == "__main__":
    main()

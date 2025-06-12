"""
Generate sample dataset for testing the Kubernetes resource recommendation engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_dataset(days: int = 30, output_path: str = "data/sample_metrics.csv"):
    """Generate sample Kubernetes metrics data.
    
    Args:
        days: Number of days of data to generate
        output_path: Path to save the generated dataset
    """
    print(f"Generating {days} days of sample metrics data...")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Generate timestamps every 5 minutes
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5T')
    
    # Generate sample data
    data = []
    
    for timestamp in timestamps:
        # Simulate realistic CPU and memory usage patterns
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Base usage with daily and weekly patterns
        base_cpu = 200  # millicores
        base_memory = 512  # MiB
        
        # Add time-based variations
        if 9 <= hour <= 17:  # Business hours
            cpu_multiplier = 1.5
            memory_multiplier = 1.3
        elif 18 <= hour <= 22:  # Evening
            cpu_multiplier = 1.2
            memory_multiplier = 1.1
        else:  # Night
            cpu_multiplier = 0.6
            memory_multiplier = 0.8
        
        # Weekend effect
        if day_of_week >= 5:  # Weekend
            cpu_multiplier *= 0.7
            memory_multiplier *= 0.8
        
        # Add some randomness
        cpu_noise = np.random.normal(0, 0.1)
        memory_noise = np.random.normal(0, 0.05)
        
        # Calculate usage
        cpu_usage = int(base_cpu * cpu_multiplier * (1 + cpu_noise))
        memory_usage = int(base_memory * memory_multiplier * (1 + memory_noise))
        
        # Current resource configuration
        current_cpu_requests = 500  # millicores
        current_memory_requests = 1024  # MiB
        current_cpu_limits = 1000  # millicores
        current_memory_limits = 2048  # MiB
        
        # Pod restarts (occasional)
        pod_restarts = np.random.poisson(0.1)  # Low restart rate
        
        # Deployment info
        deployment_replicas = 3
        pod_age_hours = np.random.uniform(1, 168)  # 1 hour to 1 week
        
        data.append({
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'cpu_requests': current_cpu_requests,
            'memory_requests': current_memory_requests,
            'cpu_limits': current_cpu_limits,
            'memory_limits': current_memory_limits,
            'pod_restarts': pod_restarts,
            'deployment_replicas': deployment_replicas,
            'pod_age_hours': pod_age_hours,
            'namespace': 'default',
            'deployment_name': 'sample-app'
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features
    df['cpu_usage_p95'] = df['cpu_usage'].rolling(window=12).quantile(0.95)
    df['cpu_usage_p99'] = df['cpu_usage'].rolling(window=12).quantile(0.99)
    df['memory_usage_p95'] = df['memory_usage'].rolling(window=12).quantile(0.95)
    df['memory_usage_p99'] = df['memory_usage'].rolling(window=12).quantile(0.99)
    
    # Time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {len(df)} data points")
    print(f"📊 Data saved to {output_path}")
    
    # Print summary statistics
    print("\n📈 Sample Data Summary:")
    print(f"CPU Usage - Mean: {df['cpu_usage'].mean():.1f}, P95: {df['cpu_usage'].quantile(0.95):.1f}")
    print(f"Memory Usage - Mean: {df['memory_usage'].mean():.1f}, P95: {df['memory_usage'].quantile(0.95):.1f}")
    print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def generate_multiple_deployments():
    """Generate sample data for multiple deployments."""
    deployments = [
        {'name': 'web-app', 'base_cpu': 300, 'base_memory': 768},
        {'name': 'api-server', 'base_cpu': 200, 'base_memory': 512},
        {'name': 'database', 'base_cpu': 500, 'base_memory': 2048},
        {'name': 'cache', 'base_cpu': 100, 'base_memory': 256}
    ]
    
    for deployment in deployments:
        output_path = f"data/sample_{deployment['name']}_metrics.csv"
        generate_sample_dataset(
            days=30, 
            output_path=output_path,
            base_cpu=deployment['base_cpu'],
            base_memory=deployment['base_memory']
        )


if __name__ == '__main__':
    # Generate sample dataset
    df = generate_sample_dataset(days=30)
    
    print("\n🎯 You can now use this data to test the recommendation engine:")
    print("python cli.py train --namespace default --deployment sample-app --days 30")
    print("python cli.py recommend --namespace default --deployment sample-app") 
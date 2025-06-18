#!/usr/bin/env python3
"""
Demo script for k8s-ml-recommender using sample data.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import DataPreprocessor
from src.models import ResourceRecommendationModel

console = Console()


def load_sample_data(filename: str) -> pd.DataFrame:
    """Load sample data from CSV file."""
    filepath = os.path.join('data', filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sample data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    console.print(f"✅ Loaded {len(df)} data points from {filename}")
    return df


def analyze_deployment(df: pd.DataFrame, deployment_name: str) -> dict:
    """Analyze deployment data and generate recommendations."""
    
    # Calculate basic statistics
    cpu_usage_stats = {
        'mean': df['cpu_usage'].mean(),
        'p95': df['cpu_usage'].quantile(0.95),
        'p99': df['cpu_usage'].quantile(0.99),
        'max': df['cpu_usage'].max()
    }
    
    memory_usage_stats = {
        'mean': df['memory_usage'].mean(),
        'p95': df['memory_usage'].quantile(0.95),
        'p99': df['memory_usage'].quantile(0.99),
        'max': df['memory_usage'].max()
    }
    
    # Get current resource configuration
    current_cpu_request = df['cpu_requests'].iloc[0]
    current_memory_request = df['memory_requests'].iloc[0]
    current_cpu_limit = df['cpu_limits'].iloc[0]
    current_memory_limit = df['memory_limits'].iloc[0]
    
    # Calculate recommended resources with safety margins
    safety_margin_cpu = 1.2  # 20% buffer
    safety_margin_memory = 1.3  # 30% buffer
    
    recommended_cpu_request = int(cpu_usage_stats['p95'] * safety_margin_cpu)
    recommended_memory_request = int(memory_usage_stats['p95'] * safety_margin_memory)
    recommended_cpu_limit = int(cpu_usage_stats['p99'] * 2.0)  # 2x p99 for limits
    recommended_memory_limit = int(memory_usage_stats['p99'] * 1.5)  # 1.5x p99 for limits
    
    # Calculate potential savings
    cpu_savings = ((current_cpu_request - recommended_cpu_request) / current_cpu_request) * 100
    memory_savings = ((current_memory_request - recommended_memory_request) / current_memory_request) * 100
    
    return {
        'deployment_name': deployment_name,
        'data_points': len(df),
        'cpu_usage_stats': cpu_usage_stats,
        'memory_usage_stats': memory_usage_stats,
        'current_resources': {
            'cpu_request': current_cpu_request,
            'memory_request': current_memory_request,
            'cpu_limit': current_cpu_limit,
            'memory_limit': current_memory_limit
        },
        'recommended_resources': {
            'cpu_request': recommended_cpu_request,
            'memory_request': recommended_memory_request,
            'cpu_limit': recommended_cpu_limit,
            'memory_limit': recommended_memory_limit
        },
        'savings': {
            'cpu_percent': cpu_savings,
            'memory_percent': memory_savings
        }
    }


def display_recommendations(analysis: dict):
    """Display recommendations in a nice table format."""
    
    # Create main table
    table = Table(title=f"Resource Recommendations for {analysis['deployment_name']}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Current", style="yellow")
    table.add_column("Recommended", style="green")
    table.add_column("Savings", style="red")
    
    # CPU recommendations
    current_cpu = f"{analysis['current_resources']['cpu_request']}m"
    recommended_cpu = f"{analysis['recommended_resources']['cpu_request']}m"
    cpu_savings = f"{analysis['savings']['cpu_percent']:.1f}%"
    
    table.add_row("CPU Request", current_cpu, recommended_cpu, cpu_savings)
    
    # Memory recommendations
    current_memory = f"{analysis['current_resources']['memory_request']}Mi"
    recommended_memory = f"{analysis['recommended_resources']['memory_request']}Mi"
    memory_savings = f"{analysis['savings']['memory_percent']:.1f}%"
    
    table.add_row("Memory Request", current_memory, recommended_memory, memory_savings)
    
    console.print(table)
    
    # Create usage statistics table
    stats_table = Table(title="Usage Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("CPU (millicores)", style="yellow")
    stats_table.add_column("Memory (MiB)", style="yellow")
    
    stats_table.add_row("Mean", 
                       f"{analysis['cpu_usage_stats']['mean']:.0f}",
                       f"{analysis['memory_usage_stats']['mean']:.0f}")
    stats_table.add_row("95th Percentile", 
                       f"{analysis['cpu_usage_stats']['p95']:.0f}",
                       f"{analysis['memory_usage_stats']['p95']:.0f}")
    stats_table.add_row("99th Percentile", 
                       f"{analysis['cpu_usage_stats']['p99']:.0f}",
                       f"{analysis['memory_usage_stats']['p99']:.0f}")
    stats_table.add_row("Maximum", 
                       f"{analysis['cpu_usage_stats']['max']:.0f}",
                       f"{analysis['memory_usage_stats']['max']:.0f}")
    
    console.print(stats_table)
    
    # Display insights
    insights = []
    if analysis['savings']['cpu_percent'] > 10:
        insights.append(f"💡 CPU savings of {analysis['savings']['cpu_percent']:.1f}% possible")
    if analysis['savings']['memory_percent'] > 10:
        insights.append(f"💡 Memory savings of {analysis['savings']['memory_percent']:.1f}% possible")
    
    if analysis['cpu_usage_stats']['p99'] > analysis['current_resources']['cpu_request'] * 0.8:
        insights.append("⚠️  CPU usage approaching current requests")
    if analysis['memory_usage_stats']['p99'] > analysis['current_resources']['memory_request'] * 0.8:
        insights.append("⚠️  Memory usage approaching current requests")
    
    if insights:
        console.print(Panel("\n".join(insights), title="Insights", border_style="blue"))


def main():
    """Main demo function."""
    console.print(Panel.fit("🤖 K8s ML Recommender Demo", style="bold blue"))
    console.print("Using sample data to demonstrate resource recommendations\n")
    
    # Available sample deployments
    sample_deployments = [
        ('sample_web-app_metrics.csv', 'web-app'),
        ('sample_api-server_metrics.csv', 'api-server'),
        ('sample_database_metrics.csv', 'database'),
        ('sample_cache_metrics.csv', 'cache'),
        ('sample_batch-processor_metrics.csv', 'batch-processor')
    ]
    
    # Display available deployments
    console.print("📊 Available sample deployments:")
    for i, (filename, name) in enumerate(sample_deployments, 1):
        console.print(f"  {i}. {name}")
    
    console.print("\n" + "="*50 + "\n")
    
    # Analyze each deployment
    for filename, deployment_name in sample_deployments:
        try:
            with console.status(f"[bold green]Analyzing {deployment_name}..."):
                # Load data
                df = load_sample_data(filename)
                
                # Analyze deployment
                analysis = analyze_deployment(df, deployment_name)
                
                # Display results
                display_recommendations(analysis)
                
                console.print("\n" + "-"*50 + "\n")
                
        except Exception as e:
            console.print(f"[red]❌ Error analyzing {deployment_name}: {e}[/red]")
            console.print("\n" + "-"*50 + "\n")
    
    # Summary
    console.print(Panel.fit(
        "🎯 Demo completed! This shows how the k8s-ml-recommender can analyze "
        "resource usage patterns and suggest optimizations to reduce costs while "
        "maintaining performance.",
        title="Summary",
        border_style="green"
    ))


if __name__ == "__main__":
    main() 
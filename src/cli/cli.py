"""
Command-line interface for Kubernetes resource recommendations.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any
import click
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from ..data import MetricsCollector, DataPreprocessor
from ..models import ResourceRecommendationModel
from ..api import RecommendationAPI

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    """Kubernetes Resource Recommendation Engine CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--namespace', '-n', required=True, help='Kubernetes namespace')
@click.option('--deployment', '-d', required=True, help='Deployment name')
@click.option('--days', default=30, help='Number of days of historical data to analyze')
@click.option('--output', '-o', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.option('--include-costs', is_flag=True, help='Include cost analysis')
@click.pass_context
def recommend(ctx, namespace: str, deployment: str, days: int, output: str, include_costs: bool):
    """Get resource recommendations for a deployment."""
    config_path = ctx.obj['config']
    
    with console.status(f"[bold green]Analyzing {namespace}/{deployment}..."):
        try:
            # Initialize components
            collector = MetricsCollector(config_path)
            preprocessor = DataPreprocessor(config_path)
            
            # Collect data
            console.print(f"📊 Collecting {days} days of metrics data...")
            df = asyncio.run(collector.collect_deployment_metrics(namespace, deployment, days))
            
            if df.empty:
                console.print(f"[red]❌ No data found for {namespace}/{deployment}[/red]")
                sys.exit(1)
            
            console.print(f"✅ Collected {len(df)} data points")
            
            # Preprocess data
            console.print("🔧 Preprocessing data...")
            processed_df, metadata = preprocessor.preprocess_data(df)
            
            # Try to load existing model
            model_key = f"{namespace}_{deployment}"
            model_path = f"models/{model_key}.h5"
            
            try:
                model = ResourceRecommendationModel()
                model.load_model(model_path)
                console.print("🤖 Using trained model for predictions")
            except Exception:
                console.print("⚠️  No trained model found, using statistical recommendations")
                # Use statistical approach
                recommendations = _get_statistical_recommendations(processed_df)
                _display_recommendations(recommendations, output, include_costs)
                return
            
            # Make predictions
            console.print("🧠 Making predictions...")
            X, _, feature_names = preprocessor.prepare_training_data(processed_df)
            predictions = model.predict(X[-1:])  # Use latest data point
            
            # Convert predictions to recommendations
            recommendations = _convert_predictions_to_recommendations(predictions[0])
            
            # Add metadata
            recommendations.update({
                'namespace': namespace,
                'deployment': deployment,
                'data_points': len(df),
                'confidence': 0.85,  # Placeholder
                'model_used': model_path
            })
            
            _display_recommendations(recommendations, output, include_costs)
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.option('--namespace', '-n', required=True, help='Kubernetes namespace')
@click.option('--deployment', '-d', required=True, help='Deployment name')
@click.option('--days', default=30, help='Number of days of historical data to analyze')
@click.option('--model-type', type=click.Choice(['regression', 'lstm', 'ensemble']), default='regression', help='Model type')
@click.option('--force', is_flag=True, help='Force retraining even if model exists')
@click.pass_context
def train(ctx, namespace: str, deployment: str, days: int, model_type: str, force: bool):
    """Train a new model for a deployment."""
    config_path = ctx.obj['config']
    
    with console.status(f"[bold green]Training model for {namespace}/{deployment}..."):
        try:
            # Initialize components
            collector = MetricsCollector(config_path)
            preprocessor = DataPreprocessor(config_path)
            
            # Check if model exists
            model_key = f"{namespace}_{deployment}"
            model_path = f"models/{model_key}.h5"
            
            if not force:
                try:
                    model = ResourceRecommendationModel()
                    model.load_model(model_path)
                    console.print(f"✅ Model already exists at {model_path}")
                    console.print("Use --force to retrain")
                    return
                except Exception:
                    pass
            
            # Collect training data
            console.print(f"📊 Collecting {days} days of training data...")
            df = asyncio.run(collector.collect_deployment_metrics(namespace, deployment, days))
            
            if df.empty:
                console.print(f"[red]❌ No data found for {namespace}/{deployment}[/red]")
                sys.exit(1)
            
            console.print(f"✅ Collected {len(df)} data points")
            
            # Preprocess data
            console.print("🔧 Preprocessing data...")
            processed_df, metadata = preprocessor.preprocess_data(df)
            
            # Prepare training data
            X, y, feature_names = preprocessor.prepare_training_data(processed_df)
            
            if len(X) < 100:
                console.print(f"[red]❌ Insufficient data for training. Need at least 100 samples, got {len(X)}[/red]")
                sys.exit(1)
            
            console.print(f"✅ Prepared {len(X)} training samples with {len(feature_names)} features")
            
            # Train model
            console.print(f"🧠 Training {model_type} model...")
            model = ResourceRecommendationModel()
            target_names = ['target_cpu_request', 'target_memory_request', 
                          'target_cpu_limit', 'target_memory_limit']
            
            training_metrics = model.train(X, y, feature_names, target_names)
            
            # Save model
            console.print(f"💾 Saving model to {model_path}...")
            model.save_model(model_path)
            
            # Display results
            _display_training_results(training_metrics, model_path, len(X))
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            sys.exit(1)


@cli.command()
@click.option('--namespace', help='Filter by namespace')
@click.pass_context
def list_deployments(ctx, namespace: Optional[str]):
    """List available deployments."""
    config_path = ctx.obj['config']
    
    try:
        collector = MetricsCollector(config_path)
        deployments = collector.get_available_deployments(namespace)
        
        if not deployments:
            console.print("No deployments found")
            return
        
        table = Table(title="Available Deployments")
        table.add_column("Namespace", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Replicas", style="yellow")
        table.add_column("Available", style="blue")
        
        for dep in deployments:
            table.add_row(
                dep['namespace'],
                dep['name'],
                str(dep['replicas']),
                str(dep['available_replicas'])
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.pass_context
def serve(ctx, host: str, port: int):
    """Start the recommendation API server."""
    config_path = ctx.obj['config']
    
    console.print(f"🚀 Starting API server on {host}:{port}")
    console.print("📖 API documentation will be available at http://localhost:8000/docs")
    
    try:
        api = RecommendationAPI(config_path)
        api.run(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--namespace', '-n', required=True, help='Kubernetes namespace')
@click.option('--deployment', '-d', required=True, help='Deployment name')
@click.option('--days', default=30, help='Number of days of historical data to analyze')
@click.pass_context
def analyze(ctx, namespace: str, deployment: str, days: int):
    """Analyze deployment resource usage patterns."""
    config_path = ctx.obj['config']
    
    with console.status(f"[bold green]Analyzing {namespace}/{deployment}..."):
        try:
            collector = MetricsCollector(config_path)
            df = asyncio.run(collector.collect_deployment_metrics(namespace, deployment, days))
            
            if df.empty:
                console.print(f"[red]❌ No data found for {namespace}/{deployment}[/red]")
                sys.exit(1)
            
            _display_usage_analysis(df)
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            sys.exit(1)


def _get_statistical_recommendations(df) -> Dict[str, Any]:
    """Get recommendations using statistical methods."""
    recommendations = {}
    
    if 'cpu_usage' in df.columns:
        cpu_p95 = df['cpu_usage'].quantile(0.95)
        cpu_p99 = df['cpu_usage'].quantile(0.99)
        recommendations['cpu_request'] = int(cpu_p95 * 1.2)  # 20% safety margin
        recommendations['cpu_limit'] = int(cpu_p99 * 2.0)    # 2x P99 for limits
    
    if 'memory_usage' in df.columns:
        memory_p95 = df['memory_usage'].quantile(0.95)
        memory_p99 = df['memory_usage'].quantile(0.99)
        recommendations['memory_request'] = int(memory_p95 * 1.3)  # 30% safety margin
        recommendations['memory_limit'] = int(memory_p99 * 1.5)    # 1.5x P99 for limits
    
    return recommendations


def _convert_predictions_to_recommendations(predictions) -> Dict[str, Any]:
    """Convert model predictions to recommendations."""
    return {
        'cpu_request': int(predictions[0]),
        'memory_request': int(predictions[1]),
        'cpu_limit': int(predictions[2]),
        'memory_limit': int(predictions[3])
    }


def _display_recommendations(recommendations: Dict[str, Any], output: str, include_costs: bool):
    """Display recommendations in the specified format."""
    if output == 'json':
        console.print(json.dumps(recommendations, indent=2))
    elif output == 'yaml':
        console.print(yaml.dump(recommendations, default_flow_style=False))
    else:  # table
        table = Table(title="Resource Recommendations")
        table.add_column("Resource", style="cyan")
        table.add_column("Request", style="green")
        table.add_column("Limit", style="yellow")
        
        table.add_row(
            "CPU",
            f"{recommendations.get('cpu_request', 'N/A')}m",
            f"{recommendations.get('cpu_limit', 'N/A')}m"
        )
        table.add_row(
            "Memory",
            f"{recommendations.get('memory_request', 'N/A')}Mi",
            f"{recommendations.get('memory_limit', 'N/A')}Mi"
        )
        
        console.print(table)
        
        # Additional info
        info_panel = Panel(
            f"📊 Data points analyzed: {recommendations.get('data_points', 'N/A')}\n"
            f"🎯 Confidence: {recommendations.get('confidence', 'N/A')}\n"
            f"🤖 Model: {recommendations.get('model_used', 'Statistical')}",
            title="Analysis Info"
        )
        console.print(info_panel)


def _display_training_results(metrics: Dict[str, Any], model_path: str, data_points: int):
    """Display training results."""
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model Path", model_path)
    table.add_row("Data Points", str(data_points))
    table.add_row("Training Loss", f"{metrics['train_loss']:.4f}")
    table.add_row("Validation Loss", f"{metrics['val_loss']:.4f}")
    table.add_row("Training MAE", f"{metrics['train_mae']:.4f}")
    table.add_row("Validation MAE", f"{metrics['val_mae']:.4f}")
    table.add_row("Training MAPE", f"{metrics['train_mape']:.2f}%")
    table.add_row("Validation MAPE", f"{metrics['val_mape']:.2f}%")
    
    console.print(table)
    
    # Success message
    success_panel = Panel(
        "🎉 Model trained successfully!\n"
        "You can now use 'recommend' command to get predictions.",
        style="green"
    )
    console.print(success_panel)


def _display_usage_analysis(df):
    """Display resource usage analysis."""
    console.print("\n📈 Resource Usage Analysis")
    
    # CPU Analysis
    if 'cpu_usage' in df.columns:
        cpu_stats = df['cpu_usage'].describe()
        console.print("\n🖥️  CPU Usage (millicores):")
        console.print(f"  Mean: {cpu_stats['mean']:.1f}")
        console.print(f"  P50: {cpu_stats['50%']:.1f}")
        console.print(f"  P95: {cpu_stats['95%']:.1f}")
        console.print(f"  P99: {cpu_stats['99%']:.1f}")
        console.print(f"  Max: {cpu_stats['max']:.1f}")
    
    # Memory Analysis
    if 'memory_usage' in df.columns:
        memory_stats = df['memory_usage'].describe()
        console.print("\n💾 Memory Usage (MiB):")
        console.print(f"  Mean: {memory_stats['mean']:.1f}")
        console.print(f"  P50: {memory_stats['50%']:.1f}")
        console.print(f"  P95: {memory_stats['95%']:.1f}")
        console.print(f"  P99: {memory_stats['99%']:.1f}")
        console.print(f"  Max: {memory_stats['max']:.1f}")
    
    # Utilization Analysis
    if all(col in df.columns for col in ['cpu_usage', 'cpu_requests']):
        cpu_utilization = df['cpu_usage'] / (df['cpu_requests'] * 1000)
        console.print(f"\n📊 CPU Utilization:")
        console.print(f"  Average: {cpu_utilization.mean():.1%}")
        console.print(f"  P95: {cpu_utilization.quantile(0.95):.1%}")
        console.print(f"  Max: {cpu_utilization.max():.1%}")
    
    if all(col in df.columns for col in ['memory_usage', 'memory_requests']):
        memory_utilization = df['memory_usage'] / df['memory_requests']
        console.print(f"\n📊 Memory Utilization:")
        console.print(f"  Average: {memory_utilization.mean():.1%}")
        console.print(f"  P95: {memory_utilization.quantile(0.95):.1%}")
        console.print(f"  Max: {memory_utilization.max():.1%}")


if __name__ == '__main__':
    cli() 
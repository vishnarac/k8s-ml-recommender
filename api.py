#!/usr/bin/env python3
"""
Main API entry point for Kubernetes Resource Recommendation Engine.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api import RecommendationAPI

if __name__ == '__main__':
    api = RecommendationAPI()
    api.run() 
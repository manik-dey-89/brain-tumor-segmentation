"""
Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path for all tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

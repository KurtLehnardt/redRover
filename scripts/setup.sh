#!/bin/bash
# Setup script for redRover development environment

set -e

echo "=== redRover Setup ==="

# Check Python version
python3 --version | grep -q "3.11\|3.12\|3.13" || {
    echo "ERROR: Python 3.11+ required"
    exit 1
}

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff

# Create data directory
mkdir -p data

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "Ollama found. Pulling gemma3..."
    ollama pull gemma3 || echo "WARNING: Could not pull gemma3. Make sure Ollama is running."
else
    echo "WARNING: Ollama not installed. Install from https://ollama.ai"
    echo "Then run: ollama pull gemma3"
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source venv/bin/activate"
echo "Run patrol:    python -m src.main --simulate"
echo "Run dashboard: python -m src.dashboard.app"
echo "Run tests:     pytest"

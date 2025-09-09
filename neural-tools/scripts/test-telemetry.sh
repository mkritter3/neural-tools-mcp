#!/bin/bash
"""
OpenTelemetry Integration Test Runner

Executes comprehensive telemetry tests with proper environment setup.
Validates tracing, metrics, and cross-component integration.
"""

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NEURAL_TOOLS_ROOT="$(dirname "$PROJECT_ROOT")/neural-tools"

cd "$NEURAL_TOOLS_ROOT"

echo "🧪 Running OpenTelemetry Integration Tests"
echo "Project root: $PROJECT_ROOT"
echo "Neural tools: $NEURAL_TOOLS_ROOT"
echo

# Set test environment
export PYTHONPATH="$NEURAL_TOOLS_ROOT/src:$PYTHONPATH"
export ENVIRONMENT="test"
export LOG_LEVEL="INFO"

echo "📊 Environment Setup:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  ENVIRONMENT: $ENVIRONMENT"
echo

# Install test dependencies if needed
if ! python3 -c "import pytest, opentelemetry" 2>/dev/null; then
    echo "📦 Installing test dependencies..."
    pip3 install pytest pytest-asyncio opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
fi

echo "🔍 Running OpenTelemetry Tests..."
echo

# Run telemetry integration tests
python3 -m pytest \
    tests/integration/test_opentelemetry_integration.py \
    -v \
    --tb=short \
    --capture=no \
    --strict-markers \
    --disable-warnings

echo
echo "✅ OpenTelemetry Integration Tests Complete"

# Optionally run performance impact tests separately
echo
echo "⚡ Running Performance Impact Tests..."
python3 -m pytest \
    tests/integration/test_opentelemetry_integration.py::TestTelemetryPerformanceImpact \
    -v \
    --tb=short \
    --capture=no

echo
echo "✅ All Telemetry Tests Complete!"
echo "📈 OpenTelemetry instrumentation is production-ready"
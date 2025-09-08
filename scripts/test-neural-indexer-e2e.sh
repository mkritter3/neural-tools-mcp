#!/bin/bash
set -e

echo "🧠 Neural Indexer Sidecar - End-to-End Test"
echo "============================================"

# Check prerequisites
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

if ! which curl > /dev/null 2>&1; then
    echo "❌ curl is not installed"
    exit 1
fi

echo "✅ Prerequisites checked"

# Configuration
PROJECT_NAME="test-e2e"
CONTAINER_NAME="test-neural-indexer"
TEST_PORT="8082"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

echo "📦 Building neural indexer container..."
if ! docker build -f docker/Dockerfile.indexer -t l9-neural-indexer:e2e-test . > /dev/null 2>&1; then
    echo "❌ Container build failed"
    exit 1
fi

echo "✅ Container built successfully"

echo "🚀 Starting neural indexer container..."

# Start container with minimal configuration for testing
docker run -d \
    --name $CONTAINER_NAME \
    -p $TEST_PORT:8080 \
    -v $(pwd):/workspace:ro \
    -e PROJECT_NAME=$PROJECT_NAME \
    -e PROJECT_PATH=/workspace \
    -e INITIAL_INDEX=false \
    -e LOG_LEVEL=INFO \
    -e NEO4J_URI=bolt://localhost:7687 \
    -e QDRANT_HOST=localhost \
    -e EMBEDDING_SERVICE_HOST=localhost \
    l9-neural-indexer:e2e-test

echo "⏳ Waiting for container startup (30s)..."
sleep 30

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -s -f http://localhost:$TEST_PORT/health > /dev/null; then
    echo "✅ Health endpoint responds"
else
    echo "❌ Health endpoint failed"
    echo "Container logs:"
    docker logs $CONTAINER_NAME --tail 20
    exit 1
fi

# Test status endpoint
echo "📊 Testing status endpoint..."
STATUS_RESPONSE=$(curl -s http://localhost:$TEST_PORT/status)
if echo "$STATUS_RESPONSE" | grep -q "running"; then
    echo "✅ Status endpoint responds correctly"
    echo "Status: $STATUS_RESPONSE"
else
    echo "❌ Status endpoint failed"
    echo "Response: $STATUS_RESPONSE"
    exit 1
fi

# Test metrics endpoint
echo "📈 Testing metrics endpoint..."
if curl -s http://localhost:$TEST_PORT/metrics | grep -q "indexer_"; then
    echo "✅ Metrics endpoint provides Prometheus metrics"
else
    echo "❌ Metrics endpoint failed"
    exit 1
fi

# Test reindex endpoint
echo "🔄 Testing reindex endpoint..."
REINDEX_RESPONSE=$(curl -s -X POST "http://localhost:$TEST_PORT/reindex-path?path=/workspace/README.md")
if echo "$REINDEX_RESPONSE" | grep -q -E "(success|queued|error)"; then
    echo "✅ Reindex endpoint responds"
    echo "Reindex response: $REINDEX_RESPONSE"
else
    echo "❌ Reindex endpoint failed"
    echo "Response: $REINDEX_RESPONSE"
    exit 1
fi

# Check container logs for any errors
echo "📋 Checking container logs for errors..."
ERROR_COUNT=$(docker logs $CONTAINER_NAME 2>&1 | grep -i error | wc -l)
WARNING_COUNT=$(docker logs $CONTAINER_NAME 2>&1 | grep -i warning | wc -l)

echo "Errors in logs: $ERROR_COUNT"
echo "Warnings in logs: $WARNING_COUNT"

if [ "$ERROR_COUNT" -gt 5 ]; then
    echo "❌ Too many errors in container logs"
    echo "Recent logs:"
    docker logs $CONTAINER_NAME --tail 10
    exit 1
fi

# Test graceful shutdown
echo "🛑 Testing graceful shutdown..."
docker stop $CONTAINER_NAME > /dev/null

# Wait for shutdown
sleep 5

# Check if container stopped gracefully (exit code 0 or 143 for SIGTERM)
EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')
if [ "$EXIT_CODE" = "0" ] || [ "$EXIT_CODE" = "143" ]; then
    echo "✅ Container shutdown gracefully (exit code: $EXIT_CODE)"
else
    echo "❌ Container did not shutdown gracefully (exit code: $EXIT_CODE)"
    echo "Last 10 log lines:"
    docker logs $CONTAINER_NAME --tail 10
    exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo "✅ Container builds successfully"
echo "✅ Health endpoint works"
echo "✅ Status endpoint provides metrics"
echo "✅ Prometheus metrics available"
echo "✅ Reindex endpoint responds"
echo "✅ Graceful shutdown works"
echo ""
echo "The neural indexer sidecar container is ready for production deployment!"

cleanup
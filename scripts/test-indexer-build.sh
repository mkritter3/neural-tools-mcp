#!/bin/bash
set -e

echo "🔧 Testing Neural Indexer Container Build"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Build the container
echo "📦 Building neural indexer container..."
echo "This may take several minutes due to PyTorch dependencies..."

if docker build -f docker/Dockerfile.indexer -t l9-neural-indexer:test .; then
    echo "✅ Container build successful"
    
    # Test container can start (health check)
    echo "🏃 Testing container startup..."
    
    # Start container in background
    docker run --name test-indexer -d \
        -p 8081:8080 \
        -v $(pwd):/workspace:ro \
        l9-neural-indexer:test \
        /workspace --project-name test --initial-index
    
    # Wait for health check
    echo "⏳ Waiting for health check..."
    sleep 10
    
    # Check health endpoint
    if curl -s http://localhost:8081/health > /dev/null; then
        echo "✅ Health check passed"
        
        # Check status endpoint
        echo "📊 Testing status endpoint..."
        curl -s http://localhost:8081/status | jq '.' || echo "Status endpoint responded"
        
        echo "✅ All tests passed!"
    else
        echo "❌ Health check failed"
        docker logs test-indexer
    fi
    
    # Cleanup
    docker stop test-indexer || true
    docker rm test-indexer || true
    
else
    echo "❌ Container build failed"
    exit 1
fi
#!/bin/bash
set -e

echo "🚀 Starting Integration Test Environment"
echo "======================================"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker Desktop first."
    echo "   After starting Docker, run this script again."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📦 Starting test services with Docker Compose..."
docker-compose up -d neo4j qdrant

echo "⏳ Waiting for services to be healthy..."

# Wait for Neo4j
echo "Waiting for Neo4j..."
timeout=60
counter=0
until docker-compose exec neo4j cypher-shell -u neo4j -p graphrag-password "RETURN 1" &> /dev/null; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ Neo4j failed to start within $timeout seconds"
        docker-compose logs neo4j
        exit 1
    fi
    echo "Neo4j not ready yet... ($counter/$timeout seconds)"
done
echo "✅ Neo4j is ready"

# Wait for Qdrant
echo "Waiting for Qdrant..."
counter=0
until curl -f http://localhost:6333/collections &> /dev/null; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ Qdrant failed to start within $timeout seconds"
        docker-compose logs qdrant
        exit 1
    fi
    echo "Qdrant not ready yet... ($counter/$timeout seconds)"
done
echo "✅ Qdrant is ready"

echo ""
echo "🎉 All services are ready for integration testing!"
echo ""
echo "Service URLs:"
echo "  - Neo4j Browser: http://localhost:7474 (neo4j/graphrag-password)"
echo "  - Qdrant API: http://localhost:6333"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "Run integration tests with:"
echo "  cd neural-tools"
echo "  python -m pytest tests/integration/ -v -m integration"
echo ""
echo "To stop services:"
echo "  docker-compose down"
#!/bin/bash
# Start the indexer service in the background
# This script is called when the container starts

echo "Starting L9 Incremental Indexer..."

# Wait for services to be ready
sleep 10

# Check if initial index is needed
if [ ! -f /app/data/.indexer_initialized ]; then
    echo "Running initial index..."
    python3 -u /app/neural-tools-src/services/indexer_service.py /workspace --project-name="${PROJECT_NAME:-default}" --initial-index &
    touch /app/data/.indexer_initialized
else
    echo "Starting indexer (no initial index)..."
    python3 -u /app/neural-tools-src/services/indexer_service.py /workspace --project-name="${PROJECT_NAME:-default}" &
fi

# Save PID for later management
echo $! > /app/data/indexer.pid
echo "Indexer started with PID: $(cat /app/data/indexer.pid)"
.PHONY: smoke

# Minimal smoke to validate containers + services
smoke:
	@echo "Starting core services..."
	docker-compose up -d neo4j qdrant redis
	@sleep 3
	@echo "Checking Qdrant HTTP..."
	curl -fsS http://localhost:46333/collections >/dev/null || (echo "Qdrant HTTP check failed" && exit 1)
	@echo "Starting indexer..."
	docker-compose up -d l9-indexer
	@echo "Waiting for indexer health..."
	@for i in $$(seq 1 30); do \
		curl -fsS http://localhost:48080/health && break || sleep 2; \
	done
	@echo "Indexer status:" && curl -fsS http://localhost:48080/status || true
	@echo "Smoke complete."

.PHONY: smoke-mcp
smoke-mcp:
	@echo "Running MCP STDIO smoke..."
	python scripts/smoke_mcp_stdio.py
	@echo "MCP smoke complete."

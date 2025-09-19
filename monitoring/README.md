# Neural Tools Production Monitoring Stack

## Overview

This directory contains the production monitoring configuration for the Neural Tools MCP server, implementing ADR-053 requirements for WriteSynchronizationManager monitoring.

## Components

### 1. Prometheus (`prometheus.yml`)
- **Purpose**: Metrics collection and storage
- **Port**: 9090
- **Features**:
  - Collects metrics from all neural-tools services
  - 15-second scrape interval for real-time monitoring
  - Alert rule evaluation

### 2. Grafana Dashboard (`grafana-dashboard.json`)
- **Purpose**: Visualization and monitoring
- **Port**: 3000 (default credentials: admin/neural-tools-2025)
- **Panels**:
  - Synchronization Rate gauge (must stay >95% per ADR-053)
  - Total Writes per Minute
  - Failed Syncs counter
  - Rollbacks counter
  - Sync Operations Timeline
  - Sync Latency (P50, P95, P99)
  - Database Consistency (Neo4j vs Qdrant drift)
  - Service Health timeline

### 3. Alert Rules (`prometheus-alerts.yml`)
- **Critical Alerts**:
  - `LowSyncRate`: Sync rate < 95% (ADR-053 requirement)
  - `CriticalSyncRate`: Sync rate < 80% (pages on-call)
  - `DatabaseDrift`: Neo4j-Qdrant chunk count difference > 100

- **Warning Alerts**:
  - `HighRollbackRate`: > 0.1 rollbacks/sec
  - `HighSyncLatency`: P95 latency > 1s
  - `IndexerUnhealthy`: Health check failing > 5min

### 4. AlertManager (`alertmanager.yml`)
- **Purpose**: Alert routing and notification
- **Port**: 9093
- **Features**:
  - Groups alerts by severity
  - Routes critical alerts to PagerDuty
  - Email notifications for warnings

## Quick Start

### 1. Start Monitoring Stack
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/neural-tools-2025)
- **AlertManager**: http://localhost:9093

### 3. Import Grafana Dashboard
1. Login to Grafana
2. Go to Dashboards → Import
3. Upload `grafana-dashboard.json`
4. Select Prometheus datasource
5. Click Import

## Key Metrics

### ADR-053 Compliance Metrics
```promql
# Sync success rate (must be >95%)
(rate(neural_tools_sync_successful[5m]) / rate(neural_tools_sync_total[5m])) * 100

# Database consistency check
abs(neural_tools_neo4j_chunk_count - neural_tools_qdrant_point_count)

# Rollback rate (should be near 0)
rate(neural_tools_sync_rollbacks[5m])
```

### Performance Metrics
```promql
# P95 sync latency
histogram_quantile(0.95, rate(neural_tools_sync_duration_seconds_bucket[5m]))

# Writes per minute
rate(neural_tools_sync_total[1m]) * 60
```

## Alert Response Playbook

### Low Sync Rate Alert (<95%)
1. Check Neo4j and Qdrant service health
2. Review recent error logs: `docker logs neural-tools-metrics-exporter`
3. Check network connectivity between services
4. Consider enabling degraded mode if persistent

### Database Drift Alert
1. Run reconciliation: `python scripts/reconcile-databases.py`
2. Check for stuck transactions in Neo4j
3. Verify Qdrant collection health
4. Review indexer logs for errors

### High Rollback Rate
1. Check Neo4j transaction logs
2. Review Qdrant write failures
3. Check for resource constraints (memory, CPU)
4. Consider increasing retry delays

## Production Deployment

### Prerequisites
- Docker and Docker Compose installed
- Ports 9090, 3000, 9093 available
- At least 2GB RAM for monitoring stack

### Environment Variables
Create `.env` file for production:
```env
GF_SECURITY_ADMIN_PASSWORD=<strong-password>
ALERTMANAGER_SLACK_WEBHOOK=<webhook-url>
PAGERDUTY_SERVICE_KEY=<service-key>
```

### Scaling Considerations
- Prometheus retention: 15 days default (adjust `--storage.tsdb.retention.time`)
- Grafana datasource limits: Configure query timeouts for large datasets
- Alert throttling: Configured in alertmanager.yml to prevent alert storms

## Troubleshooting

### No Metrics Appearing
```bash
# Check exporter is running
docker ps | grep neural-metrics-exporter

# Test metrics endpoint
curl http://localhost:9200/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Grafana Connection Issues
```bash
# Check datasource configuration
curl -u admin:neural-tools-2025 http://localhost:3000/api/datasources

# Test Prometheus from Grafana container
docker exec neural-tools-grafana curl http://prometheus:9090/api/v1/query?query=up
```

### Alert Not Firing
```bash
# Check alert state in Prometheus
curl http://localhost:9090/api/v1/alerts

# Verify AlertManager is receiving alerts
curl http://localhost:9093/api/v1/alerts
```

## Maintenance

### Backup Grafana Dashboards
```bash
# Export all dashboards
for db in $(curl -s -u admin:neural-tools-2025 http://localhost:3000/api/search | jq -r '.[].uid'); do
  curl -s -u admin:neural-tools-2025 http://localhost:3000/api/dashboards/uid/$db > dashboard-$db.json
done
```

### Update Alert Thresholds
1. Edit `prometheus-alerts.yml`
2. Reload Prometheus config: `curl -X POST http://localhost:9090/-/reload`

### Clean Up Old Data
```bash
# Prometheus data
docker exec neural-tools-prometheus promtool tsdb clean /prometheus

# Grafana data (be careful!)
docker volume rm neural-tools_grafana_data
```

## Integration with CI/CD

The monitoring stack is automatically validated in the CI/CD pipeline:
- GitHub Actions workflow tests metric endpoints
- Deploy script checks monitoring health before deployment
- Alerts are sent to Slack/PagerDuty on production issues

## References
- [ADR-053: WriteSynchronizationManager Monitoring](../docs/adr/0053-production-grade-neo4j-qdrant-synchronization.md)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
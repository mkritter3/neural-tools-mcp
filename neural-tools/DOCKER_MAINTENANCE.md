# Docker Maintenance Guide

## ⚠️ The Problem: Docker Storage Debt

Continuous development with Docker can quickly consume disk space:
- Each rebuild creates new image layers
- Old containers accumulate
- Build cache grows unbounded
- Dangling images pile up
- Unused volumes persist

**Your current usage: 123GB of images with 92GB reclaimable!**

## 🧹 Quick Cleanup

### Immediate Space Recovery
```bash
# Safe cleanup (removes only dangling images)
./docker-cleanup.sh

# Aggressive cleanup (removes all unused images)
./docker-cleanup.sh --aggressive

# Preview what would be cleaned
./docker-cleanup.sh --dry-run
```

### Cleanup During Build
```bash
# Clean before rebuilding
./build-and-run.sh --clean --rebuild
```

## 📊 Storage Management Strategy

### 1. Regular Maintenance Schedule

**Daily Development:**
```bash
# Use dev mode to avoid rebuilds
./build-and-run.sh --dev
```

**Weekly Cleanup:**
```bash
# Remove dangling images and stopped containers
./docker-cleanup.sh
```

**Monthly Deep Clean:**
```bash
# Remove all unused images
./docker-cleanup.sh --aggressive
```

### 2. Prevent Storage Debt

#### Use Development Mode
Avoid unnecessary rebuilds during development:
```bash
# Code changes apply immediately without rebuild
./build-and-run.sh --dev
```

#### Smart Rebuilding
Only rebuild when necessary:
```bash
# Check if rebuild is needed
docker images | grep l9-mcp-enhanced

# Force rebuild only when required
./build-and-run.sh --rebuild
```

#### Layer Caching
Dockerfile is optimized for layer caching:
- System dependencies cached
- Python packages cached separately
- Source code copied last

## 🔧 Docker Configuration

### Set Global Limits
Create/edit `~/.docker/daemon.json`:
```json
{
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "20GB"
    }
  },
  "storage-opts": [
    "dm.basesize=50G"
  ]
}
```

Then restart Docker Desktop.

### Prune Automation
Add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
# Docker cleanup alias
alias dclean='docker system prune -f'
alias dcleanall='docker system prune -a -f --volumes'

# Weekly cleanup reminder
if [[ $(date +%u) -eq 1 ]]; then
  echo "🧹 Monday reminder: Run ./docker-cleanup.sh"
fi
```

## 📈 Monitoring Commands

### Check Current Usage
```bash
# Overview
docker system df

# Detailed image list
docker images -a

# Sort images by size
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | sort -k3 -h

# Find large containers
docker ps -a --size
```

### Identify Culprits
```bash
# Largest images
docker images --format "{{.Size}}\t{{.Repository}}:{{.Tag}}" | sort -rh | head -10

# Dangling images
docker images -f dangling=true

# Build cache size
docker builder du
```

## 🛡️ Safe Cleanup Practices

### What's Safe to Remove

✅ **Always Safe:**
- Stopped containers
- Dangling images
- Unused networks
- Build cache

⚠️ **Check First:**
- Named volumes (may contain data)
- Tagged images (may be in use)
- Running container volumes

❌ **Never Remove:**
- Running containers
- Images for active projects
- Volumes with persistent data

### Before Aggressive Cleanup

1. **Stop all services:**
```bash
docker-compose -f config/docker-compose.neural-tools.yml down
```

2. **List what will be kept:**
```bash
docker images --filter "reference=*neural*"
docker images --filter "reference=*qdrant*"
docker images --filter "reference=*neo4j*"
```

3. **Run cleanup:**
```bash
./docker-cleanup.sh --aggressive
```

## 🚀 Best Practices

### 1. Development Workflow
```bash
# Monday: Clean start
./docker-cleanup.sh
./build-and-run.sh --rebuild

# Tue-Fri: Dev mode
./build-and-run.sh --dev

# Friday: Cleanup before weekend
docker-compose -f config/docker-compose.neural-tools.yml down
./docker-cleanup.sh
```

### 2. CI/CD Practices
```bash
# Build script for CI
./build-and-run.sh --clean --rebuild --build-only

# Cleanup after tests
docker-compose down --volumes --remove-orphans
```

### 3. Multi-Project Setup
```bash
# Use project-specific names
export PROJECT_NAME=project1
./build-and-run.sh

# Clean project-specific resources
docker-compose -p project1 down --volumes
```

## 📋 Troubleshooting

### "No space left on device"
```bash
# Emergency cleanup
docker system prune -a -f --volumes
./docker-cleanup.sh --aggressive
```

### Rebuild Issues After Cleanup
```bash
# Clear builder cache
docker builder prune -a -f

# Rebuild with no cache
docker-compose build --no-cache
```

### Slow Builds
```bash
# Check builder cache
docker builder du

# Use targeted cleanup
docker builder prune --filter until=24h
```

## 📊 Metrics & Goals

### Target Metrics
- Docker images: < 20GB total
- Build cache: < 5GB
- Dangling images: 0
- Stopped containers: 0

### Weekly Health Check
```bash
#!/bin/bash
# Add to crontab: 0 9 * * MON /path/to/weekly-docker-health.sh

echo "📊 Docker Health Check - $(date)"
echo "=========================="
docker system df
echo ""
echo "Action items:"
docker images -f dangling=true -q | wc -l | xargs -I {} echo "- {} dangling images to clean"
docker ps -a -q -f status=exited | wc -l | xargs -I {} echo "- {} stopped containers to remove"
```

## 🎯 Summary

**Prevention > Cleanup:**
1. Use `--dev` mode during development
2. Run weekly cleanups
3. Set Docker daemon limits
4. Monitor usage regularly

**Quick Commands:**
- `./docker-cleanup.sh` - Weekly cleanup
- `./build-and-run.sh --clean --rebuild` - Clean rebuild
- `docker system df` - Check usage
- `./docker-cleanup.sh --aggressive` - Monthly deep clean

**Remember:** 
- Development mode = No rebuilds = No storage debt
- Regular cleanup = Healthy system
- Monitor before it's too late!
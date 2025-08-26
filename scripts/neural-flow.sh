#!/bin/bash
# Neural Flow Multi-Project Management Script
# Provides seamless project switching with Docker containerization

set -e

# Configuration
NEURAL_FLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECTS_DIR="${NEURAL_FLOW_DIR}/projects"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Show usage
usage() {
    cat << EOF
🔮 Neural Flow - Multi-Project Intelligence System

USAGE:
  $(basename "$0") [COMMAND] [PROJECT_NAME]

COMMANDS:
  init <project>     Initialize new project with neural flow
  start <project>    Start neural flow for specific project  
  stop <project>     Stop neural flow container
  switch <project>   Switch to different project
  status             Show running containers and project status
  logs <project>     Show container logs
  shell <project>    Open shell in container
  clean              Clean up unused containers and volumes
  update             Update neural flow system

EXAMPLES:
  $(basename "$0") init my-awesome-app
  $(basename "$0") start my-awesome-app
  $(basename "$0") status
  $(basename "$0") switch my-other-project

EOF
}

# Ensure Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"  
        exit 1
    fi
}

# Initialize new project
init_project() {
    local project_name="$1"
    if [[ -z "$project_name" ]]; then
        log_error "Project name required"
        exit 1
    fi
    
    local project_dir="${PROJECTS_DIR}/${project_name}"
    
    log_info "Initializing project: $project_name"
    
    # Create project structure
    mkdir -p "$project_dir"/{src,.claude/chroma,.claude/memory}
    
    # Create project-specific .mcp.json
    cat > "$project_dir/.mcp.json" << EOF
{
  "neural-flow": {
    "command": "docker",
    "args": [
      "compose",
      "--project-directory", 
      "$NEURAL_FLOW_DIR",
      "--project-name",
      "neural-flow-${project_name}",
      "exec",
      "-T",
      "neural-flow",
      "/app/docker-entrypoint.sh"
    ],
    "env": {
      "PROJECT_NAME": "$project_name",
      "USE_QODO_EMBED": "true",
      "ENABLE_AB_TESTING": "false",
      "ENABLE_PERFORMANCE_MONITORING": "true"
    }
  }
}
EOF
    
    # Create project-specific Claude Code settings
    cat > "$project_dir/.claude/settings.json" << EOF
{
  "enableAllProjectMcpServers": true,
  "env": {
    "PROJECT_NAME": "$project_name",
    "USE_QODO_EMBED": "true"
  },
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Edit(*)",
      "Write(*)"
    ]
  }
}
EOF

    # Create sample README
    cat > "$project_dir/README.md" << EOF
# $project_name

Neural Flow enabled project with L9-grade intelligence.

## Getting Started

\`\`\`bash
# Start neural flow
../scripts/neural-flow.sh start $project_name

# Open Claude Code in this directory
cd $(realpath "$project_dir")
claude
\`\`\`

## Neural Capabilities

- 🧠 **Semantic Memory**: Conversation context preservation
- 🔍 **Code Intelligence**: AST-aware project understanding  
- ⚡ **Fast Search**: Vector-based similarity search
- 🏷️  **Smart Tagging**: Automated categorization
- 🔄 **Multi-Model**: A/B testing between embedding models

EOF

    log_success "Project $project_name initialized at $project_dir"
    log_info "Next steps:"
    log_info "  1. cd $project_dir && claude"
    log_info "  2. $(basename "$0") start $project_name"
}

# Start project container
start_project() {
    local project_name="$1"
    if [[ -z "$project_name" ]]; then
        log_error "Project name required"
        exit 1
    fi
    
    local project_dir="${PROJECTS_DIR}/${project_name}"
    if [[ ! -d "$project_dir" ]]; then
        log_error "Project $project_name not found. Run 'init $project_name' first."
        exit 1
    fi
    
    log_info "Starting neural flow for project: $project_name"
    
    cd "$NEURAL_FLOW_DIR"
    export PROJECT_NAME="$project_name"
    
    # Build if needed
    if [[ "$(docker images -q neural-flow_neural-flow 2> /dev/null)" == "" ]]; then
        log_info "Building neural flow container..."
        docker-compose build neural-flow
    fi
    
    # Start container
    docker-compose --project-name "neural-flow-$project_name" up -d neural-flow
    
    # Wait for health check
    log_info "Waiting for container to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose --project-name "neural-flow-$project_name" ps neural-flow | grep -q "Up (healthy)"; then
            log_success "Neural flow ready for project $project_name"
            log_info "Container: neural-flow-${project_name}_neural-flow_1"
            return 0
        fi
        
        sleep 2
        ((attempt++))
    done
    
    log_error "Container failed to start properly"
    docker-compose --project-name "neural-flow-$project_name" logs neural-flow
    exit 1
}

# Stop project container
stop_project() {
    local project_name="$1"
    if [[ -z "$project_name" ]]; then
        log_error "Project name required"
        exit 1
    fi
    
    log_info "Stopping neural flow for project: $project_name"
    
    cd "$NEURAL_FLOW_DIR"
    docker-compose --project-name "neural-flow-$project_name" down
    
    log_success "Stopped neural flow for project $project_name"
}

# Show status
show_status() {
    log_info "Neural Flow System Status"
    echo
    
    # Show running containers
    local containers=$(docker ps --filter "label=com.docker.compose.service=neural-flow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    
    if [[ -n "$containers" ]]; then
        echo "🔮 Running Neural Flow Containers:"
        echo "$containers"
    else
        echo "📪 No neural flow containers running"
    fi
    
    echo
    
    # Show available projects
    if [[ -d "$PROJECTS_DIR" ]]; then
        echo "📁 Available Projects:"
        for project_dir in "$PROJECTS_DIR"/*/; do
            if [[ -d "$project_dir" ]]; then
                local project_name=$(basename "$project_dir")
                local mcp_file="$project_dir/.mcp.json"
                if [[ -f "$mcp_file" ]]; then
                    echo "  ✅ $project_name (configured)"
                else
                    echo "  ⚪ $project_name (needs configuration)"
                fi
            fi
        done
    else
        echo "📁 No projects directory found"
    fi
}

# Main command dispatcher
main() {
    check_docker
    
    case "${1:-}" in
        init)
            init_project "$2"
            ;;
        start)
            start_project "$2"
            ;;
        stop)
            stop_project "$2"
            ;;
        status)
            show_status
            ;;
        logs)
            if [[ -z "$2" ]]; then
                log_error "Project name required"
                exit 1
            fi
            cd "$NEURAL_FLOW_DIR"
            docker-compose --project-name "neural-flow-$2" logs -f neural-flow
            ;;
        shell)
            if [[ -z "$2" ]]; then
                log_error "Project name required"
                exit 1
            fi
            cd "$NEURAL_FLOW_DIR"
            docker-compose --project-name "neural-flow-$2" exec neural-flow bash
            ;;
        clean)
            log_info "Cleaning up neural flow containers and volumes..."
            docker system prune -f --filter "label=com.docker.compose.project" 
            log_success "Cleanup complete"
            ;;
        update)
            log_info "Updating neural flow system..."
            cd "$NEURAL_FLOW_DIR"
            docker-compose build --no-cache neural-flow
            log_success "Update complete"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            echo
            usage
            exit 1
            ;;
    esac
}

main "$@"
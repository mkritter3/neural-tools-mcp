#!/bin/bash
# Global Neural Flow Installation Script
# Makes neural-init available system-wide

set -e

INSTALL_DIR="$HOME/.neural-flow"
BIN_DIR="$HOME/.local/bin"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "🔮 Neural Flow Global Installation"
echo "=================================="

# Create directories
mkdir -p "$INSTALL_DIR" "$BIN_DIR"

# Copy Neural Flow system
log_info "Copying Neural Flow to $INSTALL_DIR..."
cp -r "$(pwd)"/* "$INSTALL_DIR/"

# Create global neural-init command
log_info "Creating global neural-init command..."
cat > "$BIN_DIR/neural-init" << EOF
#!/bin/bash
exec "$INSTALL_DIR/scripts/neural-init" "\$@"
EOF
chmod +x "$BIN_DIR/neural-init"

# Create global neural-flow command  
log_info "Creating global neural-flow command..."
cat > "$BIN_DIR/neural-flow" << EOF
#!/bin/bash
exec "$INSTALL_DIR/scripts/neural-flow.sh" "\$@"  
EOF
chmod +x "$BIN_DIR/neural-flow"

# Create global neural-setup command
log_info "Creating global neural-setup command..."
cat > "$BIN_DIR/neural-setup" << EOF
#!/bin/bash
exec python3 "$INSTALL_DIR/neural-setup.py" "\$@"
EOF
chmod +x "$BIN_DIR/neural-setup"

# Create global neural command (MAIN UX)
log_info "Creating global neural command..."
cat > "$BIN_DIR/neural" << EOF
#!/bin/bash
# L9 Neural - Zero-effort neural memory for vibe coders
exec "$INSTALL_DIR/scripts/neural-here" "\$@"
EOF
chmod +x "$BIN_DIR/neural"

# Create global mcp add command
log_info "Creating global mcp add command..."
cat > "$BIN_DIR/mcp" << EOF
#!/bin/bash
# L9 MCP wrapper - intercepts 'add' to provide L9 neural memory
case "\$1" in
    add)
        shift
        exec "$INSTALL_DIR/mcp-add" "\$@"
        ;;
    *)
        # Pass through to claude mcp for other commands
        exec claude mcp "\$@"
        ;;
esac
EOF
chmod +x "$BIN_DIR/mcp"

log_success "🎯 L9 Neural Flow installed globally!"
echo
log_success "🧠 MAIN COMMAND (for vibe coders):"
log_info "  neural         # Zero-effort neural memory (just works!)"
echo  
log_info "📚 All Commands:"
log_info "  neural         # Enable neural memory in any directory"
log_info "  neural stop    # Stop neural memory for current project"
log_info "  neural status  # Show all neural containers"
log_info "  neural-init <project> [template]   # Initialize new neural project"
log_info "  neural-flow <command> [project]    # Manage neural containers"
log_info "  neural-setup [path] [flags]        # Add neural memory to existing projects"
log_info "  mcp add [options] [path]           # Add L9 neural memory MCP server"
echo
log_warn "Make sure $BIN_DIR is in your PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  # Add to your ~/.bashrc or ~/.zshrc"
echo
log_success "🚀 ULTRA-SIMPLE Quick Start:"
log_info "  # In ANY project directory:"
log_info "  cd /path/to/your/project"
log_info "  neural                     # That's it! Neural memory enabled"
log_info "  claude                     # Start coding with L9 intelligence"
echo
log_info "📖 Advanced Usage:"
log_info "  # New managed project:"
log_info "  neural-init my-awesome-app python"
log_info "  cd my-awesome-app && ./neural-start.sh"
echo
log_info "  # Existing project with full setup:"
log_info "  cd /path/to/existing-project"
log_info "  neural-setup              # Full setup with hooks"
log_info "  neural-setup --no-hooks   # Memory only, no hooks"
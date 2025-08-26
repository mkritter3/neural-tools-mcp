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

log_success "Neural Flow installed globally!"
echo
log_info "Commands available:"
log_info "  neural-init <project> [template]  # Initialize new neural project"
log_info "  neural-flow <command> [project]   # Manage neural containers"
echo
log_warn "Make sure $BIN_DIR is in your PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "  # Add to your ~/.bashrc or ~/.zshrc"
echo
log_info "Quick start:"
log_info "  neural-init my-awesome-app python"
log_info "  cd my-awesome-app"
log_info "  ./neural-start.sh"
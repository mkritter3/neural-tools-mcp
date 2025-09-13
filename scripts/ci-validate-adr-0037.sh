#!/bin/bash
# CI/CD ADR-0037 Validation Script
# Automatically validates container configuration compliance in CI/CD pipelines
#
# Usage:
#   ./scripts/ci-validate-adr-0037.sh
#   ./scripts/ci-validate-adr-0037.sh --strict        # Fail on warnings too
#   ./scripts/ci-validate-adr-0037.sh --code-only     # Only validate code patterns

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALIDATION_SCRIPT="$PROJECT_ROOT/scripts/validate-adr-0037.py"
LOG_FILE="$PROJECT_ROOT/adr-0037-validation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
STRICT_MODE=false
CODE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --code-only)
            CODE_ONLY=true
            shift
            ;;
        --help)
            echo "ADR-0037 CI/CD Validation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --strict     Fail on warnings (default: fail only on errors)"
            echo "  --code-only  Only validate code patterns (skip container validation)"
            echo "  --help       Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  CI                 Set to 'true' for CI mode (less verbose output)"
            echo "  SKIP_CONTAINER_CHECK  Skip running container validation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Initialize log file
echo "=== ADR-0037 CI/CD Validation Started: $(date) ===" > "$LOG_FILE"

log_info "Starting ADR-0037 configuration compliance validation"
log_info "Project root: $PROJECT_ROOT"
log_info "Strict mode: $STRICT_MODE"
log_info "Code only: $CODE_ONLY"

# Validation counters
total_checks=0
passed_checks=0
warnings=0
errors=0
exit_code=0

# Check if validation script exists
if [[ ! -f "$VALIDATION_SCRIPT" ]]; then
    log_error "ADR-0037 validation script not found: $VALIDATION_SCRIPT"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required for ADR-0037 validation"
    exit 1
fi

# Make validation script executable
chmod +x "$VALIDATION_SCRIPT"

# Function to run validation and capture results
run_validation() {
    local cmd="$1"
    local description="$2"
    
    log_info "Running: $description"
    
    # Run validation and capture output
    if output=$(python3 "$VALIDATION_SCRIPT" $cmd 2>&1); then
        local validation_exit_code=0
    else
        local validation_exit_code=$?
    fi
    
    # Parse validation output for metrics
    if echo "$output" | grep -q "SUMMARY:"; then
        local summary_line=$(echo "$output" | grep "SUMMARY:" | tail -1)
        local passed=$(echo "$summary_line" | sed 's/.*SUMMARY: \([0-9]*\) passed.*/\1/')
        local warns=$(echo "$summary_line" | sed 's/.*passed, \([0-9]*\) warnings.*/\1/')
        local errs=$(echo "$summary_line" | sed 's/.*warnings, \([0-9]*\) errors.*/\1/')
        
        passed_checks=$((passed_checks + passed))
        warnings=$((warnings + warns))
        errors=$((errors + errs))
        total_checks=$((total_checks + passed + warns + errs))
        
        log_info "Results: $passed passed, $warns warnings, $errs errors"
    fi
    
    # Log detailed output in CI mode
    if [[ "${CI:-false}" == "true" ]]; then
        echo "$output" >> "$LOG_FILE"
    else
        echo "$output"
    fi
    
    # Check for compliance failure
    if echo "$output" | grep -q "🚨 ADR-0037 COMPLIANCE: FAILED"; then
        log_error "$description: FAILED"
        exit_code=1
    elif echo "$output" | grep -q "🎉 ADR-0037 COMPLIANCE: PASSED"; then
        log_success "$description: PASSED"
    else
        log_warning "$description: Unexpected output format"
        if [[ $validation_exit_code -ne 0 ]]; then
            exit_code=1
        fi
    fi
    
    return $validation_exit_code
}

# 1. Validate code patterns in key directories
log_info "=== Phase 1: Code Pattern Validation ==="

code_dirs=("docker/scripts" "neural-tools/src/servers")
for dir in "${code_dirs[@]}"; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
        run_validation "--validate-code $dir" "Code patterns in $dir"
    else
        log_warning "Directory not found, skipping: $dir"
    fi
done

# 2. Validate running containers (unless code-only mode)
if [[ "$CODE_ONLY" == "false" && "${SKIP_CONTAINER_CHECK:-false}" != "true" ]]; then
    log_info "=== Phase 2: Container Configuration Validation ==="
    
    # Check if Docker is available
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        run_validation "--all" "All running containers"
    else
        log_warning "Docker not available, skipping container validation"
        log_warning "Set SKIP_CONTAINER_CHECK=true to suppress this warning"
    fi
else
    log_info "Skipping container validation (code-only mode or SKIP_CONTAINER_CHECK set)"
fi

# 3. Generate summary report
log_info "=== Validation Summary ==="
log_info "Total checks: $total_checks"
log_info "Passed: $passed_checks"
log_info "Warnings: $warnings"
log_info "Errors: $errors"

# Apply strict mode logic
if [[ "$STRICT_MODE" == "true" && $warnings -gt 0 ]]; then
    log_error "Strict mode: Failing due to $warnings warnings"
    exit_code=1
fi

# Final result
if [[ $exit_code -eq 0 ]]; then
    if [[ $errors -eq 0 ]]; then
        log_success "🎉 ADR-0037 CI/CD VALIDATION: PASSED"
        if [[ $warnings -gt 0 ]]; then
            log_info "Note: $warnings warnings found but not blocking in non-strict mode"
        fi
    else
        log_error "🚨 ADR-0037 CI/CD VALIDATION: FAILED ($errors errors)"
        exit_code=1
    fi
else
    log_error "🚨 ADR-0037 CI/CD VALIDATION: FAILED"
fi

# CI-specific output
if [[ "${CI:-false}" == "true" ]]; then
    # GitHub Actions / GitLab CI friendly output
    echo "::notice title=ADR-0037 Validation::Passed: $passed_checks, Warnings: $warnings, Errors: $errors"
    
    if [[ $exit_code -ne 0 ]]; then
        echo "::error title=ADR-0037 Compliance::Configuration compliance validation failed. Check container environment variables and code patterns."
    fi
fi

log_info "=== ADR-0037 CI/CD Validation Completed: $(date) ==="
echo "Full log available at: $LOG_FILE"

exit $exit_code
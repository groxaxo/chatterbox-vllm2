#!/bin/bash
# Start script for Chatterbox TTS API Server
# This script provides easy configuration presets for different GPU configurations

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed or not in PATH"
    exit 1
fi

# Display usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Start the Chatterbox TTS API Server with different configurations.

OPTIONS:
    --low-vram          Optimize for GPUs with 8GB VRAM (RTX 3060, RTX 2070, etc.)
    --medium-vram       Optimize for GPUs with 12GB VRAM (RTX 3060Ti, RTX 3080, etc.)
    --high-vram         Optimize for GPUs with 24GB+ VRAM (RTX 3090, RTX 4090, etc.)
    --english           Use English-only model (lower VRAM usage)
    --multilingual      Use multilingual model (default)
    --port PORT         Set server port (default: 8000)
    --host HOST         Set server host (default: 0.0.0.0)
    --help              Show this help message

EXAMPLES:
    # Start with default settings (multilingual, low VRAM)
    $0

    # Start with low VRAM optimization (8GB GPU)
    $0 --low-vram

    # Start with English model and medium VRAM
    $0 --english --medium-vram

    # Start on custom port
    $0 --port 8080

GPU REQUIREMENTS:
    Low VRAM (8GB):     RTX 3060, RTX 2070, RTX 2060 Super
    Medium VRAM (12GB): RTX 3060Ti, RTX 3080 (10GB), RTX 2080Ti
    High VRAM (24GB+):  RTX 3090, RTX 4090, A100

EOF
}

# Default settings (optimized for low VRAM)
MODEL="multilingual"
MAX_BATCH_SIZE=1
MAX_MODEL_LEN=800
PORT=8000
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --low-vram)
            MAX_BATCH_SIZE=1
            MAX_MODEL_LEN=800
            print_info "Low VRAM mode: max_batch_size=1, max_model_len=800"
            shift
            ;;
        --medium-vram)
            MAX_BATCH_SIZE=2
            MAX_MODEL_LEN=1000
            print_info "Medium VRAM mode: max_batch_size=2, max_model_len=1000"
            shift
            ;;
        --high-vram)
            MAX_BATCH_SIZE=3
            MAX_MODEL_LEN=1200
            print_info "High VRAM mode: max_batch_size=3, max_model_len=1200"
            shift
            ;;
        --english)
            MODEL="english"
            print_info "Using English-only model"
            shift
            ;;
        --multilingual)
            MODEL="multilingual"
            print_info "Using multilingual model"
            shift
            ;;
        --port)
            PORT="$2"
            print_info "Using port: $PORT"
            shift 2
            ;;
        --host)
            HOST="$2"
            print_info "Using host: $HOST"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Print configuration
echo ""
echo "=========================================="
echo "Chatterbox TTS API Server"
echo "=========================================="
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Max Batch Size:  $MAX_BATCH_SIZE"
echo "  Max Model Len:   $MAX_MODEL_LEN"
echo "  Host:            $HOST"
echo "  Port:            $PORT"
echo "=========================================="
echo ""

# Check if CUDA is available
print_info "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else '')" || {
    print_error "Failed to check GPU. Make sure PyTorch with CUDA support is installed."
    exit 1
}

echo ""
print_info "Starting API server..."
echo ""

# Export environment variables and start server
export CHATTERBOX_MODEL="$MODEL"
export CHATTERBOX_MAX_BATCH_SIZE="$MAX_BATCH_SIZE"
export CHATTERBOX_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export CHATTERBOX_HOST="$HOST"
export CHATTERBOX_PORT="$PORT"

# Start the server
python3 api_server.py

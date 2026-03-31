#!/bin/bash

# Enhanced Dinosaur DNA Reconstruction - WSL Quick Start Script
# This script helps you set up and run the project on WSL

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project settings - USE EXISTING DIRECTORY
PROJECT_NAME="Dinosaur DNA Reconstruction"
PROJECT_ROOT="/mnt/d/MY WORK/Dinosaur DNA Reconstruction"  # Your actual nested folder path
VENV_PATH="${PROJECT_ROOT}/venv"

echo -e "${BLUE}🧬 Enhanced Dinosaur DNA Reconstruction - WSL Setup${NC}"
echo -e "${BLUE}==========================================================${NC}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running on WSL
check_wsl() {
    if ! grep -q microsoft /proc/version; then
        print_error "This script is designed for WSL (Windows Subsystem for Linux)"
        print_error "Please run this from WSL Ubuntu"
        exit 1
    fi
    print_status "Running on WSL"
}

# Check if D drive is accessible
check_d_drive() {
    if [ ! -d "/mnt/d" ]; then
        print_error "D drive not accessible at /mnt/d"
        print_error "Please ensure your D drive is mounted in WSL"
        exit 1
    fi
    print_status "D drive accessible at /mnt/d"
}

# Create project directory structure
create_project_structure() {
    echo -e "${BLUE}Setting up in existing directory...${NC}"
    
    # Use current directory
    cd "$PROJECT_ROOT"
    
    # Create only missing subdirectories
    mkdir -p data models outputs cache logs
    
    print_status "Using existing project structure at $PROJECT_ROOT"
    print_status "Enhanced files are already in place"
}

# Setup Python virtual environment
setup_python_env() {
    echo -e "${BLUE}Setting up Python environment...${NC}"
    
    # Check Python version
    if ! python3.11 --version > /dev/null 2>&1; then
        if ! python3 --version > /dev/null 2>&1; then
            print_error "Python 3 not found. Installing..."
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv
        else
            print_warning "Python 3.11 not found, using $(python3 --version)"
        fi
    else
        print_status "Python 3.11 found"
    fi
    
    # Create virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_PATH"
        print_status "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_status "Python environment ready"
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    source "$VENV_PATH/bin/activate"
    
    # Install PyTorch (CPU version for compatibility)
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install core scientific libraries
    echo "Installing scientific libraries..."
    pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly
    
    # Install bio libraries
    echo "Installing bioinformatics libraries..."
    pip install biopython transformers datasets accelerate
    
    # Install utilities
    echo "Installing utilities..."
    pip install tqdm wandb tensorboard streamlit fastapi uvicorn
    pip install requests aiohttp pyyaml jsonlines
    pip install psutil rich loguru h5py
    
    print_status "Dependencies installed"
}

# Configure the project
configure_project() {
    echo -e "${BLUE}Configuring project...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Check if config file exists
    if [ ! -f "enhanced_config.py" ]; then
        print_error "Configuration file not found!"
        print_error "Enhanced files should be in ${PROJECT_ROOT}/"
        exit 1
    fi
    
    # Remind user to set email
    print_warning "IMPORTANT: Please edit enhanced_config.py to set your email for NCBI API"
    print_warning "Change: 'email': 'your_email@example.com' to your actual email"
    
    read -p "Press Enter after you've set your email in the config file..."
    
    print_status "Project configured"
}

# Test installation
test_installation() {
    echo -e "${BLUE}Testing installation...${NC}"
    
    source "$VENV_PATH/bin/activate"
    cd "$PROJECT_ROOT"
    
    # Test Python imports
    python3 -c "
import torch
import numpy
import pandas
try:
    import Bio
    biopython_status = '✅ BioPython available'
except ImportError:
    biopython_status = '⚠️  BioPython not installed (run: pip install biopython)'

print('✅ All core libraries imported successfully')
print(f'📊 PyTorch version: {torch.__version__}')
print(f'🧮 NumPy version: {numpy.__version__}')
print(f'🐼 Pandas version: {pandas.__version__}')
print(biopython_status)
print(f'💻 Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Show usage instructions
show_usage() {
    echo -e "${BLUE}Usage Instructions:${NC}"
    echo "1. To activate the environment:"
    echo "   cd $PROJECT_ROOT"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. To run the complete pipeline:"
    echo "   python3 main_pipeline.py"
    echo ""
    echo "3. To run specific phases:"
    echo "   python3 main_pipeline.py --phase data    # Data collection only"
    echo "   python3 main_pipeline.py --phase train   # Training only"
    echo "   python3 main_pipeline.py --phase eval    # Evaluation only"
    echo ""
    echo "4. To check system info:"
    echo "   python3 main_pipeline.py --info"
    echo ""
    echo "5. File locations:"
    echo "   📁 Project root: $PROJECT_ROOT"
    echo "   🐍 Python files: $PROJECT_ROOT/ (no src subdirectory)"
    echo "   📊 Data: $PROJECT_ROOT/data/"
    echo "   🤖 Models: $PROJECT_ROOT/models/"
    echo "   📈 Results: $PROJECT_ROOT/outputs/"
    echo "   📝 Logs: $PROJECT_ROOT/logs/"
}

# Main execution
main() {
    case "${1:-}" in
        "setup"|"")
            check_wsl
            check_d_drive
            create_project_structure
            setup_python_env
            install_dependencies
            configure_project
            if test_installation; then
                echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
                show_usage
            else
                print_error "Setup failed during testing phase"
                exit 1
            fi
            ;;
        "test")
            source "$VENV_PATH/bin/activate"
            test_installation
            ;;
        "run")
            source "$VENV_PATH/bin/activate"
            cd "$PROJECT_ROOT/src"
            python3 main_pipeline.py "${@:2}"
            ;;
        "clean")
            print_warning "This will delete the entire project directory"
            read -p "Are you sure? (type 'yes' to confirm): " confirm
            if [ "$confirm" = "yes" ]; then
                rm -rf "$PROJECT_ROOT"
                print_status "Project directory cleaned"
            else
                echo "Cancelled"
            fi
            ;;
        "help"|"-h"|"--help")
            echo "Enhanced Dinosaur DNA Reconstruction - WSL Setup Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  setup     Set up the complete project (default)"
            echo "  test      Test the installation"
            echo "  run       Run the main pipeline"
            echo "  clean     Remove the project directory"
            echo "  help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 setup                    # Initial setup"
            echo "  $0 run                      # Run complete pipeline"
            echo "  $0 run --phase data         # Run only data collection"
            echo "  $0 test                     # Test installation"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
#!/usr/bin/env python3
"""
Enhanced Dinosaur DNA Reconstruction - Main Pipeline
Run this script to execute the complete pipeline from data collection to evaluation.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
import torch

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_config import config, validate_config
from enhanced_data_collection import main as collect_data
from enhanced_training import main as train_model
from enhanced_evaluation import main as evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(module)s | %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_DIR / 'main_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} - Device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        logger.info(f"✓ Transformers available")
    except ImportError:
        logger.warning("⚠ Transformers not available - using basic implementation")
    
    try:
        import Bio
        logger.info(f"✓ BioPython available")
    except ImportError:
        missing_deps.append("biopython")
    
    try:
        import numpy
        import pandas  
        import tqdm
        logger.info(f"✓ Scientific libraries available")
    except ImportError:
        missing_deps.append("scientific libraries (numpy, pandas, tqdm)")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install with: pip install -r enhanced_requirements.txt")
        return False
    
    logger.info("✓ All dependencies satisfied")
    return True

def setup_environment():
    """Setup environment and validate configuration."""
    logger.info("Setting up environment...")
    
    # Validate configuration
    try:
        validate_config()
        logger.info("✓ Configuration validated")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    
    # Check NCBI email configuration
    if config.NCBI_CONFIG['email'] == 'your_email@example.com':
        logger.error("❌ Please set your email in enhanced_config.py for NCBI API access")
        logger.error("Edit NCBI_CONFIG['email'] = 'your_actual_email@domain.com'")
        return False
    
    logger.info(f"✓ NCBI email configured: {config.NCBI_CONFIG['email']}")
    
    # Check disk space
    base_dir = config.BASE_DIR
    if not base_dir.exists():
        try:
            base_dir.mkdir(parents=True)
            logger.info(f"✓ Created project directory: {base_dir}")
        except Exception as e:
            logger.error(f"Failed to create project directory: {e}")
            return False
    
    # Check available space (estimate 10GB needed)
    try:
        import shutil
        free_space = shutil.disk_usage(base_dir).free / (1024**3)  # GB
        logger.info(f"✓ Available disk space: {free_space:.1f} GB")
        
        if free_space < 10:
            logger.warning(f"⚠ Low disk space. Recommend at least 10GB free")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    return True

def run_data_collection():
    """Run data collection pipeline."""
    logger.info("=" * 50)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("=" * 50)
    
    # Check if data already exists
    data_file = config.DATA_DIR / "multi_species_examples.json.gz"
    if data_file.exists():
        logger.info(f"✓ Data already exists: {data_file}")
        
        response = input("Data already exists. Redownload? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping data collection")
            return True
    
    try:
        import asyncio
        asyncio.run(collect_data())
        logger.info("✓ Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def run_training():
    """Run training pipeline."""
    logger.info("=" * 50)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 50)
    
    # Check if data exists
    data_file = config.DATA_DIR / "multi_species_examples.json.gz"
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        logger.error("Please run data collection first")
        return False
    
    # Check if model already exists
    model_file = config.MODEL_DIR / "checkpoints" / "best_model.pt"
    if model_file.exists():
        logger.info(f"✓ Model already exists: {model_file}")
        
        response = input("Model already exists. Retrain? (y/n): ").lower().strip()
        if response != 'y':
            logger.info("Skipping training")
            return True
    
    try:
        train_model()
        logger.info("✓ Training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_evaluation():
    """Run evaluation pipeline."""
    logger.info("=" * 50)
    logger.info("PHASE 3: MODEL EVALUATION")
    logger.info("=" * 50)
    
    # Check if model exists
    model_file = config.MODEL_DIR / "checkpoints" / "best_model.pt"
    if not model_file.exists():
        logger.error(f"Trained model not found: {model_file}")
        logger.error("Please run training first")
        return False
    
    try:
        evaluate_model()
        logger.info("✓ Evaluation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False

def print_system_info():
    """Print system information for debugging."""
    logger.info("=" * 50)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 50)
    
    import platform
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Architecture: {platform.machine()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.info("Memory info not available (install psutil)")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.info("GPU: Not available (using CPU)")
    
    logger.info(f"Project Directory: {config.BASE_DIR}")

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Enhanced Dinosaur DNA Reconstruction Pipeline")
    parser.add_argument('--phase', choices=['all', 'data', 'train', 'eval'], default='all',
                       help='Which phase to run (default: all)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and environment checks')
    parser.add_argument('--info', action='store_true',
                       help='Show system information and exit')
    
    args = parser.parse_args()
    
    # Print system info if requested
    if args.info:
        print_system_info()
        return
    
    logger.info("🧬 Enhanced Dinosaur DNA Reconstruction Pipeline")
    logger.info("=" * 50)
    
    # Check dependencies and environment
    if not args.skip_checks:
        if not check_dependencies():
            logger.error("Dependency check failed")
            sys.exit(1)
        
        if not setup_environment():
            logger.error("Environment setup failed")
            sys.exit(1)
    
    print_system_info()
    
    # Run requested phases
    phases = {
        'data': run_data_collection,
        'train': run_training, 
        'eval': run_evaluation
    }
    
    if args.phase == 'all':
        # Run all phases in sequence
        for phase_name, phase_func in phases.items():
            logger.info(f"\n🚀 Starting {phase_name.upper()} phase...")
            success = phase_func()
            if not success:
                logger.error(f"❌ {phase_name.upper()} phase failed")
                sys.exit(1)
            logger.info(f"✅ {phase_name.upper()} phase completed")
    else:
        # Run specific phase
        if args.phase in phases:
            logger.info(f"\n🚀 Starting {args.phase.upper()} phase...")
            success = phases[args.phase]()
            if not success:
                logger.error(f"❌ {args.phase.upper()} phase failed")
                sys.exit(1)
            logger.info(f"✅ {args.phase.upper()} phase completed")
        else:
            logger.error(f"Unknown phase: {args.phase}")
            sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info(f"📁 Results available in: {config.OUTPUT_DIR}")
    logger.info(f"📊 Models saved in: {config.MODEL_DIR}")
    logger.info(f"📝 Logs available in: {config.LOG_DIR}")
    
    # Final summary
    logger.info("\n📋 SUMMARY:")
    logger.info(f"📁 Project Location: {config.BASE_DIR}")
    logger.info(f"✅ Device Used: {config.DEVICE}")  
    logger.info(f"✅ Target Species: {len(config.NCBI_CONFIG['target_species'])}")
    logger.info("\n🔬 Next Steps:")
    logger.info("1. Review evaluation results in outputs directory")
    logger.info("2. Analyze model performance metrics")
    logger.info("3. Use trained model for DNA reconstruction tasks")
    logger.info(f"4. Compare with your original system results")

if __name__ == "__main__":
    main()

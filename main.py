#!/usr/bin/env python3
"""
Main execution script for Dinosaur DNA Reconstruction System
Coordinates data collection, training, evaluation, and inference
"""

import argparse
import logging
import sys
import os
import json
import torch
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import PROJECT_CONFIG, GENOMIC_FEATURES
from data_collection import main as collect_data
from training import DinosaurDNATrainer
from inference import DinosaurDNAReconstructor
from evaluation import run_comprehensive_evaluation
from evolutionary_constraints import test_mutation_models
from web_interface import run_streamlit, run_fastapi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dinosaur_dna_reconstruction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        PROJECT_CONFIG["paths"]["data_dir"],
        PROJECT_CONFIG["paths"]["raw_data"], 
        PROJECT_CONFIG["paths"]["processed_data"],
        PROJECT_CONFIG["paths"]["models"],
        PROJECT_CONFIG["paths"]["outputs"],
        PROJECT_CONFIG["paths"]["logs"],
        PROJECT_CONFIG["paths"]["figures"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'biopython', 'matplotlib', 
        'seaborn', 'plotly', 'scikit-learn', 'transformers',
        'streamlit', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed")
    return True

def collect_and_process_data(args):
    """Data collection and preprocessing pipeline"""
    logger.info("🧬 Starting data collection and preprocessing...")
    
    try:
        collect_data()
        logger.info("✅ Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return False

def train_model(args):
    """Model training pipeline"""
    logger.info("🏋️ Starting model training...")
    
    try:
        trainer = DinosaurDNATrainer(PROJECT_CONFIG)
        
        data_path = os.path.join(PROJECT_CONFIG["paths"]["processed_data"], "training_data.json")
        if not os.path.exists(data_path):
            logger.error(f"Training data not found at {data_path}")
            logger.error("Please run data collection first: python main.py collect")
            return False
        
        best_model_path = trainer.train(
            train_data_path=data_path,
            model_type=args.model_type,
            resume_from=args.resume_from
        )
        
        logger.info(f"✅ Training completed. Best model saved to: {best_model_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def evaluate_model(args):
    """Model evaluation pipeline"""
    logger.info("📊 Starting model evaluation...")
    
    try:
        model_path = args.model_path or os.path.join(PROJECT_CONFIG["paths"]["models"], "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train a model first: python main.py train")
            return False
        
        test_data_path = os.path.join(PROJECT_CONFIG["paths"]["processed_data"], "training_data.json")
        if not os.path.exists(test_data_path):
            logger.error(f"Test data not found at {test_data_path}")
            logger.error("Please run data collection first: python main.py collect")
            return False
        output_dir = args.output_dir or os.path.join(PROJECT_CONFIG["paths"]["outputs"], "evaluation")
        results = run_comprehensive_evaluation(model_path, test_data_path, output_dir)
        
        print("\n" + "="*50)
        print("📊 EVALUATION SUMMARY")
        print("="*50)
        
        if 'reconstruction_accuracy' in results:
            acc = results['reconstruction_accuracy']
            print(f"🎯 Sequence Identity: {acc.get('mean_sequence_identity', 0):.4f}")
            print(f"🔧 Position Accuracy: {acc.get('mean_position_accuracy', 0):.4f}")
            print(f"💥 Damaged Position Recovery: {acc.get('mean_damaged_position_accuracy', 0):.4f}")
        
        if 'benchmark_comparison' in results:
            print(f"\n📈 BENCHMARK COMPARISON:")
            for method, metrics in results['benchmark_comparison'].items():
                identity = metrics.get('mean_sequence_identity', 0)
                print(f"  {method}: {identity:.4f}")
        
        print(f"\n📁 Detailed results saved to: {output_dir}")
        
        logger.info("✅ Evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return False

def reconstruct_sequence(args):
    """Single sequence reconstruction"""
    logger.info("🧬 Starting sequence reconstruction...")
    
    try:
        model_path = args.model_path or os.path.join(PROJECT_CONFIG["paths"]["models"], "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train a model first: python main.py train")
            return False
        reconstructor = DinosaurDNAReconstructor(model_path)
        
        if args.sequence:
            input_sequence = args.sequence
        elif args.file:
            with open(args.file, 'r') as f:
                input_sequence = f.read().strip()
        else:
            logger.error("Please provide either --sequence or --file argument")
            return False
        
        logger.info(f"Reconstructing sequence of length {len(input_sequence)}")
        result = reconstructor.reconstruct_sequence(
            damaged_sequence=input_sequence,
            target_species=args.target_species,
            reference_species=args.reference_species,
            confidence_threshold=args.confidence_threshold,
            num_samples=args.num_samples
        )
        
        print("\n" + "="*50)
        print("🧬 RECONSTRUCTION RESULTS")
        print("="*50)
        print(f"Target Species: {args.target_species}")
        print(f"Reference Species: {args.reference_species}")
        print(f"Input Length: {len(input_sequence)}")
        print(f"Output Length: {len(result.get('reconstructed_sequence', ''))}")
        
        print(f"\n📝 Original (first 100 bp):")
        print(input_sequence[:100] + ("..." if len(input_sequence) > 100 else ""))
        
        print(f"\n🔧 Reconstructed (first 100 bp):")
        reconstructed = result.get('reconstructed_sequence', '')
        print(reconstructed[:100] + ("..." if len(reconstructed) > 100 else ""))
        
        if 'quality_metrics' in result:
            metrics = result['quality_metrics']
            print(f"\n📊 Quality Metrics:")
            print(f"  Average Confidence: {metrics.get('avg_confidence', 0):.3f}")
            print(f"  Reconstruction Rate: {metrics.get('reconstruction_rate', 0):.1f}%")
            print(f"  Quality Score: {metrics.get('quality_score', 0):.3f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
        
        logger.info("✅ Reconstruction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reconstruction failed: {e}")
        return False

def launch_web_interface(args):
    """Launch web interface"""
    logger.info("🌐 Launching web interface...")
    
    try:
        if args.interface_type == 'streamlit':
            logger.info("Starting Streamlit interface...")
            run_streamlit()
        elif args.interface_type == 'fastapi':
            logger.info("Starting FastAPI interface...")
            run_fastapi(host=args.host, port=args.port)
        else:
            logger.error(f"Unknown interface type: {args.interface_type}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Web interface failed: {e}")
        return False

def test_system(args):
    """Run system tests"""
    logger.info("🧪 Running system tests...")
    
    try:
        logger.info("Testing evolutionary constraints...")
        test_result = test_mutation_models()
        if test_result:
            logger.info("✅ Evolutionary constraints test passed")
        else:
            logger.error("❌ Evolutionary constraints test failed")
            return False
        
        if torch.cuda.is_available():
            logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("💻 Using CPU (GPU not available)")
        
        for path_name, path_value in PROJECT_CONFIG["paths"].items():
            if os.path.exists(path_value):
                logger.info(f"✅ {path_name}: {path_value}")
            else:
                logger.info(f"➖ {path_name}: {path_value} (will be created)")
        
        logger.info("✅ System tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System tests failed: {e}")
        return False

def print_project_info():
    """Print project information"""
    print("\n" + "="*60)
    print("🦕 DINOSAUR DNA RECONSTRUCTION SYSTEM")
    print("="*60)
    print(f"Version: {PROJECT_CONFIG['version']}")
    print(f"Description: {PROJECT_CONFIG['description']}")
    print("\nFeatures:")
    print("  🧬 Advanced DNA sequence reconstruction")
    print("  🔬 Evolutionary constraints modeling")
    print("  🌳 Phylogenetic context integration")
    print("  🎯 Ancient DNA damage patterns")
    print("  🤖 Hybrid Transformer + Markov Chain models")
    print("  📊 Comprehensive evaluation metrics")
    print("  🌐 Web interface (Streamlit + FastAPI)")
    print("  💾 Multiple export formats")
    
    print(f"\nTarget Species: {PROJECT_CONFIG['data_sources']['archosaur_genomes']['key_bird_species']}")
    print(f"Reference Species: {PROJECT_CONFIG['data_sources']['archosaur_genomes']['crocodilian']}")
    print("="*60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🦕 Dinosaur DNA Reconstruction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup                    # Setup directories and check dependencies
  python main.py collect                  # Collect and process data
  python main.py train                    # Train the model
  python main.py train --model-type hybrid --resume-from models/checkpoint.pth
  python main.py evaluate                 # Evaluate the model
  python main.py reconstruct --sequence "ATGCATGCN" --target-species "Theropod_ancestor"
  python main.py reconstruct --file input.fasta --output results.json
  python main.py web --interface streamlit
  python main.py web --interface fastapi --port 8000
  python main.py test                     # Run system tests
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    setup_parser = subparsers.add_parser('setup', help='Setup directories and check dependencies')
    
    collect_parser = subparsers.add_parser('collect', help='Collect and process genomic data')
    
    train_parser = subparsers.add_parser('train', help='Train the DNA reconstruction model')
    train_parser.add_argument('--model-type', choices=['transformer', 'markov', 'hybrid'], 
                             default='hybrid', help='Type of model to train')
    train_parser.add_argument('--resume-from', type=str, help='Path to checkpoint to resume from')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the trained model')
    eval_parser.add_argument('--model-path', type=str, help='Path to trained model')
    eval_parser.add_argument('--output-dir', type=str, help='Output directory for evaluation results')
    
    recon_parser = subparsers.add_parser('reconstruct', help='Reconstruct a DNA sequence')
    recon_parser.add_argument('--sequence', type=str, help='DNA sequence to reconstruct')
    recon_parser.add_argument('--file', type=str, help='File containing DNA sequence')
    recon_parser.add_argument('--target-species', default='Theropod_ancestor', 
                             help='Target species for reconstruction')
    recon_parser.add_argument('--reference-species', default='Gallus gallus',
                             help='Reference species for phylogenetic context')
    recon_parser.add_argument('--confidence-threshold', type=float, default=0.7,
                             help='Confidence threshold for accepting predictions')
    recon_parser.add_argument('--num-samples', type=int, default=10,
                             help='Number of Monte Carlo samples')
    recon_parser.add_argument('--output', type=str, help='Output file for results')
    recon_parser.add_argument('--model-path', type=str, help='Path to trained model')
    
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--interface', dest='interface_type', 
                           choices=['streamlit', 'fastapi'], default='streamlit',
                           help='Type of web interface to launch')
    web_parser.add_argument('--host', default='localhost', help='Host to bind to (FastAPI only)')
    web_parser.add_argument('--port', type=int, default=8000, help='Port to bind to (FastAPI only)')
    
    test_parser = subparsers.add_parser('test', help='Run system tests')
    
    info_parser = subparsers.add_parser('info', help='Show project information')
    
    args = parser.parse_args()
    
    if args.command is None:
        print_project_info()
        parser.print_help()
        return
    
    if args.command != 'setup':
        if not check_dependencies():
            sys.exit(1)
    
    if args.command in ['setup', 'collect', 'train', 'evaluate', 'reconstruct']:
        setup_directories()
    
    success = True
    
    if args.command == 'setup':
        logger.info("🔧 Setting up Dinosaur DNA Reconstruction System...")
        success = check_dependencies()
        if success:
            setup_directories()
            logger.info("✅ Setup completed successfully")
    
    elif args.command == 'collect':
        success = collect_and_process_data(args)
    
    elif args.command == 'train':
        success = train_model(args)
    
    elif args.command == 'evaluate':
        success = evaluate_model(args)
    
    elif args.command == 'reconstruct':
        success = reconstruct_sequence(args)
    
    elif args.command == 'web':
        launch_web_interface(args)  
    
    elif args.command == 'test':
        success = test_system(args)
    
    elif args.command == 'info':
        print_project_info()
    
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False
    
    if success:
        logger.info("🎉 Command completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Command failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

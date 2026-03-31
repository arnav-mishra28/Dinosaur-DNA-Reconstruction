"""
Web Interface for Dinosaur DNA Reconstruction
Provides both Streamlit and FastAPI interfaces for easy interaction with the models
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import io
import base64
from typing import Dict, List, Optional, Tuple
import logging
import json
import os
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Local imports
from config import PROJECT_CONFIG, GENOMIC_FEATURES
from models import create_model
from inference import DinosaurDNAReconstructor
from evolutionary_constraints import (
    create_mutation_context, 
    AdvancedMutationModel,
    TransitionMatrixGenerator
)
from data_collection import SequencePreprocessor, PhylogeneticDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Models
class SequenceInput(BaseModel):
    sequence: str
    target_species: str = "Theropod_ancestor"
    reference_species: str = "Gallus gallus"
    confidence_threshold: float = 0.7
    num_samples: int = 10

class ReconstructionResult(BaseModel):
    original_sequence: str
    reconstructed_sequence: str
    confidence_scores: List[float]
    mutation_probabilities: Dict
    phylogenetic_context: Dict
    quality_metrics: Dict

# Initialize FastAPI
app = FastAPI(
    title="Dinosaur DNA Reconstruction API",
    description="API for reconstructing ancient dinosaur DNA using machine learning",
    version="1.0.0"
)

# Global variables for model
MODEL = None
RECONSTRUCTOR = None

def load_model_cache():
    """Load model into memory for faster inference"""
    global MODEL, RECONSTRUCTOR
    try:
        # Check if model exists
        model_path = os.path.join(PROJECT_CONFIG["paths"]["models"], "best_model.pth")
        if os.path.exists(model_path):
            RECONSTRUCTOR = DinosaurDNAReconstructor(model_path)
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Please train the model first.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

def create_sequence_visualization(original_seq: str, reconstructed_seq: str, 
                                confidence_scores: np.ndarray) -> go.Figure:
    """Create interactive visualization of sequence reconstruction"""
    
    seq_len = min(len(original_seq), len(reconstructed_seq), len(confidence_scores))
    positions = list(range(seq_len))
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'Original vs Reconstructed Sequence',
            'Confidence Scores',
            'Base Composition'
        ],
        vertical_spacing=0.1
    )
    
    # Sequence comparison
    fig.add_trace(
        go.Scatter(
            x=positions[:100],  # Show first 100 bases
            y=[1] * min(100, seq_len),
            mode='text',
            text=list(original_seq[:100]),
            textposition='middle center',
            name='Original',
            textfont=dict(size=10, family="monospace")
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=positions[:100],
            y=[0] * min(100, seq_len),
            mode='text',
            text=list(reconstructed_seq[:100]),
            textposition='middle center',
            name='Reconstructed',
            textfont=dict(size=10, family="monospace")
        ),
        row=1, col=1
    )
    
    # Confidence scores
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=confidence_scores,
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Base composition
    bases = ['A', 'T', 'G', 'C']
    original_counts = [original_seq.count(base) for base in bases]
    reconstructed_counts = [reconstructed_seq.count(base) for base in bases]
    
    fig.add_trace(
        go.Bar(
            x=bases,
            y=original_counts,
            name='Original Composition',
            marker_color='lightblue'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=bases,
            y=reconstructed_counts,
            name='Reconstructed Composition',
            marker_color='darkblue'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Dinosaur DNA Reconstruction Analysis"
    )
    
    return fig

def create_mutation_heatmap(mutation_probs: Dict) -> go.Figure:
    """Create heatmap of mutation probabilities"""
    bases = ['A', 'T', 'G', 'C']
    
    # Convert mutation probabilities to matrix format
    if isinstance(mutation_probs, dict):
        matrix = np.random.rand(4, 4) * 0.1  # Placeholder
        for i, from_base in enumerate(bases):
            for j, to_base in enumerate(bases):
                key = f"{from_base}_to_{to_base}"
                if key in mutation_probs:
                    matrix[i, j] = mutation_probs[key]
    else:
        matrix = np.array(mutation_probs) if hasattr(mutation_probs, 'shape') else np.random.rand(4, 4) * 0.1
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=bases,
        y=bases,
        colorscale='Viridis',
        colorbar=dict(title="Mutation Probability"),
        text=matrix,
        texttemplate="%{text:.3f}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Mutation Probability Matrix",
        xaxis_title="To Base",
        yaxis_title="From Base",
        width=500,
        height=400
    )
    
    return fig

def create_phylogenetic_tree() -> go.Figure:
    """Create interactive phylogenetic tree visualization"""
    # Simplified tree structure for visualization
    species_data = {
        'species': ['Archosaur Ancestor', 'Crocodilian', 'Theropod', 'Modern Bird', 'Reconstructed Dinosaur'],
        'divergence_time': [240, 240, 150, 65, 150],
        'x': [0, 1, 1, 2, 1.5],
        'y': [2, 1, 3, 4, 3.5]
    }
    
    fig = go.Figure()
    
    # Add tree branches
    branches = [
        ([0, 1], [2, 1]),  # Archosaur to Crocodilian
        ([0, 1], [2, 3]),  # Archosaur to Theropod
        ([1, 2], [3, 4]),  # Theropod to Modern Bird
        ([1, 1.5], [3, 3.5])  # Theropod to Reconstructed
    ]
    
    for branch in branches:
        fig.add_trace(go.Scatter(
            x=branch[0], y=branch[1],
            mode='lines',
            line=dict(color='brown', width=3),
            showlegend=False
        ))
    
    # Add species nodes
    fig.add_trace(go.Scatter(
        x=species_data['x'],
        y=species_data['y'],
        mode='markers+text',
        marker=dict(size=20, color='green'),
        text=species_data['species'],
        textposition='top center',
        name='Species'
    ))
    
    fig.update_layout(
        title="Phylogenetic Relationships",
        xaxis_title="Evolutionary Distance",
        yaxis_title="Lineage",
        showlegend=False,
        width=600,
        height=400
    )
    
    return fig

# Streamlit Interface
def streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🦕 Dinosaur DNA Reconstruction",
        page_icon="🦕",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🦕 Dinosaur DNA Reconstruction System")
    st.markdown("### Reconstruct ancient dinosaur DNA using machine learning and evolutionary constraints")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Load model
    if st.sidebar.button("Load Model"):
        load_model_cache()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload DNA Sequence File",
        type=['txt', 'fasta', 'fas', 'fa'],
        help="Upload a DNA sequence file for reconstruction"
    )
    
    # Parameters
    st.sidebar.subheader("Reconstruction Parameters")
    target_species = st.sidebar.selectbox(
        "Target Species",
        ["Theropod_ancestor", "Tyrannosaurus_rex", "Triceratops", "Velociraptor"],
        index=0
    )
    
    reference_species = st.sidebar.selectbox(
        "Reference Species",
        ["Gallus gallus", "Struthio camelus", "Alligator mississippiensis"],
        index=0
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    num_samples = st.sidebar.slider(
        "Number of Monte Carlo Samples",
        min_value=1,
        max_value=50,
        value=10
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input DNA Sequence")
        
        if uploaded_file is not None:
            # Process uploaded file
            if uploaded_file.name.endswith(('.fasta', '.fas', '.fa')):
                sequences = SeqIO.parse(uploaded_file, "fasta")
                sequence_list = list(sequences)
                if sequence_list:
                    input_sequence = str(sequence_list[0].seq)
                    st.success(f"Loaded sequence: {len(input_sequence)} bases")
                else:
                    st.error("No sequences found in file")
                    input_sequence = ""
            else:
                input_sequence = uploaded_file.read().decode("utf-8").strip()
        else:
            input_sequence = st.text_area(
                "Enter DNA sequence (or upload file):",
                height=150,
                placeholder="ATGCGATCGTAGC...",
                help="Enter a DNA sequence using A, T, G, C, N characters"
            )
        
        # Validate sequence
        if input_sequence:
            cleaned_seq = ''.join(c for c in input_sequence.upper() if c in 'ATGCN')
            if len(cleaned_seq) != len(input_sequence.replace('\n', '').replace(' ', '')):
                st.warning("⚠️ Sequence contains invalid characters. Only A, T, G, C, N are allowed.")
            
            st.info(f"📊 Sequence length: {len(cleaned_seq)} bases")
            
            # Show sequence composition
            if cleaned_seq:
                composition = {
                    'A': cleaned_seq.count('A'),
                    'T': cleaned_seq.count('T'), 
                    'G': cleaned_seq.count('G'),
                    'C': cleaned_seq.count('C'),
                    'N': cleaned_seq.count('N')
                }
                
                st.write("Base composition:")
                col_a, col_t, col_g, col_c, col_n = st.columns(5)
                col_a.metric("A", composition['A'])
                col_t.metric("T", composition['T'])
                col_g.metric("G", composition['G'])
                col_c.metric("C", composition['C'])
                col_n.metric("N (Unknown)", composition['N'])
    
    with col2:
        st.subheader("🧬 Model Information")
        if RECONSTRUCTOR:
            st.success("✅ Model Status: Loaded")
            st.info("📈 Model Type: Hybrid Transformer + Markov Chain")
            st.info("🔬 Training Species: Birds + Crocodilians")
        else:
            st.error("❌ Model Status: Not Loaded")
        
        st.subheader("📚 Features")
        st.markdown("""
        - **Evolutionary Constraints**: Transition/transversion ratios
        - **Phylogenetic Context**: Species relationships
        - **Ancient DNA Damage**: C→T deamination patterns
        - **Uncertainty Quantification**: Confidence scores
        - **Monte Carlo Sampling**: Multiple reconstructions
        """)
    
    # Reconstruction
    if st.button("🔬 Reconstruct DNA Sequence", type="primary"):
        if not input_sequence or not cleaned_seq:
            st.error("Please enter a DNA sequence first!")
        elif not RECONSTRUCTOR:
            st.error("Please load the model first!")
        else:
            with st.spinner("🧬 Reconstructing ancient DNA sequence..."):
                try:
                    # Perform reconstruction
                    result = RECONSTRUCTOR.reconstruct_sequence(
                        damaged_sequence=cleaned_seq,
                        target_species=target_species,
                        reference_species=reference_species,
                        confidence_threshold=confidence_threshold,
                        num_samples=num_samples
                    )
                    
                    # Display results
                    st.success("✅ Reconstruction completed!")
                    
                    # Results tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results", "📊 Visualization", "🧮 Statistics", "💾 Export"])
                    
                    with tab1:
                        st.subheader("Reconstructed Sequence")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area(
                                "Original (Damaged):",
                                value=input_sequence[:500] + ("..." if len(input_sequence) > 500 else ""),
                                height=100,
                                disabled=True
                            )
                        
                        with col2:
                            reconstructed = result.get('reconstructed_sequence', 'N/A')
                            st.text_area(
                                "Reconstructed:",
                                value=reconstructed[:500] + ("..." if len(reconstructed) > 500 else ""),
                                height=100,
                                disabled=True
                            )
                        
                        # Quality metrics
                        st.subheader("Quality Metrics")
                        if 'quality_metrics' in result:
                            metrics = result['quality_metrics']
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.3f}")
                            col2.metric("Reconstruction %", f"{metrics.get('reconstruction_rate', 0):.1f}%")
                            col3.metric("Sequence Identity", f"{metrics.get('identity', 0):.1f}%")
                            col4.metric("Quality Score", f"{metrics.get('quality_score', 0):.3f}")
                    
                    with tab2:
                        st.subheader("Sequence Analysis Visualization")
                        
                        # Create visualization
                        confidence_scores = result.get('confidence_scores', [0.5] * len(cleaned_seq))
                        fig = create_sequence_visualization(
                            cleaned_seq, 
                            result.get('reconstructed_sequence', cleaned_seq),
                            np.array(confidence_scores)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mutation probabilities heatmap
                        if 'mutation_probabilities' in result:
                            st.subheader("Mutation Probability Matrix")
                            mut_fig = create_mutation_heatmap(result['mutation_probabilities'])
                            st.plotly_chart(mut_fig)
                        
                        # Phylogenetic tree
                        st.subheader("Phylogenetic Context")
                        phylo_fig = create_phylogenetic_tree()
                        st.plotly_chart(phylo_fig)
                    
                    with tab3:
                        st.subheader("Statistical Analysis")
                        
                        # Detailed statistics
                        if 'confidence_scores' in result:
                            conf_scores = np.array(result['confidence_scores'])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Confidence Score Distribution**")
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.hist(conf_scores, bins=20, alpha=0.7, color='blue')
                                ax.set_xlabel('Confidence Score')
                                ax.set_ylabel('Frequency')
                                ax.set_title('Distribution of Confidence Scores')
                                st.pyplot(fig)
                            
                            with col2:
                                st.write("**Summary Statistics**")
                                stats_df = pd.DataFrame({
                                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                    'Value': [
                                        f"{conf_scores.mean():.4f}",
                                        f"{np.median(conf_scores):.4f}",
                                        f"{conf_scores.std():.4f}",
                                        f"{conf_scores.min():.4f}",
                                        f"{conf_scores.max():.4f}"
                                    ]
                                })
                                st.dataframe(stats_df, use_container_width=True)
                    
                    with tab4:
                        st.subheader("Export Results")
                        
                        # Export options
                        export_format = st.selectbox(
                            "Choose export format:",
                            ["FASTA", "JSON", "CSV", "TXT"]
                        )
                        
                        if export_format == "FASTA":
                            fasta_content = f">Reconstructed_{target_species}_{datetime.now().strftime('%Y%m%d_%H%M%S')}\n"
                            fasta_content += result.get('reconstructed_sequence', '')
                            st.download_button(
                                "📁 Download FASTA",
                                data=fasta_content,
                                file_name=f"reconstructed_{target_species}.fasta",
                                mime="text/plain"
                            )
                        
                        elif export_format == "JSON":
                            json_data = json.dumps(result, indent=2)
                            st.download_button(
                                "📁 Download JSON",
                                data=json_data,
                                file_name=f"reconstruction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        elif export_format == "CSV":
                            # Create CSV with position-wise data
                            if 'confidence_scores' in result:
                                csv_data = pd.DataFrame({
                                    'Position': range(len(cleaned_seq)),
                                    'Original': list(cleaned_seq),
                                    'Reconstructed': list(result.get('reconstructed_sequence', cleaned_seq)),
                                    'Confidence': result['confidence_scores']
                                })
                                csv_string = csv_data.to_csv(index=False)
                                st.download_button(
                                    "📁 Download CSV",
                                    data=csv_string,
                                    file_name=f"reconstruction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        # Show preview
                        st.subheader("Preview")
                        if export_format == "FASTA":
                            st.code(fasta_content[:200] + "..." if len(fasta_content) > 200 else fasta_content)
                        elif export_format == "JSON":
                            st.json(result)
                
                except Exception as e:
                    st.error(f"❌ Error during reconstruction: {e}")
                    logger.error(f"Reconstruction error: {e}")

# FastAPI Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    html_content = """
    <html>
        <head>
            <title>Dinosaur DNA Reconstruction API</title>
        </head>
        <body>
            <h1>🦕 Dinosaur DNA Reconstruction API</h1>
            <p>This API provides endpoints for reconstructing ancient dinosaur DNA sequences.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>POST /reconstruct</code> - Reconstruct a DNA sequence</li>
                <li><code>GET /docs</code> - API documentation</li>
                <li><code>GET /health</code> - Health check</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": RECONSTRUCTOR is not None}

@app.post("/reconstruct", response_model=ReconstructionResult)
async def reconstruct_sequence(input_data: SequenceInput):
    """Reconstruct a DNA sequence"""
    if not RECONSTRUCTOR:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = RECONSTRUCTOR.reconstruct_sequence(
            damaged_sequence=input_data.sequence,
            target_species=input_data.target_species,
            reference_species=input_data.reference_species,
            confidence_threshold=input_data.confidence_threshold,
            num_samples=input_data.num_samples
        )
        
        return ReconstructionResult(
            original_sequence=input_data.sequence,
            reconstructed_sequence=result.get('reconstructed_sequence', ''),
            confidence_scores=result.get('confidence_scores', []),
            mutation_probabilities=result.get('mutation_probabilities', {}),
            phylogenetic_context=result.get('phylogenetic_context', {}),
            quality_metrics=result.get('quality_metrics', {})
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_sequence_file(file: UploadFile = File(...)):
    """Upload and process a sequence file"""
    if not file.filename.endswith(('.txt', '.fasta', '.fas', '.fa')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    content = await file.read()
    sequence_text = content.decode('utf-8')
    
    # Parse FASTA if needed
    if file.filename.endswith(('.fasta', '.fas', '.fa')):
        sequences = []
        for line in sequence_text.split('\n'):
            if not line.startswith('>'):
                sequences.append(line.strip())
        sequence_text = ''.join(sequences)
    
    return {"sequence": sequence_text, "length": len(sequence_text)}

def run_streamlit():
    """Run Streamlit app"""
    streamlit_app()

def run_fastapi(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server"""
    global RECONSTRUCTOR
    
    # Load model on startup
    model_path = os.path.join(PROJECT_CONFIG["paths"]["models"], "best_model.pth")
    if os.path.exists(model_path):
        try:
            RECONSTRUCTOR = DinosaurDNAReconstructor(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        run_fastapi()
    else:
        # Run Streamlit by default
        run_streamlit()

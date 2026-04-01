"""
Visual DNA Reconstruction Simulation
Real-time visualization of DNA sequence reconstruction, mutations, and evolution
Compatible with VS Code and existing model architecture
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
import torch
import time
from typing import Dict, List, Tuple, Optional
import colorsys
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Enhanced configuration import
try:
    from enhanced_config import config
    from enhanced_models import create_model, EnhancedDinosaurDNAModel
except ImportError:
    print("Warning: Enhanced modules not found. Using basic configuration.")
    config = None

class DNAColorScheme:
    """Color schemes for DNA visualization."""
    
    # Standard DNA base colors
    BASE_COLORS = {
        'A': '#FF6B6B',  # Red - Adenine
        'T': '#4ECDC4',  # Teal - Thymine  
        'G': '#45B7D1',  # Blue - Guanine
        'C': '#FFA726',  # Orange - Cytosine
        'N': '#9E9E9E',  # Gray - Unknown
        '-': '#FFFFFF',  # White - Gap
    }
    
    # Mutation type colors
    MUTATION_COLORS = {
        'transition': '#E74C3C',     # Red
        'transversion': '#3498DB',   # Blue
        'insertion': '#27AE60',      # Green
        'deletion': '#F39C12',       # Orange
        'silent': '#95A5A6',         # Gray
    }
    
    # Quality/confidence colors (gradient)
    CONFIDENCE_COLORMAP = 'viridis'
    
    # Species colors for phylogenetic display
    SPECIES_COLORS = {
        'Gallus_gallus': '#FF5733',
        'Anas_platyrhynchos': '#33FF57', 
        'Alligator_mississippiensis': '#3357FF',
        'Struthio_camelus': '#FF33F5',
        'Crocodylus_porosus': '#F5FF33',
        'Python_bivittatus': '#33F5FF',
        'ancient_dinosaur': '#8B4513',
        'reconstructed': '#DAA520',
    }

class DNASequenceVisualizer:
    """Core DNA sequence visualization engine."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.colors = DNAColorScheme()
        self.base_width = 20
        self.base_height = 30
        
    def create_sequence_image(self, sequence: str, confidence: Optional[List[float]] = None,
                            mutations: Optional[List[Dict]] = None, title: str = "") -> Image.Image:
        """Create a visual representation of a DNA sequence."""
        
        # Calculate dimensions
        seq_len = len(sequence)
        cols = min(60, seq_len)  # Max 60 bases per row
        rows = (seq_len + cols - 1) // cols
        
        img_width = cols * self.base_width + 100
        img_height = rows * self.base_height + 100
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        if title:
            draw.text((10, 10), title, fill='black', font=title_font)
        
        # Draw sequence
        start_y = 40
        for i, base in enumerate(sequence):
            row = i // cols
            col = i % cols
            
            x = col * self.base_width + 50
            y = row * self.base_height + start_y
            
            # Get base color
            base_color = self.colors.BASE_COLORS.get(base.upper(), '#CCCCCC')
            
            # Modify color by confidence if available
            if confidence and i < len(confidence):
                conf = confidence[i]
                # Blend with white based on confidence
                base_color = self._blend_with_confidence(base_color, conf)
            
            # Draw base rectangle
            draw.rectangle([x, y, x + self.base_width - 2, y + self.base_height - 2], 
                         fill=base_color, outline='black')
            
            # Draw base letter
            text_x = x + self.base_width // 2 - 5
            text_y = y + self.base_height // 2 - 8
            draw.text((text_x, text_y), base.upper(), fill='white', font=font)
            
            # Mark mutations if available
            if mutations:
                for mut in mutations:
                    if mut.get('position') == i:
                        # Draw mutation marker
                        marker_color = self.colors.MUTATION_COLORS.get(
                            mut.get('type', 'unknown'), '#FF0000'
                        )
                        draw.ellipse([x + self.base_width - 8, y, x + self.base_width, y + 8],
                                   fill=marker_color)
        
        return img
    
    def _blend_with_confidence(self, color: str, confidence: float) -> str:
        """Blend base color with confidence level."""
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16) 
        b = int(color[5:7], 16)
        
        # Blend with white based on confidence (low confidence = more white)
        blend_factor = confidence
        r = int(r * blend_factor + 255 * (1 - blend_factor))
        g = int(g * blend_factor + 255 * (1 - blend_factor))
        b = int(b * blend_factor + 255 * (1 - blend_factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def create_alignment_view(self, sequences: Dict[str, str], 
                            names: List[str] = None) -> Image.Image:
        """Create multiple sequence alignment visualization."""
        
        if not sequences:
            return self.create_sequence_image("", title="No sequences to display")
        
        seq_names = names or list(sequences.keys())
        max_len = max(len(seq) for seq in sequences.values())
        
        # Pad sequences to same length
        padded_sequences = {}
        for name, seq in sequences.items():
            padded_sequences[name] = seq + '-' * (max_len - len(seq))
        
        # Calculate dimensions
        cols = min(80, max_len)
        rows_per_seq = (max_len + cols - 1) // cols
        total_rows = len(sequences) * rows_per_seq + len(sequences) - 1  # Space between sequences
        
        img_width = cols * self.base_width + 200
        img_height = total_rows * self.base_height + 100
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 10)
            name_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            name_font = ImageFont.load_default()
        
        # Draw sequences
        current_y = 20
        for seq_idx, (name, sequence) in enumerate(padded_sequences.items()):
            # Draw sequence name
            species_color = self.colors.SPECIES_COLORS.get(name, '#000000')
            draw.text((10, current_y), name[:20], fill=species_color, font=name_font)
            
            # Draw sequence
            for i, base in enumerate(sequence):
                row = i // cols
                col = i % cols
                
                x = col * self.base_width + 150
                y = current_y + row * self.base_height
                
                base_color = self.colors.BASE_COLORS.get(base.upper(), '#FFFFFF')
                
                # Highlight differences from first sequence (if not first sequence)
                if seq_idx > 0:
                    ref_base = list(padded_sequences.values())[0][i]
                    if base != ref_base and base != '-' and ref_base != '-':
                        # Add mutation highlighting
                        draw.rectangle([x-1, y-1, x + self.base_width + 1, y + self.base_height + 1], 
                                     outline='red', width=2)
                
                draw.rectangle([x, y, x + self.base_width - 2, y + self.base_height - 2], 
                             fill=base_color, outline='black')
                
                if base != '-':
                    text_x = x + self.base_width // 2 - 4
                    text_y = y + self.base_height // 2 - 6
                    draw.text((text_x, text_y), base.upper(), fill='black', font=font)
            
            current_y += rows_per_seq * self.base_height + 10
        
        return img

class LiveDNAReconstruction:
    """Live DNA reconstruction simulation engine."""
    
    def __init__(self, model=None):
        self.model = model
        self.visualizer = DNASequenceVisualizer()
        self.reconstruction_history = []
        self.mutation_events = []
        self.confidence_history = []
        
    def simulate_reconstruction_step(self, damaged_sequence: str, 
                                   step: int, total_steps: int) -> Dict:
        """Simulate one step of DNA reconstruction."""
        
        # Calculate reconstruction progress
        progress = step / total_steps
        
        # Simulate model prediction for this step
        if self.model and hasattr(self.model, 'predict_single_step'):
            # Use actual model if available
            step_result = self.model.predict_single_step(damaged_sequence, step)
        else:
            # Simulate reconstruction progress
            step_result = self._simulate_step(damaged_sequence, progress)
        
        # Record history
        self.reconstruction_history.append(step_result['sequence'])
        self.confidence_history.append(step_result['confidence'])
        
        if step_result.get('mutations'):
            self.mutation_events.extend(step_result['mutations'])
        
        return step_result
    
    def _simulate_step(self, damaged_sequence: str, progress: float) -> Dict:
        """Simulate reconstruction step (when no real model available)."""
        
        # Define target "ancient" sequence (simulated)
        target_bases = ['A', 'T', 'G', 'C']
        target_sequence = ''.join(np.random.choice(target_bases, len(damaged_sequence)))
        
        # Interpolate between damaged and target based on progress
        reconstructed = []
        confidence_scores = []
        mutations = []
        
        for i, (damaged_base, target_base) in enumerate(zip(damaged_sequence, target_sequence)):
            # Gradually reconstruct bases
            if damaged_base == 'N' or damaged_base == '-':
                if np.random.random() < progress:
                    # Reconstruct this position
                    reconstructed.append(target_base)
                    confidence_scores.append(min(0.9, progress + np.random.normal(0, 0.1)))
                    
                    mutations.append({
                        'position': i,
                        'from': damaged_base,
                        'to': target_base,
                        'type': 'reconstruction',
                        'step': progress
                    })
                else:
                    reconstructed.append(damaged_base)
                    confidence_scores.append(0.1)
            else:
                # Keep existing base but maybe improve confidence
                reconstructed.append(damaged_base)
                confidence_scores.append(min(0.95, 0.5 + progress * 0.5))
        
        return {
            'sequence': ''.join(reconstructed),
            'confidence': confidence_scores,
            'mutations': mutations,
            'progress': progress
        }
    
    def create_reconstruction_animation(self, damaged_sequence: str, 
                                     total_steps: int = 50) -> List[Image.Image]:
        """Create animation frames for reconstruction process."""
        
        frames = []
        self.reconstruction_history = []
        self.mutation_events = []
        self.confidence_history = []
        
        for step in range(total_steps + 1):
            # Simulate reconstruction step
            result = self.simulate_reconstruction_step(damaged_sequence, step, total_steps)
            
            # Create visualization frame
            title = f"Reconstruction Step {step}/{total_steps} (Progress: {result['progress']:.1%})"
            frame = self.visualizer.create_sequence_image(
                result['sequence'],
                confidence=result['confidence'],
                mutations=result.get('mutations', []),
                title=title
            )
            
            frames.append(frame)
        
        return frames

class PhylogeneticTree3D:
    """3D phylogenetic tree visualization."""
    
    def __init__(self):
        self.species_data = {}
        self.colors = DNAColorScheme()
    
    def create_3d_tree(self, species_sequences: Dict[str, str]) -> go.Figure:
        """Create 3D phylogenetic tree visualization."""
        
        # Calculate distance matrix
        species_names = list(species_sequences.keys())
        n_species = len(species_names)
        
        if n_species == 0:
            return go.Figure()
        
        # Calculate pairwise distances
        distances = np.zeros((n_species, n_species))
        for i in range(n_species):
            for j in range(n_species):
                if i != j:
                    seq1 = species_sequences[species_names[i]]
                    seq2 = species_sequences[species_names[j]]
                    distances[i][j] = self._sequence_distance(seq1, seq2)
        
        # Convert to 3D coordinates using MDS-like approach
        coordinates = self._distance_to_3d(distances)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add species points
        for i, species in enumerate(species_names):
            color = self.colors.SPECIES_COLORS.get(species, '#000000')
            
            fig.add_trace(go.Scatter3d(
                x=[coordinates[i, 0]],
                y=[coordinates[i, 1]],
                z=[coordinates[i, 2]],
                mode='markers+text',
                marker=dict(size=10, color=color),
                text=species,
                textposition='top center',
                name=species,
                hovertemplate=f'<b>{species}</b><br>' +
                             f'Sequence Length: {len(species_sequences[species])}<br>' +
                             '<extra></extra>'
            ))
        
        # Add connecting lines (simplified tree)
        self._add_tree_connections(fig, coordinates, species_names, distances)
        
        fig.update_layout(
            title='3D Phylogenetic Tree',
            scene=dict(
                xaxis_title='Evolutionary Distance X',
                yaxis_title='Evolutionary Distance Y',
                zaxis_title='Evolutionary Distance Z'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _sequence_distance(self, seq1: str, seq2: str) -> float:
        """Calculate simple p-distance between sequences."""
        if len(seq1) != len(seq2):
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
        
        if len(seq1) == 0:
            return 1.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
    
    def _distance_to_3d(self, distances: np.ndarray) -> np.ndarray:
        """Convert distance matrix to 3D coordinates."""
        n = distances.shape[0]
        
        if n < 2:
            return np.random.rand(n, 3)
        
        # Simple MDS-like projection
        # Center the distance matrix
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (distances ** 2) @ H
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)
        
        # Take top 3 eigenvalues
        idx = np.argsort(eigenvals)[::-1][:3]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Create coordinates
        coordinates = eigenvecs @ np.diag(np.sqrt(np.maximum(eigenvals, 0)))
        
        return coordinates
    
    def _add_tree_connections(self, fig, coordinates, species_names, distances):
        """Add tree connections to the 3D plot."""
        # Simple star tree for now (connect all to centroid)
        centroid = np.mean(coordinates, axis=0)
        
        for i, coord in enumerate(coordinates):
            fig.add_trace(go.Scatter3d(
                x=[centroid[0], coord[0]],
                y=[centroid[1], coord[1]],
                z=[centroid[2], coord[2]],
                mode='lines',
                line=dict(color='gray', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

class MutationEvolutionSimulator:
    """Simulate mutations and evolutionary changes in real-time."""
    
    def __init__(self):
        self.mutation_rates = {
            'A->G': 0.25, 'G->A': 0.25, 'C->T': 0.25, 'T->C': 0.25,  # Transitions
            'A->C': 0.0625, 'A->T': 0.0625, 'C->G': 0.0625, 'G->C': 0.0625,  # Transversions
            'G->T': 0.0625, 'T->G': 0.0625, 'C->A': 0.0625, 'T->A': 0.0625
        }
        self.colors = DNAColorScheme()
    
    def simulate_evolution_timeline(self, ancestral_sequence: str, 
                                  time_points: List[int]) -> Dict[int, str]:
        """Simulate sequence evolution over time."""
        
        sequences = {0: ancestral_sequence}
        current_seq = ancestral_sequence
        
        for time_point in sorted(time_points):
            if time_point <= 0:
                continue
                
            # Calculate mutations for this time period
            prev_time = max([t for t in sequences.keys() if t < time_point], default=0)
            time_diff = time_point - prev_time
            
            # Apply mutations
            current_seq = self._apply_mutations(current_seq, time_diff)
            sequences[time_point] = current_seq
        
        return sequences
    
    def _apply_mutations(self, sequence: str, time_units: int) -> str:
        """Apply mutations to sequence based on time and rates."""
        
        mutated = list(sequence)
        
        for i, base in enumerate(sequence):
            if base in ['A', 'T', 'G', 'C']:
                # Calculate probability of mutation at this position
                mut_prob = time_units * 0.01  # Base mutation rate
                
                if np.random.random() < mut_prob:
                    # Choose mutation type based on transition/transversion bias
                    possible_mutations = []
                    for mutation, rate in self.mutation_rates.items():
                        if mutation.startswith(base + '->'):
                            possible_mutations.extend([mutation.split('->')[-1]] * int(rate * 1000))
                    
                    if possible_mutations:
                        mutated[i] = np.random.choice(possible_mutations)
        
        return ''.join(mutated)
    
    def create_mutation_heatmap(self, sequences: Dict, names: List[str] = None) -> go.Figure:
        """Create heatmap showing mutation patterns."""
        
        seq_names = names or [f"Sequence_{i}" for i in range(len(sequences))]
        
        if not sequences:
            return go.Figure()
        
        # Get all sequences as list
        seq_values = list(sequences.values()) if isinstance(sequences, dict) else sequences
        
        # Calculate mutation matrix
        mutation_matrix = []
        for seq in seq_values:
            base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
            for base in seq:
                if base in base_counts:
                    base_counts[base] += 1
            
            total = sum(base_counts.values())
            if total > 0:
                mutation_matrix.append([base_counts[b] / total for b in ['A', 'T', 'G', 'C']])
            else:
                mutation_matrix.append([0.25, 0.25, 0.25, 0.25])
        
        fig = go.Figure(data=go.Heatmap(
            z=mutation_matrix,
            x=['A', 'T', 'G', 'C'],
            y=seq_names,
            colorscale='viridis',
            text=[[f"{val:.2%}" for val in row] for row in mutation_matrix],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Base Composition Heatmap',
            xaxis_title='DNA Bases',
            yaxis_title='Sequences'
        )
        
        return fig

class RealTimeDNADashboard:
    """Main dashboard for real-time DNA reconstruction visualization."""
    
    def __init__(self, model=None):
        self.model = model
        self.reconstruction_sim = LiveDNAReconstruction(model)
        self.phylo_viz = PhylogeneticTree3D()
        self.mutation_sim = MutationEvolutionSimulator()
        self.visualizer = DNASequenceVisualizer()
        
    def create_dashboard(self):
        """Create Streamlit dashboard for live DNA simulation."""
        
        st.set_page_config(
            page_title="🧬 Live DNA Reconstruction Simulator",
            page_icon="🧬",
            layout="wide"
        )
        
        st.title("🧬 Live DNA Reconstruction Simulator")
        st.markdown("Real-time visualization of dinosaur DNA reconstruction using enhanced AI models")
        
        # Sidebar controls
        st.sidebar.header("🎛️ Simulation Controls")
        
        # Input sequence
        input_sequence = st.sidebar.text_area(
            "Input Damaged Sequence",
            value="ATGCN-NATGC-NNCGATNNNAAATTT",
            help="Enter a DNA sequence with N for unknown bases and - for gaps"
        )
        
        # Simulation parameters
        reconstruction_steps = st.sidebar.slider("Reconstruction Steps", 10, 100, 50)
        mutation_rate = st.sidebar.slider("Mutation Rate", 0.001, 0.1, 0.01)
        species_count = st.sidebar.slider("Species Count", 2, 8, 4)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🔄 Live Reconstruction")
            
            if st.button("🚀 Start Reconstruction Simulation"):
                self._run_reconstruction_simulation(input_sequence, reconstruction_steps)
        
        with col2:
            st.header("📊 Statistics")
            self._display_statistics()
        
        # Additional visualizations
        st.header("🌟 Advanced Visualizations")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🌳 3D Phylogenetic Tree")
            self._display_phylogenetic_tree()
        
        with col4:
            st.subheader("🔥 Mutation Heatmap")
            self._display_mutation_heatmap()
        
        # Sequence alignment view
        st.header("🧬 Multi-Species Alignment")
        self._display_sequence_alignment()
    
    def _run_reconstruction_simulation(self, sequence: str, steps: int):
        """Run and display reconstruction simulation."""
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        image_placeholder = st.empty()
        
        # Run simulation
        frames = self.reconstruction_sim.create_reconstruction_animation(sequence, steps)
        
        for i, frame in enumerate(frames):
            progress = (i + 1) / len(frames)
            progress_bar.progress(progress)
            status_text.text(f"Step {i+1}/{steps}: Reconstructing DNA sequence...")
            
            # Convert PIL image to displayable format
            img_buffer = io.BytesIO()
            frame.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            image_placeholder.image(img_buffer, caption=f"Reconstruction Step {i+1}")
            
            # Add small delay for visualization
            time.sleep(0.1)
        
        status_text.text("✅ Reconstruction completed!")
    
    def _display_statistics(self):
        """Display reconstruction statistics."""
        
        if hasattr(self.reconstruction_sim, 'reconstruction_history') and self.reconstruction_sim.reconstruction_history:
            latest_seq = self.reconstruction_sim.reconstruction_history[-1]
            
            # Calculate statistics
            base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
            for base in latest_seq:
                if base in base_counts:
                    base_counts[base] += 1
            
            st.metric("Sequence Length", len(latest_seq))
            st.metric("Reconstructed Bases", base_counts['A'] + base_counts['T'] + base_counts['G'] + base_counts['C'])
            st.metric("Unknown Bases", base_counts['N'])
            
            # GC content
            gc_content = (base_counts['G'] + base_counts['C']) / len(latest_seq) if len(latest_seq) > 0 else 0
            st.metric("GC Content", f"{gc_content:.2%}")
            
            # Average confidence
            if hasattr(self.reconstruction_sim, 'confidence_history') and self.reconstruction_sim.confidence_history:
                avg_confidence = np.mean(self.reconstruction_sim.confidence_history[-1])
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
        else:
            st.info("Run a reconstruction simulation to see statistics")
    
    def _display_phylogenetic_tree(self):
        """Display 3D phylogenetic tree."""
        
        # Sample sequences for demonstration
        sample_sequences = {
            'Gallus_gallus': 'ATGCGATCGATCGATCG',
            'Alligator_mississippiensis': 'ATGCGATCGATCGATCG',
            'Struthio_camelus': 'ATGCGATCGATCGATCA',
            'Reconstructed_Dinosaur': 'ATGCGATCGATCGATCT'
        }
        
        fig = self.phylo_viz.create_3d_tree(sample_sequences)
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_mutation_heatmap(self):
        """Display mutation pattern heatmap."""
        
        # Sample data for demonstration
        sample_sequences = [
            'ATGCGATCGATCGATCG',
            'ATGCGATCGATCGATCA',
            'ATGCGATCGATCGATCT',
            'ATGCGATCGATCGATCG'
        ]
        
        fig = self.mutation_sim.create_mutation_heatmap(
            sample_sequences,
            ['Original', 'After 1Mya', 'After 5Mya', 'Reconstructed']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_sequence_alignment(self):
        """Display multi-species sequence alignment."""
        
        sample_sequences = {
            'T_rex_reconstructed': 'ATGCGATCGATCGATCG',
            'Gallus_gallus': 'ATGCGATCGATCGATCG',
            'Alligator_mississippiensis': 'ATGCGATCGATCGATCA',
            'Struthio_camelus': 'ATGCGATCGATCGATCT'
        }
        
        alignment_img = self.visualizer.create_alignment_view(sample_sequences)
        
        # Convert to displayable format
        img_buffer = io.BytesIO()
        alignment_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.image(img_buffer, caption="Multi-Species Sequence Alignment")

# VS Code Integration Functions
def create_vscode_extension():
    """Create HTML dashboard that can be embedded in VS Code."""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DNA Reconstruction Visualizer</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .sequence-display { font-family: monospace; font-size: 14px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; background: #007acc; color: white; border: none; cursor: pointer; }
            button:hover { background: #005a9e; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 DNA Reconstruction Visualizer</h1>
            
            <div class="controls">
                <button onclick="startReconstruction()">Start Reconstruction</button>
                <button onclick="showPhylogeny()">Show Phylogeny</button>
                <button onclick="exportResults()">Export Results</button>
            </div>
            
            <div id="sequence-display" class="sequence-display">
                <p>Input your damaged sequence and click Start Reconstruction</p>
            </div>
            
            <div id="visualization-area">
                <!-- Visualizations will be inserted here -->
            </div>
        </div>
        
        <script>
            function startReconstruction() {
                // Integration with Python backend
                const vscode = acquireVsCodeApi();
                vscode.postMessage({
                    command: 'startReconstruction',
                    sequence: 'ATGCN-NATGC-NNCGATNNNAAATTT'
                });
            }
            
            function showPhylogeny() {
                const vscode = acquireVsCodeApi();
                vscode.postMessage({
                    command: 'showPhylogeny'
                });
            }
            
            function exportResults() {
                const vscode = acquireVsCodeApi();
                vscode.postMessage({
                    command: 'exportResults'
                });
            }
            
            // Listen for messages from VS Code extension
            window.addEventListener('message', event => {
                const message = event.data;
                switch (message.command) {
                    case 'updateSequence':
                        document.getElementById('sequence-display').innerHTML = message.html;
                        break;
                    case 'updateVisualization':
                        document.getElementById('visualization-area').innerHTML = message.html;
                        break;
                }
            });
        </script>
    </body>
    </html>
    """
    
    return html_template

def run_streamlit_dashboard():
    """Main function to run the Streamlit dashboard."""
    dashboard = RealTimeDNADashboard()
    dashboard.create_dashboard()

if __name__ == "__main__":
    # Run the dashboard
    print("🧬 Starting DNA Reconstruction Visualizer...")
    print("📊 Dashboard will open in your browser")
    print("🔗 Run: streamlit run visual_dna_simulation.py")
    
    # For VS Code integration, you can also create HTML output
    vscode_html = create_vscode_extension()
    with open("dna_visualizer.html", "w", encoding="utf-8") as f:
        f.write(vscode_html)
    
    print("💻 VS Code HTML file created: dna_visualizer.html")

"""
Interactive DNA Reconstruction Notebook
For VS Code Jupyter integration - Live DNA visualization and simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import torch
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# Enhanced imports
try:
    from enhanced_config import config
    from enhanced_models import create_model
    from visual_dna_simulation import DNASequenceVisualizer, LiveDNAReconstruction, PhylogeneticTree3D
    ENHANCED_AVAILABLE = True
except ImportError:
    print("Enhanced modules not found. Using basic visualization.")
    ENHANCED_AVAILABLE = False

class InteractiveDNANotebook:
    """Interactive DNA reconstruction for Jupyter notebooks in VS Code."""
    
    def __init__(self, model=None):
        self.model = model
        self.current_sequence = ""
        self.reconstruction_history = []
        self.mutation_events = []
        
        # Initialize visualizers
        if ENHANCED_AVAILABLE:
            self.visualizer = DNASequenceVisualizer()
            self.reconstruction_sim = LiveDNAReconstruction(model)
            self.phylo_viz = PhylogeneticTree3D()
        else:
            self.visualizer = BasicDNAVisualizer()
    
    def create_interactive_widgets(self):
        """Create interactive widgets for DNA reconstruction."""
        
        # Input widgets
        sequence_input = widgets.Textarea(
            value='ATGCN-NATGC-NNCGATNNNAAATTT',
            placeholder='Enter damaged DNA sequence...',
            description='DNA Sequence:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%', height='100px')
        )
        
        steps_slider = widgets.IntSlider(
            value=20,
            min=5,
            max=100,
            step=5,
            description='Reconstruction Steps:',
            style={'description_width': 'initial'}
        )
        
        mutation_rate = widgets.FloatSlider(
            value=0.01,
            min=0.001,
            max=0.1,
            step=0.001,
            description='Mutation Rate:',
            style={'description_width': 'initial'}
        )
        
        species_selector = widgets.SelectMultiple(
            options=['Gallus_gallus', 'Alligator_mississippiensis', 'Struthio_camelus', 
                    'Crocodylus_porosus', 'Python_bivittatus'],
            value=['Gallus_gallus', 'Alligator_mississippiensis'],
            description='Species:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='120px')
        )
        
        # Control buttons
        start_button = widgets.Button(
            description='🚀 Start Reconstruction',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )
        
        phylo_button = widgets.Button(
            description='🌳 Show Phylogeny',
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        
        export_button = widgets.Button(
            description='💾 Export Results',
            button_style='warning',
            layout=widgets.Layout(width='200px')
        )
        
        # Output areas
        self.output_area = widgets.Output()
        self.plot_area = widgets.Output()
        
        # Event handlers
        start_button.on_click(lambda b: self._start_reconstruction(
            sequence_input.value, steps_slider.value, mutation_rate.value
        ))
        
        phylo_button.on_click(lambda b: self._show_phylogeny(species_selector.value))
        export_button.on_click(lambda b: self._export_results())
        
        # Layout
        input_box = widgets.VBox([
            widgets.HTML("<h3>🧬 DNA Reconstruction Parameters</h3>"),
            sequence_input,
            widgets.HBox([steps_slider, mutation_rate]),
            species_selector
        ])
        
        control_box = widgets.HBox([start_button, phylo_button, export_button])
        
        main_interface = widgets.VBox([
            input_box,
            control_box,
            widgets.HTML("<h3>📊 Live Visualization</h3>"),
            self.output_area,
            self.plot_area
        ])
        
        return main_interface
    
    def _start_reconstruction(self, sequence: str, steps: int, mutation_rate: float):
        """Start the reconstruction simulation."""
        
        with self.output_area:
            clear_output(wait=True)
            print("🚀 Starting DNA reconstruction simulation...")
            print(f"📝 Input sequence: {sequence}")
            print(f"🔄 Reconstruction steps: {steps}")
            print(f"🧬 Mutation rate: {mutation_rate}")
            print("\n" + "="*50 + "\n")
        
        # Run reconstruction simulation
        self._run_live_reconstruction(sequence, steps)
    
    def _run_live_reconstruction(self, sequence: str, steps: int):
        """Run live reconstruction with real-time updates."""
        
        # Initialize
        current_seq = sequence
        self.reconstruction_history = [current_seq]
        
        # Create figure for live plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        with self.plot_area:
            clear_output(wait=True)
            
            for step in range(steps + 1):
                # Simulate reconstruction step
                if ENHANCED_AVAILABLE:
                    result = self.reconstruction_sim.simulate_reconstruction_step(
                        current_seq, step, steps
                    )
                    current_seq = result['sequence']
                    confidence = result['confidence']
                else:
                    current_seq, confidence = self._basic_reconstruction_step(current_seq, step, steps)
                
                self.reconstruction_history.append(current_seq)
                
                # Update visualization
                if step % 5 == 0:  # Update every 5 steps for performance
                    self._update_live_plot(ax1, ax2, ax3, current_seq, confidence, step, steps)
                    
                    # Update output
                    with self.output_area:
                        print(f"Step {step:2d}/{steps}: {current_seq}")
                        if step > 0:
                            print(f"Progress: {'█' * (step * 20 // steps)}{'░' * (20 - step * 20 // steps)} {step*100//steps}%")
                
                time.sleep(0.1)  # Brief pause for visualization
            
            # Final update
            with self.output_area:
                print(f"\n✅ Reconstruction completed!")
                print(f"📊 Final sequence: {current_seq}")
                self._display_final_statistics(sequence, current_seq)
    
    def _update_live_plot(self, ax1, ax2, ax3, sequence: str, confidence: List[float], step: int, total_steps: int):
        """Update live reconstruction plots."""
        
        # Clear previous plots
        ax1.clear()
        ax2.clear() 
        ax3.clear()
        
        # Plot 1: Sequence composition
        bases = ['A', 'T', 'G', 'C', 'N', '-']
        counts = [sequence.count(base) for base in bases]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726', '#9E9E9E', '#FFFFFF']
        
        ax1.bar(bases, counts, color=colors, edgecolor='black')
        ax1.set_title(f'Step {step}/{total_steps}: Base Composition')
        ax1.set_ylabel('Count')
        
        # Plot 2: Confidence scores
        if confidence:
            ax2.plot(range(len(confidence)), confidence, 'b-', alpha=0.7)
            ax2.fill_between(range(len(confidence)), confidence, alpha=0.3)
            ax2.set_title('Reconstruction Confidence')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Confidence')
            ax2.set_ylim(0, 1)
        
        # Plot 3: Progress tracking
        progress = step / total_steps
        ax3.barh(['Progress'], [progress], color='green', alpha=0.7)
        ax3.set_xlim(0, 1)
        ax3.set_title(f'Overall Progress: {progress:.1%}')
        
        plt.tight_layout()
        plt.show()
    
    def _basic_reconstruction_step(self, sequence: str, step: int, total_steps: int) -> Tuple[str, List[float]]:
        """Basic reconstruction simulation when enhanced modules not available."""
        
        progress = step / total_steps
        reconstructed = []
        confidence_scores = []
        
        target_bases = ['A', 'T', 'G', 'C']
        
        for i, base in enumerate(sequence):
            if base in ['N', '-']:
                if np.random.random() < progress:
                    # Reconstruct this position
                    new_base = np.random.choice(target_bases)
                    reconstructed.append(new_base)
                    confidence_scores.append(min(0.9, progress + np.random.normal(0, 0.1)))
                else:
                    reconstructed.append(base)
                    confidence_scores.append(0.1)
            else:
                reconstructed.append(base)
                confidence_scores.append(min(0.95, 0.5 + progress * 0.5))
        
        return ''.join(reconstructed), confidence_scores
    
    def _show_phylogeny(self, selected_species: tuple):
        """Show phylogenetic analysis."""
        
        with self.plot_area:
            clear_output(wait=True)
            
            # Sample sequences for selected species
            sample_sequences = {}
            for species in selected_species:
                # Generate sample sequence (in real implementation, use actual data)
                seq_length = 20
                sample_sequences[species] = ''.join(np.random.choice(['A', 'T', 'G', 'C'], seq_length))
            
            # Add reconstructed sequence
            if self.reconstruction_history:
                sample_sequences['Reconstructed'] = self.reconstruction_history[-1]
            
            if ENHANCED_AVAILABLE:
                fig = self.phylo_viz.create_3d_tree(sample_sequences)
                fig.show()
            else:
                # Basic phylogenetic visualization
                self._basic_phylogeny_plot(sample_sequences)
    
    def _basic_phylogeny_plot(self, sequences: Dict[str, str]):
        """Basic phylogenetic plot when enhanced modules not available."""
        
        # Calculate distance matrix
        species = list(sequences.keys())
        n = len(species)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                seq1, seq2 = sequences[species[i]], sequences[species[j]]
                min_len = min(len(seq1), len(seq2))
                if min_len > 0:
                    differences = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len]) if a != b)
                    distances[i, j] = differences / min_len
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(distances, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Genetic Distance')
        plt.xticks(range(n), species, rotation=45)
        plt.yticks(range(n), species)
        plt.title('Phylogenetic Distance Matrix')
        plt.tight_layout()
        plt.show()
    
    def _display_final_statistics(self, original: str, reconstructed: str):
        """Display final reconstruction statistics."""
        
        print("\n📊 RECONSTRUCTION STATISTICS")
        print("=" * 40)
        
        # Basic statistics
        orig_bases = {base: original.count(base) for base in 'ATGCN-'}
        recon_bases = {base: reconstructed.count(base) for base in 'ATGCN-'}
        
        print(f"Original unknown (N): {orig_bases.get('N', 0)}")
        print(f"Reconstructed unknown (N): {recon_bases.get('N', 0)}")
        print(f"Reconstruction rate: {(1 - recon_bases.get('N', 0) / len(reconstructed)):.1%}")
        
        # GC content
        orig_gc = (orig_bases.get('G', 0) + orig_bases.get('C', 0)) / len(original)
        recon_gc = (recon_bases.get('G', 0) + recon_bases.get('C', 0)) / len(reconstructed)
        print(f"GC content change: {orig_gc:.1%} → {recon_gc:.1%}")
        
        # Sequence identity (for comparable positions)
        matches = sum(1 for a, b in zip(original, reconstructed) 
                     if a == b and a not in 'N-')
        comparable = sum(1 for a in original if a not in 'N-')
        if comparable > 0:
            identity = matches / comparable
            print(f"Sequence identity: {identity:.1%}")
    
    def _export_results(self):
        """Export reconstruction results."""
        
        with self.output_area:
            print("💾 Exporting reconstruction results...")
            
            if self.reconstruction_history:
                # Save to file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"dna_reconstruction_{timestamp}.txt"
                
                with open(filename, 'w') as f:
                    f.write("DNA Reconstruction Results\n")
                    f.write("=" * 30 + "\n\n")
                    
                    for i, seq in enumerate(self.reconstruction_history):
                        f.write(f"Step {i:2d}: {seq}\n")
                    
                    f.write(f"\nFinal sequence: {self.reconstruction_history[-1]}\n")
                
                print(f"✅ Results saved to: {filename}")
            else:
                print("❌ No reconstruction data to export")

class BasicDNAVisualizer:
    """Basic DNA visualizer when enhanced modules not available."""
    
    def __init__(self):
        self.base_colors = {
            'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#FFA726',
            'N': '#9E9E9E', '-': '#FFFFFF'
        }
    
    def plot_sequence(self, sequence: str, title: str = "DNA Sequence"):
        """Basic sequence plotting."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot 1: Base composition
        bases = ['A', 'T', 'G', 'C', 'N', '-']
        counts = [sequence.count(base) for base in bases]
        colors = [self.base_colors[base] for base in bases]
        
        ax1.bar(bases, counts, color=colors, edgecolor='black')
        ax1.set_title(f'{title}: Base Composition')
        ax1.set_ylabel('Count')
        
        # Plot 2: Sequence visualization (first 100 bases)
        display_seq = sequence[:100] if len(sequence) > 100 else sequence
        x_positions = range(len(display_seq))
        y_positions = [0] * len(display_seq)
        colors = [self.base_colors.get(base, '#CCCCCC') for base in display_seq]
        
        ax2.scatter(x_positions, y_positions, c=colors, s=100, alpha=0.8)
        ax2.set_xlim(-1, len(display_seq))
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Position')
        ax2.set_title(f'{title}: Sequence View (first 100 bases)')
        ax2.set_yticks([])
        
        # Add base labels
        for i, base in enumerate(display_seq):
            ax2.text(i, 0, base, ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Main notebook functions
def create_dna_reconstruction_notebook():
    """Main function to create the interactive DNA reconstruction notebook."""
    
    print("🧬 Initializing Interactive DNA Reconstruction Notebook...")
    
    # Load model if available
    model = None
    if ENHANCED_AVAILABLE:
        try:
            model = create_model('hybrid')
            print("✅ Enhanced model loaded")
        except Exception as e:
            print(f"⚠️ Could not load enhanced model: {e}")
    
    # Create interactive interface
    notebook = InteractiveDNANotebook(model)
    interface = notebook.create_interactive_widgets()
    
    print("🎛️ Interactive interface ready!")
    return interface

def quick_dna_visualization(sequence: str = "ATGCN-NATGC-NNCGATNNNAAATTT"):
    """Quick DNA sequence visualization function."""
    
    visualizer = BasicDNAVisualizer()
    visualizer.plot_sequence(sequence, "Quick DNA Visualization")

def demo_reconstruction_simulation():
    """Demo reconstruction simulation."""
    
    print("🧬 Demo: DNA Reconstruction Simulation")
    
    # Sample damaged sequence
    damaged_seq = "ATGCN-NATGC-NNCGATNNNAAATTT"
    print(f"🔬 Original damaged: {damaged_seq}")
    
    # Simulate reconstruction steps
    current_seq = damaged_seq
    
    for step in range(0, 21, 5):  # Show every 5th step
        progress = step / 20
        
        # Basic reconstruction simulation
        reconstructed = []
        for base in current_seq:
            if base in ['N', '-']:
                if np.random.random() < progress:
                    reconstructed.append(np.random.choice(['A', 'T', 'G', 'C']))
                else:
                    reconstructed.append(base)
            else:
                reconstructed.append(base)
        
        current_seq = ''.join(reconstructed)
        print(f"Step {step:2d}/20: {current_seq}")
    
    print(f"✅ Final result: {current_seq}")

# Interactive widgets for direct use
@interact(
    sequence=widgets.Textarea(value="ATGCN-NATGC-NNCGATNNNAAATTT", description="DNA Sequence:"),
    steps=widgets.IntSlider(value=20, min=5, max=50, description="Steps:")
)
def interactive_reconstruction(sequence, steps):
    """Interactive reconstruction widget."""
    
    print(f"🧬 Running reconstruction with {steps} steps...")
    
    current_seq = sequence
    for step in range(steps + 1):
        if step % 5 == 0:  # Show every 5th step
            progress = step / steps
            
            # Basic reconstruction
            reconstructed = []
            for base in current_seq:
                if base in ['N', '-']:
                    if np.random.random() < progress:
                        reconstructed.append(np.random.choice(['A', 'T', 'G', 'C']))
                    else:
                        reconstructed.append(base)
                else:
                    reconstructed.append(base)
            
            current_seq = ''.join(reconstructed)
            print(f"Step {step:2d}: {current_seq}")

# Display instructions
def show_notebook_instructions():
    """Display instructions for using the notebook."""
    
    instructions = """
    🧬 Interactive DNA Reconstruction Notebook
    ==========================================
    
    This notebook provides interactive DNA reconstruction visualization.
    
    Quick Start:
    1. Run: interface = create_dna_reconstruction_notebook()
    2. Display: display(interface)
    3. Enter your DNA sequence and click "Start Reconstruction"
    
    Available Functions:
    • create_dna_reconstruction_notebook() - Main interactive interface
    • quick_dna_visualization(sequence) - Quick sequence plot
    • demo_reconstruction_simulation() - Demo simulation
    • interactive_reconstruction() - Widget-based reconstruction
    
    Features:
    🔄 Live reconstruction simulation
    📊 Real-time statistics
    🌳 Phylogenetic analysis
    💾 Export results
    🎛️ Interactive controls
    
    Compatible with VS Code Jupyter extension!
    """
    
    print(instructions)

if __name__ == "__main__":
    # Show instructions when notebook is loaded
    show_notebook_instructions()
    
    # Create the main interface
    print("\n🚀 Creating main interface...")
    interface = create_dna_reconstruction_notebook()
    
    print("📱 To use the interface, run:")
    print("display(interface)")

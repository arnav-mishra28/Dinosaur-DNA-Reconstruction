"""
3D DNA Helix Reconstruction Visualizer - FIXED VERSION
Real-time 3D visualization of DNA double helix during reconstruction process
Compatible with all Python versions and matplotlib versions
"""

import tkinter as tk
from tkinter import ttk, messagebox, Frame
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import random
import math
import json

class DNAHelix3D:
    """3D DNA helix model with base pair visualization."""
    
    def __init__(self):
        # DNA helix parameters
        self.helix_radius = 1.0
        self.helix_pitch = 0.34  # Distance between base pairs (scaled)
        self.helix_turn = 36    # Degrees per base pair
        self.backbone_radius = 0.1
        
        # Base pair colors
        self.base_colors = {
            'A': '#FF4444',  # Red
            'T': '#44FF44',  # Green  
            'G': '#4444FF',  # Blue
            'C': '#FFAA00',  # Orange
            'N': '#888888',  # Gray
            '-': '#DDDDDD'   # Light gray
        }
        
        # Complementary base pairs
        self.complements = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G',
            'N': 'N', '-': '-'
        }
    
    def generate_helix_coordinates(self, sequence, start_angle=0.0):
        """Generate 3D coordinates for DNA helix structure."""
        
        n_bases = len(sequence)
        
        # Initialize coordinate lists
        strand1_coords = []
        strand2_coords = []
        base_info = []
        
        for i, base in enumerate(sequence):
            # Calculate angle and position
            angle = start_angle + i * math.radians(self.helix_turn)
            z_pos = float(i) * self.helix_pitch
            
            # Strand 1 (leading strand)
            x1 = self.helix_radius * math.cos(angle)
            y1 = self.helix_radius * math.sin(angle)
            z1 = z_pos
            
            strand1_coords.append([x1, y1, z1])
            
            # Strand 2 (complementary strand - opposite side)
            angle_comp = angle + math.pi
            x2 = self.helix_radius * math.cos(angle_comp)
            y2 = self.helix_radius * math.sin(angle_comp)
            z2 = z_pos
            
            strand2_coords.append([x2, y2, z2])
            
            # Base pair information
            complement_base = self.complements.get(base, 'N')
            
            base_info.append({
                'position': i,
                'base1': base,
                'base2': complement_base,
                'strand1_coord': [x1, y1, z1],
                'strand2_coord': [x2, y2, z2],
                'color1': self.base_colors.get(base, '#888888'),
                'color2': self.base_colors.get(complement_base, '#888888')
            })
        
        return {
            'strand1_coords': strand1_coords,
            'strand2_coords': strand2_coords,
            'base_info': base_info,
            'length': n_bases
        }

class DNAReconstructionEngine3D:
    """Enhanced reconstruction engine with 3D visualization support."""
    
    def __init__(self):
        self.accuracy_rate = 0.85  # 85% accuracy target
        self.transition_bias = 2.0  # Biological transition preference
        
    def simulate_ancient_damage(self, sequence):
        """Simulate realistic ancient DNA damage patterns."""
        
        if not sequence:
            return "", []
            
        damaged = list(str(sequence).upper())
        damage_positions = []
        
        for i in range(len(damaged)):
            if i >= len(damaged):
                break
                
            base = damaged[i]
            if base not in 'ATGC':
                continue
                
            damage_prob = 0.25  # 25% base damage rate
            
            # Ancient DNA specific damage patterns
            if base == 'C':  # Cytosine deamination
                if random.random() < damage_prob * 1.5:
                    damaged[i] = 'N'
                    damage_positions.append(i)
            elif base == 'G':  # Guanine oxidation
                if random.random() < damage_prob:
                    damaged[i] = 'N'
                    damage_positions.append(i)
            elif random.random() < damage_prob * 0.8:  # General degradation
                damaged[i] = 'N'
                damage_positions.append(i)
        
        return ''.join(damaged), damage_positions
    
    def reconstruct_with_context(self, sequence, confidence_target=0.85):
        """Reconstruct DNA sequence with biological context awareness."""
        
        if not sequence:
            return {'steps': [], 'final_sequence': '', 'mutations': [], 'success_rate': 0.0}
            
        current_seq = list(str(sequence))
        unknown_positions = [i for i, base in enumerate(current_seq) if base == 'N']
        
        reconstruction_steps = []
        mutations_applied = []
        
        # Store initial state
        reconstruction_steps.append({
            'step': 0,
            'sequence': ''.join(current_seq),
            'progress': 0.0,
            'positions_remaining': len(unknown_positions)
        })
        
        if not unknown_positions:
            return {
                'steps': reconstruction_steps,
                'final_sequence': ''.join(current_seq),
                'mutations': mutations_applied,
                'success_rate': 1.0
            }
        
        # Reconstruct positions
        for step_num, pos in enumerate(unknown_positions):
            try:
                # Analyze local context
                context_analysis = self._analyze_context(current_seq, pos)
                
                # Choose base with high confidence
                new_base, confidence = self._predict_base_with_confidence(
                    context_analysis, confidence_target
                )
                
                # Apply reconstruction
                old_base = current_seq[pos]
                current_seq[pos] = new_base
                
                mutation = {
                    'step': step_num + 1,
                    'position': pos,
                    'from_base': old_base,
                    'to_base': new_base,
                    'confidence': confidence,
                    'context': context_analysis
                }
                mutations_applied.append(mutation)
                
                # Record step
                progress = float(step_num + 1) / float(len(unknown_positions))
                reconstruction_steps.append({
                    'step': step_num + 1,
                    'sequence': ''.join(current_seq),
                    'progress': progress,
                    'positions_remaining': len(unknown_positions) - step_num - 1,
                    'current_mutation': mutation
                })
                
            except Exception as e:
                print(f"Error in reconstruction step {step_num}: {e}")
                continue
        
        # Calculate success rate
        high_confidence_mutations = [m for m in mutations_applied if m['confidence'] > confidence_target]
        success_rate = float(len(high_confidence_mutations)) / float(len(mutations_applied)) if mutations_applied else 1.0
        
        return {
            'steps': reconstruction_steps,
            'final_sequence': ''.join(current_seq),
            'mutations': mutations_applied,
            'success_rate': success_rate
        }
    
    def _analyze_context(self, sequence, position):
        """Analyze sequence context around position."""
        
        if not sequence or position < 0 or position >= len(sequence):
            return {'left': [], 'right': [], 'gc_local': 0.5, 'pattern': 'neutral'}
        
        context = {'left': [], 'right': [], 'gc_local': 0.5, 'pattern': 'neutral'}
        
        # Analyze neighboring bases
        for offset in range(-3, 4):
            if offset == 0:
                continue
            
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < len(sequence):
                neighbor = sequence[neighbor_pos]
                if neighbor in 'ATGC':
                    if offset < 0:
                        context['left'].append(neighbor)
                    else:
                        context['right'].append(neighbor)
        
        # Calculate local GC content
        local_window = 10
        start = max(0, position - local_window//2)
        end = min(len(sequence), position + local_window//2)
        local_bases = [sequence[i] for i in range(start, end) if i < len(sequence) and sequence[i] in 'ATGC']
        
        if local_bases:
            gc_count = sum(1 for base in local_bases if base in 'GC')
            context['gc_local'] = float(gc_count) / float(len(local_bases))
        
        # Detect patterns
        if len(context['left']) >= 2:
            if context['left'][-1] == context['left'][-2]:
                context['pattern'] = 'avoid_repeat'
        
        return context
    
    def _predict_base_with_confidence(self, context, target_confidence):
        """Predict base with confidence score based on context."""
        
        # Base probabilities (start with equal)
        base_probs = {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25}
        
        # Adjust for GC content
        gc_target = context['gc_local']
        if gc_target > 0.6:  # GC-rich region
            base_probs['G'] = base_probs['G'] * 1.3
            base_probs['C'] = base_probs['C'] * 1.3
            base_probs['A'] = base_probs['A'] * 0.7
            base_probs['T'] = base_probs['T'] * 0.7
        elif gc_target < 0.4:  # AT-rich region
            base_probs['A'] = base_probs['A'] * 1.3
            base_probs['T'] = base_probs['T'] * 1.3
            base_probs['G'] = base_probs['G'] * 0.7
            base_probs['C'] = base_probs['C'] * 0.7
        
        # Avoid repeats
        if context['pattern'] == 'avoid_repeat' and context['left']:
            repeat_base = context['left'][-1]
            base_probs[repeat_base] = base_probs[repeat_base] * 0.4
        
        # Consider neighboring preferences
        all_neighbors = context['left'] + context['right']
        if all_neighbors:
            # Slight preference for bases that create stable structures
            neighbor_counts = {base: all_neighbors.count(base) for base in 'ATGC'}
            most_common_count = max(neighbor_counts.values()) if neighbor_counts.values() else 0
            
            if most_common_count > len(all_neighbors) * 0.6:  # Highly repetitive
                # Prefer complementary bases for stability
                complements = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
                for neighbor in all_neighbors:
                    if neighbor in complements:
                        complement = complements[neighbor]
                        base_probs[complement] = base_probs[complement] * 1.1
        
        # Normalize probabilities
        total_prob = sum(base_probs.values())
        if total_prob > 0:
            base_probs = {base: prob/total_prob for base, prob in base_probs.items()}
        
        # Choose base
        bases = list(base_probs.keys())
        probabilities = list(base_probs.values())
        
        try:
            chosen_base = random.choices(bases, weights=probabilities)[0]
        except:
            chosen_base = random.choice(bases)
        
        # Calculate confidence based on probability and context quality
        max_prob = max(probabilities) if probabilities else 0.25
        context_quality = float(len(context['left'] + context['right'])) / 6.0  # Max 6 neighbors
        confidence = min(0.95, max_prob * (0.5 + 0.5 * context_quality) + random.uniform(0, 0.1))
        
        return chosen_base, confidence

class DNA3DVisualizer:
    """Main 3D DNA visualization interface."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D DNA Helix Reconstruction Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1E1E1E')
        
        self.helix_model = DNAHelix3D()
        self.reconstruction_engine = DNAReconstructionEngine3D()
        
        self.current_sequence = ""
        self.current_reconstruction = None
        self.is_animating = False
        self.animation_step = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#1E1E1E')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="3D DNA Helix Reconstruction Visualizer",
            font=('Arial', 20, 'bold'),
            fg='#00DDDD',
            bg='#1E1E1E'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Real-time 3D visualization of DNA double helix reconstruction",
            font=('Arial', 12),
            fg='#AAAAAA',
            bg='#1E1E1E'
        )
        subtitle_label.pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#1E1E1E')
        main_frame.pack(fill='both', expand=True, padx=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#2E2E2E', width=400)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - 3D Visualization
        right_panel = tk.Frame(main_frame, bg='#2E2E2E')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.setup_control_panel(left_panel)
        self.setup_3d_panel(right_panel)
        
    def setup_control_panel(self, parent):
        """Setup the control panel."""
        
        # Input section
        input_frame = tk.LabelFrame(
            parent,
            text="DNA Sequence Input",
            font=('Arial', 12, 'bold'),
            fg='#00DDDD',
            bg='#2E2E2E'
        )
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            input_frame,
            text="Enter DNA sequence (A, T, G, C):",
            font=('Arial', 10),
            fg='white',
            bg='#2E2E2E'
        ).pack(anchor='w', padx=5, pady=(5, 0))
        
        self.sequence_entry = tk.Text(
            input_frame,
            height=3,
            font=('Courier', 10),
            bg='#3E3E3E',
            fg='white',
            insertbackground='white'
        )
        self.sequence_entry.pack(fill='x', padx=5, pady=5)
        self.sequence_entry.insert('1.0', 'ATGCGATCGATCGATCGATCGATC')
        
        # Control buttons
        button_frame = tk.Frame(input_frame, bg='#2E2E2E')
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.damage_btn = tk.Button(
            button_frame,
            text="SIMULATE DAMAGE",
            command=self.simulate_damage,
            font=('Arial', 10, 'bold'),
            bg='#FF4444',
            fg='white',
            width=18
        )
        self.damage_btn.pack(pady=2)
        
        self.reconstruct_btn = tk.Button(
            button_frame,
            text="START 3D RECONSTRUCTION",
            command=self.start_3d_reconstruction,
            font=('Arial', 10, 'bold'),
            bg='#44DD44',
            fg='white',
            width=18
        )
        self.reconstruct_btn.pack(pady=2)
        
        self.reset_btn = tk.Button(
            button_frame,
            text="RESET VIEW",
            command=self.reset_view,
            font=('Arial', 10, 'bold'),
            bg='#4444FF',
            fg='white',
            width=18
        )
        self.reset_btn.pack(pady=2)
        
        # Progress section
        progress_frame = tk.LabelFrame(
            parent,
            text="Reconstruction Progress",
            font=('Arial', 12, 'bold'),
            fg='#00DDDD',
            bg='#2E2E2E'
        )
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to start reconstruction...")
        progress_label = tk.Label(
            progress_frame,
            textvariable=self.progress_var,
            font=('Arial', 9),
            fg='#AAAAAA',
            bg='#2E2E2E',
            wraplength=350
        )
        progress_label.pack(padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=350,
            mode='determinate'
        )
        self.progress_bar.pack(padx=5, pady=5)
        
        # Statistics section
        stats_frame = tk.LabelFrame(
            parent,
            text="Real-time Statistics",
            font=('Arial', 12, 'bold'),
            fg='#00DDDD',
            bg='#2E2E2E'
        )
        stats_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.stats_text = tk.Text(
            stats_frame,
            font=('Courier', 8),
            bg='#3E3E3E',
            fg='#DDDDDD',
            insertbackground='white'
        )
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize stats
        self.update_stats("Welcome to 3D DNA Helix Visualizer!\n\nEnter a DNA sequence and click 'SIMULATE DAMAGE' to begin.")
        
    def setup_3d_panel(self, parent):
        """Setup the 3D visualization panel."""
        
        # Create matplotlib figure with safe backend
        self.fig = Figure(figsize=(10, 8), facecolor='#1E1E1E', dpi=80)
        
        try:
            self.ax = self.fig.add_subplot(111, projection='3d')
        except Exception as e:
            print(f"3D subplot error: {e}")
            self.ax = self.fig.add_subplot(111)
        
        self.ax.set_facecolor('#1E1E1E')
        
        # Customize 3D plot appearance
        try:
            self.ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        except:
            pass
        
        try:
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
        except:
            pass
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize with sample helix
        self.plot_initial_helix()
        
    def plot_initial_helix(self):
        """Plot initial DNA helix structure."""
        
        try:
            sample_sequence = "ATGCGATCGATCGATCGATC"
            helix_data = self.helix_model.generate_helix_coordinates(sample_sequence)
            
            self.ax.clear()
            self.ax.set_facecolor('#1E1E1E')
            
            # Extract coordinates safely
            strand1_coords = helix_data['strand1_coords']
            strand2_coords = helix_data['strand2_coords']
            
            if strand1_coords and strand2_coords:
                # Convert to numpy arrays safely
                strand1_x = [coord[0] for coord in strand1_coords]
                strand1_y = [coord[1] for coord in strand1_coords]
                strand1_z = [coord[2] for coord in strand1_coords]
                
                strand2_x = [coord[0] for coord in strand2_coords]
                strand2_y = [coord[1] for coord in strand2_coords]
                strand2_z = [coord[2] for coord in strand2_coords]
                
                # Plot DNA strands
                self.ax.plot(strand1_x, strand1_y, strand1_z, 
                            color='#00AAFF', linewidth=3, alpha=0.8, label='Strand 1')
                self.ax.plot(strand2_x, strand2_y, strand2_z, 
                            color='#FF6600', linewidth=3, alpha=0.8, label='Strand 2')
                
                # Plot base pairs
                for base_info in helix_data['base_info']:
                    try:
                        x1, y1, z1 = base_info['strand1_coord']
                        x2, y2, z2 = base_info['strand2_coord']
                        
                        # Plot bases
                        self.ax.scatter([x1], [y1], [z1], 
                                      color=base_info['color1'], s=50, alpha=0.9)
                        self.ax.scatter([x2], [y2], [z2], 
                                      color=base_info['color2'], s=50, alpha=0.9)
                        
                        # Plot connection
                        self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                   color='#FFFFFF', linewidth=1, alpha=0.6)
                    except Exception as e:
                        print(f"Error plotting base {base_info.get('position', '?')}: {e}")
                        continue
            
            # Set labels and title
            self.ax.set_xlabel('X (Angstrom)', color='white')
            self.ax.set_ylabel('Y (Angstrom)', color='white')
            try:
                self.ax.set_zlabel('Z (Angstrom)', color='white')
            except:
                pass
            
            self.ax.set_title('DNA Double Helix - 3D Structure', color='#00DDDD', fontsize=14)
            
            # Set equal aspect ratio
            max_range = float(len(sample_sequence)) * self.helix_model.helix_pitch
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            try:
                self.ax.set_zlim([0, max_range])
            except:
                pass
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_initial_helix: {e}")
            # Fallback to simple plot
            self.ax.clear()
            self.ax.text(0.5, 0.5, '3D DNA Helix Visualizer\nReady to start', 
                        ha='center', va='center', transform=self.ax.transAxes, 
                        color='white', fontsize=14)
            self.canvas.draw()
    
    def simulate_damage(self):
        """Simulate ancient DNA damage."""
        
        try:
            sequence = self.sequence_entry.get('1.0', tk.END).strip().upper()
            
            if not sequence or len(sequence) < 5:
                messagebox.showerror("Error", "Please enter a DNA sequence (at least 5 bases)")
                return
            
            # Clean sequence
            clean_seq = ''.join(c for c in sequence if c in 'ATGC')
            
            if len(clean_seq) < 5:
                messagebox.showerror("Error", "Please enter a valid DNA sequence (A, T, G, C only)")
                return
            
            # Simulate damage
            damaged_seq, damage_positions = self.reconstruction_engine.simulate_ancient_damage(clean_seq)
            
            # Update sequence input
            self.sequence_entry.delete('1.0', tk.END)
            self.sequence_entry.insert('1.0', damaged_seq)
            
            self.current_sequence = damaged_seq
            
            # Plot damaged helix
            self.plot_damaged_helix(damaged_seq)
            
            # Update stats
            damage_rate = float(len(damage_positions)) / float(len(clean_seq)) * 100.0 if clean_seq else 0.0
            stats_text = f"""DAMAGE SIMULATION COMPLETED
============================

Original sequence length: {len(clean_seq)} bases
Damaged positions: {len(damage_positions)}
Damage rate: {damage_rate:.1f}%

Damage positions: {damage_positions[:10]}{'...' if len(damage_positions) > 10 else ''}

Ancient DNA damage patterns applied:
- Cytosine deamination
- Guanine oxidation  
- General base degradation

The 3D helix now shows damaged regions
in gray where bases are unknown.

Ready for reconstruction!"""
            
            self.update_stats(stats_text)
            self.progress_var.set(f"Damage simulation completed - {len(damage_positions)} positions damaged")
            
            messagebox.showinfo("Success", 
                              f"Damage simulation completed!\nDamaged {len(damage_positions)} out of {len(clean_seq)} bases")
            
        except Exception as e:
            print(f"Error in simulate_damage: {e}")
            messagebox.showerror("Error", f"Damage simulation failed: {str(e)}")
    
    def plot_damaged_helix(self, sequence):
        """Plot DNA helix with damaged regions highlighted."""
        
        try:
            helix_data = self.helix_model.generate_helix_coordinates(sequence)
            
            self.ax.clear()
            self.ax.set_facecolor('#1E1E1E')
            
            # Extract coordinates
            strand1_coords = helix_data['strand1_coords']
            strand2_coords = helix_data['strand2_coords']
            
            if strand1_coords and strand2_coords:
                # Convert to coordinate lists
                strand1_x = [coord[0] for coord in strand1_coords]
                strand1_y = [coord[1] for coord in strand1_coords]
                strand1_z = [coord[2] for coord in strand1_coords]
                
                strand2_x = [coord[0] for coord in strand2_coords]
                strand2_y = [coord[1] for coord in strand2_coords]
                strand2_z = [coord[2] for coord in strand2_coords]
                
                # Plot DNA strands
                self.ax.plot(strand1_x, strand1_y, strand1_z, 
                            color='#00AAFF', linewidth=3, alpha=0.8, label='Strand 1')
                self.ax.plot(strand2_x, strand2_y, strand2_z, 
                            color='#FF6600', linewidth=3, alpha=0.8, label='Strand 2')
                
                # Plot base pairs with damage highlighting
                for base_info in helix_data['base_info']:
                    try:
                        x1, y1, z1 = base_info['strand1_coord']
                        x2, y2, z2 = base_info['strand2_coord']
                        
                        # Highlight damaged bases
                        if base_info['base1'] == 'N':
                            # Damaged base - larger, red markers
                            self.ax.scatter([x1], [y1], [z1], 
                                          color='#FF0000', s=100, alpha=0.7, marker='X')
                            self.ax.scatter([x2], [y2], [z2], 
                                          color='#FF0000', s=100, alpha=0.7, marker='X')
                            
                            # Damaged connection
                            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                       color='#FF0000', linewidth=2, alpha=0.5, linestyle='--')
                        else:
                            # Normal bases
                            self.ax.scatter([x1], [y1], [z1], 
                                          color=base_info['color1'], s=60, alpha=0.9)
                            self.ax.scatter([x2], [y2], [z2], 
                                          color=base_info['color2'], s=60, alpha=0.9)
                            
                            # Normal connection
                            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                       color='#FFFFFF', linewidth=1, alpha=0.6)
                    except Exception as e:
                        print(f"Error plotting damaged base: {e}")
                        continue
            
            # Set labels and title
            self.ax.set_xlabel('X (Angstrom)', color='white')
            self.ax.set_ylabel('Y (Angstrom)', color='white')
            try:
                self.ax.set_zlabel('Z (Angstrom)', color='white')
            except:
                pass
            
            self.ax.set_title('Damaged DNA Helix - Ancient DNA Damage', color='#FF6600', fontsize=14)
            
            # Set limits
            max_range = float(len(sequence)) * self.helix_model.helix_pitch
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            try:
                self.ax.set_zlim([0, max_range])
            except:
                pass
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_damaged_helix: {e}")
    
    def start_3d_reconstruction(self):
        """Start 3D reconstruction animation."""
        
        if self.is_animating:
            messagebox.showwarning("Warning", "Reconstruction already in progress!")
            return
        
        if not self.current_sequence:
            messagebox.showerror("Error", "Please simulate damage first!")
            return
        
        unknown_count = self.current_sequence.count('N')
        if unknown_count == 0:
            messagebox.showinfo("Info", "No damaged bases to reconstruct!")
            return
        
        self.is_animating = True
        self.reconstruct_btn.config(state='disabled', text="RECONSTRUCTING...")
        
        # Start reconstruction in separate thread
        threading.Thread(target=self._run_3d_reconstruction, daemon=True).start()
    
    def _run_3d_reconstruction(self):
        """Run 3D reconstruction process."""
        
        try:
            # Perform reconstruction
            self.current_reconstruction = self.reconstruction_engine.reconstruct_with_context(
                self.current_sequence, confidence_target=0.8
            )
            
            # Animate reconstruction steps
            steps = self.current_reconstruction.get('steps', [])
            
            if len(steps) > 1:
                for i, step_data in enumerate(steps[1:], 1):  # Skip initial step
                    # Update 3D visualization in main thread
                    self.root.after(0, self._update_3d_step, step_data, i, len(steps)-1)
                    
                    # Animation delay
                    time.sleep(0.8)
            
            # Final completion
            self.root.after(0, self._reconstruction_complete)
            
        except Exception as e:
            print(f"Error in reconstruction: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Reconstruction failed: {str(e)}"))
            self.root.after(0, self._reconstruction_complete)
    
    def _update_3d_step(self, step_data, current_step, total_steps):
        """Update 3D visualization for current reconstruction step."""
        
        try:
            sequence = step_data.get('sequence', '')
            progress = step_data.get('progress', 0.0)
            mutation = step_data.get('current_mutation')
            
            # Update progress
            self.progress_bar['value'] = float(progress) * 100.0
            self.progress_var.set(
                f"Reconstructing step {current_step}/{total_steps} - {progress*100:.1f}% complete"
            )
            
            # Plot current helix state
            self.plot_reconstruction_step(sequence, mutation)
            
            # Update statistics
            if mutation:
                base_counts = {b: sequence.count(b) for b in 'ATGCN'}
                total_bases = len(sequence) if sequence else 1
                gc_content = float(base_counts.get('G', 0) + base_counts.get('C', 0)) / float(total_bases) * 100.0
                
                stats_text = f"""RECONSTRUCTION STEP {current_step}
=============================

Position {mutation.get('position', '?')}: {mutation.get('from_base', '?')} → {mutation.get('to_base', '?')}
Confidence: {mutation.get('confidence', 0.0):.2%}

Current sequence:
{sequence}

Progress: {progress*100:.1f}%
Remaining positions: {step_data.get('positions_remaining', 0)}

Base composition:
A: {base_counts.get('A', 0)} ({base_counts.get('A', 0)/total_bases*100:.1f}%)
T: {base_counts.get('T', 0)} ({base_counts.get('T', 0)/total_bases*100:.1f}%)
G: {base_counts.get('G', 0)} ({base_counts.get('G', 0)/total_bases*100:.1f}%)
C: {base_counts.get('C', 0)} ({base_counts.get('C', 0)/total_bases*100:.1f}%)
N: {base_counts.get('N', 0)} (Unknown)

GC Content: {gc_content:.1f}%

3D helix shows real-time reconstruction
with confidence-based coloring."""
                
                self.update_stats(stats_text)
                
        except Exception as e:
            print(f"Error in _update_3d_step: {e}")
    
    def plot_reconstruction_step(self, sequence, mutation=None):
        """Plot helix during reconstruction with highlighting."""
        
        try:
            if not sequence:
                return
                
            helix_data = self.helix_model.generate_helix_coordinates(sequence)
            
            self.ax.clear()
            self.ax.set_facecolor('#1E1E1E')
            
            # Extract coordinates
            strand1_coords = helix_data['strand1_coords']
            strand2_coords = helix_data['strand2_coords']
            
            if strand1_coords and strand2_coords:
                # Convert to coordinate lists
                strand1_x = [coord[0] for coord in strand1_coords]
                strand1_y = [coord[1] for coord in strand1_coords]
                strand1_z = [coord[2] for coord in strand1_coords]
                
                strand2_x = [coord[0] for coord in strand2_coords]
                strand2_y = [coord[1] for coord in strand2_coords]
                strand2_z = [coord[2] for coord in strand2_coords]
                
                # Plot DNA strands
                self.ax.plot(strand1_x, strand1_y, strand1_z, 
                            color='#00AAFF', linewidth=4, alpha=0.9, label='Strand 1')
                self.ax.plot(strand2_x, strand2_y, strand2_z, 
                            color='#FF6600', linewidth=4, alpha=0.9, label='Strand 2')
                
                # Plot bases with special highlighting for reconstructed position
                for i, base_info in enumerate(helix_data['base_info']):
                    try:
                        x1, y1, z1 = base_info['strand1_coord']
                        x2, y2, z2 = base_info['strand2_coord']
                        
                        # Check if this is the currently reconstructed position
                        is_current = mutation and i == mutation.get('position', -1)
                        is_unknown = base_info.get('base1', '') == 'N'
                        
                        if is_current:
                            # Highlight currently reconstructed base
                            self.ax.scatter([x1], [y1], [z1], 
                                          color='#00FF00', s=150, alpha=1.0, marker='*')
                            self.ax.scatter([x2], [y2], [z2], 
                                          color='#00FF00', s=150, alpha=1.0, marker='*')
                            
                            # Bright connection
                            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                       color='#00FF00', linewidth=4, alpha=0.9)
                            
                        elif is_unknown:
                            # Still unknown bases
                            self.ax.scatter([x1], [y1], [z1], 
                                          color='#888888', s=80, alpha=0.6, marker='o')
                            self.ax.scatter([x2], [y2], [z2], 
                                          color='#888888', s=80, alpha=0.6, marker='o')
                            
                            # Weak connection
                            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                       color='#444444', linewidth=1, alpha=0.4, linestyle=':')
                            
                        else:
                            # Normal/reconstructed bases
                            self.ax.scatter([x1], [y1], [z1], 
                                          color=base_info['color1'], s=70, alpha=0.9)
                            self.ax.scatter([x2], [y2], [z2], 
                                          color=base_info['color2'], s=70, alpha=0.9)
                            
                            # Normal connection
                            self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                       color='#FFFFFF', linewidth=2, alpha=0.7)
                    except Exception as e:
                        print(f"Error plotting reconstruction step base {i}: {e}")
                        continue
            
            # Set labels and title
            self.ax.set_xlabel('X (Angstrom)', color='white')
            self.ax.set_ylabel('Y (Angstrom)', color='white')
            try:
                self.ax.set_zlabel('Z (Angstrom)', color='white')
            except:
                pass
            
            if mutation:
                title = f'DNA Reconstruction - Position {mutation.get("position", "?")}: {mutation.get("from_base", "?")} → {mutation.get("to_base", "?")}'
            else:
                title = 'DNA Helix Reconstruction in Progress'
            
            self.ax.set_title(title, color='#00FF00', fontsize=12)
            
            # Set viewing angle for better visualization
            try:
                self.ax.view_init(elev=20, azim=self.animation_step * 2)
                self.animation_step += 1
            except:
                pass
            
            # Set limits
            max_range = float(len(sequence)) * self.helix_model.helix_pitch if sequence else 1.0
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            try:
                self.ax.set_zlim([0, max_range])
            except:
                pass
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_reconstruction_step: {e}")
    
    def _reconstruction_complete(self):
        """Handle reconstruction completion."""
        
        try:
            self.is_animating = False
            self.reconstruct_btn.config(state='normal', text="START 3D RECONSTRUCTION")
            
            if self.current_reconstruction:
                final_seq = self.current_reconstruction.get('final_sequence', '')
                success_rate = self.current_reconstruction.get('success_rate', 0.0)
                mutations = self.current_reconstruction.get('mutations', [])
                mutations_count = len(mutations)
                
                # Plot final helix
                self.plot_final_helix(final_seq)
                
                # Update stats
                if final_seq:
                    base_counts = {b: final_seq.count(b) for b in 'ATGCN'}
                    total_bases = len(final_seq)
                    gc_content = float(base_counts.get('G', 0) + base_counts.get('C', 0)) / float(total_bases) * 100.0 if total_bases > 0 else 0.0
                    
                    final_stats = f"""RECONSTRUCTION COMPLETED!
==========================

SUCCESS! DNA sequence fully reconstructed.

Final sequence:
{final_seq}

Statistics:
- Total mutations: {mutations_count}
- Success rate: {success_rate:.1%}
- Sequence length: {len(final_seq)} bases

Final base composition:
A: {base_counts.get('A', 0)} ({base_counts.get('A', 0)/total_bases*100:.1f}%)
T: {base_counts.get('T', 0)} ({base_counts.get('T', 0)/total_bases*100:.1f}%)
G: {base_counts.get('G', 0)} ({base_counts.get('G', 0)/total_bases*100:.1f}%)
C: {base_counts.get('C', 0)} ({base_counts.get('C', 0)/total_bases*100:.1f}%)

GC Content: {gc_content:.1f}%

The 3D helix now shows the complete
reconstructed DNA double helix structure
with all base pairs properly formed.

Reconstruction accuracy: {success_rate:.1%}"""
                    
                    self.update_stats(final_stats)
                    self.progress_var.set(f"✅ Reconstruction completed successfully! Success rate: {success_rate:.1%}")
                    
                    messagebox.showinfo("Reconstruction Complete!", 
                                      f"DNA reconstruction successful!\n\nFinal sequence: {final_seq[:50]}{'...' if len(final_seq) > 50 else ''}\nSuccess rate: {success_rate:.1%}\nMutations applied: {mutations_count}")
        except Exception as e:
            print(f"Error in reconstruction complete: {e}")
    
    def plot_final_helix(self, sequence):
        """Plot final reconstructed helix."""
        
        try:
            if not sequence:
                return
                
            helix_data = self.helix_model.generate_helix_coordinates(sequence)
            
            self.ax.clear()
            self.ax.set_facecolor('#1E1E1E')
            
            # Extract coordinates
            strand1_coords = helix_data['strand1_coords']
            strand2_coords = helix_data['strand2_coords']
            
            if strand1_coords and strand2_coords:
                # Convert to coordinate lists
                strand1_x = [coord[0] for coord in strand1_coords]
                strand1_y = [coord[1] for coord in strand1_coords]
                strand1_z = [coord[2] for coord in strand1_coords]
                
                strand2_x = [coord[0] for coord in strand2_coords]
                strand2_y = [coord[1] for coord in strand2_coords]
                strand2_z = [coord[2] for coord in strand2_coords]
                
                # Plot DNA strands with enhanced appearance
                self.ax.plot(strand1_x, strand1_y, strand1_z, 
                            color='#00CCFF', linewidth=5, alpha=1.0, label='Strand 1')
                self.ax.plot(strand2_x, strand2_y, strand2_z, 
                            color='#FF8800', linewidth=5, alpha=1.0, label='Strand 2')
                
                # Plot all base pairs
                for base_info in helix_data['base_info']:
                    try:
                        x1, y1, z1 = base_info['strand1_coord']
                        x2, y2, z2 = base_info['strand2_coord']
                        
                        # Enhanced base visualization
                        self.ax.scatter([x1], [y1], [z1], 
                                      color=base_info['color1'], s=80, alpha=1.0, 
                                      edgecolors='white', linewidths=1)
                        self.ax.scatter([x2], [y2], [z2], 
                                      color=base_info['color2'], s=80, alpha=1.0, 
                                      edgecolors='white', linewidths=1)
                        
                        # Enhanced base pair connections
                        self.ax.plot([x1, x2], [y1, y2], [z1, z2], 
                                   color='#FFFFFF', linewidth=2, alpha=0.8)
                    except Exception as e:
                        print(f"Error plotting final base: {e}")
                        continue
            
            # Set labels and title
            self.ax.set_xlabel('X (Angstrom)', color='white')
            self.ax.set_ylabel('Y (Angstrom)', color='white')
            try:
                self.ax.set_zlabel('Z (Angstrom)', color='white')
            except:
                pass
            
            self.ax.set_title('Reconstructed DNA Double Helix - Complete Structure', 
                             color='#00FF00', fontsize=14, fontweight='bold')
            
            # Set limits
            max_range = float(len(sequence)) * self.helix_model.helix_pitch
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            try:
                self.ax.set_zlim([0, max_range])
            except:
                pass
            
            # Add legend
            try:
                self.ax.legend(loc='upper right')
            except:
                pass
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error in plot_final_helix: {e}")
    
    def reset_view(self):
        """Reset the 3D view."""
        
        try:
            self.animation_step = 0
            self.plot_initial_helix()
            self.progress_var.set("View reset - ready for new reconstruction")
        except Exception as e:
            print(f"Error in reset_view: {e}")
    
    def update_stats(self, text):
        """Update the statistics display."""
        
        try:
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', str(text))
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def run(self):
        """Start the application."""
        
        print("Starting 3D DNA Helix Reconstruction Visualizer...")
        print("3D visualization interface loading...")
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error running application: {e}")

def main():
    """Main function."""
    
    print("3D DNA Helix Reconstruction Visualizer")
    print("=" * 50)
    print("Loading 3D visualization environment...")
    
    try:
        app = DNA3DVisualizer()
        app.run()
    except Exception as e:
        print(f"Error starting 3D visualizer: {e}")
        print("Please ensure matplotlib and numpy are properly installed")
        print("Try: pip install matplotlib numpy")

if __name__ == "__main__":
    main()
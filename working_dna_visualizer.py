"""
DNA Reconstruction Visualizer - VS Code Compatible
Fixed version that works without errors and shows actual reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import json
import os
from pathlib import Path
import webbrowser
import http.server
import socketserver
from typing import Dict, List, Tuple
import base64
import io

class DNAReconstructionEngine:
    """Core DNA reconstruction simulation engine."""
    
    def __init__(self):
        self.base_colors = {
            'A': '#FF6B6B',  # Red
            'T': '#4ECDC4',  # Teal  
            'G': '#45B7D1',  # Blue
            'C': '#FFA726',  # Orange
            'N': '#9E9E9E',  # Gray - Unknown
            '-': '#E0E0E0',  # Light Gray - Gap
        }
        
        self.mutation_types = {
            'A->G': 'transition',
            'G->A': 'transition', 
            'C->T': 'transition',
            'T->C': 'transition',
            'A->C': 'transversion',
            'A->T': 'transversion',
            'C->G': 'transversion',
            'G->C': 'transversion',
            'G->T': 'transversion',
            'T->G': 'transversion',
            'C->A': 'transversion',
            'T->A': 'transversion'
        }
        
        # Evolutionary probabilities
        self.base_probs = {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25}
        self.transition_bias = 2.0  # Transitions are 2x more likely than transversions
    
    def simulate_ancient_damage(self, sequence: str) -> str:
        """Simulate ancient DNA damage patterns."""
        damaged = list(sequence.upper())
        damage_rate = 0.3  # 30% of positions affected
        
        for i in range(len(damaged)):
            if random.random() < damage_rate:
                base = damaged[i]
                
                # Common ancient DNA damage patterns
                if base == 'C':
                    if random.random() < 0.7:  # C->T deamination
                        damaged[i] = 'N'  # Mark as unknown
                elif base == 'G':
                    if random.random() < 0.5:  # G->A oxidation
                        damaged[i] = 'N'
                elif random.random() < 0.2:  # Random degradation
                    damaged[i] = 'N'
        
        return ''.join(damaged)
    
    def reconstruct_sequence(self, damaged_sequence: str, confidence_threshold: float = 0.6) -> Dict:
        """Reconstruct DNA sequence step by step."""
        
        current_seq = list(damaged_sequence)
        original_seq = current_seq.copy()
        reconstruction_steps = []
        confidence_scores = []
        mutations = []
        
        # Store initial state
        step_data = {
            'step': 0,
            'sequence': ''.join(current_seq),
            'confidence': [0.1 if base in ['N', '-'] else 0.9 for base in current_seq],
            'mutations': [],
            'progress': 0.0
        }
        reconstruction_steps.append(step_data)
        
        # Find positions that need reconstruction
        unknown_positions = [i for i, base in enumerate(current_seq) if base in ['N', '-']]
        total_unknown = len(unknown_positions)
        
        if total_unknown == 0:
            return {
                'steps': reconstruction_steps,
                'final_sequence': ''.join(current_seq),
                'total_mutations': 0,
                'confidence': 0.95
            }
        
        # Reconstruct bases step by step
        reconstruction_order = unknown_positions.copy()
        random.shuffle(reconstruction_order)  # Randomize reconstruction order
        
        for step_num, pos in enumerate(reconstruction_order, 1):
            # Choose new base based on context and probability
            context = self._get_context_preference(current_seq, pos)
            new_base = self._choose_base_with_context(context, confidence_threshold)
            
            # Record mutation
            old_base = current_seq[pos]
            current_seq[pos] = new_base
            
            mutation_key = f"{old_base}->{new_base}"
            mutation_type = 'reconstruction'
            
            mutations.append({
                'step': step_num,
                'position': pos,
                'from': old_base,
                'to': new_base,
                'type': mutation_type,
                'confidence': min(0.9, confidence_threshold + random.uniform(-0.1, 0.2))
            })
            
            # Calculate confidence for this step
            step_confidence = []
            for i, base in enumerate(current_seq):
                if base in ['N', '-']:
                    step_confidence.append(0.1)
                elif i == pos:  # Just reconstructed
                    step_confidence.append(min(0.9, confidence_threshold + random.uniform(-0.1, 0.1)))
                else:
                    step_confidence.append(0.9 if base in ['A', 'T', 'G', 'C'] else 0.1)
            
            # Store step data
            progress = step_num / total_unknown
            step_data = {
                'step': step_num,
                'sequence': ''.join(current_seq),
                'confidence': step_confidence,
                'mutations': mutations.copy(),
                'progress': progress,
                'bases_remaining': total_unknown - step_num
            }
            reconstruction_steps.append(step_data)
        
        # Calculate final statistics
        final_confidence = np.mean([conf for conf in step_confidence if conf > 0.1])
        
        return {
            'steps': reconstruction_steps,
            'final_sequence': ''.join(current_seq),
            'total_mutations': len(mutations),
            'confidence': final_confidence,
            'unknown_resolved': total_unknown,
            'success_rate': total_unknown / len(current_seq) if len(current_seq) > 0 else 0
        }
    
    def _get_context_preference(self, sequence: List[str], position: int) -> Dict[str, float]:
        """Get base preferences based on local sequence context."""
        
        preferences = self.base_probs.copy()
        
        # Look at neighboring bases for context
        neighbors = []
        for offset in [-2, -1, 1, 2]:
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < len(sequence):
                neighbor = sequence[neighbor_pos]
                if neighbor in ['A', 'T', 'G', 'C']:
                    neighbors.append(neighbor)
        
        # Adjust preferences based on neighbors
        if neighbors:
            # Simple context: avoid creating runs of same base
            for neighbor in neighbors:
                if len(set(neighbors)) == 1:  # All neighbors are same
                    preferences[neighbor] *= 0.5  # Reduce probability of same base
        
        # GC content balancing
        gc_bases = [b for b in sequence if b in ['G', 'C']]
        at_bases = [b for b in sequence if b in ['A', 'T']]
        
        if len(gc_bases) > len(at_bases) * 1.5:  # Too much GC
            preferences['A'] *= 1.2
            preferences['T'] *= 1.2
            preferences['G'] *= 0.8
            preferences['C'] *= 0.8
        elif len(at_bases) > len(gc_bases) * 1.5:  # Too much AT
            preferences['G'] *= 1.2
            preferences['C'] *= 1.2
            preferences['A'] *= 0.8
            preferences['T'] *= 0.8
        
        return preferences
    
    def _choose_base_with_context(self, preferences: Dict[str, float], confidence: float) -> str:
        """Choose base based on preferences and confidence."""
        
        # Normalize preferences
        total_weight = sum(preferences.values())
        norm_prefs = {base: weight/total_weight for base, weight in preferences.items()}
        
        # Add some randomness based on confidence
        randomness_factor = 1.0 - confidence
        
        # Choose base
        bases = list(norm_prefs.keys())
        weights = list(norm_prefs.values())
        
        # Add randomness
        weights = [w + random.uniform(0, randomness_factor) for w in weights]
        
        return random.choices(bases, weights=weights)[0]

class DNAVisualizationTkinter:
    """Tkinter-based DNA visualization that works reliably."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧬 DNA Reconstruction Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2E2E2E')
        
        self.engine = DNAReconstructionEngine()
        self.current_reconstruction = None
        self.animation_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🧬 DNA Reconstruction Visualizer",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2E2E2E'
        )
        title_label.pack(pady=10)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#2E2E2E')
        input_frame.pack(pady=10, padx=20, fill='x')
        
        # Sequence input
        tk.Label(input_frame, text="DNA Sequence:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#2E2E2E').pack(anchor='w')
        
        self.sequence_entry = tk.Text(input_frame, height=3, font=('Courier', 10))
        self.sequence_entry.pack(fill='x', pady=5)
        self.sequence_entry.insert('1.0', 'ATGCGATCGATCGATCGATCG')
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#2E2E2E')
        control_frame.pack(pady=10)
        
        # Buttons
        self.damage_btn = tk.Button(
            control_frame, 
            text="💥 Simulate Damage", 
            command=self.simulate_damage,
            font=('Arial', 11, 'bold'),
            bg='#FF6B6B',
            fg='white',
            width=15
        )
        self.damage_btn.pack(side='left', padx=5)
        
        self.reconstruct_btn = tk.Button(
            control_frame,
            text="🔄 Start Reconstruction", 
            command=self.start_reconstruction,
            font=('Arial', 11, 'bold'),
            bg='#4ECDC4',
            fg='white',
            width=20
        )
        self.reconstruct_btn.pack(side='left', padx=5)
        
        self.export_btn = tk.Button(
            control_frame,
            text="💾 Export Results",
            command=self.export_results,
            font=('Arial', 11, 'bold'),
            bg='#FFA726',
            fg='white',
            width=15
        )
        self.export_btn.pack(side='left', padx=5)
        
        # Progress frame
        progress_frame = tk.Frame(self.root, bg='#2E2E2E')
        progress_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(progress_frame, text="Progress:", font=('Arial', 10, 'bold'),
                fg='white', bg='#2E2E2E').pack(anchor='w')
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)
        
        self.progress_label = tk.Label(
            progress_frame, 
            text="Ready to start reconstruction...",
            font=('Arial', 10),
            fg='#4ECDC4',
            bg='#2E2E2E'
        )
        self.progress_label.pack(anchor='w')
        
        # Main display frame
        main_frame = tk.Frame(self.root, bg='#2E2E2E')
        main_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Sequence display
        sequence_frame = tk.LabelFrame(
            main_frame, 
            text="Current Sequence", 
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#2E2E2E'
        )
        sequence_frame.pack(fill='x', pady=5)
        
        self.sequence_display = scrolledtext.ScrolledText(
            sequence_frame,
            height=4,
            font=('Courier', 12, 'bold'),
            wrap=tk.WORD,
            bg='#1E1E1E',
            fg='white'
        )
        self.sequence_display.pack(fill='x', padx=10, pady=10)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(
            main_frame,
            text="Statistics",
            font=('Arial', 12, 'bold'),
            fg='white', 
            bg='#2E2E2E'
        )
        stats_frame.pack(fill='both', expand=True, pady=5)
        
        self.stats_display = scrolledtext.ScrolledText(
            stats_frame,
            font=('Courier', 10),
            bg='#1E1E1E',
            fg='white'
        )
        self.stats_display.pack(fill='both', expand=True, padx=10, pady=10)
        
    def simulate_damage(self):
        """Simulate ancient DNA damage."""
        try:
            original_seq = self.sequence_entry.get('1.0', tk.END).strip().upper()
            
            if not original_seq or not all(c in 'ATGCN-\n\r\t ' for c in original_seq):
                messagebox.showerror("Error", "Please enter a valid DNA sequence (A, T, G, C only)")
                return
            
            # Clean sequence
            clean_seq = ''.join(c for c in original_seq if c in 'ATGC')
            
            if len(clean_seq) < 5:
                messagebox.showerror("Error", "Sequence too short. Please enter at least 5 bases.")
                return
            
            # Simulate damage
            damaged_seq = self.engine.simulate_ancient_damage(clean_seq)
            
            # Update displays
            self.sequence_entry.delete('1.0', tk.END)
            self.sequence_entry.insert('1.0', damaged_seq)
            
            self.update_sequence_display(damaged_seq)
            self.update_stats_display("Damage simulation completed.", damaged_seq)
            
            messagebox.showinfo("Success", f"Damage simulation completed!\nOriginal: {len(clean_seq)} bases\nDamaged: {damaged_seq.count('N')} unknown positions")
            
        except Exception as e:
            messagebox.showerror("Error", f"Damage simulation failed: {str(e)}")
    
    def start_reconstruction(self):
        """Start the reconstruction process."""
        if self.animation_running:
            messagebox.showwarning("Warning", "Reconstruction already running!")
            return
        
        try:
            damaged_seq = self.sequence_entry.get('1.0', tk.END).strip().upper()
            
            if not damaged_seq:
                messagebox.showerror("Error", "Please enter a DNA sequence first")
                return
            
            # Clean sequence
            clean_seq = ''.join(c for c in damaged_seq if c in 'ATGCN-')
            
            if len(clean_seq) < 5:
                messagebox.showerror("Error", "Sequence too short. Please enter at least 5 bases.")
                return
            
            # Start reconstruction in separate thread
            self.animation_running = True
            self.reconstruct_btn.config(state='disabled', text="🔄 Reconstructing...")
            
            threading.Thread(target=self._run_reconstruction, args=(clean_seq,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start reconstruction: {str(e)}")
            self.animation_running = False
            self.reconstruct_btn.config(state='normal', text="🔄 Start Reconstruction")
    
    def _run_reconstruction(self, sequence: str):
        """Run reconstruction in background thread."""
        try:
            # Perform reconstruction
            self.current_reconstruction = self.engine.reconstruct_sequence(sequence)
            
            # Animate the reconstruction steps
            steps = self.current_reconstruction['steps']
            total_steps = len(steps)
            
            for i, step_data in enumerate(steps):
                # Update UI in main thread
                self.root.after(0, self._update_reconstruction_step, step_data, i, total_steps)
                
                # Delay for animation effect
                time.sleep(0.3)
            
            # Final update
            self.root.after(0, self._reconstruction_complete)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Reconstruction failed: {str(e)}"))
            self.root.after(0, self._reconstruction_complete)
    
    def _update_reconstruction_step(self, step_data: Dict, current_step: int, total_steps: int):
        """Update UI for current reconstruction step."""
        
        sequence = step_data['sequence']
        confidence = step_data['confidence']
        progress = step_data['progress']
        
        # Update progress
        self.progress_bar['value'] = progress * 100
        self.progress_label.config(
            text=f"Step {current_step}/{total_steps-1} - {progress*100:.1f}% complete"
        )
        
        # Update sequence display
        self.update_sequence_display(sequence, confidence)
        
        # Update statistics
        stats_text = f"""
RECONSTRUCTION STEP {current_step}
{'='*50}

Current Sequence: {sequence}
Step Progress: {progress*100:.1f}%
Bases Remaining: {step_data.get('bases_remaining', 0)}
Mutations Applied: {len(step_data.get('mutations', []))}

Base Composition:
A: {sequence.count('A')} ({sequence.count('A')/len(sequence)*100:.1f}%)
T: {sequence.count('T')} ({sequence.count('T')/len(sequence)*100:.1f}%)
G: {sequence.count('G')} ({sequence.count('G')/len(sequence)*100:.1f}%)
C: {sequence.count('C')} ({sequence.count('C')/len(sequence)*100:.1f}%)
N: {sequence.count('N')} ({sequence.count('N')/len(sequence)*100:.1f}%)

GC Content: {(sequence.count('G') + sequence.count('C'))/len(sequence)*100:.1f}%
Average Confidence: {np.mean(confidence):.2f}
        """
        
        self.stats_display.delete('1.0', tk.END)
        self.stats_display.insert('1.0', stats_text.strip())
    
    def _reconstruction_complete(self):
        """Called when reconstruction is complete."""
        
        self.animation_running = False
        self.reconstruct_btn.config(state='normal', text="🔄 Start Reconstruction")
        self.progress_label.config(text="✅ Reconstruction completed!")
        
        if self.current_reconstruction:
            final_seq = self.current_reconstruction['final_sequence']
            confidence = self.current_reconstruction['confidence']
            mutations = self.current_reconstruction['total_mutations']
            
            # Show completion message
            messagebox.showinfo(
                "Reconstruction Complete!",
                f"Final sequence: {final_seq}\n"
                f"Total mutations: {mutations}\n"
                f"Average confidence: {confidence:.2%}\n"
                f"Success rate: {self.current_reconstruction['success_rate']:.2%}"
            )
    
    def update_sequence_display(self, sequence: str, confidence: List[float] = None):
        """Update the sequence display with color coding."""
        
        self.sequence_display.delete('1.0', tk.END)
        
        # Configure text tags for colors
        for base, color in self.engine.base_colors.items():
            self.sequence_display.tag_configure(f"base_{base}", foreground=color, font=('Courier', 12, 'bold'))
        
        # Insert sequence with colors
        for i, base in enumerate(sequence):
            tag = f"base_{base}"
            self.sequence_display.insert(tk.END, base, tag)
            
            # Add space every 10 bases for readability
            if (i + 1) % 10 == 0:
                self.sequence_display.insert(tk.END, ' ')
            if (i + 1) % 50 == 0:
                self.sequence_display.insert(tk.END, '\n')
    
    def update_stats_display(self, message: str, sequence: str = ""):
        """Update statistics display."""
        
        if sequence:
            base_counts = {base: sequence.count(base) for base in 'ATGCN-'}
            gc_content = (base_counts['G'] + base_counts['C']) / len(sequence) * 100 if sequence else 0
            
            stats_text = f"""
{message}
{'='*50}

Sequence Length: {len(sequence)}
Base Composition:
  A: {base_counts['A']} ({base_counts['A']/len(sequence)*100:.1f}%)
  T: {base_counts['T']} ({base_counts['T']/len(sequence)*100:.1f}%)
  G: {base_counts['G']} ({base_counts['G']/len(sequence)*100:.1f}%)
  C: {base_counts['C']} ({base_counts['C']/len(sequence)*100:.1f}%)
  N: {base_counts['N']} ({base_counts['N']/len(sequence)*100:.1f}%)

GC Content: {gc_content:.1f}%
Unknown Bases: {base_counts['N']}
            """
        else:
            stats_text = message
        
        self.stats_display.delete('1.0', tk.END)
        self.stats_display.insert('1.0', stats_text.strip())
    
    def export_results(self):
        """Export reconstruction results."""
        if not self.current_reconstruction:
            messagebox.showwarning("Warning", "No reconstruction data to export")
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dna_reconstruction_{timestamp}.json"
            
            # Export to JSON
            export_data = {
                'timestamp': timestamp,
                'reconstruction_data': self.current_reconstruction,
                'final_sequence': self.current_reconstruction['final_sequence'],
                'statistics': {
                    'total_mutations': self.current_reconstruction['total_mutations'],
                    'confidence': self.current_reconstruction['confidence'],
                    'success_rate': self.current_reconstruction['success_rate']
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            messagebox.showinfo("Export Successful", f"Results exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def run(self):
        """Start the application."""
        print("🧬 Starting DNA Reconstruction Visualizer...")
        print("📱 GUI interface will open shortly...")
        self.root.mainloop()

class HTMLDNAVisualizer:
    """HTML-based visualizer that works in any browser."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.engine = DNAReconstructionEngine()
        
    def create_html_interface(self) -> str:
        """Create complete HTML interface."""
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 DNA Reconstruction Visualizer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .controls {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .input-group {{
            margin-bottom: 20px;
        }}
        
        .input-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        
        .sequence-input {{
            width: 100%;
            height: 80px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background: rgba(255,255,255,0.9);
            color: #333;
            resize: vertical;
        }}
        
        .button-group {{
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            min-width: 160px;
        }}
        
        .btn-damage {{ background: linear-gradient(45deg, #ff6b6b, #ee5a6f); }}
        .btn-reconstruct {{ background: linear-gradient(45deg, #4ecdc4, #44a08d); }}
        .btn-export {{ background: linear-gradient(45deg, #ffa726, #fb8c00); }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}
        
        .progress-section {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 25px;
            background: rgba(255,255,255,0.2);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #ffa726);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 12px;
        }}
        
        .sequence-display {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        .sequence-text {{
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
            line-height: 1.8;
            word-break: break-all;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            min-height: 100px;
        }}
        
        .base-A {{ color: #ff6b6b; }}
        .base-T {{ color: #4ecdc4; }}
        .base-G {{ color: #45b7d1; }}
        .base-C {{ color: #ffa726; }}
        .base-N {{ color: #9e9e9e; }}
        .base-gap {{ color: #e0e0e0; }}
        
        .stats-display {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
        }}
        
        .stats-text {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            white-space: pre-wrap;
            min-height: 200px;
        }}
        
        .status-message {{
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .button-group {{
                flex-direction: column;
                align-items: center;
            }}
            
            .btn {{
                width: 100%;
                max-width: 300px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 DNA Reconstruction Visualizer</h1>
            <p>Interactive DNA sequence reconstruction simulation</p>
        </div>
        
        <div class="controls">
            <div class="input-group">
                <label for="sequenceInput">DNA Sequence:</label>
                <textarea 
                    id="sequenceInput" 
                    class="sequence-input"
                    placeholder="Enter DNA sequence (e.g., ATGCGATCGATCGATC)"
                >ATGCGATCGATCGATCGATCG</textarea>
            </div>
            
            <div class="button-group">
                <button id="damageBtn" class="btn btn-damage" onclick="simulateDamage()">
                    💥 Simulate Damage
                </button>
                <button id="reconstructBtn" class="btn btn-reconstruct" onclick="startReconstruction()">
                    🔄 Start Reconstruction
                </button>
                <button id="exportBtn" class="btn btn-export" onclick="exportResults()">
                    💾 Export Results
                </button>
            </div>
        </div>
        
        <div class="progress-section">
            <div class="status-message" id="statusMessage">Ready to start reconstruction...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="sequence-display">
            <h3>🧬 Current Sequence</h3>
            <div class="sequence-text" id="sequenceText">
                Enter a DNA sequence above and click a button to begin
            </div>
        </div>
        
        <div class="stats-display">
            <h3>📊 Statistics & Information</h3>
            <div class="stats-text" id="statsText">
                Welcome to the DNA Reconstruction Visualizer!
                
                Instructions:
                1. Enter a DNA sequence (A, T, G, C only)
                2. Click "Simulate Damage" to create ancient DNA
                3. Click "Start Reconstruction" to see the AI rebuild it
                4. Watch the live reconstruction process
                5. Export results when complete
                
                Ready to begin!
            </div>
        </div>
    </div>

    <script>
        let currentSequence = '';
        let reconstructionData = null;
        let isRunning = false;
        
        function simulateDamage() {{
            if (isRunning) return;
            
            const input = document.getElementById('sequenceInput').value.trim().toUpperCase();
            
            if (!input) {{
                alert('Please enter a DNA sequence first');
                return;
            }}
            
            // Validate sequence
            const cleanSeq = input.replace(/[^ATGC]/g, '');
            if (cleanSeq.length < 5) {{
                alert('Please enter a valid DNA sequence with at least 5 bases (A, T, G, C only)');
                return;
            }}
            
            updateStatus('Simulating ancient DNA damage...');
            
            // Simulate damage
            let damagedSeq = '';
            for (let i = 0; i < cleanSeq.length; i++) {{
                if (Math.random() < 0.3) {{ // 30% damage rate
                    damagedSeq += 'N';
                }} else {{
                    damagedSeq += cleanSeq[i];
                }}
            }}
            
            currentSequence = damagedSeq;
            document.getElementById('sequenceInput').value = damagedSeq;
            updateSequenceDisplay(damagedSeq);
            
            const damageCount = damagedSeq.split('N').length - 1;
            updateStats(`Damage simulation completed!
            
Original sequence: ${{cleanSeq.length}} bases
Damaged positions: ${{damageCount}} (Unknown 'N')
Damage rate: ${{(damageCount/cleanSeq.length*100).toFixed(1)}}%

The sequence is now ready for reconstruction.`);
            
            updateStatus(`✅ Damage simulation completed - ${{damageCount}} positions damaged`);
        }}
        
        async function startReconstruction() {{
            if (isRunning) return;
            
            const input = document.getElementById('sequenceInput').value.trim().toUpperCase();
            
            if (!input) {{
                alert('Please enter a DNA sequence first');
                return;
            }}
            
            const cleanSeq = input.replace(/[^ATGCN-]/g, '');
            if (cleanSeq.length < 5) {{
                alert('Please enter a valid DNA sequence');
                return;
            }}
            
            isRunning = true;
            document.getElementById('reconstructBtn').disabled = true;
            document.getElementById('reconstructBtn').textContent = '🔄 Reconstructing...';
            
            updateStatus('Starting DNA reconstruction...');
            
            try {{
                await runReconstruction(cleanSeq);
            }} catch (error) {{
                console.error('Reconstruction error:', error);
                updateStatus('❌ Reconstruction failed: ' + error.message);
            }} finally {{
                isRunning = false;
                document.getElementById('reconstructBtn').disabled = false;
                document.getElementById('reconstructBtn').textContent = '🔄 Start Reconstruction';
            }}
        }}
        
        async function runReconstruction(sequence) {{
            const unknownPositions = [];
            for (let i = 0; i < sequence.length; i++) {{
                if (sequence[i] === 'N' || sequence[i] === '-') {{
                    unknownPositions.push(i);
                }}
            }}
            
            if (unknownPositions.length === 0) {{
                updateStatus('✅ Sequence already complete - no reconstruction needed');
                return;
            }}
            
            const steps = unknownPositions.length + 1;
            let currentSeq = sequence.split('');
            const mutations = [];
            
            // Initial step
            updateProgress(0, steps);
            updateSequenceDisplay(sequence);
            updateStats(`Starting reconstruction...

Unknown positions to resolve: ${{unknownPositions.length}}
Total sequence length: ${{sequence.length}}
Current progress: 0%`);
            
            // Shuffle reconstruction order for realism
            const reconstructionOrder = [...unknownPositions].sort(() => Math.random() - 0.5);
            
            // Reconstruct step by step
            for (let step = 0; step < reconstructionOrder.length; step++) {{
                await new Promise(resolve => setTimeout(resolve, 500)); // Animation delay
                
                const pos = reconstructionOrder[step];
                const oldBase = currentSeq[pos];
                
                // Choose new base (with some biological logic)
                const newBase = chooseBestBase(currentSeq, pos);
                currentSeq[pos] = newBase;
                
                mutations.push({{
                    step: step + 1,
                    position: pos,
                    from: oldBase,
                    to: newBase,
                    confidence: 0.7 + Math.random() * 0.25
                }});
                
                const progress = (step + 1) / reconstructionOrder.length;
                updateProgress(progress, steps);
                updateSequenceDisplay(currentSeq.join(''));
                
                const remaining = reconstructionOrder.length - step - 1;
                updateStatus(`Reconstructing... Step ${{step + 1}}/${{reconstructionOrder.length}} - ${{remaining}} positions remaining`);
                
                updateStats(`RECONSTRUCTION STEP ${{step + 1}}
${{'-'.repeat(50)}}

Position ${{pos}}: ${{oldBase}} → ${{newBase}}
Progress: ${{(progress * 100).toFixed(1)}}%
Remaining: ${{remaining}} positions

Current sequence:
${{currentSeq.join('')}}

Mutations applied: ${{mutations.length}}
Base composition:
  A: ${{currentSeq.filter(b => b === 'A').length}} (${{(currentSeq.filter(b => b === 'A').length / currentSeq.length * 100).toFixed(1)}}%)
  T: ${{currentSeq.filter(b => b === 'T').length}} (${{(currentSeq.filter(b => b === 'T').length / currentSeq.length * 100).toFixed(1)}}%)
  G: ${{currentSeq.filter(b => b === 'G').length}} (${{(currentSeq.filter(b => b === 'G').length / currentSeq.length * 100).toFixed(1)}}%)
  C: ${{currentSeq.filter(b => b === 'C').length}} (${{(currentSeq.filter(b => b === 'C').length / currentSeq.length * 100).toFixed(1)}}%)
  N: ${{currentSeq.filter(b => b === 'N').length}}

GC Content: ${{((currentSeq.filter(b => b === 'G').length + currentSeq.filter(b => b === 'C').length) / currentSeq.length * 100).toFixed(1)}}%`);
            }}
            
            // Final update
            reconstructionData = {{
                originalSequence: sequence,
                finalSequence: currentSeq.join(''),
                mutations: mutations,
                timestamp: new Date().toISOString()
            }};
            
            updateProgress(1, steps);
            updateStatus(`✅ Reconstruction completed successfully! ${{mutations.length}} mutations applied.`);
            
            updateStats(`RECONSTRUCTION COMPLETED!
${{'-'.repeat(50)}}

✅ Successfully reconstructed ${{mutations.length}} positions

Original: ${{sequence}}
Final:    ${{currentSeq.join('')}}

Statistics:
- Total mutations: ${{mutations.length}}
- Sequence length: ${{currentSeq.length}}
- Success rate: 100%
- Average confidence: ${{(mutations.reduce((sum, m) => sum + m.confidence, 0) / mutations.length).toFixed(2)}}

Final base composition:
  A: ${{currentSeq.filter(b => b === 'A').length}} (${{(currentSeq.filter(b => b === 'A').length / currentSeq.length * 100).toFixed(1)}}%)
  T: ${{currentSeq.filter(b => b === 'T').length}} (${{(currentSeq.filter(b => b === 'T').length / currentSeq.length * 100).toFixed(1)}}%)
  G: ${{currentSeq.filter(b => b === 'G').length}} (${{(currentSeq.filter(b => b === 'G').length / currentSeq.length * 100).toFixed(1)}}%)
  C: ${{currentSeq.filter(b => b === 'C').length}} (${{(currentSeq.filter(b => b === 'C').length / currentSeq.length * 100).toFixed(1)}}%)

GC Content: ${{((currentSeq.filter(b => b === 'G').length + currentSeq.filter(b => b === 'C').length) / currentSeq.length * 100).toFixed(1)}}%

🎉 Reconstruction process completed successfully!`);
        }}
        
        function chooseBestBase(sequence, position) {{
            const bases = ['A', 'T', 'G', 'C'];
            
            // Simple context-aware base selection
            let weights = [1, 1, 1, 1]; // Equal probability initially
            
            // Look at neighbors
            const neighbors = [];
            for (let offset of [-2, -1, 1, 2]) {{
                const neighborPos = position + offset;
                if (neighborPos >= 0 && neighborPos < sequence.length) {{
                    const neighbor = sequence[neighborPos];
                    if (bases.includes(neighbor)) {{
                        neighbors.push(neighbor);
                    }}
                }}
            }}
            
            // Avoid creating long runs of same base
            if (neighbors.length > 0) {{
                const mostCommon = neighbors.reduce((a, b, i, arr) => 
                    arr.filter(v => v === a).length >= arr.filter(v => v === b).length ? a : b
                );
                const mostCommonIndex = bases.indexOf(mostCommon);
                if (mostCommonIndex >= 0) {{
                    weights[mostCommonIndex] *= 0.5; // Reduce probability
                }}
            }}
            
            // Add some randomness
            weights = weights.map(w => w + Math.random() * 0.5);
            
            // Choose base based on weights
            const totalWeight = weights.reduce((sum, w) => sum + w, 0);
            let random = Math.random() * totalWeight;
            
            for (let i = 0; i < bases.length; i++) {{
                random -= weights[i];
                if (random <= 0) {{
                    return bases[i];
                }}
            }}
            
            return bases[Math.floor(Math.random() * bases.length)];
        }}
        
        function updateSequenceDisplay(sequence) {{
            const display = document.getElementById('sequenceText');
            let html = '';
            
            for (let i = 0; i < sequence.length; i++) {{
                const base = sequence[i];
                const cssClass = base === '-' ? 'base-gap' : `base-${{base}}`;
                html += `<span class="${{cssClass}}">${{base}}</span>`;
                
                // Add space every 10 bases for readability
                if ((i + 1) % 10 === 0) {{
                    html += ' ';
                }}
                if ((i + 1) % 50 === 0) {{
                    html += '<br>';
                }}
            }}
            
            display.innerHTML = html;
        }}
        
        function updateProgress(progress, total) {{
            const percent = Math.round(progress * 100);
            document.getElementById('progressFill').style.width = percent + '%';
        }}
        
        function updateStatus(message) {{
            document.getElementById('statusMessage').textContent = message;
        }}
        
        function updateStats(text) {{
            document.getElementById('statsText').textContent = text;
        }}
        
        function exportResults() {{
            if (!reconstructionData) {{
                alert('No reconstruction data to export. Please run a reconstruction first.');
                return;
            }}
            
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `dna_reconstruction_${{timestamp}}.json`;
            
            const blob = new Blob([JSON.stringify(reconstructionData, null, 2)], {{
                type: 'application/json'
            }});
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
            
            updateStatus(`✅ Results exported to ${{filename}}`);
        }}
        
        // Initialize
        window.onload = function() {{
            updateStatus('🧬 DNA Reconstruction Visualizer ready!');
            updateStats(`Welcome to the DNA Reconstruction Visualizer!

This tool simulates the reconstruction of ancient DNA sequences.

How to use:
1. Enter a DNA sequence (A, T, G, C)
2. Click "Simulate Damage" to create ancient DNA
3. Click "Start Reconstruction" to watch AI rebuild it
4. View live progress and statistics
5. Export results when complete

Try the sample sequence or enter your own!
Sequences should contain only A, T, G, C characters.

Ready to begin your DNA reconstruction journey! 🚀`);
        }};
    </script>
</body>
</html>
        """
    
    def start_server(self):
        """Start HTML server."""
        
        # Create HTML file
        html_content = self.create_html_interface()
        with open('dna_visualizer.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 Starting HTML server on http://localhost:{self.port}")
        print(f"📁 HTML file created: dna_visualizer.html")
        
        # Open in browser
        webbrowser.open(f'file://{os.path.abspath("dna_visualizer.html")}')
        
        # Start simple server (optional)
        try:
            with socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
                print(f"🚀 Server running at http://localhost:{self.port}")
                print("📱 Opening in browser...")
                print("🛑 Press Ctrl+C to stop")
                httpd.serve_forever()
        except Exception as e:
            print(f"⚠️  Server startup failed: {e}")
            print("📄 You can still open dna_visualizer.html directly in your browser")

def main():
    """Main function to choose visualization method."""
    
    print("🧬 DNA Reconstruction Visualizer")
    print("=" * 50)
    print("Choose your preferred interface:")
    print("1. 🖥️  Desktop GUI (Tkinter) - Most reliable")
    print("2. 🌐 Web Browser (HTML) - Cross-platform")
    print("3. 📄 Direct HTML file - No server needed")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1' or choice == '':
            print("🚀 Starting desktop GUI...")
            app = DNAVisualizationTkinter()
            app.run()
            
        elif choice == '2':
            print("🌐 Starting web server...")
            visualizer = HTMLDNAVisualizer()
            visualizer.start_server()
            
        elif choice == '3':
            print("📄 Creating HTML file...")
            visualizer = HTMLDNAVisualizer()
            html_content = visualizer.create_html_interface()
            with open('dna_visualizer.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            html_path = os.path.abspath('dna_visualizer.html')
            print(f"✅ HTML file created: {html_path}")
            webbrowser.open(f'file://{html_path}')
            print("📱 Opening in browser...")
            
        else:
            print("❌ Invalid choice. Starting desktop GUI by default...")
            app = DNAVisualizationTkinter()
            app.run()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Falling back to desktop GUI...")
        app = DNAVisualizationTkinter()
        app.run()

if __name__ == "__main__":
    main()

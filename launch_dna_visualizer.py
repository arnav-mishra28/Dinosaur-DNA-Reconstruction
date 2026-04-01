#!/usr/bin/env python3
"""
DNA Reconstruction Visualizer - Fixed Launcher
This script fixes encoding issues and ensures everything works
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"SUCCESS: Python version: {sys.version.split()[0]}")
    return True

def install_required_packages():
    """Install only essential packages."""
    required_packages = [
        'numpy',
        'matplotlib', 
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"SUCCESS: {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         capture_output=True, text=True, check=True)
            print(f"SUCCESS: {package} installed successfully")

def create_simple_visualizer():
    """Create the simplest possible working visualizer."""
    
    code = '''
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import random
import json

class SimpleDNAVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DNA Reconstruction Visualizer")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2E2E2E')
        
        self.is_running = False
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="DNA Reconstruction Visualizer", 
                        font=('Arial', 18, 'bold'), fg='white', bg='#2E2E2E')
        title.pack(pady=15)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#2E2E2E')
        input_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(input_frame, text="DNA Sequence:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#2E2E2E').pack(anchor='w')
        
        self.sequence_entry = tk.Text(input_frame, height=2, font=('Courier', 12))
        self.sequence_entry.pack(fill='x', pady=5)
        self.sequence_entry.insert('1.0', 'ATGCGATCGATCGATC')
        
        # Buttons
        button_frame = tk.Frame(self.root, bg='#2E2E2E')
        button_frame.pack(pady=15)
        
        self.damage_btn = tk.Button(button_frame, text="SIMULATE DAMAGE", 
                                   command=self.simulate_damage, font=('Arial', 11, 'bold'),
                                   bg='#FF6B6B', fg='white', width=15)
        self.damage_btn.pack(side='left', padx=10)
        
        self.reconstruct_btn = tk.Button(button_frame, text="START RECONSTRUCTION", 
                                        command=self.start_reconstruction, font=('Arial', 11, 'bold'),
                                        bg='#4ECDC4', fg='white', width=20)
        self.reconstruct_btn.pack(side='left', padx=10)
        
        self.export_btn = tk.Button(button_frame, text="EXPORT RESULTS", 
                                   command=self.export_results, font=('Arial', 11, 'bold'),
                                   bg='#FFA726', fg='white', width=15)
        self.export_btn.pack(side='left', padx=10)
        
        # Progress
        progress_frame = tk.Frame(self.root, bg='#2E2E2E')
        progress_frame.pack(pady=10, padx=20, fill='x')
        
        self.progress_var = tk.StringVar(value="Ready to start...")
        tk.Label(progress_frame, textvariable=self.progress_var, font=('Arial', 10),
                fg='#4ECDC4', bg='#2E2E2E').pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=600, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Sequence display
        seq_frame = tk.LabelFrame(self.root, text="Current Sequence", font=('Arial', 12, 'bold'),
                                 fg='white', bg='#2E2E2E')
        seq_frame.pack(pady=10, padx=20, fill='x')
        
        self.sequence_display = tk.Text(seq_frame, height=3, font=('Courier', 14, 'bold'),
                                       bg='#1E1E1E', fg='white')
        self.sequence_display.pack(fill='x', padx=10, pady=10)
        
        # Stats
        stats_frame = tk.LabelFrame(self.root, text="Statistics", font=('Arial', 12, 'bold'),
                                   fg='white', bg='#2E2E2E')
        stats_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        self.stats_display = scrolledtext.ScrolledText(stats_frame, font=('Courier', 10),
                                                      bg='#1E1E1E', fg='white')
        self.stats_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure colors
        self.setup_colors()
        
        # Initial display
        self.update_sequence_display("ATGCGATCGATCGATC")
        self.update_stats("Welcome! Enter a DNA sequence and try the buttons.")
    
    def setup_colors(self):
        self.sequence_display.tag_config("A", foreground="#FF6B6B")  # Red
        self.sequence_display.tag_config("T", foreground="#4ECDC4")  # Teal
        self.sequence_display.tag_config("G", foreground="#45B7D1")  # Blue
        self.sequence_display.tag_config("C", foreground="#FFA726")  # Orange
        self.sequence_display.tag_config("N", foreground="#9E9E9E")  # Gray
    
    def simulate_damage(self):
        if self.is_running:
            return
        
        sequence = self.sequence_entry.get('1.0', tk.END).strip().upper()
        if not sequence or len(sequence) < 5:
            messagebox.showerror("Error", "Please enter a DNA sequence (at least 5 bases)")
            return
        
        # Clean sequence
        clean_seq = ''.join(c for c in sequence if c in 'ATGC')
        
        # Simulate damage
        damaged_seq = ''
        damage_count = 0
        for base in clean_seq:
            if random.random() < 0.25:  # 25% damage rate
                damaged_seq += 'N'
                damage_count += 1
            else:
                damaged_seq += base
        
        self.sequence_entry.delete('1.0', tk.END)
        self.sequence_entry.insert('1.0', damaged_seq)
        
        self.update_sequence_display(damaged_seq)
        self.update_stats(f"Damage simulation completed!\\nOriginal: {len(clean_seq)} bases\\nDamaged: {damage_count} positions\\nDamage rate: {damage_count/len(clean_seq)*100:.1f}%")
        self.progress_var.set(f"Damage simulation completed - {damage_count} positions damaged")
        
        messagebox.showinfo("Success", f"Damage simulation completed!\\nOriginal: {len(clean_seq)} bases\\nDamaged: {damage_count} unknown positions")
    
    def start_reconstruction(self):
        if self.is_running:
            return
        
        sequence = self.sequence_entry.get('1.0', tk.END).strip().upper()
        if not sequence:
            messagebox.showerror("Error", "Please enter a sequence first")
            return
        
        clean_seq = ''.join(c for c in sequence if c in 'ATGCN')
        unknown_count = clean_seq.count('N')
        
        if unknown_count == 0:
            messagebox.showinfo("Info", "No unknown bases to reconstruct")
            return
        
        self.is_running = True
        self.reconstruct_btn.config(state='disabled', text="RECONSTRUCTING...")
        
        # Start reconstruction in thread
        threading.Thread(target=self._run_reconstruction, args=(clean_seq,), daemon=True).start()
    
    def _run_reconstruction(self, sequence):
        try:
            current_seq = list(sequence)
            unknown_positions = [i for i, base in enumerate(current_seq) if base == 'N']
            total_steps = len(unknown_positions)
            
            print(f"Starting reconstruction of {total_steps} unknown positions...")
            
            for step, pos in enumerate(unknown_positions):
                # Choose base with some biological logic
                context = self.get_context(current_seq, pos)
                new_base = self.choose_base_with_context(context)
                current_seq[pos] = new_base
                
                progress = (step + 1) / total_steps * 100
                reconstructed_seq = ''.join(current_seq)
                
                print(f"Step {step+1}/{total_steps}: Position {pos} changed to {new_base}")
                
                # Update UI in main thread
                self.root.after(0, self._update_reconstruction_step, 
                               reconstructed_seq, step + 1, total_steps, progress)
                
                time.sleep(0.5)  # Animation delay
            
            # Complete
            self.root.after(0, self._reconstruction_complete, ''.join(current_seq), total_steps)
            
        except Exception as e:
            print(f"Reconstruction error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Reconstruction failed: {e}"))
            self.root.after(0, self._reconstruction_complete, sequence, 0)
    
    def get_context(self, sequence, position):
        """Get neighboring bases for context."""
        context = []
        for offset in [-2, -1, 1, 2]:
            neighbor_pos = position + offset
            if 0 <= neighbor_pos < len(sequence):
                neighbor = sequence[neighbor_pos]
                if neighbor in 'ATGC':
                    context.append(neighbor)
        return context
    
    def choose_base_with_context(self, context):
        """Choose base based on biological context."""
        # Base frequencies in typical DNA
        base_weights = {'A': 0.25, 'T': 0.25, 'G': 0.25, 'C': 0.25}
        
        # Adjust weights based on context
        if context:
            # Avoid creating long runs of same base
            most_common = max(set(context), key=context.count)
            base_weights[most_common] *= 0.7
            
            # Complement pairing preference (weak)
            complements = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
            for base in context:
                if base in complements:
                    base_weights[complements[base]] *= 1.1
        
        # Choose base based on weights
        bases = list(base_weights.keys())
        weights = list(base_weights.values())
        
        import random
        return random.choices(bases, weights=weights)[0]
    
    def _update_reconstruction_step(self, sequence, step, total, progress):
        self.progress_bar['value'] = progress
        self.progress_var.set(f"Reconstructing... Step {step}/{total} - {progress:.1f}% complete")
        
        self.update_sequence_display(sequence)
        
        base_counts = {b: sequence.count(b) for b in 'ATGCN'}
        gc_content = (base_counts['G'] + base_counts['C']) / len(sequence) * 100
        
        stats = f"""RECONSTRUCTION STEP {step}
========================================

Current sequence: {sequence}
Progress: {progress:.1f}%
Remaining: {total - step} positions

Base composition:
A: {base_counts['A']} ({base_counts['A']/len(sequence)*100:.1f}%)
T: {base_counts['T']} ({base_counts['T']/len(sequence)*100:.1f}%)
G: {base_counts['G']} ({base_counts['G']/len(sequence)*100:.1f}%)
C: {base_counts['C']} ({base_counts['C']/len(sequence)*100:.1f}%)
N: {base_counts['N']} ({base_counts['N']/len(sequence)*100:.1f}%)

GC Content: {gc_content:.1f}%

Reconstruction using biological context and
base frequency analysis for optimal results.
"""
        self.update_stats(stats)
    
    def _reconstruction_complete(self, final_sequence, mutations_applied):
        self.is_running = False
        self.reconstruct_btn.config(state='normal', text="START RECONSTRUCTION")
        self.progress_var.set("Reconstruction completed successfully!")
        
        if mutations_applied > 0:
            messagebox.showinfo("Success!", 
                              f"Reconstruction completed!\\nFinal sequence: {final_sequence}\\nPositions reconstructed: {mutations_applied}")
        
        # Store results for export
        self.last_result = {
            'final_sequence': final_sequence,
            'mutations_applied': mutations_applied,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def export_results(self):
        """Export reconstruction results."""
        if not hasattr(self, 'last_result'):
            messagebox.showwarning("Warning", "No reconstruction data to export. Run a reconstruction first.")
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dna_reconstruction_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.last_result, f, indent=2)
            
            messagebox.showinfo("Export Successful", f"Results exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def update_sequence_display(self, sequence):
        self.sequence_display.delete('1.0', tk.END)
        
        for i, base in enumerate(sequence):
            self.sequence_display.insert(tk.END, base, base)
            if (i + 1) % 10 == 0:
                self.sequence_display.insert(tk.END, ' ')
            if (i + 1) % 50 == 0:
                self.sequence_display.insert(tk.END, '\\n')
    
    def update_stats(self, text):
        self.stats_display.delete('1.0', tk.END)
        self.stats_display.insert('1.0', text)
    
    def run(self):
        print("Starting DNA Reconstruction Visualizer...")
        print("GUI interface opening...")
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleDNAVisualizer()
    app.run()
'''
    
    # Write with UTF-8 encoding explicitly
    with open('simple_dna_visualizer.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    return 'simple_dna_visualizer.py'

def main():
    """Main launcher function."""
    
    print("DNA Reconstruction Visualizer - Launcher")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Install packages
    try:
        install_required_packages()
    except Exception as e:
        print(f"WARNING: Package installation failed: {e}")
        print("Continuing with basic functionality...")
    
    # Create simple visualizer
    try:
        visualizer_file = create_simple_visualizer()
        print(f"SUCCESS: Created working visualizer: {visualizer_file}")
    except Exception as e:
        print(f"ERROR: Failed to create visualizer: {e}")
        input("Press Enter to exit...")
        return
    
    print("Starting DNA Reconstruction Visualizer...")
    print("The GUI will open in a few seconds...")
    
    try:
        # Import and run
        import sys
        sys.path.append('.')
        
        # Execute the visualizer
        exec(open('simple_dna_visualizer.py', encoding='utf-8').read())
        
    except Exception as e:
        print(f"ERROR: Failed to start visualizer: {e}")
        print("\\nAlternative: Run this command manually:")
        print(f"python simple_dna_visualizer.py")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
"""
VS Code Extension Integration for DNA Reconstruction Visualizer
Provides seamless integration with VS Code for real-time DNA visualization
"""

import json
import os
import webbrowser
from pathlib import Path
import subprocess
import sys
from typing import Dict, Any
import threading
import time

class VSCodeDNAExtension:
    """VS Code integration for DNA reconstruction visualization."""
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path or os.getcwd()
        self.extension_name = "dna-reconstruction-visualizer"
        self.port = 8501  # Streamlit default port
        self.jupyter_port = 8888  # Jupyter default port
        
    def create_extension_config(self) -> Dict[str, Any]:
        """Create VS Code extension configuration."""
        
        config = {
            "name": self.extension_name,
            "displayName": "DNA Reconstruction Visualizer",
            "description": "Interactive DNA sequence reconstruction and visualization",
            "version": "1.0.0",
            "engines": {
                "vscode": "^1.60.0"
            },
            "categories": ["Other", "Data Science", "Visualization"],
            "activationEvents": [
                "onCommand:dnaVisualizer.start",
                "onLanguage:python",
                "onFileSystem:file"
            ],
            "main": "./extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "dnaVisualizer.start",
                        "title": "🧬 Start DNA Visualizer",
                        "category": "DNA"
                    },
                    {
                        "command": "dnaVisualizer.openNotebook",
                        "title": "📓 Open DNA Notebook",
                        "category": "DNA"
                    },
                    {
                        "command": "dnaVisualizer.runReconstruction",
                        "title": "🔄 Run Reconstruction",
                        "category": "DNA"
                    },
                    {
                        "command": "dnaVisualizer.showPhylogeny",
                        "title": "🌳 Show Phylogeny",
                        "category": "DNA"
                    }
                ],
                "menus": {
                    "explorer/context": [
                        {
                            "when": "resourceExtname == .py",
                            "command": "dnaVisualizer.start",
                            "group": "dna"
                        }
                    ],
                    "editor/context": [
                        {
                            "when": "editorLangId == python",
                            "command": "dnaVisualizer.runReconstruction",
                            "group": "dna"
                        }
                    ]
                },
                "keybindings": [
                    {
                        "command": "dnaVisualizer.start",
                        "key": "ctrl+alt+d",
                        "when": "editorTextFocus"
                    }
                ],
                "configuration": {
                    "title": "DNA Visualizer",
                    "properties": {
                        "dnaVisualizer.autoStart": {
                            "type": "boolean",
                            "default": true,
                            "description": "Automatically start visualizer when opening DNA files"
                        },
                        "dnaVisualizer.port": {
                            "type": "number",
                            "default": 8501,
                            "description": "Port for the visualization server"
                        }
                    }
                }
            }
        }
        
        return config
    
    def create_extension_script(self) -> str:
        """Create the main extension script."""
        
        script = """
const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');

function activate(context) {
    console.log('DNA Reconstruction Visualizer extension is now active!');
    
    // Register commands
    let startCommand = vscode.commands.registerCommand('dnaVisualizer.start', () => {
        startDNAVisualizer();
    });
    
    let notebookCommand = vscode.commands.registerCommand('dnaVisualizer.openNotebook', () => {
        openDNANotebook();
    });
    
    let reconstructionCommand = vscode.commands.registerCommand('dnaVisualizer.runReconstruction', () => {
        runReconstruction();
    });
    
    let phylogenyCommand = vscode.commands.registerCommand('dnaVisualizer.showPhylogeny', () => {
        showPhylogeny();
    });
    
    context.subscriptions.push(startCommand, notebookCommand, reconstructionCommand, phylogenyCommand);
    
    // Auto-start if enabled
    const config = vscode.workspace.getConfiguration('dnaVisualizer');
    if (config.get('autoStart')) {
        // Check if current file is DNA-related
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor) {
            const fileName = activeEditor.document.fileName;
            if (fileName.includes('dna') || fileName.includes('sequence') || fileName.includes('enhanced_')) {
                vscode.window.showInformationMessage(
                    'DNA Visualizer available! Press Ctrl+Alt+D to start.',
                    'Start Now'
                ).then(selection => {
                    if (selection === 'Start Now') {
                        startDNAVisualizer();
                    }
                });
            }
        }
    }
}

function startDNAVisualizer() {
    const workspaceFolder = vscode.workspace.workspaceFolders[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('Please open a workspace folder first');
        return;
    }
    
    const workspacePath = workspaceFolder.uri.fsPath;
    
    // Start Streamlit dashboard
    vscode.window.showInformationMessage('Starting DNA Visualizer...');
    
    const command = `cd "${workspacePath}" && python -m streamlit run visual_dna_simulation.py`;
    
    exec(command, (error, stdout, stderr) => {
        if (error) {
            vscode.window.showErrorMessage(`Failed to start DNA Visualizer: ${error.message}`);
            return;
        }
        
        // Open browser to visualizer
        setTimeout(() => {
            vscode.env.openExternal(vscode.Uri.parse('http://localhost:8501'));
        }, 3000);
    });
}

function openDNANotebook() {
    const workspaceFolder = vscode.workspace.workspaceFolders[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('Please open a workspace folder first');
        return;
    }
    
    // Create/open DNA notebook
    const notebookPath = path.join(workspaceFolder.uri.fsPath, 'DNA_Reconstruction.ipynb');
    
    // Create notebook if it doesn't exist
    const fs = require('fs');
    if (!fs.existsSync(notebookPath)) {
        const notebookContent = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# 🧬 Interactive DNA Reconstruction\\n\\n", "This notebook provides interactive DNA reconstruction visualization."]
                },
                {
                    "cell_type": "code",
                    "execution_count": null,
                    "metadata": {},
                    "source": [
                        "# Import the interactive DNA notebook\\n",
                        "from interactive_dna_notebook import create_dna_reconstruction_notebook\\n",
                        "from IPython.display import display\\n\\n",
                        "# Create and display the interface\\n",
                        "interface = create_dna_reconstruction_notebook()\\n",
                        "display(interface)"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        };
        
        fs.writeFileSync(notebookPath, JSON.stringify(notebookContent, null, 2));
    }
    
    // Open the notebook
    vscode.commands.executeCommand('vscode.open', vscode.Uri.file(notebookPath));
}

function runReconstruction() {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('Please open a Python file first');
        return;
    }
    
    // Get selected text or entire document
    const selection = activeEditor.selection;
    const text = selection.isEmpty 
        ? activeEditor.document.getText() 
        : activeEditor.document.getText(selection);
    
    // Extract DNA sequence (simple pattern matching)
    const dnaPattern = /[ATGCN\\-]{10,}/gi;
    const matches = text.match(dnaPattern);
    
    if (!matches || matches.length === 0) {
        vscode.window.showErrorMessage('No DNA sequences found in the selected text');
        return;
    }
    
    const sequence = matches[0];
    vscode.window.showInformationMessage(`Running reconstruction on sequence: ${sequence.substring(0, 20)}...`);
    
    // Run reconstruction script
    const workspaceFolder = vscode.workspace.workspaceFolders[0];
    const command = `cd "${workspaceFolder.uri.fsPath}" && python -c "
from enhanced_training import main as train_model
from enhanced_evaluation import main as evaluate_model
print('🧬 Starting DNA reconstruction...')
train_model()
evaluate_model()
print('✅ Reconstruction completed!')
"`;
    
    const terminal = vscode.window.createTerminal('DNA Reconstruction');
    terminal.sendText(command);
    terminal.show();
}

function showPhylogeny() {
    vscode.window.showInformationMessage('Opening phylogenetic analysis...');
    
    // Open phylogeny visualization
    const workspaceFolder = vscode.workspace.workspaceFolders[0];
    const command = `cd "${workspaceFolder.uri.fsPath}" && python -c "
from visual_dna_simulation import PhylogeneticTree3D
import plotly.graph_objects as go

phylo = PhylogeneticTree3D()
sample_sequences = {
    'Gallus_gallus': 'ATGCGATCGATCGATCG',
    'Alligator_mississippiensis': 'ATGCGATCGATCGATCA',
    'Reconstructed': 'ATGCGATCGATCGATCT'
}
fig = phylo.create_3d_tree(sample_sequences)
fig.show()
"`;
    
    const terminal = vscode.window.createTerminal('DNA Phylogeny');
    terminal.sendText(command);
    terminal.show();
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
        """
        
        return script
    
    def setup_vscode_integration(self) -> None:
        """Set up complete VS Code integration."""
        
        print("🔧 Setting up VS Code integration...")
        
        # Create extension directory
        ext_dir = Path(self.workspace_path) / ".vscode" / "extensions" / self.extension_name
        ext_dir.mkdir(parents=True, exist_ok=True)
        
        # Write package.json
        config = self.create_extension_config()
        with open(ext_dir / "package.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Write extension script
        script = self.create_extension_script()
        with open(ext_dir / "extension.js", 'w') as f:
            f.write(script)
        
        # Create settings for current workspace
        self.create_workspace_settings()
        
        print(f"✅ VS Code extension created at: {ext_dir}")
        print("📱 Extension features:")
        print("   • Ctrl+Alt+D - Start DNA Visualizer")
        print("   • Right-click Python files - DNA options")
        print("   • Command palette: 'DNA' commands")
    
    def create_workspace_settings(self) -> None:
        """Create VS Code workspace settings for DNA project."""
        
        vscode_dir = Path(self.workspace_path) / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        # Settings
        settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.terminal.activateEnvironment": True,
            "jupyter.defaultKernel": "python3",
            "dnaVisualizer.autoStart": True,
            "dnaVisualizer.port": self.port,
            "files.associations": {
                "*.dna": "plaintext",
                "*.fasta": "plaintext",
                "*.fa": "plaintext"
            },
            "workbench.colorCustomizations": {
                "[Default Dark+]": {
                    "editorLineNumber.foreground": "#4ECDC4"
                }
            }
        }
        
        with open(vscode_dir / "settings.json", 'w') as f:
            json.dump(settings, f, indent=2)
        
        # Tasks
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Start DNA Visualizer",
                    "type": "shell",
                    "command": "python",
                    "args": ["-m", "streamlit", "run", "visual_dna_simulation.py"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "new"
                    },
                    "problemMatcher": []
                },
                {
                    "label": "Run DNA Training",
                    "type": "shell", 
                    "command": "python",
                    "args": ["enhanced_training.py"],
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": True,
                        "panel": "new"
                    }
                },
                {
                    "label": "Collect DNA Data",
                    "type": "shell",
                    "command": "python", 
                    "args": ["enhanced_data_collection.py"],
                    "group": "build"
                }
            ]
        }
        
        with open(vscode_dir / "tasks.json", 'w') as f:
            json.dump(tasks, f, indent=2)
        
        # Launch configuration
        launch = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Debug DNA Training",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/enhanced_training.py",
                    "console": "integratedTerminal",
                    "justMyCode": True
                },
                {
                    "name": "Debug DNA Visualization",
                    "type": "python",
                    "request": "launch",
                    "program": "${workspaceFolder}/visual_dna_simulation.py",
                    "console": "integratedTerminal"
                }
            ]
        }
        
        with open(vscode_dir / "launch.json", 'w') as f:
            json.dump(launch, f, indent=2)
        
        print("⚙️ VS Code workspace settings created")
    
    def create_html_viewer(self) -> str:
        """Create HTML viewer for embedded visualization."""
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 DNA Reconstruction Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .control-group {
            flex: 1;
            min-width: 200px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, textarea, select, button {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
        }
        button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            color: white;
            cursor: pointer;
            transition: transform 0.2s;
            font-weight: bold;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .visualization-area {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            color: #333;
        }
        .sequence-display {
            font-family: 'Courier New', monospace;
            font-size: 16px;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .base-A { color: #ff6b6b; font-weight: bold; }
        .base-T { color: #4ecdc4; font-weight: bold; }
        .base-G { color: #45b7d1; font-weight: bold; }
        .base-C { color: #ffa726; font-weight: bold; }
        .base-N { color: #9e9e9e; font-weight: bold; }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #ffa726);
            width: 0%;
            transition: width 0.3s ease;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 DNA Reconstruction Visualizer</h1>
            <p>Interactive DNA sequence reconstruction using AI models</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="sequenceInput">DNA Sequence:</label>
                <textarea id="sequenceInput" rows="3" placeholder="Enter DNA sequence (e.g., ATGCN-NATGC)">ATGCN-NATGC-NNCGATNNNAAATTT</textarea>
            </div>
            
            <div class="control-group">
                <label for="reconstructionSteps">Reconstruction Steps:</label>
                <input type="range" id="reconstructionSteps" min="10" max="100" value="30">
                <span id="stepsValue">30</span>
            </div>
            
            <div class="control-group">
                <label for="mutationRate">Mutation Rate:</label>
                <input type="range" id="mutationRate" min="0.001" max="0.1" step="0.001" value="0.01">
                <span id="mutationValue">0.01</span>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="startReconstruction()">🚀 Start Reconstruction</button>
            <button onclick="showPhylogeny()">🌳 Show Phylogeny</button>
            <button onclick="exportResults()">💾 Export Results</button>
            <button onclick="resetSimulation()">🔄 Reset</button>
        </div>
        
        <div class="visualization-area">
            <h3>🔄 Reconstruction Progress</h3>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div id="progressText">Ready to start reconstruction...</div>
        </div>
        
        <div class="visualization-area">
            <h3>🧬 Current Sequence</h3>
            <div class="sequence-display" id="sequenceDisplay">
                Enter a sequence and click "Start Reconstruction"
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="sequenceLength">0</div>
                <div>Sequence Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="unknownBases">0</div>
                <div>Unknown Bases</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="gcContent">0%</div>
                <div>GC Content</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="confidence">0%</div>
                <div>Confidence</div>
            </div>
        </div>
        
        <div class="visualization-area">
            <h3>📊 Visualization</h3>
            <div id="plotArea" style="width: 100%; height: 400px;"></div>
        </div>
        
        <div class="visualization-area">
            <h3>🌳 Phylogenetic Analysis</h3>
            <div id="phylogenyArea" style="width: 100%; height: 500px;"></div>
        </div>
    </div>

    <script>
        let currentSequence = '';
        let reconstructionHistory = [];
        let isRunning = false;
        
        // Update slider values
        document.getElementById('reconstructionSteps').oninput = function() {
            document.getElementById('stepsValue').textContent = this.value;
        };
        
        document.getElementById('mutationRate').oninput = function() {
            document.getElementById('mutationValue').textContent = this.value;
        };
        
        function formatSequence(sequence) {
            return sequence.split('').map(base => {
                return `<span class="base-${base}">${base}</span>`;
            }).join('');
        }
        
        function updateStats(sequence) {
            const length = sequence.length;
            const unknownCount = (sequence.match(/[N-]/g) || []).length;
            const gcCount = (sequence.match(/[GC]/g) || []).length;
            const gcContent = length > 0 ? (gcCount / length * 100).toFixed(1) : 0;
            const confidence = length > 0 ? ((length - unknownCount) / length * 100).toFixed(1) : 0;
            
            document.getElementById('sequenceLength').textContent = length;
            document.getElementById('unknownBases').textContent = unknownCount;
            document.getElementById('gcContent').textContent = gcContent + '%';
            document.getElementById('confidence').textContent = confidence + '%';
        }
        
        async function startReconstruction() {
            if (isRunning) return;
            
            const inputSequence = document.getElementById('sequenceInput').value;
            const steps = parseInt(document.getElementById('reconstructionSteps').value);
            
            if (!inputSequence.trim()) {
                alert('Please enter a DNA sequence');
                return;
            }
            
            isRunning = true;
            currentSequence = inputSequence.trim().toUpperCase();
            reconstructionHistory = [currentSequence];
            
            document.getElementById('progressText').textContent = 'Starting reconstruction...';
            
            for (let step = 0; step <= steps; step++) {
                const progress = step / steps;
                
                // Simulate reconstruction
                let reconstructed = '';
                for (let i = 0; i < currentSequence.length; i++) {
                    const base = currentSequence[i];
                    if (base === 'N' || base === '-') {
                        if (Math.random() < progress) {
                            reconstructed += ['A', 'T', 'G', 'C'][Math.floor(Math.random() * 4)];
                        } else {
                            reconstructed += base;
                        }
                    } else {
                        reconstructed += base;
                    }
                }
                
                currentSequence = reconstructed;
                reconstructionHistory.push(currentSequence);
                
                // Update UI
                const progressPercent = Math.round(progress * 100);
                document.getElementById('progressFill').style.width = progressPercent + '%';
                document.getElementById('progressText').textContent = 
                    `Step ${step}/${steps} - ${progressPercent}% complete`;
                document.getElementById('sequenceDisplay').innerHTML = formatSequence(currentSequence);
                
                updateStats(currentSequence);
                
                // Update plot
                if (step % 5 === 0) {
                    updatePlot(currentSequence);
                }
                
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            document.getElementById('progressText').textContent = '✅ Reconstruction completed!';
            isRunning = false;
            
            // Final visualization
            updatePlot(currentSequence);
            showPhylogeny();
        }
        
        function updatePlot(sequence) {
            const baseCounts = {
                'A': (sequence.match(/A/g) || []).length,
                'T': (sequence.match(/T/g) || []).length,
                'G': (sequence.match(/G/g) || []).length,
                'C': (sequence.match(/C/g) || []).length,
                'N': (sequence.match(/N/g) || []).length,
                '-': (sequence.match(/-/g) || []).length
            };
            
            const trace = {
                x: Object.keys(baseCounts),
                y: Object.values(baseCounts),
                type: 'bar',
                marker: {
                    color: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726', '#9e9e9e', '#ffffff'],
                    line: { color: '#333', width: 1 }
                }
            };
            
            const layout = {
                title: 'Base Composition',
                xaxis: { title: 'DNA Bases' },
                yaxis: { title: 'Count' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('plotArea', [trace], layout);
        }
        
        function showPhylogeny() {
            // Sample phylogenetic data
            const species = ['Gallus gallus', 'Alligator mississippiensis', 'Struthio camelus', 'Reconstructed'];
            const x = [0, 1, 0.5, 0.7];
            const y = [0, 0, 1, 0.3];
            const z = [0, 0, 0, 0.5];
            
            const trace = {
                x: x,
                y: y,
                z: z,
                mode: 'markers+text',
                type: 'scatter3d',
                text: species,
                textposition: 'top center',
                marker: {
                    size: 12,
                    color: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726']
                }
            };
            
            const layout = {
                title: '3D Phylogenetic Relationships',
                scene: {
                    xaxis: { title: 'Evolutionary Distance X' },
                    yaxis: { title: 'Evolutionary Distance Y' },
                    zaxis: { title: 'Evolutionary Distance Z' }
                },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('phylogenyArea', [trace], layout);
        }
        
        function exportResults() {
            if (reconstructionHistory.length === 0) {
                alert('No reconstruction data to export');
                return;
            }
            
            let csvContent = "Step,Sequence\\n";
            reconstructionHistory.forEach((seq, index) => {
                csvContent += `${index},${seq}\\n`;
            });
            
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dna_reconstruction_results.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        }
        
        function resetSimulation() {
            isRunning = false;
            currentSequence = '';
            reconstructionHistory = [];
            
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = 'Ready to start reconstruction...';
            document.getElementById('sequenceDisplay').textContent = 'Enter a sequence and click "Start Reconstruction"';
            
            // Reset stats
            document.getElementById('sequenceLength').textContent = '0';
            document.getElementById('unknownBases').textContent = '0';
            document.getElementById('gcContent').textContent = '0%';
            document.getElementById('confidence').textContent = '0%';
            
            // Clear plots
            document.getElementById('plotArea').innerHTML = '';
            document.getElementById('phylogenyArea').innerHTML = '';
        }
        
        // Initialize empty plots
        window.onload = function() {
            resetSimulation();
        };
    </script>
</body>
</html>
        """
        
        return html_content
    
    def install_dependencies(self) -> None:
        """Install required dependencies for VS Code integration."""
        
        print("📦 Installing VS Code integration dependencies...")
        
        # Python dependencies
        python_deps = [
            'streamlit',
            'plotly',
            'jupyter',
            'ipywidgets',
            'matplotlib',
            'seaborn',
            'Pillow',
            'opencv-python'
        ]
        
        for dep in python_deps:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         capture_output=True, text=True)
        
        print("✅ Dependencies installed")
    
    def create_launcher_script(self) -> None:
        """Create launcher script for easy startup."""
        
        launcher_content = f"""#!/usr/bin/env python3
\"\"\"
DNA Reconstruction Visualizer Launcher
Quick launcher for VS Code integration
\"\"\"

import subprocess
import sys
import webbrowser
import time
import threading
from pathlib import Path

def start_streamlit():
    \"\"\"Start Streamlit dashboard.\"\"\"
    print("🚀 Starting Streamlit dashboard...")
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        'visual_dna_simulation.py', 
        '--server.port', '{self.port}'
    ])

def start_jupyter():
    \"\"\"Start Jupyter server.\"\"\"
    print("📓 Starting Jupyter server...")
    subprocess.run([
        sys.executable, '-m', 'jupyter', 'lab', 
        '--port', '{self.jupyter_port}',
        '--no-browser'
    ])

def open_html_viewer():
    \"\"\"Open standalone HTML viewer.\"\"\"
    html_file = Path('dna_visualizer.html')
    if html_file.exists():
        webbrowser.open(f'file://{html_file.absolute()}')
    else:
        print("❌ HTML viewer not found")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DNA Visualizer Launcher')
    parser.add_argument('--mode', choices=['streamlit', 'jupyter', 'html'], 
                       default='streamlit', help='Launch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'streamlit':
        # Start Streamlit and open browser
        threading.Thread(target=start_streamlit, daemon=True).start()
        time.sleep(3)
        webbrowser.open(f'http://localhost:{self.port}')
        
    elif args.mode == 'jupyter':
        # Start Jupyter and open browser  
        threading.Thread(target=start_jupyter, daemon=True).start()
        time.sleep(3)
        webbrowser.open(f'http://localhost:{self.jupyter_port}')
        
    elif args.mode == 'html':
        open_html_viewer()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down...")

if __name__ == '__main__':
    main()
"""
        
        with open('launch_dna_visualizer.py', 'w') as f:
            f.write(launcher_content)
        
        # Make executable on Unix systems
        try:
            import stat
            st = os.stat('launch_dna_visualizer.py')
            os.chmod('launch_dna_visualizer.py', st.st_mode | stat.S_IEXEC)
        except:
            pass
        
        print("🚀 Launcher script created: launch_dna_visualizer.py")

def setup_complete_vscode_integration(workspace_path: str = None):
    """Set up complete VS Code integration for DNA visualization."""
    
    print("🔧 Setting up complete VS Code integration...")
    
    # Initialize extension
    extension = VSCodeDNAExtension(workspace_path)
    
    # Install dependencies
    extension.install_dependencies()
    
    # Set up VS Code integration
    extension.setup_vscode_integration()
    
    # Create HTML viewer
    html_content = extension.create_html_viewer()
    with open('dna_visualizer.html', 'w') as f:
        f.write(html_content)
    
    # Create launcher script
    extension.create_launcher_script()
    
    print("\n🎉 VS Code integration setup complete!")
    print("\n📋 Available options:")
    print("1. 🖥️  Streamlit Dashboard: python launch_dna_visualizer.py --mode streamlit")
    print("2. 📓 Jupyter Notebook: python launch_dna_visualizer.py --mode jupyter") 
    print("3. 🌐 HTML Viewer: python launch_dna_visualizer.py --mode html")
    print("4. ⌨️  VS Code Commands: Ctrl+Alt+D or Command Palette > 'DNA'")
    
    return extension

if __name__ == "__main__":
    setup_complete_vscode_integration()

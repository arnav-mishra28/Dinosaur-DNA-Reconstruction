# Step-by-Step Setup Guide for Existing Folder

## 🎯 Overview
This guide helps you set up the enhanced dinosaur DNA system in your **existing** `/mnt/d/dinosaur_dna` folder.

---

## 📋 Step 1: Copy Files to Your Existing Folder

### What you need to do:

1. **In Windows Explorer:**
   - Go to your `D:\dinosaur_dna\` folder (your existing project folder)
   - You should see your current files there

2. **Copy the new enhanced files:**
   From your Downloads folder (or wherever you saved them), copy these files to `D:\dinosaur_dna\`:
   
   ```
   ✅ enhanced_config.py
   ✅ enhanced_models.py  
   ✅ enhanced_data_collection.py
   ✅ enhanced_training.py
   ✅ enhanced_evaluation.py
   ✅ enhanced_requirements.txt
   ✅ main_pipeline.py
   ✅ setup_wsl.sh
   ✅ README.md
   ✅ WSL_Setup_Guide.md
   ```

3. **Your folder should now look like:**
   ```
   D:\dinosaur_dna\
   ├── config.py                    (your old files)
   ├── models.py                    (your old files)
   ├── training.py                  (your old files)
   ├── enhanced_config.py           (✨ new enhanced files)
   ├── enhanced_models.py           (✨ new enhanced files)
   ├── enhanced_training.py         (✨ new enhanced files)
   ├── enhanced_data_collection.py  (✨ new enhanced files)
   ├── enhanced_evaluation.py      (✨ new enhanced files)
   ├── main_pipeline.py             (✨ new enhanced files)
   └── setup_wsl.sh                 (✨ new enhanced files)
   ```

---

## 📋 Step 2: WSL Setup (if not already done)

### If you DON'T have WSL installed yet:

1. **Open PowerShell as Administrator** (Right-click Start → Windows PowerShell (Admin))

2. **Run these commands:**
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

3. **Restart Windows**

4. **After restart, open PowerShell again as Administrator:**
   ```powershell
   wsl --set-default-version 2
   ```

5. **Install Ubuntu:**
   - Open Microsoft Store
   - Search "Ubuntu 24.04" (or Ubuntu 22.04)
   - Click Install
   - Launch Ubuntu when done
   - Create username and password when prompted

6. **Update Ubuntu:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### If you ALREADY have WSL:
   - Just launch Ubuntu from Start menu
   - Continue to Step 3

---

## 📋 Step 3: Check Your D Drive Access

### In WSL Ubuntu terminal:

1. **Check if you can see your D drive:**
   ```bash
   ls /mnt/d/
   ```
   
2. **Check if your project folder exists:**
   ```bash
   ls /mnt/d/dinosaur_dna/
   ```
   
   You should see all your files including the new enhanced_*.py files.

3. **If D drive is not accessible:**
   ```bash
   sudo mkdir -p /mnt/d
   sudo mount -t drvfs D: /mnt/d
   ```

---

## 📋 Step 4: Set Up Python Environment

### In WSL Ubuntu terminal:

1. **Navigate to your project folder:**
   ```bash
   cd /mnt/d/dinosaur_dna
   ```

2. **Install Python 3.11 if needed:**
   ```bash
   sudo apt update
   sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
   ```

3. **Create virtual environment (in your existing folder):**
   ```bash
   python3.11 -m venv venv
   ```
   
4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```
   
   You should see `(venv)` at the beginning of your prompt.

5. **Upgrade pip:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

---

## 📋 Step 5: Install Dependencies

### With virtual environment activated:

1. **Install PyTorch (CPU version for safety):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Install core dependencies:**
   ```bash
   pip install -r enhanced_requirements.txt
   ```

   This might take 10-15 minutes. If you get memory errors, install one by one:
   ```bash
   pip install numpy pandas scipy
   pip install biopython transformers datasets
   pip install tqdm wandb matplotlib seaborn
   pip install streamlit fastapi uvicorn
   ```

3. **Install system dependencies:**
   ```bash
   sudo apt install -y build-essential libhdf5-dev libffi-dev
   ```

---

## 📋 Step 6: Configure Your Email (REQUIRED)

### This is CRITICAL for NCBI data download:

1. **Edit the configuration file:**
   ```bash
   nano enhanced_config.py
   ```

2. **Find this line (around line 150):**
   ```python
   'email': 'your_email@example.com',  # CHANGE THIS TO YOUR EMAIL!
   ```

3. **Change it to your actual email:**
   ```python
   'email': 'youremail@gmail.com',  # Your actual email
   ```

4. **Save and exit:**
   - Press `Ctrl + X`
   - Press `Y` to confirm
   - Press `Enter` to save

---

## 📋 Step 7: Test Installation

### Verify everything is working:

1. **Test Python imports:**
   ```bash
   python3 -c "
   import torch
   import numpy
   import pandas
   print('✅ Basic libraries working')
   print(f'PyTorch version: {torch.__version__}')
   print(f'Device: {torch.cuda.is_available() and \"CUDA\" or \"CPU\"}')
   "
   ```

2. **Test BioPython (optional, might fail initially):**
   ```bash
   python3 -c "
   try:
       import Bio
       print('✅ BioPython working')
   except ImportError:
       print('⚠️  BioPython not installed yet - will install during first run')
   "
   ```

---

## 📋 Step 8: Run the Enhanced System

### Now you can run the enhanced system:

1. **Make sure you're in the right directory and environment:**
   ```bash
   cd /mnt/d/dinosaur_dna
   source venv/bin/activate
   ```

2. **Run the complete pipeline:**
   ```bash
   python3 main_pipeline.py
   ```

3. **Or run specific phases:**
   ```bash
   python3 main_pipeline.py --phase data    # Data collection only
   python3 main_pipeline.py --phase train   # Training only
   python3 main_pipeline.py --phase eval    # Evaluation only
   ```

4. **Check system info:**
   ```bash
   python3 main_pipeline.py --info
   ```

---

## 📁 Your Folder Structure After Setup

```
D:\dinosaur_dna\                    (your existing folder)
├── config.py                       (your original files - keep them!)
├── models.py                       (your original files - keep them!)
├── training.py                     (your original files - keep them!)
├── enhanced_config.py              (new enhanced system)
├── enhanced_models.py              (new enhanced system)
├── enhanced_training.py            (new enhanced system)
├── enhanced_data_collection.py     (new enhanced system)
├── enhanced_evaluation.py          (new enhanced system)
├── main_pipeline.py                (new enhanced system)
├── venv/                           (Python virtual environment)
├── data/                           (downloaded datasets)
├── models/                         (trained models)  
├── outputs/                        (results)
├── cache/                          (cached sequences)
└── logs/                           (log files)
```

---

## 🚨 Important Notes

1. **Keep your original files** - the enhanced system works alongside them
2. **Everything stays in your existing D:\dinosaur_dna folder**
3. **Email configuration is REQUIRED** for NCBI data download
4. **First run will take time** to download data from NCBI
5. **Training can take several hours** depending on your system

---

## 🆘 If Something Goes Wrong

### Common Issues:

**1. Permission Errors:**
```bash
sudo chown -R $USER:$USER /mnt/d/dinosaur_dna
chmod -R 755 /mnt/d/dinosaur_dna
```

**2. Import Errors:**
```bash
source venv/bin/activate
pip install --upgrade pip
pip install torch biopython transformers
```

**3. Memory Errors:**
Edit `enhanced_config.py` and reduce:
```python
'batch_size': 4,  # Instead of 8
```

**4. NCBI Errors:**
Check your email is properly set in `enhanced_config.py`

---

## ✅ Success Checklist

- [ ] Files copied to D:\dinosaur_dna\
- [ ] WSL Ubuntu working
- [ ] Can access /mnt/d/dinosaur_dna in WSL
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Email configured in enhanced_config.py
- [ ] Test imports working
- [ ] Ready to run main_pipeline.py

**You're all set! Your enhanced system will run in your existing folder with all improvements!** 🎉

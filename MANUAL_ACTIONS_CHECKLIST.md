# 🚀 MANUAL ACTIONS REQUIRED - Quick Checklist

## What You Need to Do From Your Side:

### 📥 1. COPY FILES (5 minutes)
**Action:** Copy these files from Downloads to your existing `D:\MY WORK\Dinosaur DNA Reconstruction\` folder:

- ✅ enhanced_config.py
- ✅ enhanced_models.py  
- ✅ enhanced_data_collection.py
- ✅ enhanced_training.py
- ✅ enhanced_evaluation.py
- ✅ enhanced_requirements.txt
- ✅ main_pipeline.py
- ✅ setup_wsl.sh
- ✅ STEP_BY_STEP_SETUP.md

**Result:** All files should be in your existing `D:\MY WORK\Dinosaur DNA Reconstruction\` folder alongside your original files.

---

### 🖥️ 2. WSL SETUP (10 minutes, one-time only)
**If you don't have WSL yet:**

1. **PowerShell as Administrator:**
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. **Restart Windows**

3. **Install Ubuntu 24.04 from Microsoft Store**

4. **Launch Ubuntu, create username/password**

**If you already have WSL:** Skip this step! ✅

---

### 🐍 3. PYTHON SETUP (10 minutes)
**In WSL Ubuntu terminal:**

```bash
# Go to your existing project folder (use quotes for spaces!)
cd "/mnt/d/MY WORK/Dinosaur DNA Reconstruction"

# Create virtual environment  
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r enhanced_requirements.txt
```

---

### ✉️ 4. EMAIL CONFIGURATION (2 minutes) - CRITICAL!
**Action:** Edit `enhanced_config.py` to set your email:

```bash
nano enhanced_config.py
```

**Find this line (~line 150):**
```python
'email': 'your_email@example.com',  # CHANGE THIS TO YOUR EMAIL!
```

**Change to your real email:**
```python  
'email': 'youremail@gmail.com',
```

**Save:** Ctrl+X, Y, Enter

---

### 🏃‍♂️ 5. RUN THE SYSTEM (Automated!)
**In WSL terminal:**

```bash
cd "/mnt/d/MY WORK/Dinosaur DNA Reconstruction"
source venv/bin/activate
python3 main_pipeline.py
```

**That's it!** The enhanced system will:
- ✅ Download real NCBI data (10+ species)
- ✅ Train with multi-head attention & DNABERT
- ✅ Evaluate with comprehensive metrics
- ✅ Save everything in your existing folder

---

## 📊 Expected Timeline:

- **Setup:** 20-30 minutes (one time)
- **Data Collection:** 30-60 minutes
- **Training:** 2-6 hours (depending on your hardware)
- **Evaluation:** 10-30 minutes

---

## 🎯 What You'll Get:

### Enhanced Features vs Your Original System:
- ✨ **Multi-head attention** (12 heads) vs basic models
- ✨ **DNABERT integration** vs simple tokenization  
- ✨ **Real NCBI datasets** vs synthetic data
- ✨ **Variable-length sequences** vs fixed length
- ✨ **Multi-species phylogenetic context** vs single species
- ✨ **Advanced mutation modeling** vs basic mutations
- ✨ **Comprehensive evaluation metrics** vs basic accuracy
- ✨ **WSL optimized** for your Windows system

### File Organization:
```
D:\MY WORK\Dinosaur DNA Reconstruction\   (your existing folder - no changes!)
├── config.py                             (your original files - preserved!)
├── models.py                             (your original files - preserved!) 
├── training.py                           (your original files - preserved!)
├── enhanced_config.py                    (🆕 enhanced system)
├── enhanced_models.py                    (🆕 enhanced system)
├── enhanced_training.py                  (🆕 enhanced system)
└── ... (new enhanced files)              (🆕 enhanced system)
```

---

## 🆘 Need Help?

1. **Read:** `STEP_BY_STEP_SETUP.md` for detailed instructions
2. **Check:** Your email is set in `enhanced_config.py`  
3. **Verify:** You can access `"/mnt/d/MY WORK/Dinosaur DNA Reconstruction"` in WSL
4. **Test:** `python3 main_pipeline.py --info` to check system

**Remember:** Everything stays in your existing folder - no new directories created! 🎉
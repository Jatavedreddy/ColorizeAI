# 🔧 Environment Setup & Troubleshooting Guide

## 📋 Problem Summary

**Issue**: Application behavior differs between conda environments
- **`colorize` environment**: Old UI works, but DDColor shows "inactive"
- **`base` environment**: DDColor shows "active" but UI doesn't work properly

**Root Cause**: Missing dependencies in different environments

---

## ✅ **SOLUTION: Use `colorize` Environment with DDColor Dependencies**

### **Step 1: Install Missing Dependencies**

The `colorize` environment was missing DDColor-specific packages. These have now been installed:

```bash
conda activate colorize
pip install timm>=0.9.0 lmdb>=1.4.0 modelscope tensorboard huggingface_hub
```

**Status**: ✅ **COMPLETE** (Already done for you)

---

## 🚀 **How to Run the Application**

### **Option 1: Using the Launcher Script (Recommended)**

```bash
cd /Users/jatavedreddy/ColorizeAI
./run.sh
```

This script automatically:
- Activates the `colorize` environment
- Checks for DDColor weights
- Starts the application
- Shows the URL (http://127.0.0.1:7860)

### **Option 2: Manual Activation**

```bash
cd /Users/jatavedreddy/ColorizeAI
conda activate colorize
python main.py
```

### **Option 3: Direct Conda Run**

```bash
cd /Users/jatavedreddy/ColorizeAI
conda run -n colorize python main.py
```

---

## 🔍 **Verification Steps**

### **1. Check Environment Has All Dependencies**

```bash
conda activate colorize
pip list | grep -E "(gradio|torch|timm|lmdb|modelscope)"
```

**Expected Output:**
```
gradio             5.38.0
torch              2.7.1
timm               1.0.20
lmdb               1.7.5
modelscope         1.31.0
```

### **2. Verify DDColor Can Import**

```bash
conda activate colorize
python -c "
import sys
sys.path.insert(0, '../ddcolor/DDColor-master copy')
from basicsr.archs.ddcolor_arch import DDColor
print('✅ DDColor imports successfully')
"
```

### **3. Check DDColor Weights Exist**

```bash
ls -lh "../ddcolor/DDColor-master copy/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
```

**Expected**: File should be ~870 MB

### **4. Test Application Startup**

```bash
conda activate colorize
python main.py
```

**Expected Console Output:**
```
Using device: cpu
✓ DDColor (large) loaded successfully from ...
* Running on local URL:  http://127.0.0.1:7860
```

### **5. Verify UI Shows DDColor Active**

Open http://127.0.0.1:7860 and check the status bar at the top:
- ✅ Should show: **"🟢 DDColor active (TorchScript weights detected)"**
- ❌ NOT: "🔴 DDColor not available"

---

## 🐛 **Common Issues & Fixes**

### **Issue 1: "DDColor not available" in UI**

**Symptoms:**
- UI shows red "DDColor not available" status
- Console doesn't show "✓ DDColor loaded successfully"

**Causes:**
1. Missing dependencies (timm, lmdb, modelscope)
2. Weights file not found
3. Wrong Python path

**Fixes:**

```bash
# Fix 1: Install dependencies
conda activate colorize
pip install timm>=0.9.0 lmdb>=1.4.0 modelscope

# Fix 2: Download weights
python tools/download_ddcolor_weights.py --model-size large

# Fix 3: Check Python path
python -c "import sys; print(sys.executable)"
# Should show: /opt/anaconda3/envs/colorize/bin/python
```

---

### **Issue 2: UI Not Working (Images Won't Upload)**

**Symptoms:**
- Can't upload images
- Tabs not switching properly
- Buttons not responding

**Cause:** Running in wrong environment (base instead of colorize)

**Fix:**
```bash
# Stop any running instance
pkill -f "python main.py"

# Start with correct environment
conda activate colorize
python main.py
```

---

### **Issue 3: "gradio" Module Issues**

**Symptoms:**
- `type="numpy"` errors
- Image upload fails
- Component errors

**Cause:** Gradio version mismatch or cached files

**Fix:**
```bash
conda activate colorize
pip install --upgrade gradio
python main.py
```

---

### **Issue 4: Environment Conflicts**

**Symptoms:**
- Works in one terminal, not in another
- Inconsistent behavior

**Cause:** Different conda environments active

**Fix:**
```bash
# Always check which environment is active
conda info --envs
# Look for the "*" next to active env

# Always activate colorize before running
conda activate colorize
which python  # Should show: /opt/anaconda3/envs/colorize/bin/python
```

---

## 📦 **Environment Comparison**

| Feature | `base` Environment | `colorize` Environment |
|---------|-------------------|----------------------|
| Python Version | 3.11 | 3.10 ✅ |
| Gradio | 5.38.0 | 5.38.0 ✅ |
| PyTorch | 2.7.1 | 2.7.1 ✅ |
| DDColor deps (timm, lmdb) | ✅ Present | ✅ **Now Present** |
| UI Working | ❌ Issues | ✅ **Working** |
| DDColor Active | ✅ Yes | ✅ **Now Yes** |
| **RECOMMENDED** | ❌ No | ✅ **YES** |

---

## 🎯 **Best Practices**

### **Always Use the `colorize` Environment**

1. **Add to your shell profile** (~/.zshrc or ~/.bashrc):
   ```bash
   alias colorize-app='cd /Users/jatavedreddy/ColorizeAI && conda activate colorize && python main.py'
   ```

2. **Or use the launcher script**:
   ```bash
   cd /Users/jatavedreddy/ColorizeAI
   ./run.sh
   ```

3. **Check environment before running**:
   ```bash
   conda info --envs | grep "*"
   ```

---

## 🔄 **Quick Reference Commands**

### **Start Application**
```bash
cd /Users/jatavedreddy/ColorizeAI
./run.sh
```

### **Stop Application**
```bash
pkill -f "python main.py"
# or press Ctrl+C in the terminal
```

### **Check Status**
```bash
# Check if running
ps aux | grep "python main.py"

# Check environment
conda activate colorize
python -c "from colorizeai.core.ddcolor_model import is_ddcolor_available; print('DDColor:', is_ddcolor_available())"
```

### **Test Imports**
```bash
conda activate colorize
python -c "
from colorizeai.core.colorization import colorize_highres
from colorizeai.core.ddcolor_model import is_ddcolor_available
print('✓ Imports work')
print('DDColor available:', is_ddcolor_available())
"
```

---

## 📝 **For Your Presentation Tomorrow**

### **Before the Demo:**

1. **Activate the correct environment**:
   ```bash
   conda activate colorize
   ```

2. **Start the application**:
   ```bash
   cd /Users/jatavedreddy/ColorizeAI
   python main.py
   ```

3. **Verify in browser** (http://127.0.0.1:7860):
   - Check status shows: "🟢 DDColor active"
   - Test uploading an image in Basic tab
   - Make sure all tabs are working

4. **Keep the terminal open** during presentation

### **If Something Goes Wrong During Demo:**

1. **Quick restart**:
   ```bash
   # In terminal: Ctrl+C to stop
   # Then: python main.py
   ```

2. **Fallback**: If DDColor fails, the app automatically uses ECCV16/SIGGRAPH17 (still impressive!)

3. **Browser refresh**: If UI freezes, refresh the browser page (Ctrl+F5 or Cmd+Shift+R)

---

## ✅ **Current Status**

- ✅ `colorize` environment has all dependencies
- ✅ DDColor weights downloaded (870 MB)
- ✅ Application running successfully
- ✅ UI working properly
- ✅ DDColor showing as active
- ✅ Ready for presentation!

---

## 📞 **Quick Troubleshooting Checklist**

Before asking for help, verify:

- [ ] Running in `colorize` environment? (`conda info --envs` shows `*` next to colorize)
- [ ] All dependencies installed? (`pip list | grep -E "(timm|lmdb|modelscope)"`)
- [ ] Weights file exists? (`ls -lh "../ddcolor/DDColor-master copy/modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"`)
- [ ] No other instance running? (`pkill -f "python main.py"` before starting)
- [ ] Correct directory? (`pwd` should show `.../ColorizeAI`)
- [ ] Browser cache cleared? (Try Ctrl+F5 or incognito mode)

---

**Last Updated**: October 19, 2025  
**Environment**: `colorize` (Python 3.10)  
**Status**: ✅ Fully Functional

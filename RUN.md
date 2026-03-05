# ColorizeAI - Command Reference 🚀

Welcome to the command reference for **ColorizeAI**. This guide provides the essential terminal commands to start the application, run evaluations, and execute automated tests across the modernized Flask + Web architecture.

## 🌟 1. Running the Web Application

The architecture uses a Python Flask backend serving a static HTML/JS frontend. Wait for the server to load its robust ML models, then visit `http://127.0.0.1:8080/` in your browser.

```bash
# Start the Flask Backend API & Web Server
python main.py
```

*(Optionally, if you have your environment setup script `. run.sh`, you can also use that.)*

---

## 📊 2. Benchmarking & Evaluating Performance

Testing algorithms against localized ground-truth datasets for validation mapping and algorithmic tuning. Test metrics compute **PSNR, SSIM, LPIPS**, and **Colorfulness**.

```bash
# 1. Quick Evaluation (test_images directory)
# Benchmarks ECCV16, SIGGRAPH17, DDColor, and our Adaptive Fusion
python scripts/evaluate_test_images.py

# 2. Heavy Large-Scale Validation (500 Samples Benchmark)
# A deeper multi-model validation dumping detailed CSV rankings in outputs/
python scripts/benchmark_500.py
python scripts/validate_models_500.py
```

---

## 🛠 3. Environment & Dependencies Management

If switching machines or updating packages.

```bash
# Install the core prerequisites
pip install -r requirements.txt

# Activate the conda environment (if using Conda)
conda activate colorize
```

## 🧪 4. For colab open the link : 
https://colab.research.google.com/github/Jatavedreddy/ColorizeAI/blob/main/Colab_GPU_Deploy.ipynb


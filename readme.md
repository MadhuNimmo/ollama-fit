```md
# OllamaFit  

A simple command-line tool to check which **Ollama AI models** your system can run smoothly.  

## 🚀 What is OllamaFit?  

OllamaFit scans your system's **CPU, RAM, and GPU** to see which Ollama models will work well. It fetches the latest model info from Ollama’s website, calculates memory needs based on model size & quantization, and gives you a clear **compatibility report**.  

## 🔥 Features  

✅ **Automatic hardware detection** (CPU, RAM, GPU/VRAM)  
✅ **Search models** by name (e.g., `llama`, `code`, etc.)  
✅ **Smart memory calculations** based on model size & quantization  
✅ **Supports CPU-only inference** for small quantized models  
✅ **Handles Apple Silicon shared memory**  
✅ **Caches results** to reduce web requests  

## 🛠 Installation  

```bash
# Clone the repo
git clone https://github.com/yourusername/ollamafit.git
cd ollamafit

# Create a virtual environment
python -m venv ollamafit
source ollamafit/bin/activate  # Windows: ollamafit\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Use  

Check all compatible models:  
```bash
python ollamafit_cli.py
```

Search for specific models:  
```bash
python ollamafit_cli.py --search llama  
```

Limit results (for quick testing):  
```bash
python ollamafit_cli.py --max-models 5
```

Save results to a file:  
```bash
python ollamafit_cli.py --output my_results.json
```

## ⚡ How It Works  

1. Detects your system’s **CPU, RAM, and GPU** specs  
2. Fetches the latest **Ollama models**  
3. **Calculates memory needs** for each model based on:  
   - Model **size (parameters)**  
   - **Quantization** (bit-level optimization)  
   - **Extra overhead** for runtime needs  
4. Applies **special handling** for:  
   - **Apple Silicon (shared RAM/VRAM)**  
   - **CPU-only inference for small models**  
5. **Shows which models you can run**  

## ✅ What’s Working Well  

- **Accurate Hardware Checks**: RAM, VRAM, CPU & GPU detection  
- **Quantization Support**: Accounts for memory savings from model quantization  
- **Reasonable Memory Overhead**: +20% VRAM, +30% RAM for extra usage  
- **CPU-Only Compatibility**: Allows small models (≤3B on **4GB RAM**, ≤7B on **6GB RAM**)  

## ❌ What Can Be Improved  

🔹 **Better VRAM Estimation** → Current formula is close, but doesn’t factor in **context length** & **batch size**  
🔹 **More Detailed Quantization Handling** → Different methods (QLoRA, GPTQ, GGUF) impact memory differently  
🔹 **Context Length Awareness** → Long prompts use more memory, but we don’t account for this yet
🔹 **Better Integrated GPU Handling** → Some GPUs (especially on Windows/Linux) can use shared RAM  

## 📜 Requirements  

- **Python 3.7+**  
- **Internet connection** (to fetch model data)  
- **Dependencies**: `requests`, `beautifulsoup4`, `psutil`  

## 📝 License  

**MIT License** – Free to use, modify & improve!  

---

🔹 **Feedback & Contributions Welcome!** 🚀
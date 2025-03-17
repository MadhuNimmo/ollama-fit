# OllamaFit

A command-line tool to identify which Ollama AI models are compatible with your system's hardware.

## What is OllamaFit?

OllamaFit analyzes your computer's specifications (RAM, CPU, GPU) and determines which Ollama models will run efficiently on your system. It fetches model information directly from Ollama's website, calculates memory requirements based on model size and quantization, and provides a clear compatibility report.

## Features

- Automatic hardware detection (CPU, RAM, GPU/VRAM)
- Search functionality to find specific model types
- Detailed memory requirement calculations
- Special handling for quantized models
- Compatibility reports with RAM requirements
- Results caching to avoid excessive web requests

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ollamafit.git
cd ollamafit

# Create a virtual environment
python -m venv ollamafit
source ollamafit/bin/activate  # On Windows: ollamafit\Scripts\activate

# Install dependencies
pip install requests beautifulsoup4 psutil
```

## Usage

Basic usage:

```bash
python ollamafit_cli.py
```

Search for specific models:

```bash
python ollamafit_cli.py --search code  # Find code-oriented models
python ollamafit_cli.py --search llama  # Find LLaMA models
```

Limit results for testing:

```bash
python ollamafit_cli.py --max-models 5
```

Specify output file:

```bash
python ollamafit_cli.py --output my_results.json
```

## How It Works

1. OllamaFit detects your system's hardware specifications
2. It fetches model information from the Ollama website
3. For each model, it determines memory requirements based on:
   - Parameter count (model size)
   - Quantization level (bits per parameter)
   - Runtime overhead
4. It applies specialized rules for CPU-only systems and small/medium quantized models
5. Results show which models and tags are compatible with your hardware

## Requirements

- Python 3.7 or higher
- Internet connection to access Ollama website
- Libraries: requests, beautifulsoup4, psutil

## License
MIT License

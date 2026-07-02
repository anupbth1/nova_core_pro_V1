# 🚀 Nova Core - Post-Transformer LLM

## What is Nova Core?

A completely new architecture that replaces:

- ❌ **Attention** (O(n²)) → ✅ **Field Dynamics** (O(n))
- ❌ **Tokens** (discrete) → ✅ **Pulses** (continuous)
- ❌ **Fixed Layers** → ✅ **Adaptive Depth Cores**

## Quick Start

```bash
# Build
cargo build --release

# Run CLI
cargo run --release -- info
cargo run --release -- run --input "Hello world"
cargo run --release -- chat
cargo run --release -- bench
cargo run --release -- speed --pulses 1000

# Train Nova Core (generates data + trains)
cargo run --release -- train --examples 100 --epochs 10

# Smart Chat (with vocabulary-based readable output)
cargo run --release -- smart-chat

# Full benchmark suite
cargo run --release -- full-bench

# Auto-improve based on benchmarks
cargo run --release -- improve
```

## 🎯 Complete LLM Training System

Nova Core now has a **complete pipeline** for building, training, and managing LLMs:

### 1. Dataset Management (`nova dataset`)

Load datasets from local files or Hugging Face with flexible column mapping and filtering.

#### Load Local Files
```bash
# CSV file with custom column mapping
cargo run --release -- dataset load --file data.csv --input-col text --target-col label

# JSON file
cargo run --release -- dataset load --file data.json --input-col question --target-col answer

# JSONL file with prefix/suffix
cargo run --release -- dataset load --file data.jsonl --input-col input --target-col output --prefix "Q: " --suffix " A:"

# Text file (auto-creates next-sentence prediction pairs)
cargo run --release -- dataset load --file book.txt

# Limit rows
cargo run --release -- dataset load --file large.csv --max-rows 5000
```

#### Load from Hugging Face
```bash
# Basic dataset
cargo run --release -- dataset hf --name imdb --input-col text --target-col label --max-rows 1000

# With subset and split
cargo run --release -- dataset hf --name wikitext --subset wikitext-2-raw-v1 --split train --input-col text --target-col text

# Tiny Shakespeare
cargo run --release -- dataset hf --name tiny_shakespeare --input-col text --target-col text --max-rows 500
```

#### Dataset Utilities
```bash
# Show statistics
cargo run --release -- dataset stats

# Save to JSONL
cargo run --release -- dataset save --output my_dataset.jsonl
```

### 2. Model Management (`nova model`)

Save, load, and manage Nova Core models in `.nova` format.

```bash
# List available models
cargo run --release -- model list

# Save current model
cargo run --release -- model save --name my-model

# Load a model
cargo run --release -- model load --name my-model

# Delete a model
cargo run --release -- model delete --name my-model

# Upload to Hugging Face Hub
cargo run --release -- model upload --name my-model --repo username/my-nova-model --token hf_xxxx

# Download from Hugging Face Hub
cargo run --release -- model download --repo username/my-nova-model --name my-model --token hf_xxxx
```

### 3. Train ONE Model on MULTIPLE Datasets (`nova multi-hf-train`) 🆕

Train a **single model** on **multiple Hugging Face datasets** sequentially. The model learns from all datasets and accumulates knowledge.

```bash
# Train on 3 datasets sequentially (single model)
cargo run --release -- multi-hf-train \
  --datasets "imdb,wikitext,tiny_shakespeare" \
  --max-rows 300 \
  --dim 64 \
  --cores 5 \
  --model-name "my-universal-model"

# With Pro mode (adaptive iterations, pattern caching)
cargo run --release -- multi-hf-train \
  --datasets "imdb,wikitext,tiny_shakespeare" \
  --max-rows 500 \
  --dim 128 \
  --cores 7 \
  --model-name "my-pro-model" \
  --pro

# With NEURAL mode (real gradient-based learning through cores + field)
cargo run --release -- multi-hf-train \
  --datasets "tiny_shakespeare" \
  --max-rows 200 \
  --neural \
  --model-name "my-neural-model"

# With custom column mapping
cargo run --release -- multi-hf-train \
  --datasets "imdb" \
  --input-col text \
  --target-col label \
  --max-rows 300 \
  --model-name "imdb-sentiment"
```

**Key Features:**
- ✅ ONE model trained on ALL datasets (not separate models)
- ✅ Shared vocabulary across all datasets
- ✅ Progress bar with ETA during training
- ✅ N-gram pattern learning for text generation
- ✅ Pro mode for adaptive learning
- ✅ **Neural mode** (`--neural`): Real gradient-based training through cores + field (full vector error)

### 4. Train with Hugging Face Datasets (`nova hf-train`)


End-to-end training pipeline: download → split → train → evaluate → save.

```bash
# Train on IMDB sentiment
cargo run --release -- hf-train --dataset imdb --input-col text --target-col label --max-rows 500 --epochs 10 --model-name imdb-model

# Train on Tiny Shakespeare
cargo run --release -- hf-train --dataset tiny_shakespeare --input-col text --target-col text --max-rows 1000 --epochs 20 --model-name shakespeare-model

# Train with subset
cargo run --release -- hf-train --dataset wikitext --subset wikitext-2-raw-v1 --input-col text --target-col text --max-rows 500 --epochs 15 --model-name wikitext-model
```

### 4. Built-in Training (`nova train`)

Quick training with built-in data generator (no external files needed):

```bash
cargo run --release -- train --examples 200 --epochs 15
```

### 5. Smart Chat

Interactive chat with vocabulary-based readable output:

```bash
cargo run --release -- smart-chat
```

In Smart Chat mode:
- Type any message to get a response
- Type `train 50` to train on 50 examples (hash-based, fast)
- Type `neural 50` to train on 50 examples (neural, real learning through cores + field)
- Type `stats` to see model performance
- Type `load <name>` to load a saved model
- Type `save <name>` to save the current model
- Type `timeout <secs>` to set response timeout (default: 30s, range: 5-300)
- Type `exit` to quit

**New in Smart Chat:**
- ✅ **Input timeout**: Chat won't hang forever - auto-detects no-input after 30s
- ✅ **Neural training**: Type `neural 50` for real gradient-based learning
- ✅ **Save/Load models**: Type `save my-model` or `load my-model` directly
- ✅ **Configurable timeout**: Type `timeout 60` to change to 60 seconds

## 📊 NOVA CORE - Complete Summary

### ✅ Working Features
| Component | Status | Description |
|-----------|--------|-------------|
| Nova Field | ✅ Working | O(n) attention replacement |
| Nova Pulses | ✅ Working | Continuous tokens (no vocabulary) |
| Adaptive Cores | ✅ Working | 5 specialized cores with dynamic depth |
| CLI Interface | ✅ Working | 14+ commands |
| Chat Mode | ✅ Working | Interactive conversation |
| Training Pipeline | ✅ Working | Gradient descent with vocabulary |
| Smart Chat | ✅ Working | Readable word output via vocabulary |
| Data Generator | ✅ Working | Sentiment, math, Q&A templates |
| **Dataset Manager** | ✅ **NEW** | CSV/JSON/JSONL/Text/Parquet parsing |
| **Column Mapping** | ✅ **NEW** | Map any column to input/target |
| **Filter Conditions** | ✅ **NEW** | Equals, Contains, MinLength, MaxLength, Regex, NonEmpty |
| **HF Dataset Download** | ✅ **NEW** | Python bridge to Hugging Face datasets |
| **Model Save/Load** | ✅ **NEW** | .nova format with full state |
| **Model Upload/Download** | ✅ **NEW** | Hugging Face Hub integration |
| **HF Train Pipeline** | ✅ **NEW** | End-to-end: download → train → save |
| Rust Implementation | ✅ Working | Fast, compiled, no Python overhead |

### 📈 Performance Metrics
- Processing speed: 400-600 microseconds per query
- Memory usage: ~50-100MB
- CPU only: No GPU required
- Response time: 0.5ms for short inputs

### 🏗️ Architecture
```
Input Text → Nova Pulses → 5 Specialized Cores → Nova Field → Output
              (O(n))      (Adaptive Depth)      (O(n))      (Numbers/Words)
                                                              ↑
                                                    Vocabulary Mapper
```

### Dataset Pipeline
```
Local Files (CSV/JSON/JSONL/TXT) ─┐
                                   ├──→ Column Mapping → Filtering → Training Examples
Hugging Face Datasets ─────────────┘
                                              ↓
                                     Train/Validation Split
                                              ↓
                                     NovaTrainer (Gradient Descent)
                                              ↓
                                     Model Save (.nova format)
                                              ↓
                                     Hugging Face Hub Upload
```

## CLI Commands Reference

```bash
# === Core Commands ===

# Process text
cargo run --release -- run --input "your text"

# Interactive chat (numeric output)
cargo run --release -- chat

# Smart chat (readable word output with training)
cargo run --release -- smart-chat

# Train the model (built-in data)
cargo run --release -- train --examples 100 --epochs 10

# Run benchmarks
cargo run --release -- bench
cargo run --release -- full-bench

# Auto-improve
cargo run --release -- improve

# Speed test
cargo run --release -- speed --pulses 1000

# Architecture info
cargo run --release -- info

# === Dataset Commands ===

# Load local file
cargo run --release -- dataset load --file data.csv --input-col text --target-col label

# Load from Hugging Face
cargo run --release -- dataset hf --name imdb --input-col text --target-col label --max-rows 1000

# Show dataset stats
cargo run --release -- dataset stats

# Save dataset to JSONL
cargo run --release -- dataset save --output dataset.jsonl

# === Model Commands ===

# List models
cargo run --release -- model list

# Save model
cargo run --release -- model save --name my-model

# Load model
cargo run --release -- model load --name my-model

# Delete model
cargo run --release -- model delete --name my-model

# Upload to Hugging Face
cargo run --release -- model upload --name my-model --repo username/repo --token hf_xxx

# Download from Hugging Face
cargo run --release -- model download --repo username/repo --name my-model --token hf_xxx

# === HF Train Command ===

# End-to-end training from Hugging Face dataset
cargo run --release -- hf-train --dataset imdb --input-col text --target-col label --max-rows 500 --epochs 10 --model-name my-model
```

## Dataset Format Support

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | .csv | Comma-separated values with header row |
| JSON | .json | JSON array of objects |
| JSONL | .jsonl, .ndjson | JSON lines (one object per line) |
| Text | .txt | Plain text (auto-creates next-sentence pairs) |
| Parquet | .parquet | Columnar format (via Python bridge) |

## Filter Conditions

When loading datasets, you can apply filters to clean your data:

| Filter | Description |
|--------|-------------|
| Equals | Keep rows where column equals value |
| Contains | Keep rows where column contains substring |
| MinLength | Keep rows where column length > N |
| MaxLength | Keep rows where column length < N |
| Regex | Keep rows matching regex pattern |
| NonEmpty | Remove rows with empty columns |

## 🔄 Hugging Face Model Conversion

Convert any Hugging Face model (TinyLlama, Llama, GPT-2, etc.) to Nova `.nova` format:

### Using the Universal Converter (Recommended)

```bash
# Install dependencies
pip install transformers torch

# Convert TinyLlama (open access)
python convert_to_nova_json.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-tinyllama

# Convert Llama (requires HF token for gated models)
python convert_to_nova_json.py --model meta-llama/Llama-3.2-1B --name my-llama --token hf_your_token_here

# Custom dimension
python convert_to_nova_json.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-model --dim 128
```

### Using Specialized Converters

```bash
# TinyLlama converter
python convert_tinyllama_to_nova.py

# Llama converter
python convert_llama_to_nova.py --token hf_your_token_here
```

### Load and Test Converted Models

```bash
# Load the converted model
cargo run --release -- model load --name my-tinyllama

# Chat with it
cargo run --release -- smart-chat
```

### ⚠️ Important Notes
- All converters now output **JSON format** compatible with Rust's `ModelSnapshot`
- Old binary format (starting with `NOVA` magic bytes) is **no longer supported**
- If you have old binary `.nova` files, retrain or reconvert them using the new scripts
- The converter extracts weights from HF model layers into Nova cores and creates a vocabulary from the tokenizer

## Requirements

- **Rust** (latest stable)
- **Python 3** (only for Hugging Face dataset download/integration)
- **pip install datasets** (for Hugging Face dataset integration)
- **pip install huggingface_hub** (for model upload/download)
- **pip install transformers torch** (for HF model conversion)


## Project Structure

```
src/
├── main.rs          # CLI entry point with all commands
├── loom.rs          # NovaLoom - main orchestration engine
├── core.rs          # NovaCore - adaptive processing cores
├── field.rs         # NovaField - O(n) field dynamics
├── pulse.rs         # NovaPulse - continuous vector representation
├── trainer.rs       # Training pipeline with gradient descent
├── dataset.rs       # Dataset manager (CSV/JSON/JSONL/Text/HF)
├── model.rs         # Model manager (save/load/upload/download)
└── benchmark/       # Benchmark suite
    ├── mod.rs
    ├── tasks.rs
    ├── data.rs
    ├── improve.rs
    ├── metrics.rs
    └── compare.rs
```


Task Completed

🎯 Naya LLM Kaise Banaye - Step by Step Guide
Yeh raha complete workflow naya LLM banane ka - from scratch to trained model:

🔹 Step 1: Pehle Model Banaye (Create a fresh model)

# Kuch nahi karna - jab bhi 'nova' command chalate ho, ek naya model auto-create hota hai
# Lekin aap model save kar sakte ho:
cargo run --release -- model save --name mera-pehla-model
🔹 Step 2: Dataset Select Karein (Choose your dataset)
Option A: Local file se data lo


# Agar aapke paas CSV/JSON/JSONL file hai:
cargo run --release -- dataset load --file data.csv --input-col text --target-col label
Option B: Hugging Face se data download karo


# IMDB sentiment dataset
cargo run --release -- dataset hf --name imdb --input-col text --target-col label --max-rows 1000

# Tiny Shakespeare (text generation ke liye)
cargo run --release -- dataset hf --name tiny_shakespeare --input-col text --target-col text --max-rows 500

# WikiText (language modeling)
cargo run --release -- dataset hf --name wikitext --subset wikitext-2-raw-v1 --split train --input-col text --target-col text --max-rows 500

# Apna koi bhi HF dataset - bas name aur columns batado!
🔹 Step 3: Dataset Check Karein (Verify your data)

cargo run --release -- dataset stats
Yeh batayega: kitne examples hain, average length kya hai, unique words kitne hain.

🔹 Step 4: Model Train Karein (Train the model)
Quick training (built-in data):


cargo run --release -- train --examples 100 --epochs 10
Real training with your dataset (EK CLICK MEIN):


# Yeh ek hi command mein: download → split → train → evaluate → save karega
cargo run --release -- hf-train --dataset imdb --input-col text --target-col label --max-rows 500 --epochs 10 --model-name imdb-sentiment-model
🔹 Step 5: Model Test Karein (Test your model)

# Smart Chat - vocabulary ke saath readable output
cargo run --release -- smart-chat

# Andar type karo:
#   "hello" → model response dekhein
#   "train 50" → hash-based training (fast)
#   "neural 50" → neural training (real learning through cores + field)
#   "stats" → model performance dekhein
#   "save my-model" → model save karo
#   "load my-model" → model load karo
#   "timeout 60" → response timeout set karo (default: 30s)
#   "exit" → bahar nikle
🔹 Step 6: Model Save/Load Karein (Save & Load)

# Save karo
cargo run --release -- model save --name mera-trained-model

# List dekho
cargo run --release -- model list

# Load karo
cargo run --release -- model load --name mera-trained-model
🔹 Step 7: Model Share Karein (Upload to Hugging Face)

# Upload karo HF Hub pe
cargo run --release -- model upload --name mera-trained-model --repo mera-username/mera-nova-model --token hf_your_token_here

# Kisi aur ka model download karo
cargo run --release -- model download --repo mera-username/mera-nova-model --name mera-trained-model --token hf_your_token_here
🚀 Example: Complete Workflow (Ek Hi Baar Mein)
Yeh lo pura workflow ek saath - IMDB sentiment analysis model banane ke liye:


# 1. Dataset download karo
cargo run --release -- dataset hf --name imdb --input-col text --target-col label --max-rows 1000

# 2. Stats dekho
cargo run --release -- dataset stats

# 3. Train karo (download + split + train + save ek saath)
cargo run --release -- hf-train --dataset imdb --input-col text --target-col label --max-rows 500 --epochs 15 --model-name imdb-model

# 4. Test karo smart chat mein
cargo run --release -- smart-chat
📊 Different Types ke LLM Kaise Banaye
Model Type	Dataset	Command
Sentiment Analysis	IMDB	hf-train --dataset imdb --input-col text --target-col label
Text Generation	Tiny Shakespeare	hf-train --dataset tiny_shakespeare --input-col text --target-col text
Language Model	WikiText	hf-train --dataset wikitext --subset wikitext-2-raw-v1 --input-col text --target-col text
Q&A Model	Any QA dataset	hf-train --dataset your-dataset --input-col question --target-col answer
Custom Model	Apni CSV/JSON file	dataset load --file data.csv + train
💡 Important Tips
Zyada data = better model: --max-rows 5000 use karo agar aapke paas zyada data hai
Zyada epochs = better learning: --epochs 20 ya --epochs 50 try karo
Column mapping important hai: Har dataset ke columns alag hote hain - --input-col aur --target-col sahi set karo
Pehle chhota data try karo: 100-500 rows se start karo, phir badhao
Python required for HF: pip install datasets huggingface_hub karna mat bhoolo



Jitne zyada cores, utna better processing — but more CPU/GPU usage.

📊 Combination Examples
Use Case	dim	cores	Command
Quick testing	64	3	--dim 64 --cores 3
Balanced (default)	128	5	--dim 128 --cores 5
Quality focused	256	7	--dim 256 --cores 7
Max power	512	9	--dim 512 --cores 9
🔬 Visual Representation
text
dim = 64 means:
┌─────────────────────────────────────────────────────────────┐
│ Pulse = [0.23, -0.45, 0.67, 0.12, ..., 0.89]              │
│          └────────────────┬─────────────────┘              │
│                    64 numbers                              │
└─────────────────────────────────────────────────────────────┘

cores = 5 means:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Syntax Core  │  │ Semantic Core│  │ Memory Core  │  ──→ Output
│ (structure)  │→ │ (meaning)    │→ │ (recall)     │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                  ↓                 ↓
┌──────────────┐  ┌──────────────┐
│Reasoning Core│  │ Pattern Core │
│ (logic)      │  │ (repetition) │
└──────────────┘  └──────────────┘
⚡ Performance Impact
Config	Training Speed	Memory	Quality
dim=64, cores=3	⚡ Fastest	50MB	Basic
dim=128, cores=5	🚀 Fast	100MB	Good
dim=256, cores=7	🐌 Slower	250MB	Better
dim=512, cores=9	🐢 Slow	1GB	Best
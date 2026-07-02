#!/usr/bin/env python3
"""
Convert HuggingFace models to Nova .nova JSON format

This converter creates .nova files in the JSON format that Nova Core expects.
The output matches the ModelSnapshot struct in src/model.rs exactly.

Usage:
    python convert_to_nova_json.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-model
    python convert_to_nova_json.py --model meta-llama/Llama-3.2-1B --name llama-model --token hf_xxx
"""

import argparse
import json
import numpy as np
import os
import sys
import struct
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Convert HF model to Nova .nova format")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model name")
    parser.add_argument("--name", default="converted-model", help="Output model name")
    parser.add_argument("--token", default="", help="HF token for gated models")
    parser.add_argument("--dim", type=int, default=64, help="Nova dimension (default: 64)")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 HF Model → Nova JSON Converter")
    print("=" * 60)
    print(f"   Model: {args.model}")
    print(f"   Name:  {args.name}")
    print(f"   Dim:   {args.dim}")

    # Step 1: Download model
    print("\n📥 Step 1: Downloading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers torch")
        sys.exit(1)

    token = args.token if args.token else None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cpu",
            token=token
        )
        print(f"   ✅ Loaded: {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   💡 For gated models (like Llama), use --token hf_your_token_here")
        sys.exit(1)

    # Step 2: Extract weights
    print("\n📊 Step 2: Extracting weights...")
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
    print(f"   ✅ Extracted {len(weights)} tensors")

    # Step 3: Count layers
    num_layers = 0
    for name in weights.keys():
        if "layers." in name:
            try:
                idx = int(name.split("layers.")[1].split(".")[0])
                num_layers = max(num_layers, idx + 1)
            except:
                pass
    print(f"   Found {num_layers} layers")

    # Step 4: Build Nova ModelSnapshot
    print("\n🔄 Step 3: Building Nova format...")

    # Core names matching NovaLoom::new()
    core_names = ["syntax", "semantic", "memory", "reasoning", "pattern"]
    core_sizes = [256, 256, 512, 256, 128]  # Memory sizes from loom.rs

    # Create core snapshots
    cores = []
    for i, (cname, csize) in enumerate(zip(core_names, core_sizes)):
        # Try to extract weights from corresponding layer
        memory = np.zeros(csize, dtype=np.float32)
        internal_state = np.zeros(64, dtype=np.float32)
        gate = 0.5

        if i < num_layers:
            # Extract MLP weights for this core
            prefix = f"model.layers.{i}."
            layer_weights = []
            for wname, wval in weights.items():
                if wname.startswith(prefix):
                    layer_weights.append(wval.flatten())

            if layer_weights:
                # Aggregate layer weights into core memory
                all_w = np.concatenate(layer_weights)
                # Take mean of absolute values as a simple compression
                step = max(1, len(all_w) // csize)
                for j in range(min(csize, len(all_w) // step)):
                    memory[j] = float(np.mean(np.abs(all_w[j * step:(j + 1) * step])))

                # Internal state from first few values
                for j in range(min(64, len(all_w))):
                    internal_state[j] = float(all_w[j])

                gate = 0.8  # Higher gate for converted layers

        cores.append({
            "id": i,
            "name": cname,
            "memory": memory.tolist(),
            "internal_state": internal_state.tolist(),
            "gate": gate
        })
        print(f"   ✅ Core {i}: {cname} (size={csize})")

    # Create field state (dim=64)
    dim = args.dim
    field_state = np.zeros(dim, dtype=np.float32)
    field_momentum = np.zeros(dim, dtype=np.float32)

    # Extract attention weights into field state
    all_attn = []
    for layer in range(num_layers):
        for suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", 
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight"]:
            wname = f"model.layers.{layer}.{suffix}"
            if wname in weights:
                all_attn.append(weights[wname].flatten())

    if all_attn:
        all_attn_cat = np.concatenate(all_attn)
        step = max(1, len(all_attn_cat) // dim)
        for j in range(dim):
            field_state[j] = float(np.mean(np.abs(all_attn_cat[j * step:(j + 1) * step])))
        print(f"   ✅ Field state created from {len(all_attn)} attention matrices")

    # Create vocabulary from tokenizer
    print("\n📚 Step 4: Building vocabulary...")
    vocabulary = {}
    try:
        vocab_dict = tokenizer.get_vocab()
        # Take first 5000 words for vocabulary
        count = 0
        for word, idx in vocab_dict.items():
            if count >= 5000:
                break
            # Create a deterministic vector for each word
            seed = sum(ord(c) * (31 ** i) for i, c in enumerate(word[:10])) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            vec = rng.uniform(-0.3, 0.3, dim).astype(np.float32)
            # Normalize
            norm = np.sqrt(np.sum(vec ** 2))
            if norm > 0:
                vec = vec / norm
            vocabulary[word] = vec.tolist()
            count += 1
        print(f"   ✅ Vocabulary: {len(vocabulary)} words")
    except Exception as e:
        print(f"   ⚠️ Could not extract vocabulary: {e}")
        print("   Using empty vocabulary")

    # Build the complete ModelSnapshot
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot = {
        "config": {
            "name": args.name,
            "version": "0.1.0",
            "description": f"Converted from {args.model}",
            "dim": dim,
            "num_cores": len(cores),
            "core_names": core_names,
            "max_iterations": 8,
            "convergence_threshold": 0.08,
            "created_at": now,
            "trained_on": args.model,
            "accuracy": 0.0
        },
        "cores": cores,
        "field_state": field_state.tolist(),
        "field_momentum": field_momentum.tolist(),
        "field_update_count": 0,
        "vocabulary": vocabulary
    }

    # Step 5: Save as JSON .nova file
    print(f"\n💾 Step 5: Saving to {args.output_dir}/{args.name}.nova...")
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"{args.name}.nova")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Saved: {file_size_mb:.2f} MB")

    # Also save a config for reference
    config_path = os.path.join(args.output_dir, f"{args.name}_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "source_model": args.model,
            "nova_name": args.name,
            "dim": dim,
            "cores": len(cores),
            "vocab_size": len(vocabulary),
            "file_size_mb": round(file_size_mb, 2)
        }, f, indent=2)
    print(f"   📄 Config: {config_path}")

    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("=" * 60)
    print(f"\n📁 Model: {output_path}")
    print(f"\n🚀 To use in Nova Core:")
    print(f"   cargo run --release -- model load --name {args.name}")
    print(f"   cargo run --release -- smart-chat")
    print(f"\n💡 Or train further:")
    print(f"   cargo run --release -- hf-train --dataset imdb --input-col text --target-col label --max-rows 500 --epochs 10 --model-name {args.name}-trained")

if __name__ == "__main__":
    main()

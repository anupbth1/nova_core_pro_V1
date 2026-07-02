#!/usr/bin/env python3
"""
TinyLlama 1.1B to .nova Converter (JSON Format)

⚠️ UPDATED: Now outputs JSON format compatible with Rust's ModelSnapshot.
   Previously this script output binary format which was incompatible.

Usage:
    python convert_tinyllama_to_nova.py
    python convert_to_nova_json.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --name my-model

For more options, use: python convert_to_nova_json.py --help
"""

import torch
import numpy as np
import json
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


class TinyLlamaToNovaConverter:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.weights = {}
        
    def download_model(self):
        """Step 1: Download TinyLlama model from HuggingFace"""
        print(f"📥 Downloading {self.model_name}...")
        print("   ✅ Open access — no approval needed!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self.model
    
    def extract_weights(self):
        """Step 2: Extract all weights from model"""
        print("📊 Extracting weights...")
        for name, param in self.model.named_parameters():
            self.weights[name] = param.detach().numpy()
        print(f"✅ Extracted {len(self.weights)} weight tensors")
        return self.weights
    
    def build_nova_snapshot(self):
        """Step 3: Build Nova ModelSnapshot in JSON format"""
        print("🔄 Building Nova JSON format...")
        
        # Count layers
        num_layers = 0
        for name in self.weights.keys():
            if "layers." in name and ".self_attn." in name:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                num_layers = max(num_layers, layer_idx + 1)
        print(f"   Found {num_layers} transformer layers")
        
        dim = 64  # Nova dimension
        core_names = ["syntax", "semantic", "memory", "reasoning", "pattern"]
        core_sizes = [256, 256, 512, 256, 128]
        
        # Build cores
        cores = []
        for i, (cname, csize) in enumerate(zip(core_names, core_sizes)):
            memory = np.zeros(csize, dtype=np.float32)
            internal_state = np.zeros(64, dtype=np.float32)
            gate = 0.5
            
            if i < num_layers:
                prefix = f"model.layers.{i}."
                layer_weights = []
                for wname, wval in self.weights.items():
                    if wname.startswith(prefix):
                        layer_weights.append(wval.flatten())
                
                if layer_weights:
                    all_w = np.concatenate(layer_weights)
                    step = max(1, len(all_w) // csize)
                    for j in range(min(csize, len(all_w) // step)):
                        memory[j] = float(np.mean(np.abs(all_w[j * step:(j + 1) * step])))
                    for j in range(min(64, len(all_w))):
                        internal_state[j] = float(all_w[j])
                    gate = 0.8
            
            cores.append({
                "id": i,
                "name": cname,
                "memory": memory.tolist(),
                "internal_state": internal_state.tolist(),
                "gate": gate
            })
            print(f"   ✅ Core {i}: {cname} (size={csize})")
        
        # Build field state
        field_state = np.zeros(dim, dtype=np.float32)
        field_momentum = np.zeros(dim, dtype=np.float32)
        
        all_attn = []
        for layer in range(num_layers):
            for suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                           "self_attn.v_proj.weight", "self_attn.o_proj.weight"]:
                wname = f"model.layers.{layer}.{suffix}"
                if wname in self.weights:
                    all_attn.append(self.weights[wname].flatten())
        
        if all_attn:
            all_attn_cat = np.concatenate(all_attn)
            step = max(1, len(all_attn_cat) // dim)
            for j in range(dim):
                field_state[j] = float(np.mean(np.abs(all_attn_cat[j * step:(j + 1) * step])))
        
        # Build vocabulary
        print("\n📚 Building vocabulary...")
        vocabulary = {}
        try:
            vocab_dict = self.tokenizer.get_vocab()
            count = 0
            for word, idx in vocab_dict.items():
                if count >= 5000:
                    break
                seed = sum(ord(c) * (31 ** i) for i, c in enumerate(word[:10])) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
                vec = rng.uniform(-0.3, 0.3, dim).astype(np.float32)
                norm = np.sqrt(np.sum(vec ** 2))
                if norm > 0:
                    vec = vec / norm
                vocabulary[word] = vec.tolist()
                count += 1
            print(f"   ✅ Vocabulary: {len(vocabulary)} words")
        except Exception as e:
            print(f"   ⚠️ Could not extract vocabulary: {e}")
        
        # Build complete snapshot
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = {
            "config": {
                "name": "converted_tinyllama",
                "version": "0.1.0",
                "description": f"Converted from {self.model_name}",
                "dim": dim,
                "num_cores": len(cores),
                "core_names": core_names,
                "max_iterations": 8,
                "convergence_threshold": 0.08,
                "created_at": now,
                "trained_on": self.model_name,
                "accuracy": 0.0
            },
            "cores": cores,
            "field_state": field_state.tolist(),
            "field_momentum": field_momentum.tolist(),
            "field_update_count": 0,
            "vocabulary": vocabulary
        }
        
        return snapshot
    
    def save_nova_format(self, snapshot, output_path="models/converted_tinyllama.nova"):
        """Step 4: Save as JSON .nova file"""
        print(f"💾 Saving to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Saved: {file_size_mb:.2f} MB")
        
        # Save config for reference
        config_path = output_path.replace('.nova', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'source_model': self.model_name,
                'nova_name': 'converted_tinyllama',
                'dim': snapshot['config']['dim'],
                'cores': len(snapshot['cores']),
                'vocab_size': len(snapshot['vocabulary']),
                'file_size_mb': round(file_size_mb, 2)
            }, f, indent=2)
        
        return output_path


def main():
    print("=" * 60)
    print("🔄 TinyLlama 1.1B → Nova Converter (JSON Format)")
    print("=" * 60)
    print("   ⚠️  This script now outputs JSON format compatible with Rust")
    print("   💡  For more options, use: python convert_to_nova_json.py --help")
    print("=" * 60)
    
    converter = TinyLlamaToNovaConverter(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    
    # Step 1: Download
    converter.download_model()
    
    # Step 2: Extract weights
    converter.extract_weights()
    
    # Step 3: Build Nova snapshot
    snapshot = converter.build_nova_snapshot()
    
    # Step 4: Save
    converter.save_nova_format(snapshot, "models/converted_tinyllama.nova")
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("📁 Model saved as: models/converted_tinyllama.nova")
    print("=" * 60)
    print("\n🚀 To test in Nova:")
    print("   cargo run --release -- model load --name converted_tinyllama")
    print("   cargo run --release -- smart-chat")


if __name__ == "__main__":
    main()

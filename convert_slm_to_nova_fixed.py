#!/usr/bin/env python3
"""
SLM-10M to .nova Converter (Fixed)
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, GPT2Tokenizer

class SLMToNovaConverter:
    def __init__(self, model_name="liodon-ai/slm-10m"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.weights = {}
        self.dim = 64
        
    def download_model(self):
        """Download SLM model"""
        print(f"📥 Downloading {self.model_name}...")
        
        # Try to load tokenizer with GPT2
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        except:
            print("   ⚠️ Using default tokenizer")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self.model
    
    def extract_weights(self):
        """Extract all weights"""
        print("📊 Extracting weights...")
        for name, param in self.model.named_parameters():
            self.weights[name] = param.detach().numpy()
        print(f"✅ Extracted {len(self.weights)} weight tensors")
        return self.weights
    
    def build_nova_snapshot(self):
        """Build Nova JSON format"""
        print("🔄 Building Nova JSON format...")
        
        # Count layers
        num_layers = 0
        for name in self.weights.keys():
            if "layers." in name:
                try:
                    layer_idx = int(name.split("layers.")[1].split(".")[0])
                    num_layers = max(num_layers, layer_idx + 1)
                except:
                    pass
        print(f"   Found {num_layers} layers")
        
        dim = self.dim
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
                    gate = 0.8
            
            cores.append({
                "id": i,
                "name": cname,
                "memory": memory.tolist(),
                "internal_state": internal_state.tolist(),
                "gate": gate
            })
            print(f"   ✅ Core {i}: {cname}")
        
        # Field state
        field_state = np.zeros(dim, dtype=np.float32)
        field_momentum = np.zeros(dim, dtype=np.float32)
        
        all_attn = []
        for name, weight in self.weights.items():
            if "attn" in name.lower():
                all_attn.append(weight.flatten())
        
        if all_attn:
            all_attn_cat = np.concatenate(all_attn)
            step = max(1, len(all_attn_cat) // dim)
            for j in range(dim):
                field_state[j] = float(np.mean(np.abs(all_attn_cat[j * step:(j + 1) * step])))
        
        # Vocabulary
        print("📚 Building vocabulary...")
        vocabulary = {}
        try:
            if self.tokenizer:
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
            # Create dummy vocabulary
            words = ["hello", "world", "nova", "core", "ai", "intelligence", "learning", "machine", "neural", "network"]
            for i, word in enumerate(words):
                vec = np.random.randn(dim).astype(np.float32) * 0.1
                vocabulary[word] = vec.tolist()
            print(f"   ✅ Created dummy vocabulary: {len(vocabulary)} words")
        
        # Snapshot
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = {
            "config": {
                "name": "converted_slm_10m",
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
    
    def save_nova_format(self, snapshot, output_path="models/converted_slm_10m.nova"):
        """Save as .nova"""
        print(f"💾 Saving to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)
        
        size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Saved: {size:.2f} MB")
        return output_path

def main():
    print("="*60)
    print("🔄 SLM-10M → Nova Converter (Fixed)")
    print("="*60)
    
    converter = SLMToNovaConverter("liodon-ai/slm-10m")
    converter.download_model()
    converter.extract_weights()
    snapshot = converter.build_nova_snapshot()
    converter.save_nova_format(snapshot, "models/converted_slm_10m.nova")
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print("📁 models/converted_slm_10m.nova")
    print("🚀 Load: cargo run -- model load --name converted_slm_10m")

if __name__ == "__main__":
    main()
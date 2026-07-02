#!/usr/bin/env python3
"""
Llama 3.2 1B to .nova Converter (JSON Format)

⚠️ UPDATED: Now outputs JSON format compatible with Rust's ModelSnapshot.
   Previously this script output binary format which was incompatible.

Usage:
    python convert_llama_to_nova.py
    python convert_to_nova_json.py --model meta-llama/Llama-3.2-1B --name my-model --token hf_xxx

For more options, use: python convert_to_nova_json.py --help
"""

import torch
import numpy as np
import json
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaToNovaConverter:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", token=None):
        self.model_name = model_name
        self.token = token
        self.model = None
        self.tokenizer = None
        self.weights = {}
        
    def download_model(self):
        """Step 1: Download Llama model from HuggingFace"""
        print(f"📥 Downloading {self.model_name}...")
        token_kwargs = {"token": self.token} if self.token else {}
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **token_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            **token_kwargs
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
    
    def _project_ssm_parameters(self, layer_idx, d_inner, d_state=16):
        """Project SSM parameters from transformer layer weights."""
        ssm_params = {}
        
        # --- Δ (delta) projection ---
        # Use gate_proj weights (controls information flow)
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        if gate_key in self.weights:
            gate_w = self.weights[gate_key]
            delta_proj = gate_w.mean(axis=0)[:d_inner]
        else:
            delta_proj = np.ones(d_inner) * 0.1
        ssm_params["delta"] = delta_proj.tolist()
        
        # Δ bias
        ssm_params["delta_bias"] = np.zeros(d_inner).tolist()
        
        # --- A_log ---
        a_log = np.zeros((d_inner, d_state), dtype=np.float32)
        for j in range(d_inner):
            for k in range(d_state):
                a_log[j, k] = np.log(k + 1)
        ssm_params["a_log"] = a_log.tolist()
        
        # --- B (input projection) ---
        up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        if up_key in self.weights:
            up_w = self.weights[up_key]
            b_proj = up_w.mean(axis=0)[:d_inner]
        else:
            b_proj = np.ones(d_inner) * 0.01
        b_mat = np.outer(b_proj, np.ones(d_state)) * 0.01
        ssm_params["b"] = b_mat.tolist()
        
        # --- C (output projection) ---
        down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        if down_key in self.weights:
            down_w = self.weights[down_key]
            c_proj = down_w.mean(axis=1)[:d_inner]
        else:
            c_proj = np.ones(d_inner) * 0.01
        c_mat = np.outer(c_proj, np.ones(d_state)) * 0.01
        ssm_params["c"] = c_mat.tolist()
        
        # --- D (skip connection) ---
        ssm_params["d"] = np.ones(d_inner).tolist()
        
        # --- Hidden state h ---
        ssm_params["h"] = np.zeros((d_inner, d_state)).tolist()
        
        # --- RWKV time-mix parameters ---
        norm_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        if norm_key in self.weights:
            norm_vals = self.weights[norm_key][:d_inner]
        else:
            norm_vals = np.ones(d_inner) * 0.5
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        ssm_params["time_mix_x"] = (sigmoid(norm_vals) * 0.5).tolist()
        ssm_params["time_mix_w"] = np.zeros(d_inner).tolist()
        ssm_params["time_mix_key"] = (sigmoid(norm_vals * 0.8) * 0.5).tolist()
        ssm_params["time_mix_value"] = (sigmoid(norm_vals * 0.6) * 0.5).tolist()
        ssm_params["time_mix_receptance"] = (sigmoid(norm_vals * 0.7) * 0.5).tolist()
        ssm_params["prev_x"] = np.zeros(d_inner).tolist()
        
        return ssm_params
    
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
            
            # Project SSM parameters from this layer
            ssm_params = self._project_ssm_parameters(i, dim)
            
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
            
            use_time_mixing = cname in ("memory", "reasoning", "context_window")
            
            cores.append({
                "id": i,
                "name": cname,
                "memory": memory.tolist(),
                "internal_state": internal_state.tolist(),
                "gate": gate,
                # NEW: SSM parameters
                "ssm_delta": ssm_params["delta"],
                "ssm_delta_bias": ssm_params["delta_bias"],
                "ssm_a_log": ssm_params["a_log"],
                "ssm_b": ssm_params["b"],
                "ssm_c": ssm_params["c"],
                "ssm_d": ssm_params["d"],
                "ssm_h": ssm_params["h"],
                "ssm_time_mix_x": ssm_params["time_mix_x"],
                "ssm_time_mix_w": ssm_params["time_mix_w"],
                "ssm_time_mix_key": ssm_params["time_mix_key"],
                "ssm_time_mix_value": ssm_params["time_mix_value"],
                "ssm_time_mix_receptance": ssm_params["time_mix_receptance"],
                "ssm_prev_x": ssm_params["prev_x"],
                "use_ssm": True,
                "use_time_mixing": use_time_mixing,
            })
            print(f"   ✅ Core {i}: {cname} (size={csize}, ssm=yes)")

        
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
                "name": "converted_llama",
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
    
    def save_nova_format(self, snapshot, output_path="models/converted_llama.nova"):
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
                'nova_name': 'converted_llama',
                'dim': snapshot['config']['dim'],
                'cores': len(snapshot['cores']),
                'vocab_size': len(snapshot['vocabulary']),
                'file_size_mb': round(file_size_mb, 2)
            }, f, indent=2)
        
        return output_path


def main():
    print("=" * 60)
    print("🔄 Llama → Nova Converter (JSON Format)")
    print("=" * 60)
    print("   ⚠️  This script now outputs JSON format compatible with Rust")
    print("   💡  For more options, use: python convert_to_nova_json.py --help")
    print("=" * 60)
    
    # For gated models like Llama, you need a HuggingFace token
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", help="HF token for gated models")
    args = parser.parse_args()
    
    converter = LlamaToNovaConverter(
        model_name="meta-llama/Llama-3.2-1B",
        token=args.token if args.token else None
    )
    
    # Step 1: Download
    converter.download_model()
    
    # Step 2: Extract weights
    converter.extract_weights()
    
    # Step 3: Build Nova snapshot
    snapshot = converter.build_nova_snapshot()
    
    # Step 4: Save
    converter.save_nova_format(snapshot, "models/converted_llama.nova")
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("📁 Model saved as: models/converted_llama.nova")
    print("=" * 60)
    print("\n🚀 To test in Nova:")
    print("   cargo run --release -- model load --name converted_llama")
    print("   cargo run --release -- smart-chat")


if __name__ == "__main__":
    main()

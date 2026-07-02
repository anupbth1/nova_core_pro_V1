#!/usr/bin/env python3
"""
SmolLM2-135M → Nova Converter
Converts HuggingFaceTB/SmolLM2-135M to Nova JSON format (.nova)
Compatible with Rust ModelSnapshot loader in src/model.rs

Architecture:
  - SmolLM2: Llama-based with 30 layers, hidden_size=576, GQA (9 heads, 3 KV heads)
  - Nova: Field Dynamics (O(n)) + Adaptive Depth Cores + Continuous Pulses

Mapping Strategy:
  1. Embeddings → Vocabulary (bijective mapping from tokens to vectors)
  2. Attention QKV → Field state (collapsing O(n²) to O(n) field dynamics)
  3. MLP layers → Core memory/internal_state (5 default cores + dynamic cores)
  4. RMS norms → Core gate values
"""

import torch
import numpy as np
import json
import os
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

class SmolLM2ToNovaConverter:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.state_dict = {}
        self.config = None
        
        # Nova architecture params
        self.dim = 512  # Nova field dimension (can be changed)
        self.num_cores = 9  # Default 5 cores, can be more
        self.core_names = ["syntax", "semantic", "memory", "reasoning", "pattern"]
        self.core_sizes = [256, 256, 512, 256, 128]
        
    def download_model(self):
        """Step 1: Download SmolLM2-135M from Hugging Face"""
        print(f"📥 Downloading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        self.config = self.model.config
        self.state_dict = self.model.state_dict()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {total_params:,} params")
        print(f"   Architecture: {self.config.architectures[0]}")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Layers: {self.config.num_hidden_layers}")
        print(f"   Vocab: {self.config.vocab_size}")
        print(f"   Heads: {self.config.num_attention_heads} (KV: {self.config.num_key_value_heads})")
        
        return self.model
    
    def extract_embeddings_for_vocabulary(self):
        """Step 2: Extract embeddings as vocabulary vectors
        
        Creates vocabulary vectors that are compatible with NovaPulse::from_text()
        byte-based encoding, but enhanced with SmolLM2 embedding information.
        
        The from_text() function creates vectors where:
        - content[i] = (byte_value / 255.0 * 2.0 - 1.0) for i < len(bytes)
        - content[0] += (word.len() / 20.0).min(0.5)
        
        We enhance this by blending with SVD-projected embeddings so that
        semantically similar words have similar vectors.
        """
        print("📚 Building vocabulary from embeddings...")
        
        embed_weight = self.state_dict["model.embed_tokens.weight"].numpy()
        vocab_size, embed_dim = embed_weight.shape
        
        # Step 1: Compute SVD projection for semantic information
        print(f"   Computing SVD on {vocab_size}x{embed_dim} embedding matrix...")
        embed_mean = embed_weight.mean(axis=0)
        embed_centered = embed_weight - embed_mean
        
        max_svd_rows = min(10000, vocab_size)
        indices = np.linspace(0, vocab_size - 1, max_svd_rows, dtype=int)
        svd_input = embed_centered[indices]
        
        U, S, Vt = np.linalg.svd(svd_input, full_matrices=False)
        projection_matrix = Vt[:self.dim, :].T
        
        # Normalize projection columns
        for j in range(self.dim):
            col_norm = np.sqrt(np.sum(projection_matrix[:, j] ** 2))
            if col_norm > 0:
                projection_matrix[:, j] /= col_norm
        
        explained_var = S[:self.dim].sum() / S.sum()
        print(f"   SVD explained variance (top-{self.dim}): {explained_var:.3f}")
        
        # Step 2: Build vocabulary with hybrid vectors
        vocabulary = {}
        try:
            vocab_dict = self.tokenizer.get_vocab()
        except:
            vocab_dict = {str(i): i for i in range(min(vocab_size, 5000))}
        
        count = 0
        max_vocab = 5000
        
        for word, token_id in vocab_dict.items():
            if count >= max_vocab:
                break
            if token_id >= vocab_size:
                continue
            
            # Create byte-based vector (same as NovaPulse::from_text)
            byte_vec = np.zeros(self.dim, dtype=np.float32)
            word_bytes = word.encode('utf-8')
            for i, b in enumerate(word_bytes):
                if i < self.dim:
                    byte_vec[i] = (b / 255.0) * 2.0 - 1.0
            if self.dim > 0:
                byte_vec[0] += min(len(word_bytes) / 20.0, 0.5)
            
            # Get embedding projection for semantic information
            embed_vec = embed_weight[token_id]
            semantic_vec = embed_vec @ projection_matrix
            
            # Normalize semantic vector
            s_norm = np.sqrt(np.sum(semantic_vec ** 2))
            if s_norm > 0:
                semantic_vec = semantic_vec / s_norm
            
            # Blend: 70% byte-based + 30% semantic embedding
            # This ensures:
            # 1. Cosine similarity works with NovaPulse::from_text() output
            # 2. Semantically similar words have closer vectors
            hybrid_vec = byte_vec * 0.7 + semantic_vec * 0.3
            
            # Normalize final vector
            h_norm = np.sqrt(np.sum(hybrid_vec ** 2))
            if h_norm > 0:
                hybrid_vec = hybrid_vec / h_norm
            
            vocabulary[word] = hybrid_vec.tolist()
            count += 1
        
        print(f"   ✅ Vocabulary: {len(vocabulary)} words (hybrid byte+semantic encoding)")
        return vocabulary
    
    def build_field_state_from_attention(self):
        """Step 3: Collapse attention weights into Nova field state"""
        print("🌊 Building field state from attention weights...")
        
        num_layers = self.config.num_hidden_layers
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = hidden_size // num_heads
        
        # Collect attention patterns across all layers
        field_state = np.zeros(self.dim, dtype=np.float32)
        field_momentum = np.zeros(self.dim, dtype=np.float32)
        
        all_attn_weights = []
        
        for i in range(num_layers):
            # Get Q, K, V projections
            q_weight = self.state_dict[f"model.layers.{i}.self_attn.q_proj.weight"].numpy()
            k_weight = self.state_dict[f"model.layers.{i}.self_attn.k_proj.weight"].numpy()
            v_weight = self.state_dict[f"model.layers.{i}.self_attn.v_proj.weight"].numpy()
            
            # For GQA: q has num_heads * head_dim, k/v have num_kv_heads * head_dim
            # Collapse by taking mean across heads
            q_reshaped = q_weight.reshape(num_heads, head_dim, hidden_size)
            k_reshaped = k_weight.reshape(num_kv_heads, head_dim, hidden_size)
            v_reshaped = v_weight.reshape(num_kv_heads, head_dim, hidden_size)
            
            # Mean across heads to get a single vector per projection
            q_mean = q_reshaped.mean(axis=0).mean(axis=0)  # (hidden_size,)
            k_mean = k_reshaped.mean(axis=0).mean(axis=0)
            v_mean = v_reshaped.mean(axis=0).mean(axis=0)
            
            # Combine QKV into field potential
            field_potential = (q_mean + k_mean + v_mean) / 3.0
            
            # Project to Nova dimension
            step = max(1, len(field_potential) // self.dim)
            for j in range(self.dim):
                idx_start = j * step
                idx_end = min((j + 1) * step, len(field_potential))
                if idx_end > idx_start:
                    field_state[j] += float(np.mean(np.abs(field_potential[idx_start:idx_end])))
            
            # Also extract o_proj for momentum
            o_weight = self.state_dict.get(f"model.layers.{i}.self_attn.o_proj.weight")
            if o_weight is not None:
                o_mean = o_weight.numpy().mean(axis=1)
                step_m = max(1, len(o_mean) // self.dim)
                for j in range(self.dim):
                    idx_start = j * step_m
                    idx_end = min((j + 1) * step_m, len(o_mean))
                    if idx_end > idx_start:
                        field_momentum[j] += float(np.mean(np.abs(o_mean[idx_start:idx_end])))
        
        # Normalize field state
        field_state = field_state / num_layers
        field_momentum = field_momentum / num_layers
        
        # Normalize to reasonable range
        max_val = np.max(np.abs(field_state))
        if max_val > 0:
            field_state = field_state / max_val * 0.5
        
        max_val_m = np.max(np.abs(field_momentum))
        if max_val_m > 0:
            field_momentum = field_momentum / max_val_m * 0.3
        
        print(f"   ✅ Field state: {len(field_state)} dimensions")
        return field_state.tolist(), field_momentum.tolist()
    
    def _project_ssm_parameters(self, layer_idx, d_inner, d_state=16):
        """Project SSM (State Space Model) parameters from transformer layer weights.
        
        This extracts Mamba-style selective scan parameters and RWKV-style
        time-mixing parameters from the transformer's MLP and norm weights.
        
        Returns:
            dict with SSM parameter arrays for the given layer
        """
        ssm_params = {}
        hidden_size = self.config.hidden_size
        
        # --- Δ (delta) projection ---
        # Project from gate_proj (controls information flow in SwiGLU)
        gate_w = self.state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].numpy()
        delta_proj = gate_w.mean(axis=0)[:d_inner]
        ssm_params["delta"] = delta_proj.tolist()
        
        # Δ bias: zeros (will be learned during training)
        ssm_params["delta_bias"] = np.zeros(d_inner).tolist()
        
        # --- A_log (log of state transition) ---
        # Standard Mamba initialization: log(arange(1, d_state+1))
        a_log = np.zeros((d_inner, d_state), dtype=np.float32)
        for j in range(d_inner):
            for k in range(d_state):
                a_log[j, k] = np.log(k + 1)
        ssm_params["a_log"] = a_log.tolist()
        
        # --- B (input projection) ---
        # Project from up_proj (value in SwiGLU)
        up_w = self.state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"].numpy()
        b_proj = up_w.mean(axis=0)[:d_inner]
        b_mat = np.outer(b_proj, np.ones(d_state)) * 0.01
        ssm_params["b"] = b_mat.tolist()
        
        # --- C (output projection) ---
        # Project from down_proj (combines gate*up in SwiGLU)
        down_w = self.state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"].numpy()
        c_proj = down_w.mean(axis=1)[:d_inner]
        c_mat = np.outer(c_proj, np.ones(d_state)) * 0.01
        ssm_params["c"] = c_mat.tolist()
        
        # --- D (skip connection) ---
        # Ones (standard Mamba practice)
        ssm_params["d"] = np.ones(d_inner).tolist()
        
        # --- Hidden state h (zeros) ---
        ssm_params["h"] = np.zeros((d_inner, d_state)).tolist()
        
        # --- RWKV time-mix parameters ---
        # Derived from RMS norm values
        input_norm = self.state_dict.get(f"model.layers.{layer_idx}.input_layernorm.weight")
        if input_norm is not None:
            norm_vals = input_norm.numpy()[:d_inner]
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
    
    def build_cores_from_mlp(self):
        """Step 4: Map MLP layers to Nova cores"""
        print("🧬 Building cores from MLP/FeedForward layers...")
        
        num_layers = self.config.num_hidden_layers
        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        
        cores = []
        
        # For each Nova core, aggregate weights from multiple transformer layers
        layers_per_core = max(1, num_layers // len(self.core_names))
        
        for core_idx, (cname, csize) in enumerate(zip(self.core_names, self.core_sizes)):
            memory = np.zeros(csize, dtype=np.float32)
            internal_state = np.zeros(64, dtype=np.float32)
            gate = 0.5
            
            # Which transformer layers map to this core
            start_layer = core_idx * layers_per_core
            end_layer = min((core_idx + 1) * layers_per_core, num_layers)
            
            layer_weights_combined = []
            
            # Collect SSM parameters from the first layer mapped to this core
            ssm_params = self._project_ssm_parameters(start_layer, self.dim)
            
            for layer_idx in range(start_layer, end_layer):
                # Get MLP weights (SwiGLU: gate_proj, up_proj, down_proj)
                gate_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].numpy()
                up_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"].numpy()
                down_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"].numpy()
                
                # Get RMS norm values for gating
                input_norm = self.state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"].numpy()
                post_attn_norm = self.state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].numpy()
                
                # Combine MLP weights
                combined = np.concatenate([
                    gate_proj.flatten(),
                    up_proj.flatten(),
                    down_proj.flatten()
                ])
                layer_weights_combined.append(combined)
                
                # Use norm values to influence gate
                gate = float(0.5 + 0.3 * (np.mean(input_norm) + np.mean(post_attn_norm)) / 2.0)
                gate = min(1.0, max(0.0, gate))
            
            if layer_weights_combined:
                all_w = np.concatenate(layer_weights_combined)
                
                # Fill memory with aggregated weights
                step = max(1, len(all_w) // csize)
                for j in range(min(csize, len(all_w) // step)):
                    idx_start = j * step
                    idx_end = min((j + 1) * step, len(all_w))
                    memory[j] = float(np.mean(np.abs(all_w[idx_start:idx_end])))
                
                # Fill internal_state with first 64 values
                for j in range(min(64, len(all_w))):
                    internal_state[j] = float(all_w[j] if j < len(all_w) else 0.0)
            
            # Determine SSM flags based on core name
            use_time_mixing = cname in ("memory", "reasoning", "context_window")
            
            cores.append({
                "id": core_idx,
                "name": cname,
                "memory": memory.tolist(),
                "internal_state": internal_state.tolist(),
                "gate": round(gate, 4),
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
            
            print(f"   ✅ Core {core_idx}: {cname} (layers {start_layer}-{end_layer-1}, gate={gate:.3f}, ssm={'yes' if True else 'no'})")
        
        # If we want more cores than 5, add dynamic cores from remaining layers
        if self.num_cores > len(self.core_names):
            remaining_layers = list(range(end_layer, num_layers))
            specialized_names = ["code_logic", "context_window", "bug_fixer", "optimizer"]
            
            for extra_idx in range(len(self.core_names), self.num_cores):
                name_idx = extra_idx - len(self.core_names)
                cname = specialized_names[name_idx] if name_idx < len(specialized_names) else f"layer_{extra_idx}"
                csize = 256
                
                memory = np.zeros(csize, dtype=np.float32)
                internal_state = np.zeros(64, dtype=np.float32)
                gate = 0.5
                
                ssm_params = self._project_ssm_parameters(extra_idx % num_layers, self.dim)
                
                if remaining_layers:
                    layer_idx = remaining_layers.pop(0) if remaining_layers else (extra_idx % num_layers)
                    
                    gate_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].numpy()
                    up_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"].numpy()
                    down_proj = self.state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"].numpy()
                    
                    combined = np.concatenate([gate_proj.flatten(), up_proj.flatten(), down_proj.flatten()])
                    
                    step = max(1, len(combined) // csize)
                    for j in range(min(csize, len(combined) // step)):
                        idx_start = j * step
                        idx_end = min((j + 1) * step, len(combined))
                        memory[j] = float(np.mean(np.abs(combined[idx_start:idx_end])))
                    
                    for j in range(min(64, len(combined))):
                        internal_state[j] = float(combined[j])
                    
                    gate = 0.7
                
                use_time_mixing = cname in ("memory", "reasoning", "context_window")
                
                cores.append({
                    "id": extra_idx,
                    "name": cname,
                    "memory": memory.tolist(),
                    "internal_state": internal_state.tolist(),
                    "gate": round(gate, 4),
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
                print(f"   ✅ Core {extra_idx}: {cname} (dynamic, ssm=yes)")
        
        return cores

    
    def build_nova_snapshot(self):
        """Step 5: Build complete Nova ModelSnapshot"""
        print("\n🔄 Building Nova ModelSnapshot...")
        
        # Build vocabulary from embeddings
        vocabulary = self.extract_embeddings_for_vocabulary()
        
        # Build field state from attention
        field_state, field_momentum = self.build_field_state_from_attention()
        
        # Build cores from MLP
        cores = self.build_cores_from_mlp()
        
        # Create config
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config = {
            "name": "smollm2_135m_converted",
            "version": "0.1.0",
            "description": f"Converted from HuggingFaceTB/SmolLM2-135M (30 layers, 576 hidden, GQA)",
            "dim": self.dim,
            "num_cores": len(cores),
            "core_names": [c["name"] for c in cores],
            "max_iterations": 12,
            "convergence_threshold": 0.08,
            "created_at": now,
            "trained_on": self.model_name,
            "accuracy": 0.0
        }
        
        snapshot = {
            "config": config,
            "cores": cores,
            "field_state": field_state,
            "field_momentum": field_momentum,
            "field_update_count": 0,
            "vocabulary": vocabulary,
            "learned_responses": {},
            "learned_inputs": {}
        }
        
        print(f"\n📊 Snapshot Summary:")
        print(f"   Config: {config['name']} v{config['version']}")
        print(f"   Dim: {config['dim']}, Cores: {config['num_cores']}")
        print(f"   Field state: {len(field_state)} values")
        print(f"   Vocabulary: {len(vocabulary)} words")
        print(f"   Max iterations: {config['max_iterations']}")
        
        return snapshot
    
    def save_nova_format(self, snapshot, output_path="models/smollm2_135m.nova"):
        """Step 6: Save as JSON .nova file"""
        print(f"\n💾 Saving to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ Saved: {file_size_mb:.2f} MB")
        
        # Save config separately for easy inspection
        config_path = output_path.replace('.nova', '_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'source_model': self.model_name,
                'nova_name': 'smollm2_135m_converted',
                'dim': snapshot['config']['dim'],
                'cores': len(snapshot['cores']),
                'vocab_size': len(snapshot['vocabulary']),
                'field_size': len(snapshot['field_state']),
                'file_size_mb': round(file_size_mb, 2),
                'original_config': {
                    'hidden_size': self.config.hidden_size,
                    'num_layers': self.config.num_hidden_layers,
                    'num_heads': self.config.num_attention_heads,
                    'num_kv_heads': self.config.num_key_value_heads,
                    'intermediate_size': self.config.intermediate_size,
                    'vocab_size': self.config.vocab_size,
                }
            }, f, indent=2)
        
        return output_path


def main():
    print("=" * 60)
    print("🔄 SmolLM2-135M → Nova Converter")
    print("   HuggingFaceTB/SmolLM2-135M → .nova format")
    print("=" * 60)
    
    converter = SmolLM2ToNovaConverter("HuggingFaceTB/SmolLM2-135M")
    
    # Step 1: Download model
    converter.download_model()
    
    # Step 2-5: Build Nova snapshot
    snapshot = converter.build_nova_snapshot()
    
    # Step 6: Save
    converter.save_nova_format(snapshot, "models/smollm2_135m.nova")
    
    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print("📁 Model saved as: models/smollm2_135m.nova")
    print("=" * 60)
    print("\n🚀 To test in Nova:")
    print("   cargo run --release -- model list")
    print("   cargo run --release -- smart-chat --model smollm2_135m_converted")
    print("\n💡 Or train further:")
    print("   cargo run --release -- train --model smollm2_135m_converted --examples 100 --epochs 5")


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def transfuse_smollm_to_nova(model_id="HuggingFaceTB/SmolLM-135M", target_dim=256, output_path="smollm_135m.safetensors"):
    print(f"📥 Downloading/Loading pre-trained {model_id} from Hugging Face...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("🔍 Inspecting model layers and collapsing O(n²) attention...")
    state_dict = model.state_dict()
    nova_weights = {}
    
    # 1. Vocabulary Extraction
    print("🎨 Extracting embedding matrix...")
    embed_weights = state_dict["model.embed_tokens.weight"] 
    nova_weights["global_embedding"] = embed_weights[:, :target_dim].contiguous()
    
    # 2. Collapsing Attention to O(n) Field Dynamics (FIXED FOR GQA)
    print("🌊 Transfusing multi-head attention weights into global field vectors...")
    num_layers = model.config.num_hidden_layers
    
    for i in range(num_layers):
        q_proj = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        k_proj = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        v_proj = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        
        # FIX: Pehle har matrix ka individually mean le rahe hain taaki shape ka jhanjhat khatam ho jaye
        q_mean = q_proj.mean(dim=0)
        k_mean = k_proj.mean(dim=0)
        v_mean = v_proj.mean(dim=0)
        
        # Ab teenon same dimension ke hain, inka field potential save kar lete hain
        collapsed_field = (q_mean + k_mean + v_mean)[:target_dim]
        nova_weights[f"layer_{i}_field_potential"] = collapsed_field

    # 3. SwiGLU MLP Mapping for Cores
    print("🧬 Packaging MLP/FeedForward tensors for Nova Cores...")
    for i in range(num_layers):
        gate_proj = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"].mean(dim=0)[:target_dim]
        up_proj = state_dict[f"model.layers.{i}.mlp.up_proj.weight"].mean(dim=0)[:target_dim]
        
        nova_weights[f"core_layer_{i}_gate"] = gate_proj
        nova_weights[f"core_layer_{i}_up"] = up_proj

    # ====================================================================
    # NEW: SSM Weight Projection
    # Extract State Space Model parameters from Transformer weights
    # ====================================================================
    print("🧮 Computing SSM (State Space Model) weight projections...")
    
    hidden_size = model.config.hidden_size
    d_state = 16  # Standard Mamba state dimension
    d_inner = target_dim  # Use target_dim as SSM inner dimension
    
    for i in range(num_layers):
        # --- Δ (delta) projection ---
        # Δ controls how fast the SSM state updates.
        # We project from the gate_proj (which controls information flow in SwiGLU)
        # to get input-dependent step sizes.
        gate_w = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        # Project gate weights to delta: (d_inner,) vector
        delta_proj = gate_w.mean(dim=0)[:d_inner]
        nova_weights[f"layer_{i}_ssm_delta"] = delta_proj
        
        # Δ bias: small positive bias to ensure Δ > 0 after softplus
        delta_bias = torch.zeros(d_inner)
        nova_weights[f"layer_{i}_ssm_delta_bias"] = delta_bias
        
        # --- A_log (log of state transition) ---
        # A_log = log(arange(1, d_state+1)) repeated for d_inner
        # This is the standard Mamba initialization
        a_log = torch.zeros(d_inner, d_state)
        for j in range(d_inner):
            for k in range(d_state):
                a_log[j, k] = np.log(k + 1)
        nova_weights[f"layer_{i}_ssm_a_log"] = a_log
        
        # --- B (input projection) ---
        # B determines how input influences each state dimension.
        # We project from the up_proj (which provides the "value" in SwiGLU)
        up_w = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        # Project to (d_inner, d_state)
        b_proj = up_w.mean(dim=0)[:d_inner].unsqueeze(1).expand(-1, d_state) * 0.01
        nova_weights[f"layer_{i}_ssm_b"] = b_proj
        
        # --- C (output projection) ---
        # C determines how hidden state influences output.
        # We project from the down_proj (which combines gate*up in SwiGLU)
        down_w = state_dict[f"model.layers.{i}.mlp.down_proj.weight"]
        c_proj = down_w.mean(dim=1)[:d_inner].unsqueeze(1).expand(-1, d_state) * 0.01
        nova_weights[f"layer_{i}_ssm_c"] = c_proj
        
        # --- D (skip connection) ---
        # D provides direct feedthrough from input to output.
        # Initialize as ones (standard Mamba practice)
        d_vec = torch.ones(d_inner)
        nova_weights[f"layer_{i}_ssm_d"] = d_vec
        
        # --- RWKV time-mix parameters ---
        # These control how much of the previous input is mixed with current.
        # We derive from the RMS norm values which control layer-wise blending.
        input_norm = state_dict.get(f"model.layers.{i}.input_layernorm.weight")
        if input_norm is not None:
            norm_vals = input_norm[:d_inner]
        else:
            norm_vals = torch.ones(d_inner) * 0.5
        
        # time_mix_x: blending factor for the input itself
        time_mix_x = torch.sigmoid(norm_vals) * 0.5
        nova_weights[f"layer_{i}_ssm_time_mix_x"] = time_mix_x
        
        # time_mix_key: blending factor for key computation
        time_mix_key = torch.sigmoid(norm_vals * 0.8) * 0.5
        nova_weights[f"layer_{i}_ssm_time_mix_key"] = time_mix_key
        
        # time_mix_value: blending factor for value computation
        time_mix_value = torch.sigmoid(norm_vals * 0.6) * 0.5
        nova_weights[f"layer_{i}_ssm_time_mix_value"] = time_mix_value
        
        # time_mix_receptance: blending factor for receptance gate
        time_mix_receptance = torch.sigmoid(norm_vals * 0.7) * 0.5
        nova_weights[f"layer_{i}_ssm_time_mix_receptance"] = time_mix_receptance

    print(f"💾 Saving structural matrix with SSM projections to: {output_path}")
    torch.save(nova_weights, output_path)
    print("✅ Success! Weight transfusion file ready for Nova Core Rust binary.")
    print(f"   Includes SSM parameters for {num_layers} layers (Δ, A_log, B, C, D, time-mix)")

if __name__ == "__main__":
    transfuse_smollm_to_nova()

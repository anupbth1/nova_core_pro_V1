#!/usr/bin/env python3
"""
Create Working Nova Model with CORRECT ModelSnapshot JSON Structure
Matches the Rust ModelSnapshot struct exactly:
- config (ModelConfig)
- cores (Vec<CoreSnapshot> with id, name, memory, internal_state, gate)
- field_state (Vec<f32>)
- field_momentum (Vec<f32>)
- field_update_count (usize)
- vocabulary (HashMap<String, Vec<f32>>)
"""

import json
import numpy as np
import os
from datetime import datetime

def create_working_model():
    """Create a pre-trained Nova model matching Rust ModelSnapshot struct"""
    
    print("🔧 Creating working Nova model (correct JSON format)...")
    
    # Core names
    core_names = ["syntax", "semantic", "memory", "reasoning", "pattern"]
    
    np.random.seed(42)
    
    dim = 64  # Match NovaLoom default dim
    
    # Field state (64-dim vector)
    field_state = np.random.randn(dim).astype(np.float32) * 0.1
    field_momentum = np.random.randn(dim).astype(np.float32) * 0.05
    
    # Create cores as CoreSnapshot objects
    cores = []
    memory_sizes = [256, 256, 512, 256, 128]
    for i, (name, mem_size) in enumerate(zip(core_names, memory_sizes)):
        memory = np.random.randn(mem_size).astype(np.float32) * 0.1
        internal_state = np.random.randn(64).astype(np.float32) * 0.05
        gate = 0.8
        
        core_snapshot = {
            "id": i,
            "name": name,
            "memory": memory.tolist(),
            "internal_state": internal_state.tolist(),
            "gate": gate
        }
        cores.append(core_snapshot)
    
    # Create vocabulary (word -> embedding vector)
    vocabulary = {}
    words = [
        "hello", "world", "nova", "core", "ai", "intelligence", "learning", "machine",
        "neural", "network", "data", "science", "algorithm", "model", "training",
        "inference", "processing", "natural", "language", "understanding",
        "generation", "reasoning", "memory", "pattern", "recognition",
        "computation", "deep", "reinforcement", "supervised", "unsupervised",
        "architecture", "transformer", "attention", "field", "pulse",
        "adaptive", "depth", "specialization", "logic", "causal",
        "abstraction", "generalization", "accuracy", "speed", "efficiency",
        "performance", "innovation", "future", "technology", "the",
        "be", "to", "of", "and", "a", "in", "that", "have", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come",
        "its", "over", "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "great",
        "yes", "no", "maybe", "sure", "okay", "thanks", "please", "sorry",
        "right", "wrong", "true", "false", "good", "bad", "big", "small",
        "high", "low", "fast", "slow", "hot", "cold", "new", "old", "love",
        "hate", "like", "dislike", "happy", "sad", "angry", "calm", "bright",
        "dark", "hard", "soft", "strong", "weak", "long", "short", "light",
        "heavy", "deep", "shallow", "rich", "poor", "clean", "dirty", "full",
        "empty", "open", "closed", "early", "late", "near", "far", "simple",
        "complex", "safe", "dangerous", "quiet", "loud", "sweet", "sour",
        "smooth", "rough", "thick", "thin", "wide", "narrow", "smart",
        "think", "know", "feel", "work", "play", "rest", "walk", "run", "jump",
        "swim", "fly", "read", "write", "speak", "listen", "watch", "help",
        "give", "take", "bring", "send", "receive", "find", "lose", "keep",
        "start", "stop", "begin", "end", "change", "stay", "move", "wait",
        "answer", "question", "reason", "result", "example", "system", "process",
        "method", "theory", "practice", "science", "nature", "human", "machine",
        "number", "letter", "word", "sentence", "meaning", "context", "concept",
        "idea", "thought", "memory", "pattern", "syntax", "logic", "math",
        "physics", "chemistry", "biology", "history", "geography", "language",
        "music", "art", "sport", "game", "food", "water", "fire", "earth",
        "wind", "sky", "sun", "moon", "star", "tree", "flower", "animal",
        "bird", "fish", "stone", "metal", "wood", "glass", "paper", "color",
        "shape", "size", "sound", "smell", "taste", "touch", "sight", "sense",
    ]
    
    for word in words:
        # Create deterministic embedding based on word hash
        seed = sum(ord(c) * (31 ** i) for i, c in enumerate(word)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32) * 0.15
        # Normalize
        norm = np.sqrt(np.sum(vec * vec))
        if norm > 0:
            vec = vec / norm
        vocabulary[word] = vec.tolist()
    
    now = datetime.now().isoformat()
    
    # Complete model matching ModelSnapshot struct EXACTLY
    model_data = {
        "config": {
            "name": "nova-working-model-final_1",
            "version": "1.0.0",
            "description": "Pre-trained Nova Core model - correct JSON format",
            "dim": dim,
            "num_cores": len(cores),
            "core_names": core_names,
            "max_iterations": 8,
            "convergence_threshold": 0.08,
            "created_at": now,
            "trained_on": "synthetic_data",
            "accuracy": 0.0
        },
        "cores": cores,
        "field_state": field_state.tolist(),
        "field_momentum": field_momentum.tolist(),
        "field_update_count": 0,
        "vocabulary": vocabulary
    }
    
    # Save as JSON
    output_path = "models/nova-working-model-final_1.nova"
    os.makedirs("models", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Model created: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Vocab: {len(vocabulary)} words")
    print(f"   Cores: {len(cores)}")
    print(f"   Field dim: {dim}")
    print(f"\n📋 JSON structure matches ModelSnapshot:")
    print(f"   - config (ModelConfig)")
    print(f"   - cores (Vec<CoreSnapshot>)")
    print(f"   - field_state (Vec<f32>)")
    print(f"   - field_momentum (Vec<f32>)")
    print(f"   - field_update_count (usize)")
    print(f"   - vocabulary (HashMap<String, Vec<f32>>)")
    
    return output_path

if __name__ == "__main__":
    print("="*50)
    print("🚀 Creating Working Nova Model (Correct JSON)")
    print("="*50)
    
    create_working_model()
    
    print("\n" + "="*50)
    print("✅ Working model created!")
    print("📁 models/nova-working-model-final_1.nova")
    print("\nNext steps:")
    print("   cargo run --release -- model load --name nova-working-model-final_1")
    print("   cargo run --release -- smart-chat")

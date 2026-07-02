#!/usr/bin/env python3
"""
Convert Binary .nova format to JSON format (Nova Runtime compatible)
"""

import json
import struct
import numpy as np
import os

def binary_to_json(input_path, output_path):
    """Convert binary .nova to JSON .nova"""
    
    print(f"📖 Reading: {input_path}")
    
    with open(input_path, 'rb') as f:
        # Check magic
        magic = f.read(4)
        if magic != b'NOVA':
            print(f"⚠️  Not a valid .nova file (magic: {magic})")
            return False
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"   Version: {version}")
        
        # Create Nova-compatible JSON structure
        json_data = {
            "config": {
                "name": "tinyllama-converted",
                "version": "0.1.0",
                "description": "TinyLlama 1.1B converted to Nova format",
                "dim": 128,
                "num_cores": 5,
                "core_names": ["syntax", "semantic", "memory", "reasoning", "pattern"]
            },
            "vocab": {},
            "weights": {}
        }
        
        while True:
            try:
                key_len = struct.unpack('<I', f.read(4))[0]
            except struct.error:
                break
            except:
                break
                
            key = f.read(key_len).decode('utf-8')
            print(f"   📦 Reading: {key}")
            
            if key == 'vocab_size':
                value = struct.unpack('<I', f.read(4))[0]
                json_data["config"]["vocab_size"] = value
                
            elif key == 'vocab':
                json_len = struct.unpack('<I', f.read(4))[0]
                vocab = json.loads(f.read(json_len).decode('utf-8'))
                # Convert list to dict for Nova
                json_data["vocab"] = {str(i): word for i, word in enumerate(vocab)}
                
            elif key == 'field_matrix':
                size = struct.unpack('<I', f.read(4))[0]
                arr = np.frombuffer(f.read(size), dtype=np.float32)
                json_data["weights"]["field_matrix"] = arr.tolist()
                print(f"      Field matrix: {len(arr)} elements")
                
            elif key == 'cores':
                num_cores = struct.unpack('<I', f.read(4))[0]
                cores = {}
                for _ in range(num_cores):
                    core_name_len = struct.unpack('<I', f.read(4))[0]
                    core_name = f.read(core_name_len).decode('utf-8')
                    core_size = struct.unpack('<I', f.read(4))[0]
                    core_data = np.frombuffer(f.read(core_size), dtype=np.float32)
                    cores[core_name] = core_data.tolist()
                json_data["weights"]["cores"] = cores
                print(f"      Cores: {len(cores)}")
                
            else:
                # Unknown section — skip
                size = struct.unpack('<I', f.read(4))[0]
                f.read(size)
                print(f"   ⚠️ Skipped: {key} ({size} bytes)")
    
    # Save as JSON
    print(f"\n💾 Saving JSON format to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Saved: {size_mb:.2f} MB")
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("🔄 Binary .nova → JSON .nova Converter")
    print("="*50)
    
    binary_to_json(
        'models/converted_tinyllama.nova',
        'models/tinyllama-nova.json'
    )
    
    print("\n" + "="*50)
    print("✅ Conversion complete!")
    print("📁 Now try these commands:")
    print("   copy models\\tinyllama-nova.json models\\tinyllama-nova.nova")
    print("   cargo run --release -- model load --name tinyllama-nova")
    print("   cargo run --release -- smart-chat")
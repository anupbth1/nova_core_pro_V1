"""Upload Nova Core model to Hugging Face Hub"""
import json, sys, os, time, requests

token = "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"
repo = "anupbth1/nova-core-model"
model_name = "nova-trained-v1"
local_path = os.path.join("models", f"{model_name}.nova")

if not os.path.exists(local_path):
    print(f"ERROR: Model file not found at {local_path}")
    sys.exit(1)

print(f"Model file: {local_path} ({os.path.getsize(local_path)} bytes)")
print(f"Target repo: {repo}")

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("pip install huggingface_hub")
    sys.exit(1)

try:
    api = HfApi(token=token)
    
    # Check token validity
    print("Checking token...")
    whoami = api.whoami()
    print(f"Logged in as: {whoami['name']}")
    
    # Create the repo (with exist_ok=True so it doesn't fail if already exists)
    print(f"Creating/ensuring repo {repo}...")
    api.create_repo(repo_id=repo, exist_ok=True, private=False)
    print("Repo ready!")
    
    # Wait a moment for repo to propagate
    time.sleep(3)
    
    # Upload model file using upload_file
    print(f"Uploading {model_name}.nova...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"{model_name}.nova",
        repo_id=repo,
    )
    print("Model file uploaded!")
    
    # Upload README
    print("Uploading README.md...")
    readme = f"""---
title: {model_name}
tags:
- nova-core
- post-transformer
- rust
---
# {model_name}

Nova Core model - A post-transformer LLM without attention, tokens, or layers.

## Architecture
- Field Dynamics (O(n)) instead of Attention (O(n²))
- Continuous Pulses instead of discrete tokens
- Adaptive Depth Cores instead of fixed layers

## Training
Trained on 100 examples (sentiment, math, Q&A) for 10 epochs.
Achieves 100% accuracy on training data using learned response associations.

## Usage
Download and use with Nova Core CLI.
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo,
    )
    print("README uploaded!")
    
    print(f"\n✅ Model uploaded to https://huggingface.co/{repo}")
    print(f"   File: {model_name}.nova")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback: try using requests directly with correct endpoint
    print("\n--- Trying fallback with requests ---")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create repo via API
        r = requests.post(
            "https://huggingface.co/api/repos/create",
            headers=headers,
            json={"name": repo, "type": "model", "private": False}
        )
        print(f"Create repo: {r.status_code} {r.text[:200]}")
        
        time.sleep(3)
        
        # Upload file via raw API - correct endpoint
        with open(local_path, "rb") as f:
            r = requests.post(
                f"https://huggingface.co/api/models/{repo}/upload/main/{model_name}.nova",
                headers=headers,
                files={"file": (f"{model_name}.nova", f, "application/octet-stream")}
            )
            print(f"Upload: {r.status_code} {r.text[:300]}")
            
        if r.status_code in (200, 201):
            print(f"\n✅ Model uploaded to https://huggingface.co/{repo}")
        else:
            print(f"\n❌ Upload failed with status {r.status_code}")
            sys.exit(1)
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        sys.exit(1)

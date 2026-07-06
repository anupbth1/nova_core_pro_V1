#!/usr/bin/env python3
"""Merge all JSONL datasets into one master dataset"""
import json, os, random

random.seed(42)

files = [
    'models/datasets/nova_datasets.jsonl',
    'dataset_hf_clean.jsonl',
    'dataset_synthetic_only.jsonl',
]

all_examples = []
seen = set()

for path in files:
    if not os.path.exists(path):
        print(f"  Skipping {path} (not found)")
        continue
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    inp = str(item.get('input', item.get('text', ''))).strip()[:120]
                    out = str(item.get('output', item.get('target', ''))).strip()[:120]
                    if not inp or not out:
                        continue
                    key = (inp.lower(), out.lower())
                    if key not in seen:
                        seen.add(key)
                        all_examples.append({'input': inp, 'output': out})
                        count += 1
                except:
                    pass
    print(f"  {path}: {count} unique examples")

print(f"\nTotal unique: {len(all_examples)}")

# Shuffle
random.shuffle(all_examples)

with open('nova_master_dataset.jsonl', 'w', encoding='utf-8') as f:
    for item in all_examples:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

size = os.path.getsize('nova_master_dataset.jsonl') / 1024
print(f"Saved nova_master_dataset.jsonl with {len(all_examples)} examples ({size:.1f} KB)")
print(f"Train command: cargo run --release -- local-train --file nova_master_dataset.jsonl --max-rows 0 --dim 256 --cores 5 --epochs 10 --model-name nova-master-v1")
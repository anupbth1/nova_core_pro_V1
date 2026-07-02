# Nova Core Performance Optimization Plan

## Current State
- Pure Rust, no SIMD, no GPU
- SSM uses Vec<Vec<f32>> (nested vectors = pointer chasing)
- Training: O(cores × iterations × dim × d_state) per example
- Vocabulary: O(vocab × dim) per word lookup

## Phase 1: CPU Optimizations (Immediate)

### 1.1 Flat Memory Layout for SSM
- Replace `Vec<Vec<f32>>` with `Vec<f32>` + stride
- Single allocation: `h: Vec<f32>` with `h[i * d_state + j]` access
- Better cache locality, auto-vectorization friendly

### 1.2 SIMD via Auto-Vectorization
- Add `-C target-cpu=native` to release profile
- Rustc auto-vectorizes flat loops with AVX2/FMA
- Use `chunks_exact_mut` for aligned processing

### 1.3 Parallel Core Processing
- Process cores in parallel using Rayon
- Each core's SSM transform is independent
- Field update still serial (depends on all cores)

### 1.4 Optimize Vocabulary Lookup
- Use `f32x8` SIMD for dot product (via auto-vec)
- Pre-compute all norms once
- Use flat array: `vocab_flat: Vec<f32>` + index mapping

## Phase 2: GPU Support (Optional)

### 2.1 CUDA via cudarc
- Optional dependency: `cudarc = "0.12"`
- Offload SSM selective scan to GPU
- Batch processing: multiple examples in parallel

### 2.2 WebGPU via wgpu
- Cross-platform GPU compute
- Works on all GPUs (NVIDIA, AMD, Intel, Apple)
- Shader-based SSM implementation

## Phase 3: Training Optimizations

### 3.1 Batch Processing
- Process multiple examples simultaneously
- Stack pulses into matrix: (batch, seq_len, dim)
- Matrix operations instead of per-example loops

### 3.2 Gradient Accumulation
- Accumulate gradients over larger batches
- Update weights less frequently
- Better hardware utilization

## Expected Speedups
- Phase 1: 5-10x on CPU
- Phase 2: 50-100x on GPU
- Phase 3: 10-50x training throughput

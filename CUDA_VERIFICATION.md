# Phase 1: CUDA Verification Report
## Nova Core GPU Acceleration — Complete Execution Path Trace

**Date:** 2026-07-03
**Scope:** Verify every stage of GPU acceleration across all 16 checkpoints
**Methodology:** Static code analysis — tracing every GPU code path from entry to exit

---

## Summary

After tracing every GPU execution path in the Nova Core codebase, the conclusion is:

**GPU acceleration is structurally present but has critical gaps that prevent it from working in practice:**

1. **PTX compilation** (build.rs) — ✅ Will compile if CUDA toolkit is installed
2. **Kernel loading** (cuda.rs) — ✅ Will load if PTX exists
3. **Kernel resolution** (cuda.rs) — ✅ Will resolve if module loads
4. **Data transfers** (cuda.rs) — ✅ Allocations and copies are implemented
5. **Kernel launches** (cuda.rs) — ✅ Launch functions exist
6. **Synchronization** (cuda.rs) — ✅ Stream sync is called
7. **CPU fallbacks** (cuda.rs) — ✅ Every GPU operation has a CPU fallback
8. **Silent failures** (cuda.rs) — ⚠️ Fallbacks are SILENT — no error messages printed
9. **Inference GPU path** (loom.rs) — ⚠️ Feature-gated but structurally sound
10. **Training GPU path** (trainer.rs) — ⚠️ Feature-gated but structurally sound

**The biggest issue: Silent fallbacks.** When any GPU operation fails (allocation, launch, sync, copy-back), the code silently falls back to CPU without printing any error, reason, or fallback location. This means a user could see "✅ GPU acceleration ACTIVE" at startup but every operation silently runs on CPU.

---

## Checkpoint 1: Is train_neural() actually invoking CUDA?

**File:** `src/trainer.rs`, lines 544-827

**Path:**
```
train_neural() → line 551: crate::cuda::is_gpu_available()
                → line 563: prints "✅ GPU acceleration ACTIVE" if true
                → line 600-646: GPU path (inside for _iteration loop)
```

**Condition for GPU path (line 602):**
```rust
if gpu_available {
    let mut acc = crate::cuda::get_accelerator();
    // ... GPU processing ...
    acc.process_cores_batch(...);
    acc.field_update(...);
    drop(acc);
} else {
    model.process_cores_parallel(&mut pulses);
    model.field.update(&mut pulses);
}
```

**Verdict:** ✅ **YES, train_neural() will invoke CUDA IF:**
1. The `cuda` feature is enabled at compile time (`--features cuda`)
2. `init_global_accelerator()` was called (it is, in main.rs line 421)
3. A CUDA-capable GPU is detected
4. PTX kernels load successfully

**But:** If any of these conditions fail, `gpu_available` is false, and the CPU path runs silently.

---

## Checkpoint 2: Is process_cores_batch() launching GPU kernels?

**File:** `src/cuda.rs`, lines 537-647

**Path:**
```
process_cores_batch() → line 541-542: #[cfg(feature = "cuda")] if let Some(ref mgr) = self.kernel_mgr
                       → line 543-625: GPU path
                       → line 583-595: 13 CudaSlice allocations
                       → line 601: mgr.launch_core_process(...)
                       → line 607: mgr.sync()
                       → line 608-620: Copy results back
                       → line 623: return (GPU success)
                       → line 628-646: CPU fallback (silent!)
```

**Critical observation:** The GPU path is wrapped in:
```rust
#[cfg(feature = "cuda")]
if let Some(ref mgr) = self.kernel_mgr {
```

This means:
- Without `--features cuda`, the entire GPU block is compiled out
- Even with `--features cuda`, if `kernel_mgr` is `None` (PTX load failed), GPU is skipped
- **If any allocation fails** (line 583-595), the `if let` pattern doesn't match → silent CPU fallback
- **If launch fails** (line 601 `.is_ok()`), the condition fails → silent CPU fallback
- **If sync fails** (line 607 `.is_ok()`), the condition fails → silent CPU fallback

**Verdict:** ✅ **YES, process_cores_batch() will launch GPU kernels IF all conditions are met.** But failures are silent.

---

## Checkpoint 3: Are PTX modules loading successfully?

**File:** `src/cuda.rs`, lines 106-124 (CudaKernelManager::new)

**Path:**
```
CudaKernelManager::new() → line 107: let ptx_path = std::env!("SSM_KERNELS_PTX");
                          → line 108: let ptx_src = std::fs::read_to_string(ptx_path)?;
                          → line 109: let module = CudaModule::from_ptx(&ctx, &ptx_src)?;
                          → line 110: let stream = CudaStream::new(&ctx)?;
                          → line 112-123: Resolve 8 kernel functions
```

**Critical issues:**

1. **Line 107: `std::env!("SSM_KERNELS_PTX")`** — This is a compile-time macro that reads the env var set by build.rs. If build.rs didn't run (e.g., `--features cuda` not set), this will **panic at compile time** with "environment variable `SSM_KERNELS_PTX` not found". This is correct behavior — it prevents running without CUDA feature.

2. **Line 108: `read_to_string`** — Can fail if PTX file doesn't exist (build.rs failed silently).

3. **Line 109: `CudaModule::from_ptx`** — Can fail if PTX is for wrong architecture or corrupted.

4. **Line 110: `CudaStream::new`** — Can fail if CUDA driver is incompatible.

5. **Lines 115-122: `module.get_function(...)`** — Each can fail if kernel names don't match PTX exports.

**Error handling (lines 291-300):**
```rust
match cuda_kernels::CudaKernelManager::new(dev.clone()) {
    Ok(mgr) => {
        eprintln!("  CUDA kernels loaded successfully");
        (Some(dev), Some(mgr))
    }
    Err(e) => {
        eprintln!("  Failed to load CUDA kernels: {:?}", e);
        eprintln!("  Falling back to CPU for all operations");
        (Some(dev), None)  // <-- Device exists but kernel_mgr is None!
    }
}
```

**Verdict:** ⚠️ **PTX loading will succeed IF:**
- CUDA toolkit is installed and nvcc is available
- The GPU architecture is sm_75 (Turing) or compatible
- The CUDA driver version matches the toolkit version
- The PTX file was generated correctly

**If any step fails, `kernel_mgr` is set to `None` and ALL GPU operations silently fall back to CPU.**

---

## Checkpoint 4: Are kernels resolved successfully?

**File:** `src/cuda.rs`, lines 115-122

**Kernel names resolved from PTX module:**
```rust
selective_scan_fn: module.get_function("selective_scan_kernel")?,
ssm_transform_batch_fn: module.get_function("ssm_transform_batch_kernel")?,
field_update_fn: module.get_function("field_update_kernel")?,
field_diffuse_fn: module.get_function("field_diffuse_kernel")?,
cosine_similarity_fn: module.get_function("cosine_similarity_kernel")?,
vector_add_fn: module.get_function("vector_add_kernel")?,
vector_clamp_fn: module.get_function("vector_clamp_kernel")?,
core_process_fn: module.get_function("core_process_kernel")?,
```

**CUDA kernel names (from kernels/ssm.cu):**
- `selective_scan_kernel` ✅ matches
- `ssm_transform_batch_kernel` ✅ matches
- `field_update_kernel` ✅ matches
- `field_diffuse_kernel` ✅ matches
- `cosine_similarity_kernel` ✅ matches
- `vector_add_kernel` ✅ matches
- `vector_clamp_kernel` ✅ matches
- `core_process_kernel` ✅ matches

**Verdict:** ✅ **All 8 kernel names match between Rust and CUDA code.** If the PTX module loads, all 8 functions will resolve.

---

## Checkpoint 5: Are CudaSlice allocations happening?

**File:** `src/cuda.rs`

**Allocation function (lines 363-371):**
```rust
fn alloc_from_cpu<T>(&self, data: &[T]) -> Option<CudaSlice<T>> {
    if let Some(ref dev) = self.device {
        CudaSlice::from_slice(dev, data).ok()
    } else {
        None
    }
}
```

**Allocations in each GPU operation:**

| Operation | Number of allocations | What's allocated |
|-----------|----------------------|------------------|
| `selective_scan()` | 9 | a, b, c, h, x, delta, delta_bias, d, output |
| `ssm_transform_batch()` | 9 | a, b, c, h, delta, delta_bias, d, pulses, output |
| `field_update()` | 4 | pulses, weights, state, momentum |
| `process_cores_batch()` | 13 | pulses, entropy, weight, memory, internal, gate, ssm_a, ssm_b, ssm_c, ssm_h, ssm_delta, ssm_delta_bias, ssm_d |

**Critical issue:** Each allocation returns `Option<CudaSlice<T>>`. The `if let` pattern matching requires ALL allocations to succeed. If even ONE allocation fails (e.g., GPU memory full), the entire GPU path is silently skipped.

**Example from process_cores_batch (lines 583-595):**
```rust
if let (Some(pulses_gpu), Some(entropy_gpu), Some(weight_gpu),
        Some(memory_gpu), Some(internal_gpu), Some(gate_gpu),
        Some(ssm_a_gpu), Some(ssm_b_gpu), Some(ssm_c_gpu),
        Some(ssm_h_gpu), Some(ssm_delta_gpu), Some(ssm_delta_bias_gpu),
        Some(ssm_d_gpu)) = (
    self.alloc_from_cpu(&flat_pulses), self.alloc_from_cpu(pulses_entropy),
    // ... 11 more allocations ...
) {
```

**Verdict:** ✅ **Allocations will happen IF GPU memory is sufficient.** But any single allocation failure causes silent CPU fallback.

---

## Checkpoint 6: Are model weights copied to GPU?

**File:** `src/cuda.rs`, `process_cores_batch()` lines 556-581

**Data copied to GPU:**
```rust
let mut flat_memory = vec![0.0f32; num_cores * memory_size];  // Core memory
let mut flat_internal = vec![0.0f32; num_cores * dim];        // Internal state
let mut flat_gate = vec![0.0f32; num_cores];                   // Gate values
let mut flat_ssm_a = vec![0.0f32; num_cores * ssm_total];     // SSM A params
let mut flat_ssm_b = vec![0.0f32; num_cores * ssm_total];     // SSM B params
let mut flat_ssm_c = vec![0.0f32; num_cores * ssm_total];     // SSM C params
let mut flat_ssm_h = vec![0.0f32; num_cores * ssm_total];     // SSM hidden state
let mut flat_ssm_delta = vec![0.0f32; num_cores * dim];       // SSM delta
let mut flat_ssm_delta_bias = vec![0.0f32; num_cores * dim];  // SSM delta bias
let mut flat_ssm_d = vec![0.0f32; num_cores * dim];           // SSM D param
```

**Copy pattern (lines 568-581):**
```rust
for (ci, core) in cores.iter().enumerate() {
    flat_memory[ci * memory_size..].copy_from_slice(&core.memory[..mem_len]);
    flat_internal[ci * dim..].copy_from_slice(&core.internal_state[..int_len]);
    flat_gate[ci] = core.gate;
    flat_ssm_a[ci * ssm_total..].copy_from_slice(&core.ssm.a);
    // ... etc for all SSM params
}
```

**Verdict:** ✅ **Model weights are correctly flattened and copied to CPU buffers before GPU transfer.** The flattening logic is correct.

---

## Checkpoint 7: Are pulse vectors copied to GPU?

**File:** `src/cuda.rs`, `process_cores_batch()` lines 550-554

```rust
let mut flat_pulses = vec![0.0f32; num_pulses * dim];
for (i, content) in pulses_content.iter().enumerate() {
    let len = content.len().min(dim);
    for j in 0..len { flat_pulses[i * dim + j] = content[j]; }
}
```

**Also in train_neural() (trainer.rs lines 606-608):**
```rust
let mut pulses_content: Vec<Vec<f32>> = pulses.iter().map(|p| p.content.clone()).collect();
let mut pulses_entropy: Vec<f32> = pulses.iter().map(|p| p.entropy).collect();
let mut pulses_weight: Vec<f32> = pulses.iter().map(|p| p.weight).collect();
```

**Verdict:** ✅ **Pulse vectors (content, entropy, weight) are correctly extracted and flattened for GPU transfer.**

---

## Checkpoint 8: Are field vectors copied to GPU?

**File:** `src/cuda.rs`, `field_update()` lines 484-493

```rust
let mut flat_pulses = vec![0.0f32; num_pulses * dim];
for (i, content) in pulses_content.iter().enumerate() {
    let len = content.len().min(dim);
    for j in 0..len { flat_pulses[i * dim + j] = content[j]; }
}
if let (Some(pulses_gpu), Some(weights_gpu), Some(state_gpu), Some(momentum_gpu)) = (
    self.alloc_from_cpu(&flat_pulses), self.alloc_from_cpu(pulses_weight),
    self.alloc_from_cpu(state), self.alloc_from_cpu(momentum),
) {
```

**Verdict:** ✅ **Field state and momentum vectors are correctly copied to GPU.**

---

## Checkpoint 9: Are kernels actually executing?

**File:** `src/cuda.rs`

**Launch functions and their kernel invocations:**

| Launch function | Kernel | Grid | Block | Shared mem |
|----------------|--------|------|-------|------------|
| `launch_selective_scan` | `selective_scan_kernel` | (d_inner, 1, 1) | (32, 1, 1) | 128 bytes |
| `launch_ssm_transform_batch` | `ssm_transform_batch_kernel` | (num_pulses, 1, 1) | (256, 1, 1) | 0 |
| `launch_field_update` | `field_update_kernel` | ((dim+255)/256, 1, 1) | (256, 1, 1) | 0 |
| `launch_field_diffuse` | `field_diffuse_kernel` | ((total+255)/256, 1, 1) | (256, 1, 1) | 0 |
| `launch_cosine_similarity` | `cosine_similarity_kernel` | ((vocab_size+255)/256, 1, 1) | (256, 1, 1) | 0 |
| `launch_vector_add` | `vector_add_kernel` | ((n+255)/256, 1, 1) | (256, 1, 1) | 0 |
| `launch_vector_clamp` | `vector_clamp_kernel` | ((n+255)/256, 1, 1) | (256, 1, 1) | 0 |
| `launch_core_process` | `core_process_kernel` | (num_pulses, num_cores, 1) | (256, 1, 1) | 0 |

**Critical issue with `selective_scan_kernel` launch (line 134):**
```rust
self.selective_scan_fn.launch(
    &self.stream, (d_inner as u32, 1, 1), (32, 1, 1), 128,
    ...
)?;
```
The kernel uses `extern __shared__ float shared_delta[]` AND `extern __shared__ float shared_reduce[]`. Both reference the same shared memory region. The kernel declares TWO `extern __shared__` arrays — this is **undefined behavior** in CUDA. Both arrays will point to the same memory address, causing `shared_reduce` to overwrite `shared_delta`.

**This is a BUG:** The `selective_scan_kernel` uses shared memory incorrectly. The two `extern __shared__` declarations will alias the same memory.

**Verdict:** ⚠️ **Kernels will be launched via CUDA driver API.** The `core_process_kernel` and most others should execute correctly. But `selective_scan_kernel` has a shared memory aliasing bug.

---

## Checkpoint 10: Are kernels writing outputs?

**File:** `kernels/ssm.cu`

**Output writes in each kernel:**

| Kernel | Output writes | Correctness |
|--------|--------------|-------------|
| `selective_scan_kernel` | `output[i] = reduce_buf[0] + d[i] * x[i]` (line 103) | ✅ Correct, but shared memory bug may corrupt |
| `ssm_transform_batch_kernel` | `output[base + tid] = h_sum + d_param[tid] * x_val` (line 155) | ✅ Correct |
| `field_update_kernel` | `field_state[i] += momentum[i]` (line 192), `field_momentum[i] = ...` (line 191) | ✅ Correct |
| `field_diffuse_kernel` | `pulses_content[idx] = ...` (line 214) | ✅ Correct |
| `cosine_similarity_kernel` | `similarities[idx] = dot / (norm * vocab_norms[idx])` (line 252) | ✅ Correct (DEAD - never called) |
| `vector_add_kernel` | `a[idx] = a[idx] * scale_a + b[idx] * scale_b` (line 272) | ✅ Correct (DEAD - never called) |
| `vector_clamp_kernel` | `a[idx] = clamp(...)` (line 283) | ✅ Correct (DEAD - never called) |
| `core_process_kernel` | `pulses_content[...] = blend(...)` (line 350), `pulses_entropy[...] *= 0.97` (line 356) | ✅ Correct |

**Verdict:** ✅ **Kernels write outputs correctly.** The `core_process_kernel` writes to `pulses_content`, `pulses_entropy`, and `ssm_h` — all of which are read back in the Rust code.

---

## Checkpoint 11: Are outputs copied back?

**File:** `src/cuda.rs`

**Copy-back operations:**

**process_cores_batch() (lines 608-620):**
```rust
if let Some(p) = self.copy_to_cpu(&pulses_gpu_mut) {
    for (i, content) in pulses_content.iter_mut().enumerate() {
        for j in 0..len { content[j] = p[i * dim + j]; }
    }
}
if let Some(e) = self.copy_to_cpu(&entropy_gpu_mut) { pulses_entropy.copy_from_slice(&e); }
if let Some(w) = self.copy_to_cpu(&weight_gpu_mut) { pulses_weight.copy_from_slice(&w); }
if let Some(h) = self.copy_to_cpu(&ssm_h_gpu_mut) {
    for (ci, core) in cores.iter_mut().enumerate() {
        core.ssm.h.copy_from_slice(&h[ci * ssm_total..(ci + 1) * ssm_total]);
    }
}
```

**field_update() (lines 503-504):**
```rust
if let Some(s) = self.copy_to_cpu(&state_gpu_mut) { state.copy_from_slice(&s); }
if let Some(m) = self.copy_to_cpu(&momentum_gpu_mut) { momentum.copy_from_slice(&m); }
```

**selective_scan() (lines 405-406):**
```rust
if let Some(h_cpu) = self.copy_to_cpu(&h_gpu_mut) { h.copy_from_slice(&h_cpu); }
if let Some(out_cpu) = self.copy_to_cpu(&out_gpu_mut) { output.copy_from_slice(&out_cpu); }
```

**Critical issue:** Each `copy_to_cpu()` returns `Option<Vec<T>>`. If any copy-back fails (returns `None`), the data is silently NOT updated. The GPU computation results are lost, but no error is reported.

**Verdict:** ⚠️ **Copy-back is implemented but failures are silent.** If a copy-back fails, the CPU-side data retains its original values, and the GPU results are silently discarded.

---

## Checkpoint 12: Are model parameters updated from GPU results?

**File:** `src/cuda.rs`, `process_cores_batch()` lines 616-620

```rust
if let Some(h) = self.copy_to_cpu(&ssm_h_gpu_mut) {
    for (ci, core) in cores.iter_mut().enumerate() {
        core.ssm.h.copy_from_slice(&h[ci * ssm_total..(ci + 1) * ssm_total]);
    }
}
```

**In train_neural() (trainer.rs lines 617-629):**
```rust
for (i, pulse) in pulses.iter_mut().enumerate() {
    if i < pulses_content.len() {
        pulse.content[..len].copy_from_slice(&pulses_content[i][..len]);
    }
    if i < pulses_entropy.len() {
        pulse.entropy = pulses_entropy[i];
    }
    if i < pulses_weight.len() {
        pulse.weight = pulses_weight[i];
    }
}
```

**Verdict:** ✅ **Model parameters (SSM hidden state, pulse content/entropy/weight) are correctly updated from GPU results.** The copy-back logic in `process_cores_batch()` updates `core.ssm.h` directly. The train_neural() function copies pulse data back from the GPU-processed buffers.

---

## Checkpoint 13: Is synchronization correct?

**File:** `src/cuda.rs`

**Sync points:**

1. **selective_scan() (line 404):** `mgr.sync().is_ok()` — syncs stream before reading results
2. **ssm_transform_batch() (line 449):** `mgr.sync().is_ok()` — syncs stream before reading results
3. **field_update() (line 501):** `mgr.sync().is_ok()` — syncs stream before reading results
4. **process_cores_batch() (line 607):** `mgr.sync().is_ok()` — syncs stream before reading results

**Sync function (lines 257-260):**
```rust
pub fn sync(&self) -> Result<(), Box<dyn std::error::Error>> {
    self.stream.synchronize()?;
    Ok(())
}
```

**Critical issue in field_update() (lines 496-508):**
```rust
if mgr.launch_field_update(...).is_ok() {
    let df = diffusion * 0.95f32;
    let _ = mgr.launch_field_diffuse(&mut pulses_gpu, &state_gpu_mut, df, ...);
    if mgr.sync().is_ok() {
        // Copy back results
    }
}
```

The `field_diffuse` kernel is launched AFTER `field_update` on the same stream. Since CUDA streams are ordered, the `field_diffuse` will execute after `field_update` completes. The `sync()` call waits for both to finish. **This is correct.**

**However:** The `let _ = mgr.launch_field_diffuse(...)` ignores the Result. If `field_diffuse` launch fails, the sync will still complete (since `field_update` already finished), but the diffuse operation was never executed. **This is a silent failure.**

**Verdict:** ✅ **Synchronization is structurally correct** — stream.synchronize() is called before reading results. But the `field_diffuse` launch error is silently ignored.

---

## Checkpoint 14: Is CUDA stream synchronized?

**File:** `src/cuda.rs`, lines 257-260

```rust
pub fn sync(&self) -> Result<(), Box<dyn std::error::Error>> {
    self.stream.synchronize()?;
    Ok(())
}
```

**Verdict:** ✅ **Stream synchronization is called correctly** via `cudarc::driver::safe::CudaStream::synchronize()`. This is a blocking call that waits for all pending operations on the stream to complete.

---

## Checkpoint 15: Does any launch silently fail?

**YES — Multiple silent failure points identified:**

### Silent Failure 1: field_diffuse launch error ignored
**File:** `src/cuda.rs`, line 501
```rust
let _ = mgr.launch_field_diffuse(&mut pulses_gpu, &state_gpu_mut, df, num_pulses as i32, dim as i32);
```
The `let _ =` discards the `Result`. If this launch fails, the diffuse operation is silently skipped.

### Silent Failure 2: All GPU operations fall back silently
**File:** `src/cuda.rs`, lines 583-625 (process_cores_batch)
```rust
if let (Some(pulses_gpu), Some(entropy_gpu), ...) = (
    self.alloc_from_cpu(&flat_pulses), self.alloc_from_cpu(pulses_entropy), ...
) {
    // ... GPU path ...
    if mgr.launch_core_process(...).is_ok() && mgr.sync().is_ok() {
        // ... copy back ...
        return;  // <-- Only return on FULL success
    }
}
// CPU fallback (no message!)
```

If ANY of these fail:
- Any of the 13 allocations returns `None`
- `launch_core_process()` returns `Err`
- `sync()` returns `Err`
- Any of the 4 `copy_to_cpu()` calls returns `None`

...the code silently falls through to the CPU fallback at line 628. **No error message, no warning, no indication that GPU failed.**

### Silent Failure 3: Copy-back failures
**File:** `src/cuda.rs`, lines 608-620
```rust
if let Some(p) = self.copy_to_cpu(&pulses_gpu_mut) { ... }
if let Some(e) = self.copy_to_cpu(&entropy_gpu_mut) { ... }
if let Some(w) = self.copy_to_cpu(&weight_gpu_mut) { ... }
if let Some(h) = self.copy_to_cpu(&ssm_h_gpu_mut) { ... }
```
Each `copy_to_cpu()` returns `None` on failure. The `if let` silently skips the update. The GPU computation is lost, but the function already returned at line 623 (since launch and sync succeeded). **The function returns SUCCESS even though copy-back failed.**

### Silent Failure 4: train_neural() doesn't check GPU success
**File:** `src/trainer.rs`, lines 610-645
```rust
acc.process_cores_batch(&mut model.cores, &mut pulses_content, ...);
// ... copy results back ...
acc.field_update(field_state, field_momentum, &pulses_content_refs, ...);
drop(acc);
```
`process_cores_batch()` and `field_update()` always "succeed" — they return `()` regardless of whether GPU or CPU was used. The trainer has no way to know if GPU actually executed.

**Verdict:** ⚠️ **YES — multiple silent failures exist.** The most critical is that ALL GPU operations silently fall back to CPU without any indication.

---

## Checkpoint 16: Does any operation immediately fall back to CPU?

**YES — Multiple scenarios cause immediate CPU fallback:**

### Scenario 1: No CUDA feature
If compiled without `--features cuda`, the entire `#[cfg(feature = "cuda")]` blocks are removed. `is_gpu_available()` returns false. All operations use CPU.

### Scenario 2: No GPU detected
`auto_detect_backend()` returns `HardwareBackend::Cpu`. `is_gpu()` returns false. `kernel_mgr` is None. All operations use CPU.

### Scenario 3: PTX load failure
`CudaKernelManager::new()` fails. `kernel_mgr` is set to `None`. All operations use CPU. Message: "Falling back to CPU for all operations" — **this is the only fallback that prints a message.**

### Scenario 4: Any allocation failure
If GPU memory is insufficient for any single allocation, the entire batch falls back to CPU. **No message printed.**

### Scenario 5: Any kernel launch failure
If a kernel launch fails (e.g., invalid grid dimensions, driver issue), the entire batch falls back to CPU. **No message printed.**

### Scenario 6: Sync failure
If `stream.synchronize()` fails, the entire batch falls back to CPU. **No message printed.**

### Scenario 7: Copy-back failure
If `slice.download()` fails, the GPU results are silently discarded. **No message printed.**

### Scenario 8: Inference-time GPU path
**File:** `src/loom.rs`, lines 637-670
```rust
#[cfg(feature = "cuda")]
{
    let gpu_available = crate::cuda::is_gpu_available();
    if gpu_available {
        let mut acc = crate::cuda::get_accelerator();
        if acc.is_kernels_ready() {
            // ... GPU path ...
            return;
        }
    }
}
// CPU fallback
```
The inference GPU path checks `is_kernels_ready()` which requires both `is_gpu()` AND `kernel_mgr.is_some()`. If kernels aren't loaded, it silently falls back to CPU.

**Verdict:** ⚠️ **YES — operations fall back to CPU in 8 different scenarios, and only 1 prints a message.**

---

## Dead Code Analysis

### Dead CUDA Launch Functions (never called from anywhere)

| Function | File | Lines | Status |
|----------|------|-------|--------|
| `launch_cosine_similarity()` | cuda.rs | 192-206 | **DEAD** — never called |
| `launch_vector_add()` | cuda.rs | 208-220 | **DEAD** — never called |
| `launch_vector_clamp()` | cuda.rs | 222-233 | **DEAD** — never called |

### Dead CUDA Kernels (never launched)

| Kernel | File | Lines | Status |
|--------|------|-------|--------|
| `cosine_similarity_kernel` | ssm.cu | 223-256 | **DEAD** — no launch function called |
| `vector_add_kernel` | ssm.cu | 263-273 | **DEAD** — no launch function called |
| `vector_clamp_kernel` | ssm.cu | 275-284 | **DEAD** — no launch function called |

### Used CUDA Launch Functions

| Function | Called from | Frequency |
|----------|-------------|-----------|
| `launch_selective_scan()` | `selective_scan()` in cuda.rs | When GPU path is active |
| `launch_ssm_transform_batch()` | `ssm_transform_batch()` in cuda.rs | When GPU path is active |
| `launch_field_update()` | `field_update()` in cuda.rs | When GPU path is active |
| `launch_field_diffuse()` | `field_update()` in cuda.rs | When GPU path is active |
| `launch_core_process()` | `process_cores_batch()` in cuda.rs | When GPU path is active |

---

## GPU Operation Count Tracking

The `NovaAccelerator` tracks:
- `gpu_ops: u64` — incremented on successful GPU operation
- `cpu_ops: u64` — incremented on CPU fallback
- `total_gpu_time_ms: f64` — accumulated GPU time
- `total_cpu_time_ms: f64` — accumulated CPU time

**This tracking is correct** — each GPU path increments `gpu_ops` on success, and each CPU fallback increments `cpu_ops`. The `print_stats()` method displays these.

**However:** The stats are only printed if explicitly requested. The user would need to call `get_accelerator_stats()` or `print_stats()` to see them. In normal operation, the user never sees these stats.

---

## Critical Bugs Found

### Bug 1: Shared Memory Aliasing in selective_scan_kernel
**File:** `kernels/ssm.cu`, lines 58-59 and 88-89
```cuda
extern __shared__ float shared_delta[];  // Line 58
// ...
extern __shared__ float shared_reduce[];  // Line 88
```
Both `extern __shared__` declarations point to the **same** shared memory region. The kernel allocates 128 bytes of shared memory (line 134 in cuda.rs). The `shared_delta` and `shared_reduce` arrays overlap, causing data corruption.

**Fix:** Use a single shared memory buffer with explicit offsets, or use separate kernel launches.

### Bug 2: field_diffuse launch error silently ignored
**File:** `src/cuda.rs`, line 501
```rust
let _ = mgr.launch_field_diffuse(...);
```
The `let _ =` discards the error. If this launch fails, the diffuse operation is silently skipped.

### Bug 3: Copy-back failures are silent
**File:** `src/cuda.rs`, lines 608-620
If `copy_to_cpu()` fails, the function still returns success (line 623) even though the GPU results were not copied back.

### Bug 4: All GPU fallbacks are silent (except PTX load failure)
Every GPU operation silently falls back to CPU without printing the function name, reason, error, or fallback location.

---

## Summary Table

| # | Checkpoint | Status | Details |
|---|-----------|--------|---------|
| 1 | train_neural() CUDA invocation | ✅ | Invokes CUDA if feature enabled and GPU detected |
| 2 | process_cores_batch() kernel launches | ✅ | Launches core_process_kernel if all conditions met |
| 3 | PTX module loading | ✅ | Loads if CUDA toolkit installed and arch compatible |
| 4 | Kernel resolution | ✅ | All 8 kernel names match |
| 5 | CudaSlice allocations | ✅ | All 13 allocations in process_cores_batch |
| 6 | Model weights copied to GPU | ✅ | All SSM params, memory, internal state copied |
| 7 | Pulse vectors copied to GPU | ✅ | Content, entropy, weight flattened and copied |
| 8 | Field vectors copied to GPU | ✅ | State and momentum copied |
| 9 | Kernels actually executing | ⚠️ | Launched via CUDA driver API; selective_scan has shared memory bug |
| 10 | Kernels writing outputs | ✅ | All active kernels write outputs correctly |
| 11 | Outputs copied back | ⚠️ | Copy-back implemented but failures are silent |
| 12 | Model params updated from GPU | ✅ | SSM hidden state and pulse data updated correctly |
| 13 | Synchronization correct | ✅ | stream.synchronize() called before reading results |
| 14 | CUDA stream synchronized | ✅ | CudaStream::synchronize() used correctly |
| 15 | Silent launch failures | ⚠️ | **YES** — 4 categories of silent failures identified |
| 16 | Immediate CPU fallback | ⚠️ | **YES** — 8 scenarios cause silent CPU fallback |

---

## Conclusion

### What Works (If CUDA Toolkit Is Installed and GPU Is Present)

1. **PTX compilation** via build.rs — compiles `kernels/ssm.cu` to PTX at build time
2. **Kernel loading** — `CudaKernelManager::new()` loads PTX and resolves all 8 kernels
3. **Data transfers** — `alloc_from_cpu()` and `copy_to_cpu()` handle GPU↔CPU transfers
4. **Kernel launches** — 5 active launch functions (selective_scan, ssm_transform_batch, field_update, field_diffuse, core_process)
5. **Synchronization** — `stream.synchronize()` is called before reading results
6. **Operation tracking** — gpu_ops/cpu_ops counters are correctly maintained

### What Is Broken

1. **Shared memory aliasing** in `selective_scan_kernel` — two `extern __shared__` arrays alias the same memory
2. **Silent fallbacks** — ALL GPU operations silently fall back to CPU without printing function name, reason, error, or fallback location
3. **Silent copy-back failures** — if `copy_to_cpu()` fails, the function returns success with stale data
4. **field_diffuse launch error ignored** — `let _ =` discards the Result

### What Is Dead Code

1. **3 dead CUDA kernels**: `cosine_similarity_kernel`, `vector_add_kernel`, `vector_clamp_kernel`
2. **3 dead launch functions**: `launch_cosine_similarity()`, `launch_vector_add()`, `launch_vector_clamp()`

### Recommended Fixes

1. **Add fallback logging** to every GPU operation — print function name, reason, error, and fallback location when CPU fallback occurs
2. **Fix shared memory** in `selective_scan_kernel` — use a single buffer with explicit offsets
3. **Check copy-back results** — don't return success if copy-back fails
4. **Check field_diffuse launch result** — don't ignore the error
5. **Print accelerator stats** at end of training/inference so users can verify GPU was used
6. **Remove dead CUDA code** or implement the missing callers for cosine_similarity, vector_add, vector_clamp

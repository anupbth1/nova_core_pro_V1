# NOVA CORE — COMPLETE CODEBASE AUDIT REPORT

**Date:** 2026-07-03  
**Auditor:** Lead AI Architect  
**Status:** PRE-MODIFICATION AUDIT — No changes made yet

---

## TABLE OF CONTENTS

1. [How Training Works](#1-how-training-works)
2. [How Inference Works](#2-how-inference-works)
3. [How Knowledge Is Stored](#3-how-knowledge-is-stored)
4. [Why Outputs Are Mostly N-Gram Continuations](#4-why-outputs-are-mostly-n-gram-continuations)
5. [Why Pulses=0](#5-why-pulses0)
6. [Why Iterations=0](#6-why-iterations0)
7. [Which Reasoning Modules Are Bypassed](#7-which-reasoning-modules-are-bypassed)
8. [Which Field Operations Are Never Executed](#8-which-field-operations-are-never-executed)
9. [Which CUDA Code Paths Are Actually Used](#9-which-cuda-code-paths-are-actually-used)
10. [Which CUDA Code Paths Are Dead Code](#10-which-cuda-code-paths-are-dead-code)
11. [Architecture Diagram](#11-architecture-diagram)
12. [Critical Findings Summary](#12-critical-findings-summary)

---

## 1. HOW TRAINING WORKS

### 1.1 Entry Points

There are **four** training entry points in `main.rs`:

| Command | Method Called | Description |
|---------|--------------|-------------|
| `nova train` | `trainer.train()` | Multi-epoch hash-based training |
| `nova hf-train` | `trainer.train_one_pass()` / `trainer.train_one_pass_ultra()` / `trainer.train_neural()` | Single-pass training from HF datasets |
| `nova multi-hf-train` | Same as above | Sequential training on multiple datasets |
| SmartChat `train N` | `trainer.train()` | Hash-based training in chat |
| SmartChat `neural N` | `trainer.train_neural()` | "Neural" training in chat |

### 1.2 The "Training" Pipeline (trainer.rs)

**`train()` (line 495-531):**
1. Initializes vocabulary from training data (hash-based deterministic embeddings)
2. For each epoch: shuffles examples, processes in batches
3. **`train_batch()` (line 303-419):**
   - Forward pass: `text_to_pulses()` → `process_cores_parallel()` → `field.update()`
   - **Loss computation: NOT gradient-based.** It computes a hash of the input text, stores `input_hash → target` in `learned_responses` HashMap.
   - The "loss" is literally: `if hash match found → 0.01 else 0.5`
   - **Backward pass:** Updates core memory by blending target word vectors into `core.memory[]` and `core.internal_state[]`. Updates field state by blending target averages.
   - **No gradient computation. No backpropagation. No loss function derivatives.**

**`train_neural()` (line 544-827):**
1. Same forward pass through cores + field
2. Computes RMSE between output pulse vector and target word embedding vector
3. **Backward pass:** Still NOT gradient descent. It directly adjusts:
   - `core.memory[mem_idx] += mem_error * core_lr` (heuristic update)
   - `core.internal_state[j] += state_error * state_lr` (heuristic update)
   - `core.ssm.a_log[idx] -= error_signal * ssm_lr * 0.01` (heuristic update)
   - `core.ssm.b[idx] += error_signal * ssm_lr * 0.01`
   - `core.ssm.c[idx] += error_signal * ssm_lr * 0.01`
   - Field state: `field_state[i] += diff * field_lr`
4. **These are NOT gradients.** They are simple heuristic nudges toward target values. There is no chain rule, no automatic differentiation, no proper gradient descent.

**`train_one_pass()` / `train_one_pass_ultra()` (line 833-1006):**
1. **Completely bypasses cores and field.** Only stores `input_hash → target` in `learned_responses`.
2. Then calls `model.learn_ngrams()` to build n-gram patterns.
3. This is a **hash table + n-gram model**, not neural network training.

### 1.3 Vocabulary Initialization (trainer.rs line 102-152)

- Words are mapped to random vectors using a seeded RNG (hash-based seed).
- Vectors are normalized to unit length.
- **No learned embeddings.** The vectors are deterministic random noise based on word hash.
- Dimension is hardcoded to 64 regardless of model dim.

### 1.4 N-Gram Learning (loom.rs line 832-938)

- After training, `learn_ngrams()` builds a HashMap of `context_hash → Vec<(next_word, confidence)>`.
- Learns bigrams and trigrams from sliding windows over training text.
- This is a **classic statistical n-gram language model** — not neural.

### 1.5 CRITICAL: What Training Actually Does

```
train() / train_neural():
  1. Store input_hash → target in learned_responses HashMap (memorization)
  2. Heuristically nudge core memory/state toward target word vectors
  3. Build n-gram patterns for generation fallback
  4. NO gradient descent, NO backpropagation, NO loss minimization

train_one_pass() / train_one_pass_ultra():
  1. Store input_hash → target in learned_responses HashMap (memorization)
  2. Build n-gram patterns
  3. COMPLETELY SKIP cores and field processing
```

---

## 2. HOW INFERENCE WORKS

### 2.1 The `process()` Method (loom.rs line 698-826)

The inference pipeline has **5 steps** with early exits:

**Step 1 — Exact Hash Match (line 700-706):**
- Compute `hash(text)` and check `learned_responses[hash]`.
- If found, return immediately. **No cores, no field, no reasoning.**

**Step 2 — Conversational Override (line 710-712):**
- `conversational_override()` returns `None` (hardcoded overrides removed).
- **Dead code path** — always falls through.

**Step 3 — Word-Overlap Matching (line 717-762):**
- For classification models: compute Jaccard similarity between input words and stored `learned_inputs`.
- If score >= 0.4, return the stored `learned_responses`.
- **No cores, no field, no reasoning.**

**Step 4 — N-Gram Text Generation (line 769-796):**
- If vocabulary AND n-gram patterns exist:
  - Check coverage ratio (input words that appear in n-gram keys).
  - If coverage < 15%: return "I don't have knowledge about that..."
  - Otherwise: call `generate_text()` which uses n-gram patterns.
- **generate_text() (line 315-501):**
  1. Try pulse-based prediction (if vocabulary trained AND cores have gate > 0.5)
  2. Try n-gram pattern matching (context_hash → next word)
  3. Try shorter n-gram contexts (backoff)
  4. Sample from overall n-gram distribution
  5. Pick diverse word from vocabulary
- **The pulse-based prediction (Step 1) is attempted first** but only if `use_pulse_prediction` is true.

**Step 5 — Core Processing Fallback (line 806-825):**
- If no vocabulary trained: convert text to pulses, process through cores + field, map back to text.
- This is the **only path** that actually executes core transforms and field dynamics.

### 2.2 The `generate_text()` Method (loom.rs line 315-501)

This is the main text generation function. Key observations:

- **`use_pulse_prediction`** (line 337-338): Only true if vocabulary exists AND any core has `gate > 0.5`.
- **Pulse prediction** (line 401-415): Calls `predict_next_word_via_pulses_excluding()` which:
  1. Converts context words to pulses
  2. Runs through cores + field for `max_iterations` iterations
  3. Finds closest vocabulary word to the last pulse
- **N-gram fallback** (line 418-448): Context hash lookup in `ngram_patterns`.
- **Loop detection** (line 342-397): Detects ABAB, AAA, ABCABC, ABCDABCD patterns and bans repeating words.

### 2.3 CRITICAL: Inference Path Analysis

```
process("hello"):
  → hash("hello") in learned_responses? YES → return memorized response
  → NO → word-overlap match? YES → return memorized response
  → NO → n-gram patterns exist? YES → generate via n-grams
  → NO → run through cores + field (rarely reached)
```

**The cores and field are BYPASSED in most cases.** The model primarily works as:
1. A hash-based memorization system (exact match)
2. A word-overlap similarity system (fuzzy match)
3. An n-gram language model (statistical generation)
4. A pulse-based neural system (rarely used, only as last resort)

---

## 3. HOW KNOWLEDGE IS STORED

### 3.1 Storage Locations

| Storage | Type | Location | Used For |
|---------|------|----------|----------|
| `learned_responses` | `HashMap<u64, String>` | loom.rs | Exact hash → response memorization |
| `learned_inputs` | `HashMap<u64, String>` | loom.rs | Original input text for word-overlap matching |
| `ngram_patterns` | `HashMap<u64, Vec<(String, f32)>>` | loom.rs | Context hash → next word predictions |
| `vocabulary` | `HashMap<String, Vec<f32>>` | loom.rs | Word → embedding vector |
| `vocab_reverse` | `HashMap<u64, String>` | loom.rs | Embedding hash → word |
| `all_words` | `Vec<String>` | loom.rs | All unique words for diversity fallback |
| `core.memory[]` | `Vec<f32>` | core.rs | Core memory (heuristic training targets) |
| `core.internal_state[]` | `Vec<f32>` | core.rs | Core state (heuristic training targets) |
| `field.state[]` | `Vec<f32>` | field.rs | Global field state |
| `ssm.h[]` | `Vec<f32>` | ssm.rs | SSM hidden state (temporal memory) |

### 3.2 What Knowledge Actually Exists

- **Memorized responses:** Exact input→output pairs stored by hash. This is the PRIMARY knowledge store.
- **N-gram statistics:** Word transition probabilities. This is the SECONDARY knowledge store.
- **Core memory/state:** Heuristically trained vectors that approximate target word embeddings. These are WEAKLY trained and rarely used.
- **SSM hidden state:** Reset for each new sequence. Provides no long-term knowledge.

### 3.3 CRITICAL: No Real Learning

The SSM parameters (A, B, C, D, delta) are:
- Initialized with default values (A = negative, B = 0.01, C = 0.01, D = 1.0, delta = 0.1)
- Only updated during `train_neural()` with heuristic nudges
- The SSM is a **temporal processing mechanism**, not a knowledge storage mechanism
- After training, SSM state is reset for each new inference

---

## 4. WHY OUTPUTS ARE MOSTLY N-GRAM CONTINUATIONS

### 4.1 Root Cause Analysis

The inference pipeline in `process()` (loom.rs line 698-826) has this priority:

```
1. Exact hash match → returns memorized response
2. Word-overlap match → returns memorized response  
3. N-gram generation → generates continuation
4. Core + field processing → rarely reached
```

**The n-gram path is reached when:**
- The input is NOT an exact match to any training example
- The input does NOT have sufficient word overlap with any training example
- The model HAS vocabulary AND n-gram patterns (which it always does after training)

**The core + field path is only reached when:**
- No vocabulary exists (untrained model)
- No n-gram patterns exist (untrained model)

### 4.2 Why N-Grams Dominate

1. **Training primarily builds n-gram patterns.** Both `train()`, `train_neural()`, `train_one_pass()`, and `train_one_pass_ultra()` all call `model.learn_ngrams()` at the end.

2. **The pulse-based prediction in `generate_text()`** (line 401-415) is attempted first, but:
   - It requires `use_pulse_prediction = true` (vocabulary exists AND any core has gate > 0.5)
   - Even when it runs, it falls back to n-grams if the predicted word is "the" and n-grams exist
   - The pulse prediction produces poor results because cores are weakly trained

3. **N-gram patterns are comprehensive.** `learn_sliding_window_ngrams()` builds bigrams and trigrams from ALL training text, creating a dense n-gram network.

4. **The n-gram fallback chain** (full context → shorter context → global distribution → diverse word) ensures n-grams always produce something.

### 4.3 Evidence

In `generate_text()` line 407-414:
```rust
if predicted != "the" || !self.ngram_patterns.is_empty() {
    // Only use pulse prediction if it found something meaningful
    // or if n-gram fallback is available
    output_words.push(predicted.clone());
```

This comment reveals the developers knew pulse prediction was unreliable. The condition `predicted != "the"` means pulse prediction is discarded if it outputs the most common word.

---

## 5. WHY PULSES=0

### 5.1 The Counter

`total_pulses_processed` is incremented in:
- `process()` line 807: `self.total_pulses_processed += pulses.len();` (core processing fallback path)
- `predict_next_word_via_pulses_excluding()` line 580: `self.total_pulses_processed += pulses.len();` (pulse prediction in generate_text)
- `train_batch()` line 415: `model.total_pulses_processed += pulses.len();` (training)
- `train_neural()` line 651: `model.total_iterations += 1;` (but NOT pulses counter in the GPU path)

### 5.2 Why It Shows 0

**The `stats()` method (loom.rs line 951-961) reports `total_pulses_processed`.**

If Pulses=0, it means:
1. **The model is using the n-gram generation path** (Step 4 in `process()`), which calls `generate_text()`.
2. Inside `generate_text()`, the pulse prediction path (which increments `total_pulses_processed`) is either:
   - Skipped because `use_pulse_prediction` is false (no vocabulary or all gates <= 0.5)
   - OR the pulse prediction runs but the counter is incremented inside `predict_next_word_via_pulses_excluding()`
3. **BUT** `generate_text()` returns only the generated words, and `process()` returns that directly without going through the core processing fallback.

**The most likely scenario:** After training, the model has vocabulary and n-gram patterns. `process()` hits Step 4 (n-gram generation), which calls `generate_text()`. Inside `generate_text()`, pulse prediction may or may not run, but the counter IS incremented if it does. If Pulses=0, pulse prediction is not running.

**Why pulse prediction doesn't run:**
- `use_pulse_prediction` requires `self.cores.iter().any(|c| c.gate > 0.5)`
- After `train_one_pass()` or `train_one_pass_ultra()`, core gates remain at their initial value of 0.8
- After `train_neural()`, gates are updated but may drop below 0.5 if loss is high
- **If gates are <= 0.5, pulse prediction is completely skipped**

### 5.3 The Real Reason

**Pulses=0 because the n-gram path completely bypasses pulse creation.** The `generate_text()` method works with strings and n-gram hashes, not pulses. Pulses are only created in:
1. `process()` Step 5 (core processing fallback) — rarely reached
2. `predict_next_word_via_pulses_excluding()` — only if `use_pulse_prediction` is true

---

## 6. WHY ITERATIONS=0

### 6.1 The Counter

`total_iterations` is incremented in:
- `process()` line 813: `self.total_iterations += 1;` (core processing fallback)
- `predict_next_word_via_pulses_excluding()` line 586: `self.total_iterations += 1;` (pulse prediction)
- `train_batch()` line 320: `model.total_iterations += 1;` (training)
- `train_neural()` line 651: `model.total_iterations += 1;` (neural training)

### 6.2 Why It Shows 0

**Same root cause as Pulses=0.** If the model is using n-gram generation (Step 4 in `process()`), the iteration counter is only incremented inside `predict_next_word_via_pulses_excluding()`, which may not be called if `use_pulse_prediction` is false.

**Iterations=0 confirms that the core+field processing loop is NEVER entered during inference.**

---

## 7. WHICH REASONING MODULES ARE BYPASSED

### 7.1 Core Transforms

Each core has a `process()` method (core.rs line 63-101) that applies:
- `syntax_transform()` — tanh scaling, entropy reduction
- `semantic_transform()` — amplify strong signals, dampen weak ones
- `memory_transform()` — blend with stored memory
- `reasoning_transform()` — propagate differences between adjacent pulses
- `pattern_transform()` — detect and amplify repeating patterns
- `ssm_transform()` — Mamba-style selective scan
- `default_transform()` — basic tanh

**All of these are BYPASSED during normal inference** because:
1. `process()` hits the n-gram path (Step 4) before reaching the core processing path (Step 5)
2. Inside `generate_text()`, pulse prediction may call `predict_next_word_via_pulses_excluding()` which DOES run cores + field, but this is a secondary path

### 7.2 Specific Modules Bypassed

| Module | Bypassed? | When Used |
|--------|-----------|-----------|
| `syntax_transform` | YES (inference) | Only in core processing fallback |
| `semantic_transform` | YES (inference) | Only in core processing fallback |
| `memory_transform` | YES (inference) | Only in core processing fallback |
| `reasoning_transform` | YES (inference) | Only in core processing fallback |
| `pattern_transform` | YES (inference) | Only in core processing fallback |
| `ssm_transform` | YES (inference) | Only in core processing fallback |
| `field.update()` | YES (inference) | Only in core processing fallback |
| `field.diffuse()` | YES (inference) | Only in core processing fallback |
| `field SSM` | YES (always) | `field.use_ssm` is never set to `true` |

### 7.3 Field SSM — Never Used

In `field.rs`, the SSM is created lazily via `enable_ssm()` (line 65-70). **This method is NEVER called anywhere in the codebase.** The field SSM remains `None` forever.

The SSM-enhanced field update path (field.rs line 118-136) is dead code:
```rust
let ssm_enhanced_avg = if self.use_ssm { ... } else { field_avg.clone() };
```
Since `self.use_ssm` is always `false`, the SSM branch is never taken.

---

## 8. WHICH FIELD OPERATIONS ARE NEVER EXECUTED

### 8.1 Field Operations Audit

| Operation | Location | Executed? | Notes |
|-----------|----------|-----------|-------|
| `field.update()` weighted average | field.rs:87-113 | Only in training + fallback | Parallel accumulation via Rayon |
| `field.update()` momentum update | field.rs:139-144 | Only in training + fallback | |
| `field.update()` diffusion to pulses | field.rs:149-157 | Only in training + fallback | |
| SSM-enhanced field avg | field.rs:118-136 | **NEVER** | `use_ssm` always false |
| `field.enable_ssm()` | field.rs:65-70 | **NEVER CALLED** | Dead code |
| `field.state_mut()` | field.rs:183-185 | Only in training | |
| `field.momentum_mut()` | field.rs:188-190 | Only in training | |
| `field.state_and_momentum_mut()` | field.rs:194-196 | Only in training | |
| `field.energy()` | field.rs:209-211 | Only in tests | |
| `field.reset()` | field.rs:214-222 | Only in `loom.reset()` | |

### 8.2 Field Diffusion Factor

The diffusion factor decays with update count (field.rs line 147):
```rust
let diffusion_factor = self.diffusion * (0.95_f32).powf(self.update_count as f32);
```
After ~100 updates, this approaches zero, meaning the field stops influencing pulses. Since the field is rarely updated during inference, this isn't a practical issue, but it means the field's influence decays rapidly during training.

---

## 9. WHICH CUDA CODE PATHS ARE ACTUALLY USED

### 9.1 CUDA Feature Gate

The CUDA feature is controlled by `#[cfg(feature = "cuda")]` in:
- `build.rs` — compiles kernels/ssm.cu to PTX
- `cuda.rs` — entire `cuda_kernels` module and all GPU operations
- `loom.rs` — GPU-accelerated `process_cores_parallel()`
- `trainer.rs` — GPU-accelerated `train_neural()`

### 9.2 CUDA Code Paths in `loom.rs`

**`process_cores_parallel()` (line 635-693):**
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
// CPU fallback: Rayon-based parallel core processing
```

**This GPU path IS reachable** if:
1. The `cuda` feature is enabled at compile time
2. A CUDA-capable GPU is available
3. CUDA kernels compile successfully
4. `is_kernels_ready()` returns true

**However**, this function is only called from:
- `process()` Step 5 (core processing fallback) — rarely reached
- `predict_next_word_via_pulses_excluding()` — only if pulse prediction runs
- `train_batch()` — during training
- `train_neural()` — during neural training

### 9.3 CUDA Code Paths in `trainer.rs`

**`train_neural()` (line 600-646):**
```rust
if gpu_available {
    let mut acc = crate::cuda::get_accelerator();
    acc.process_cores_batch(...);
    acc.field_update(...);
    drop(acc);
} else {
    model.process_cores_parallel(&mut pulses);
    model.field.update(&mut pulses);
}
```

**This GPU path IS reachable** during neural training if CUDA is available.

### 9.4 CUDA Code Paths in `cuda.rs`

All GPU operations in `NovaAccelerator`:
- `selective_scan()` — GPU path reachable
- `ssm_transform_batch()` — GPU path reachable
- `field_update()` — GPU path reachable
- `process_cores_batch()` — GPU path reachable

Each has a CPU fallback that runs if GPU fails.

### 9.5 Summary of Used CUDA Paths

| Function | Used During | Frequency |
|----------|-------------|-----------|
| `process_cores_batch()` | Neural training | When `--neural` flag used |
| `field_update()` | Neural training | When `--neural` flag used |
| `selective_scan()` | Via process_cores_batch | When `--neural` flag used |
| `ssm_transform_batch()` | Via process_cores_batch | When `--neural` flag used |
| `process_cores_parallel()` GPU path | Inference fallback | Rarely (n-gram path preferred) |

---

## 10. WHICH CUDA CODE PATHS ARE DEAD CODE

### 10.1 CUDA Kernels (kernels/ssm.cu)

| Kernel | Used? | Notes |
|--------|-------|-------|
| `selective_scan_kernel` | USED | Via `launch_selective_scan()` |
| `ssm_transform_batch_kernel` | USED | Via `launch_ssm_transform_batch()` |
| `field_update_kernel` | USED | Via `launch_field_update()` |
| `field_diffuse_kernel` | USED | Via `launch_field_diffuse()` |
| `cosine_similarity_kernel` | **DEAD** | `launch_cosine_similarity()` never called |
| `vector_add_kernel` | **DEAD** | `launch_vector_add()` never called |
| `vector_clamp_kernel` | **DEAD** | `launch_vector_clamp()` never called |
| `core_process_kernel` | USED | Via `launch_core_process()` |

### 10.2 Dead CUDA Launch Functions

In `cuda.rs`:
- `launch_cosine_similarity()` (line 192-206) — **NEVER CALLED**
- `launch_vector_add()` (line 208-220) — **NEVER CALLED**
- `launch_vector_clamp()` (line 222-233) — **NEVER CALLED**

### 10.3 Dead CUDA Accelerator Methods

In `NovaAccelerator`:
- `selective_scan()` (line 384-418) — **NEVER CALLED directly** (only via `process_cores_batch`)
- `ssm_transform_batch()` (line 420-470) — **NEVER CALLED directly** (only via `process_cores_batch`)

### 10.4 Dead SSM Functions (src/ssm.rs)

| Function | Used? | Notes |
|----------|-------|-------|
| `selective_scan_step()` | USED | Via `ssm_transform_pulse()` |
| `selective_scan_step_raw()` | USED | Via CUDA CPU fallback |
| `selective_scan_sequence()` | **DEAD** | Never called |
| `time_mixing()` | USED | Via `ssm_transform_pulse()` when `use_time_mixing=true` |
| `channel_mixing()` | **DEAD** | Never called |
| `wkv_attention()` | **DEAD** | Never called |
| `outer_product()` | **DEAD** | Only used by `wkv_attention()` |
| `mat_vec_mul_nested()` | **DEAD** | Only used by `channel_mixing()` |
| `ssm_transform_pulses()` | **DEAD** | Never called (batch version exists but unused) |

### 10.5 Dead Code in field.rs

- `field.enable_ssm()` — **NEVER CALLED**
- `field.disable_ssm()` — **NEVER CALLED**
- SSM-enhanced field update path — **NEVER EXECUTED** (use_ssm always false)

### 10.6 Dead Code in loom.rs

- `conversational_override()` — Returns `None`, always falls through
- `find_closest_word()` — Only used in tests? Not called in main inference path
- `find_closest_word_excluding()` — Used in `predict_next_word_via_pulses_excluding()`

### 10.7 Dead Code in core.rs

- `default_transform()` — Only reached if core name doesn't match any known name
- `reset_ssm()` — Never called externally (SSM is reset via `ssm.reset()` directly)

---

## 11. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NOVA CORE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT TEXT                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    process() PIPELINE                         │   │
│  │                                                               │   │
│  │  Step 1: Exact hash match? ────YES────► Return memorized     │   │
│  │      │ NO                                                    │   │
│  │      ▼                                                       │   │
│  │  Step 2: Conversational override? ──YES────► Return canned   │   │
│  │      │ NO (always None)                                      │   │
│  │      ▼                                                       │   │
│  │  Step 3: Word-overlap match? ────YES────► Return memorized   │   │
│  │      │ NO                                                    │   │
│  │      ▼                                                       │   │
│  │  Step 4: N-gram generation? ────YES────► Generate via n-grams│   │
│  │      │ NO (no vocab/ngrams)                                  │   │
│  │      ▼                                                       │   │
│  │  Step 5: Core + Field processing ───────► Generate via pulses│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    generate_text()                            │   │
│  │                                                               │   │
│  │  For each word to generate:                                   │   │
│  │    1. Loop detection (ABAB, AAA, ABCABC)                      │   │
│  │    2. Pulse prediction (if use_pulse_prediction)              │   │
│  │    3. N-gram context match                                    │   │
│  │    4. N-gram backoff (shorter contexts)                       │   │
│  │    5. Global n-gram distribution                              │   │
│  │    6. Diverse word from vocabulary                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    CORES (5 default)                          │   │
│  │                                                               │   │
│  │  syntax: tanh scaling + SSM                                   │   │
│  │  semantic: amplify/dampen + SSM                               │   │
│  │  memory: blend with stored memory + SSM                       │   │
│  │  reasoning: propagate differences + SSM                       │   │
│  │  pattern: detect repetitions + SSM                            │   │
│  │                                                               │   │
│  │  Each core: adaptive_depth (1-12 iterations)                  │   │
│  │  Each core: SSM (d_inner=dim, d_state=16)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    FIELD (O(n) dynamics)                      │   │
│  │                                                               │   │
│  │  1. Weighted average of pulses (parallel)                     │   │
│  │  2. Update state with momentum                                │   │
│  │  3. Diffuse state back to pulses                              │   │
│  │                                                               │   │
│  │  SSM enhancement: DISABLED (never enabled)                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    KNOWLEDGE STORES                           │   │
│  │                                                               │   │
│  │  learned_responses: HashMap<u64, String>  (PRIMARY)           │   │
│  │  ngram_patterns:    HashMap<u64, Vec<(String,f32)>> (MAIN)    │   │
│  │  vocabulary:        HashMap<String, Vec<f32>>  (WEAK)         │   │
│  │  core.memory:       Vec<f32> per core         (WEAK)          │   │
│  │  field.state:       Vec<f32>                   (WEAK)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    TRAINING METHODS                           │   │
│  │                                                               │   │
│  │  train():           Hash memorization + n-gram learning       │   │
│  │  train_neural():    Hash memorization + heuristic updates     │   │
│  │  train_one_pass():  Hash memorization ONLY (skip cores)       │   │
│  │  train_one_pass_ultra(): Same as train_one_pass               │   │
│  │                                                               │   │
│  │  NO gradient descent, NO backpropagation                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. CRITICAL FINDINGS SUMMARY

### 12.1 The Model Is Not Actually Learning

**Finding:** Nova Core does NOT perform gradient-based learning. All training methods use:
1. Hash-based memorization (input → output lookup table)
2. N-gram statistical pattern learning
3. Heuristic vector nudges (not gradient descent)

**Impact:** The model cannot generalize. It can only reproduce training examples exactly or generate statistically plausible n-gram continuations.

### 12.2 The Core Architecture Is Bypassed

**Finding:** During inference, the n-gram path (Step 4 in `process()`) is reached before the core+field path (Step 5). The cores and field are only used as a last resort when no vocabulary or n-gram patterns exist.

**Impact:** The entire Nova philosophy (O(n) field dynamics, pulse-based computation, adaptive iterative reasoning, graph-based cores) is NOT being utilized during normal operation.

### 12.3 Pulses and Iterations Counters Show 0

**Finding:** The `total_pulses_processed` and `total_iterations` counters remain at 0 because:
- The n-gram generation path doesn't create pulses
- Pulse prediction in `generate_text()` may be skipped if `use_pulse_prediction` is false
- The core processing fallback is rarely reached

**Impact:** The counters accurately reflect that the neural computation path is not being used.

### 12.4 Field SSM Is Dead Code

**Finding:** `field.enable_ssm()` is never called. The field's SSM-enhanced dynamics are completely dead code.

**Impact:** The field operates as a simple momentum-based averager without state-space memory.

### 12.5 CUDA Cosine Similarity and Vector Operations Are Dead Code

**Finding:** Three CUDA kernels (`cosine_similarity_kernel`, `vector_add_kernel
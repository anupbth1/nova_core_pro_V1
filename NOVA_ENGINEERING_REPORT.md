# Nova Core — Complete Engineering & Architecture Report

**Date:** 2026-07-03  
**Author:** AI Architecture Analysis (based on real source code inspection)  
**Version:** 0.1.0  
**Repository:** github.com/anupbth1/nova_core_pro_V1

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module-by-Module Analysis](#3-module-by-module-analysis)
4. [Data Flow & Inference Path](#4-data-flow--inference-path)
5. [Training System](#5-training-system)
6. [GPU Acceleration](#6-gpu-acceleration)
7. [Memory Model & State Management](#7-memory-model--state-management)
8. [SSM (State Space Model) Implementation](#8-ssm-state-space-model-implementation)
9. [Field Dynamics](#9-field-dynamics)
10. [Pulse-Based Computation](#10-pulse-based-computation)
11. [Core Transforms](#11-core-transforms)
12. [Benchmarking & Evaluation](#12-benchmarking--evaluation)
13. [Utility Modules (Coding, Math, Tools)](#13-utility-modules-coding-math-tools)
14. [Build System & Dependencies](#14-build-system--dependencies)
15. [Known Issues & Limitations](#15-known-issues--limitations)
16. [Roadmap & Recommendations](#16-roadmap--recommendations)

---

## 1. Executive Summary

Nova Core is a **post-Transformer LLM** implemented in Rust that deliberately avoids attention mechanisms, discrete tokens, and fixed-layer architectures. Its core innovation is a **field dynamics** paradigm where:

- **Pulses** (continuous vectors) replace tokens
- **Field** (a global state vector) replaces attention
- **Cores** (adaptive-depth processors) replace fixed layers
- **SSM** (Mamba-style selective scan) provides recurrent state tracking

**Current Status:** Nova Core is a **functional prototype** with a complete inference pipeline, training system, GPU acceleration, and benchmark suite. However, it is **NOT yet competitive** with modern LLMs. The architecture is innovative but the learning mechanism is fundamentally limited — training is primarily hash-based memorization rather than gradient-based optimization of learned parameters.

**Key Finding:** Nova's "training" does NOT modify the SSM parameters (A, B, C, delta, etc.) through gradient descent. Instead, it stores input→output associations in a HashMap and performs heuristic adjustments to core memory/state vectors. The SSM parameters remain at their random initialization values. This is the single most critical limitation.

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NovaLoom (Orchestrator)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Core 0  │  │  Core 1  │  │  Core 2  │  │  Core 3,4   │ │
│  │ (syntax) │  │(semantic)│  │ (memory) │  │(reason,pat) │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
│       │              │              │               │        │
│       └──────────────┴──────┬──────┴───────────────┘        │
│                             │                                │
│                      ┌──────▼──────┐                        │
│                      │ NovaField   │                        │
│                      │ (global)    │                        │
│                      └─────────────┘                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Supporting Systems:                                 │   │
│  │  • KnowledgeStore (concepts, relations, facts)       │   │
│  │  • Vocabulary (hash-based deterministic embeddings)  │   │
│  │  • N-gram patterns (bigram/trigram language model)   │   │
│  │  • Learned responses (HashMap<u64, String>)          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Module Dependency Graph

```
main.rs ──► loom.rs ──► core.rs ──► ssm.rs
                │            │
                ├──► field.rs
                ├──► pulse.rs
                ├──► knowledge.rs
                ├──► trainer.rs ──► optimizer.rs
                ├──► context.rs
                ├──► cuda.rs
                ├──► dataset.rs
                ├──► model.rs
                ├──► coding.rs
                ├──► math.rs
                ├──► tools.rs
                └──► benchmark/
```

### 2.3 Key Design Principles

1. **O(n) Complexity:** Field dynamics scale linearly with dimension, avoiding O(n²) attention
2. **No Tokens:** Continuous pulse vectors instead of discrete token IDs
3. **No Fixed Layers:** Adaptive-depth cores that iterate based on entropy/weight
4. **Hash-Based Learning:** Fast memorization via input hashing (NOT gradient descent on SSM params)
5. **Deterministic Embeddings:** Vocabulary embeddings are hash-derived, not learned
6. **Optional GPU Acceleration:** CUDA via cudarc crate, with graceful CPU fallback

---

## 3. Module-by-Module Analysis

### 3.1 `src/main.rs` (1150 lines) — CLI Entry Point

**Status:** COMPLETE AND FUNCTIONAL

**Commands:**
| Command | Description | Status |
|---------|-------------|--------|
| `Run` | Process a single input through Nova | ✅ |
| `Bench` | Run benchmark suite | ✅ |
| `Chat` | Interactive chat mode | ✅ |
| `Info` | Show model info | ✅ |
| `Speed` | Speed benchmark | ✅ |
| `FullBench` | Full benchmark suite | ✅ |
| `Improve` | Auto-improve from benchmarks | ✅ |
| `GenData` | Generate training data | ✅ |
| `Train` | Train Nova | ✅ |
| `SmartChat` | Chat with knowledge augmentation | ✅ |
| `Dataset` | Dataset management | ✅ |
| `Model` | Model save/load/list/delete | ✅ |
| `HfTrain` | Train from Hugging Face dataset | ✅ |
| `MultiHfTrain` | Train from multiple HF datasets | ✅ |

**Startup Sequence:**
1. Parse CLI args via clap
2. `init_global_thread_pool()` — auto-detect CPU threads
3. `init_global_accelerator()` — auto-detect GPU
4. Create `NovaLoom::new(dim, cores)` with default dim=64, cores=5
5. Dispatch to command handler

### 3.2 `src/pulse.rs` (124 lines) — NovaPulse

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct NovaPulse {
    pub content: Vec<f32>,  // Continuous vector representation
    pub weight: f32,        // Importance/confidence weight
    pub entropy: f32,       // Uncertainty measure (drives adaptive depth)
    pub position: usize,    // Position in sequence
    pub parent: Option<usize>, // Parent pulse index (for hierarchical structure)
}
```

**Key Methods:**
- `new(dim)` — Creates random pulse with uniform [-1, 1] content, weight=1.0, entropy=1.0
- `from_text(text, dim)` — **Deterministic byte-to-float encoding**: `content[i] = (byte as f32) / 255.0 * 2.0 - 1.0`. NOT learned.
- `transform()` — Applies tanh to content, reduces entropy by 3%
- `reduce_entropy(factor)` — Scales entropy down
- `dominant()` — Returns index of max absolute value in content
- `similarity(other)` — Cosine similarity between two pulses

**Critical Limitation:** `from_text()` uses a bijective byte mapping that preserves information but produces embeddings with NO semantic meaning. "cat" and "dog" are no more similar than "cat" and "xyz". This is fundamentally different from learned embeddings in Transformer models.

### 3.3 `src/field.rs` (274 lines) — NovaField

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct NovaField {
    pub dim: usize,
    state: Vec<f32>,           // Global field state vector
    momentum: Vec<f32>,        // Momentum for smooth updates
    pub learning_rate: f32,    // Default: 0.1
    pub diffusion: f32,        // Default: 0.3
    pub update_count: usize,
    pub ssm: Option<StateSpace>, // Optional SSM for field-level processing
    pub use_ssm: bool,
    pub ssm_gate: f32,         // Blend ratio for SSM output (default: 0.3)
}
```

**Core Algorithm — `update()`:**
```
Step 1: Weighted Field Average (O(n) parallel via Rayon)
  field_avg[i] = Σ(pulse.content[i] * pulse.weight) / Σ(pulse.weight)

Step 2: SSM-Enhanced State Update
  if use_ssm:
    ssm_output = selective_scan_step(ssm, field_avg)
    blended = field_avg * (1 - ssm_gate) + ssm_output * ssm_gate
    momentum = momentum * 0.9 + (blended - state) * learning_rate
  else:
    momentum = momentum * 0.9 + (field_avg - state) * learning_rate
  state += momentum
  state = clamp(state, -1.0, 1.0)

Step 3: Diffuse Field to Pulses
  for each pulse:
    pulse.content[i] = pulse.content[i] * (1 - diffusion) + state[i] * diffusion
```

**Analysis:** The field acts as a **global context aggregator** — it compresses all pulse information into a single state vector, then diffuses that state back to the pulses. This is O(n*d) where n=pulses, d=dimension, compared to O(n²*d) for attention. However, the compression into a single vector is a severe bottleneck — it cannot represent multiple simultaneous contexts.

### 3.4 `src/core.rs` (357 lines) — NovaCore

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct NovaCore {
    pub id: usize,
    pub name: String,           // "syntax", "semantic", "memory", "reasoning", "pattern"
    pub memory: Vec<f32>,       // Core-specific memory buffer
    pub adaptive_depth: usize,  // Computed from pulse entropy/weight
    pub internal_state: Vec<f32>,
    pub gate: f32,              // Blend strength (default: 0.5)
    pub ssm: StateSpace,        // Per-core SSM
    pub use_ssm: bool,
    pub use_time_mixing: bool,
    pub received_messages: Vec<CoreMessage>, // PHASE 4: cross-core communication
    pub cross_core_blend: f32,  // PHASE 4: blend strength for received messages
}
```

**Core Algorithm — `process()`:**
```
1. Compute adaptive_depth from avg_entropy and avg_weight
   depth = max(1, min(10, (avg_entropy * 3.0 / avg_weight.max(0.1)) as usize))

2. For each iteration up to adaptive_depth:
   a. syntax_transform: pulse.content = tanh(pulse.content)
   b. semantic_transform: amplify high-magnitude, attenuate low-magnitude
   c. memory_transform: read from / write to core memory
   d. reasoning_transform: pairwise diff between pulses
   e. pattern_transform: similarity detection between pulses
   f. default_transform: tanh with scaling
   g. ssm_transform: selective scan step with blending
   h. knowledge_transform (PHASE 5): augment with knowledge store
```

**Transform Details:**
| Transform | Operation | Complexity |
|-----------|-----------|------------|
| Syntax | `tanh(content)` | O(d) per pulse |
| Semantic | `content[i] *= 1.5 if |content[i]| > 0.5 else *= 0.5` | O(d) per pulse |
| Memory | Read: `content += memory[offset..]`, Write: `memory[offset..] = content` | O(d) |
| Reasoning | `content[i] = content[i] - content[j]` for pulse pairs | O(n²*d) |
| Pattern | Cosine similarity between all pulse pairs | O(n²*d) |
| SSM | `selective_scan_step(ssm, content)` with blending | O(d*d_state) |
| Knowledge | Augment pulse with closest concept from KnowledgeStore | O(v*d) where v=concepts |

**Critical Issue:** The reasoning and pattern transforms are O(n²) — they iterate over all pulse pairs. This undermines the O(n) claim for the overall architecture when these transforms are active.

### 3.5 `src/ssm.rs` (619 lines) — StateSpace

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct StateSpace {
    pub d_state: usize,     // Default: 16
    pub d_inner: usize,     // Default: 64 (matches field dim)
    pub a: Vec<f32>,        // [d_inner * d_state] — State transition matrix
    pub a_log: Vec<f32>,    // Log of A (for numerical stability)
    pub b: Vec<f32>,        // [d_inner * d_state] — Input matrix
    pub c: Vec<f32>,        // [d_inner * d_state] — Output matrix
    pub h: Vec<f32>,        // [d_inner * d_state] — Hidden state
    pub output_buf: Vec<f32>, // [d_inner] — Output buffer
    pub delta: Vec<f32>,    // [d_inner] — Step size parameter
    pub delta_bias: Vec<f32>, // [d_inner] — Step size bias
    pub d: Vec<f32>,        // [d_inner] — Skip connection
    // RWKV-style time mixing parameters
    pub time_mix_x: Vec<f32>,
    pub time_mix_w: Vec<f32>,
    pub time_mix_key: Vec<f32>,
    pub time_mix_value: Vec<f32>,
    pub time_mix_receptance: Vec<f32>,
    pub prev_x: Vec<f32>,
}
```

**Flat Memory Layout:** All SSM matrices use flat `Vec<f32>` with indexing `[i * d_state + j]` where `i` is the d_inner index and `j` is the d_state index. This is the same layout as Mamba.

**Core Algorithm — `selective_scan_step()`:**
```
For each i in 0..d_inner:
  Δ = softplus(delta[i] * x[i] + delta_bias[i])
  For each j in 0..d_state:
    h[i * d_state + j] = exp(Δ * A[i * d_state + j]) * h[i * d_state + j]
                        + Δ * B[i * d_state + j] * x[i]
  y[i] = Σ(C[i * d_state + j] * h[i * d_state + j]) + D[i] * x[i]
```

**RWKV-Style Time Mixing:**
```
mixed_x = time_mix_x * x + (1 - time_mix_x) * prev_x
mixed_w = time_mix_w * w + (1 - time_mix_w) * prev_w
... (similar for key, value, receptance)
```

**Critical Issue:** The SSM parameters (A, B, C, delta, delta_bias, D) are initialized randomly and **NEVER updated during training**. The `trainer.rs` does not call any SSM parameter update functions. The SSM acts as a fixed random projection, not a learned state space model.

### 3.6 `src/loom.rs` (1082 lines) — NovaLoom (Orchestrator)

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct NovaLoom {
    pub name: String,
    pub cores: Vec<NovaCore>,       // 5 cores: syntax, semantic, memory, reasoning, pattern
    pub field: NovaField,           // Global field
    pub dim: usize,                 // Default: 64
    pub max_iterations: usize,      // Default: 6
    pub convergence_threshold: f32, // Default: 0.12
    pub total_pulses_processed: u64,
    pub total_iterations: u64,
    pub learned_responses: HashMap<u64, String>,  // Hash → response
    pub learned_inputs: HashMap<u64, String>,      // Hash → original input
    pub vocabulary: HashMap<String, Vec<f32>>,     // Word → embedding
    pub vocab_reverse: HashMap<u64, String>,       // Hash → word
    pub ngram_patterns: HashMap<u64, Vec<(String, f32)>>, // Context → predictions
    pub ngram_order: usize,         // Default: 3
    pub all_words: Vec<String>,
    pub knowledge: KnowledgeStore,
}
```

**Main Inference Path — `process()`:**
```
Step 1: Hash Lookup (Exact Match)
  input_hash = hash(input_text)
  if learned_responses contains input_hash:
    return learned_responses[input_hash]  // EARLY EXIT

Step 2: Conversational Override (REMOVED — returns None)
  // Previously had hardcoded overrides, now removed

Step 3: Word-Overlap Matching
  Find learned_input with highest word overlap
  If overlap > threshold, return associated response

Step 4: Neural Path (ALWAYS RUN)
  a. text_to_pulses(input) — split by whitespace, create NovaPulse per word
  b. For iteration in 0..max_iterations:
     - process_cores_parallel(pulses) — GPU or CPU parallel
     - field.update(pulses) — weighted average + diffusion
     - Check convergence: if avg_entropy < threshold, break
  c. pulses_to_text(pulses) — map pulses to vocabulary words

Step 5: Return Response
  Priority: exact match > overlap match > neural generation > n-gram fallback
```

**`process_cores_parallel()`:**
- **GPU path** (with `--features cuda`): Calls `accelerator.process_cores_batch()`
- **CPU path** (default): Uses Rayon `par_iter()` over cores, each core processes all pulses

**`generate_text()` — Text Generation:**
```
For each position up to max_words:
  1. Pulse prediction: process input through cores + field
  2. N-gram prediction: look up context_hash in ngram_patterns
  3. Backoff n-gram: try lower-order n-gram
  4. Distribution sampling: weighted random from vocabulary
  5. Diverse word: random word from all_words
  Priority: pulse > n-gram > backoff > distribution > diverse
```

**`pulses_to_text()` — Pulse Decoding:**
1. For each pulse, find closest vocabulary word via cosine similarity
2. Early exit at similarity > 0.95
3. Fallback: deterministic word list (150 hardcoded words) indexed by pulse hash

**Critical Issues:**
- The neural path ALWAYS runs, even for exact matches (though the result is discarded)
- `pulses_to_text()` has a hardcoded fallback list of 150 words — this is the actual output for most pulses since cosine similarity rarely exceeds 0.95 with random embeddings
- The vocabulary is hash-based deterministic, not learned — no semantic structure
- `LongContextManager` is defined in `context.rs` but **NEVER called** from loom.rs

### 3.7 `src/trainer.rs` (1139 lines) — NovaTrainer

**Status:** COMPLETE AND FUNCTIONAL (but fundamentally limited)

**Training Methods:**

| Method | Description | Status |
|--------|-------------|--------|
| `init_vocabulary()` | Creates hash-based deterministic embeddings | ✅ |
| `train_batch()` | Forward pass + hash association + backward pass | ✅ |
| `train_epoch()` | Shuffle, batch, train, compute accuracy | ✅ |
| `train()` | Multi-epoch training loop | ✅ |
| `train_neural()` | Full vector error training with GPU support | ✅ |
| `train_one_pass()` | Ultra-fast hash-based learning | ✅ |
| `train_one_pass_ultra()` | Same as one_pass | ✅ |
| `compute_loss()` | MSE against target word embeddings | ✅ |
| `pulse_to_word()` | Cosine similarity or deterministic fallback | ✅ |

**What Training Actually Does:**

1. **Hash Association:** Stores `hash(input) → target_text` in `learned_responses`
2. **N-gram Learning:** Builds bigram/trigram transition probabilities
3. **Core Memory Update:** Heuristic adjustment of core memory vectors:
   ```rust
   // From train_batch():
   for (j, val) in core.memory.iter_mut().enumerate() {
       *val += error_signal * 0.01;  // Heuristic, NOT gradient
   }
   ```
4. **Field State Update:** Direct state modification:
   ```rust
   // From train_batch():
   field_state[i] += error_signal * 0.005;  // Heuristic, NOT gradient
   ```
5. **SSM Parameters:** **NOT UPDATED** — remain at random initialization
6. **Vocabulary:** **NOT UPDATED** — remains hash-based deterministic

**`train_neural()` — The "Neural" Training:**
Despite the name, this method:
- Computes MSE loss between pulse outputs and target embeddings
- Updates core memory/state/gate with scaled error signals
- Updates field state with scaled error signals
- **Does NOT compute gradients through SSM parameters**
- **Does NOT use backpropagation through time**
- **Does NOT update vocabulary embeddings**

**`compute_loss()`:**
```rust
fn compute_loss(output_pulses: &[NovaPulse], target_embedding: &[f32]) -> f32 {
    let mut loss = 0.0;
    for pulse in output_pulses {
        for i in 0..pulse.content.len().min(target_embedding.len()) {
            let diff = pulse.content[i] - target_embedding[i];
            loss += diff * diff;
        }
    }
    loss / output_pulses.len() as f32
}
```

**Critical Issue:** This is MSE loss but there's no gradient computation or backpropagation. The "backward pass" is heuristic scaling of error signals, not true gradient descent. The `NovaOptimizer` (AdamW) is defined in `optimizer.rs` but **NEVER CALLED** from trainer.rs.

### 3.8 `src/optimizer.rs` (613 lines) — NovaOptimizer

**Status:** DEFINED BUT NOT INTEGRATED

**Features:**
- AdamW optimizer with weight decay
- Gradient clipping
- Learning rate scheduling (Constant, Cosine, LinearWarmupDecay, StepDecay)
- Finite-difference gradient computation (central difference)
- AdamW state management (m, v, t per parameter)

**Why It's Not Used:**
The optimizer requires proper gradient computation, but the training system doesn't compute gradients. The `compute_gradients_finite_diff()` function uses central difference approximation:
```rust
fn compute_gradients_finite_diff(params: &mut [f32], ...) {
    for i in 0..params.len() {
        let original = params[i];
        params[i] = original + epsilon;
        let loss_plus = compute_loss(...);
        params[i] = original - epsilon;
        let loss_minus = compute_loss(...);
        params[i] = original;
        gradients[i] = (loss_plus - loss_minus) / (2.0 * epsilon);
    }
}
```
This is O(n) forward passes per parameter — extremely expensive and not practical for real training.

### 3.9 `src/cuda.rs` (900+ lines) — GPU Acceleration

**Status:** COMPLETE AND FUNCTIONAL (with graceful CPU fallback)

**Components:**
- `HardwareBackend` enum: Cuda, Hip, Cpu, None
- `NovaAccelerator`: Global singleton with GPU/CPU operation tracking
- `CudaKernelManager`: PTX kernel loading and launching
- `BatchProfile` / `CumulativeProfile`: Detailed GPU profiling
- Persistent GPU buffer cache for memory reuse
- Async CUDA streams for overlapping operations

**GPU-Accelerated Operations:**
| Operation | CUDA Kernel | CPU Fallback |
|-----------|-------------|--------------|
| SSM Selective Scan | `selective_scan_kernel` | `selective_scan_step_raw()` |
| SSM Transform Batch | `ssm_transform_batch_kernel` | `ssm_transform_pulse()` |
| Field Update | `field_update_kernel` | Manual weighted average |
| Field Diffuse | `field_diffuse_kernel` | Manual diffusion |
| Cosine Similarity | `cosine_similarity_kernel` | Manual dot product |
| Vector Add | `vector_add_kernel` | Manual addition |
| Vector Clamp | `vector_clamp_kernel` | Manual clamp |
| Core Process | `core_process_kernel` | Rayon parallel |

**Profiling System:**
- Per-batch profiling with timing breakdown (preprocess, upload, kernel, download, sync)
- Cumulative profiling across all batches
- Fallback tracking with function name, reason, error, and location
- Memory allocation/reuse tracking

**Critical Issue:** The GPU path has extensive error handling that falls back to CPU on any failure. In practice, if CUDA is not properly configured, ALL operations fall back to CPU silently (with profiling tracking).

### 3.10 `src/knowledge.rs` (448 lines) — KnowledgeStore

**Status:** COMPLETE AND FUNCTIONAL

**Struct:**
```rust
pub struct KnowledgeStore {
    pub concepts: HashMap<String, Concept>,
    pub relations: HashMap<String, Vec<(String, String, f32)>>,  // relation → [(source, target, strength)]
    pub reverse_relations: HashMap<String, Vec<(String, String, f32)>>,
    pub facts: Vec<Fact>,
    pub facts_by_category: HashMap<String, Vec<Fact>>,
    pub dim: usize,
    pub max_concepts: usize,
    pub learning_rate: f32,
}
```

**Key Methods:**
- `add_concept(name, category, description)` — Creates deterministic byte-based embedding
- `add_relation(source, relation, target, strength)` — Adds directed relation
- `add_fact(subject, predicate, object, category)` — Adds factual triple
- `find_closest_concept(pulse)` — Cosine similarity search
- `augment_pulse_with_knowledge(pulse)` — Blends closest concept embedding into pulse
- `learn_from_example(input, target)` — Extracts concepts from words, adds relations

**Integration:** Called from `core.rs` in `knowledge_transform()` and from `trainer.rs` in `train_neural()`. However, the knowledge augmentation is a simple vector blend — it does not perform reasoning or inference over the knowledge graph.

### 3.11 `src/context.rs` (395 lines) — LongContextManager

**Status:** DEFINED BUT NOT INTEGRATED

**Components:**
- `LongContextManager`: Sliding window SSM, hierarchical fields, context compression
- `SlidingWindowSSM`: Window-based SSM with overlap
- `ContextCompressor`: Compression ratio and SSM-based compression
- `HierarchicalField`: Local + global field with blending

**Why It's Not Integrated:**
The `LongContextManager` is never instantiated or called from `loom.rs` or `trainer.rs`. It's a standalone module with no integration points. The main inference path processes all input at once without any windowing or compression.

### 3.12 `src/dataset.rs` (1022 lines) — NovaDataset

**Status:** COMPLETE AND FUNCTIONAL

**Supported Formats:**
- CSV (comma-separated)
- JSON (array of objects)
- JSONL (JSON lines)
- Parquet (via Python bridge)
- Text (plain text, split by paragraphs)

**Features:**
- Auto-detect input/target columns from common patterns
- Multi-column concatenation
- Prompt template support (`"User: {user}\nAssistant: {assistant}"`)
- Filter conditions (Equals, Contains, MinLength, MaxLength, Regex, NonEmpty)
- Hugging Face dataset download via Python bridge (3 fallback strategies)
- Train/validation split
- Save to JSONL

**HF Download Strategy:**
1. Try HF API to find Parquet/JSONL files, download and parse
2. Try raw data URLs on Hugging Face
3. Try `huggingface_hub` Python library

### 3.13 `src/model.rs` (583 lines) — NovaModelManager

**Status:** COMPLETE AND FUNCTIONAL

**Features:**
- Save/load models in `.nova` format (JSON)
- List available models
- Delete models
- Upload to Hugging Face Hub (via Python bridge)
- Download from Hugging Face Hub (via Python bridge)

**ModelSnapshot Structure:**
```rust
pub struct ModelSnapshot {
    pub config: ModelConfig,
    pub cores: Vec<CoreSnapshot>,     // Full SSM state included
    pub field_state: Vec<f32>,
    pub field_momentum: Vec<f32>,
    pub vocabulary: HashMap<String, Vec<f32>>,
    pub vocab_reverse: HashMap<u64, String>,
    pub ngram_patterns: HashMap<u64, Vec<(String, f32)>>,
    pub all_words: Vec<String>,
    pub learned_responses: HashMap<u64, String>,
    pub learned_inputs: HashMap<u64, String>,
    pub knowledge: Option<KnowledgeStore>,
}
```

### 3.14 `src/coding.rs` (756 lines) — CodingEngine

**Status:** COMPLETE BUT STANDALONE (not integrated into inference)

**Features:**
- Code analysis: pattern detection via string matching
- Code generation: template-based (hello, fibonacci, sort, quicksort)
- Code debugging: rule-based checks (unwrap(), unsafe, mutable defaults, etc.)

**Not Integrated:** CodingEngine is never called from the main inference path. It's a standalone utility accessible only through direct API calls.

### 3.15 `src/math.rs` (650+ lines) — MathEngine

**Status:** COMPLETE BUT STANDALONE (not integrated into inference)

**Features:**
- Expression evaluation (recursive descent)
- Linear equation solving
- Quadratic equation solving
- Logical deduction (modus ponens, modus tollens, hypothetical syllogism)
- Number theory (is_prime, gcd, lcm, prime_factors)
- Statistics (mean, median, mode, std_dev, correlation)

**Not Integrated:** MathEngine is never called from the main inference path.

### 3.16 `src/tools.rs` (550+ lines) — ToolEngine

**Status:** COMPLETE BUT STANDALONE (not integrated into inference)

**Features:**
- File read/write
- HTTP GET/POST (feature-gated with `--features http`)
- Calculator (recursive expression evaluator)
- Data transform (JSON ↔ CSV)
- Web search (placeholder — returns "Web search not implemented")
- Shell command (safe command whitelist)
- Code execution (placeholder — returns "not implemented")

**Not Integrated:** ToolEngine is never called from the main inference path.

### 3.17 `src/benchmark/` — Benchmark Suite

**Status:** COMPLETE AND FUNCTIONAL

**Components:**
- `mod.rs` (172 lines): Main benchmark orchestrator
- `tasks.rs` (209 lines): Benchmark task definitions
- `metrics.rs` (43 lines): Accuracy, precision, recall, F1, perplexity
- `data.rs` (54 lines): Training data generation from weak tasks
- `compare.rs` (15 lines): Comparison with other LLMs (placeholder)
- `improve.rs` (38 lines): Auto-improvement (placeholder)

**Benchmark Tasks:**
| Category | Tasks |
|----------|-------|
| Language Understanding | Sentiment analysis, named entity, paraphrase detection |
| Reasoning | Logical deduction, mathematical reasoning, analogical reasoning |
| Code | Code completion, bug detection |
| Long Context | Long summary, information retrieval |
| Memory | Short-term memory, working memory |
| Efficiency | Speed (tok/s), memory efficiency |

**Critical Issue:** All benchmark tasks use simple string matching for evaluation (`answer.contains(expected)`). This is not a rigorous evaluation methodology. The tasks are also very limited in scope and difficulty.

---

## 4. Data Flow & Inference Path

### 4.1 Complete Inference Pipeline

```
Input Text
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 1. Hash Lookup (Exact Match)                         │
│    hash(input) → learned_responses[hash]             │
│    If found: return response (but still run neural)  │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 2. Word-Overlap Matching                             │
│    Compare input words with learned_inputs           │
│    If overlap > threshold: return associated response │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 3. Text → Pulses                                     │
│    Split by whitespace                                │
│    For each word: NovaPulse::from_text(word, dim)     │
│    Result: Vec<NovaPulse> with deterministic content  │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 4. Iterative Core + Field Processing                 │
│    For iteration in 0..max_iterations (default: 6):  │
│    │                                                 │
│    ├── process_cores_parallel(pulses)                │
│    │   GPU: accelerator.process_cores_batch()        │
│    │   CPU: rayon::par_iter() over cores             │
│    │   Each core applies all transforms to pulses    │
│    │                                                 │
│    ├── field.update(pulses)                          │
│    │   Weighted average → SSM blend → momentum       │
│    │   → state update → diffuse to pulses            │
│    │                                                 │
│    └── Check convergence                             │
│        If avg_entropy < threshold: break early       │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 5. Pulses → Text                                     │
│    For each pulse:                                   │
│    ├── Cosine similarity with vocabulary             │
│    ├── If similarity > 0.95: return matching word    │
│    └── Fallback: deterministic word list (150 words) │
└──────────────────────────────────────────────────────┘
    │
    ▼
Output Text
```

### 4.2 Complexity Analysis

| Step | Complexity | Notes |
|------|-----------|-------|
| Hash Lookup | O(1) | HashMap operation |
| Word-Overlap | O(v * w) | v=vocab size, w=input words |
| Text→Pulses | O(w * d) | w=words, d=dimension |
| Core Processing | O(i * c * n * d * t) | i=iterations, c=cores, n=pulses, d=dim, t=transforms |
| Reasoning Transform | O(i * n² * d) | Pairwise pulse operations — O(n²) |
| Pattern Transform | O(i * n² * d) | Pairwise similarity — O(n²) |
| Field Update | O(i * n * d) | Weighted average + diffusion |
| Pulses→Text | O(n * v * d) | n=pulses, v=vocab size, d=dimension |
| **Total (worst case)** | **O(i * n² * d)** | Dominated by reasoning/pattern transforms |

**Key Finding:** Despite the O(n) claim, the architecture has O(n²) components in the reasoning and pattern transforms. For small n (< 20 pulses), this is negligible. For large n, it becomes the bottleneck.

### 4.3 Convergence Behavior

The adaptive iteration mechanism:
1. Computes `avg_entropy` across all pulses after each iteration
2. If `avg_entropy < convergence_threshold` (default: 0.12), stops early
3. `max_iterations` (default: 6) caps the maximum iterations

In practice, entropy decreases by ~3% per iteration per core (from `pulse.transform()`), so convergence typically happens in 3-5 iterations for short inputs.

---

## 5. Training System

### 5.1 Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NovaTrainer                              │
│                                                             │
│  ┌─────────────────────┐   ┌─────────────────────────────┐  │
│  │  Hash-Based Learning │   │  Neural Training            │  │
│  │                     │   │                             │  │
│  │  • Store input→output│   │  • Forward pass through    │  │
│  │    in HashMap        │   │    cores + field           │  │
│  │  • Build n-gram      │   │  • Compute MSE loss        │  │
│  │    patterns          │   │  • Heuristic backward pass │  │
│  │  • O(1) inference    │   │  • Update core memory/state│  │
│  │    for exact matches │   │  • Update field state      │  │
│  └─────────────────────┘   └─────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Knowledge Learning                                  │   │
│  │  • Extract concepts from words                       │   │
│  │  • Add "followed_by" and "predicts" relations        │   │
│  │  • Store input→target as fact                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 What Gets Updated During Training

| Component | Updated? | Method |
|-----------|----------|--------|
| `learned_responses` | ✅ YES | Direct HashMap insert |
| `ngram_patterns` | ✅ YES | Frequency counting |
| `knowledge` (KnowledgeStore) | ✅ YES | Concept/relation/fact extraction |
| Core `memory` | ✅ YES | Heuristic error signal scaling |
| Core `internal_state` | ✅ YES | Heuristic error signal scaling |
| Core `gate` | ✅ YES | Heuristic error signal scaling |
| Field `state` | ✅ YES | Heuristic error signal scaling |
| SSM `a, b, c, delta, delta_bias, d` | ❌ NO | Never updated |
| SSM `h` (hidden state) | ❌ NO | Reset per forward pass |
| Vocabulary embeddings | ❌ NO | Hash-based deterministic |
| Pulse `from_text()` encoding | ❌ NO | Deterministic byte mapping |

### 5.3 The Fundamental Training Problem

Nova's training does NOT perform gradient-based optimization of its core parameters. The "backward pass" is:

```rust
// From trainer.rs train_batch() — simplified
let error_signal = target_embedding[i] - pulse_output[i];
core.memory[j] += error_signal * 0.01;     // Heuristic scaling
field_state[i] += error_signal * 0.005;     // Heuristic scaling
```

This is equivalent to a hand-tuned update rule, NOT gradient descent. The SSM parameters (which are the most expressive component) remain at their random initialization values forever.

**Why this matters:** The SSM has 64 * 16 * 5 = 5,120 parameters per core (A, B, C matrices) plus 64 * 5 = 320 parameters for delta/bias/D, totaling ~27,000 parameters across 5 cores. These are never optimized. The only "learned" components are the core memory buffers (64 * 5 = 320 floats) and field state (64 floats), which are updated with heuristic rules.

### 5.4 Training Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `train_one_pass()` | Fastest | Lowest | Quick memorization |
| `train()` | Medium | Medium | Balanced training |
| `train_neural()` | Slowest | Highest (relatively) | "Deep" learning |

---

## 6. GPU Acceleration

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NovaAccelerator (Singleton)               │
│                                                             │
│  ┌─────────────────────┐   ┌─────────────────────────────┐  │
│  │  Hardware Detection  │   │  CUDA Kernel Manager        │  │
│  │                     │   │                             │  │
│  │  • Auto-detect GPU  │   │  • Load PTX modules         │  │
│  │  • Fallback to CPU  │   │  • Launch kernels           │  │
│  │  • Backend enum     │   │  • Manage streams           │  │
│  └─────────────────────┘   └─────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────┐   ┌─────────────────────────────┐  │
│  │  Buffer Cache        │   │  Profiling System           │  │
│  │                     │   │                             │  │
│  │  • Reuse GPU memory │   │  • Per-batch timing         │  │
│  │  • Reduce allocs    │   │  • Cumulative stats         │  │
│  │  • Key-based lookup │   │  • Fallback tracking        │  │
│  └─────────────────────┘   └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 CUDA Kernels (kernels/ssm.cu)

| Kernel | Grid | Block | Shared Memory | Description |
|--------|------|-------|---------------|-------------|
| `selective_scan_kernel` | (d_inner, 1, 1) | (32, 1, 1) | 128 bytes | Single SSM step |
| `ssm_transform_batch_kernel` | (num_pulses, 1, 1) | (256, 1, 1) | 0 | Batched SSM transform |
| `field_update_kernel` | ((dim+255)/256, 1, 1) | (256, 1, 1) | 0 | Weighted average + momentum |
| `field_diffuse_kernel` | ((total+255)/256, 1, 1) | (256, 1, 1) | 0 | Diffuse field to pulses |
| `cosine_similarity_kernel` | ((vocab+255)/256, 1, 1) | (256, 1, 1) | 0 | Vocabulary matching |
| `vector_add_kernel` | ((n+255)/256, 1, 1) | (256, 1, 1) | 0 | Element-wise add |
| `vector_clamp_kernel` | ((n+255)/256, 1, 1) | (256, 1, 1) | 0 | Element-wise clamp |
| `core_process_kernel` | (n_pulses, n_cores, 1) | (256, 1, 1) | 0 | Full core processing |

### 6.3 Build-Time Compilation (build.rs)

- Compiles `kernels/ssm.cu` to PTX using `nvcc`
- Target architecture: `sm_75` (Turing/T4/RTX 20xx)
- Also compiles for `sm_80` (Ampere/A100/RTX 30xx) if available
- PTX path embedded via `SSM_KERNELS_PTX` environment variable
- Only runs when `--features cuda` is enabled

### 6.4 GPU vs CPU Performance

The GPU acceleration provides benefit primarily for:
1. **Batch processing:** Multiple pulses through SSM in parallel
2. **Field operations:** Weighted average across many pulses
3. **Vocabulary matching:** Cosine similarity against large vocabulary

For single-inference (the common case with < 20 pulses), the CPU overhead of GPU upload/download may outweigh the kernel execution speedup.

---

## 7. Memory Model & State Management

### 7.1 State Components

| Component | Size | Persistence | Description |
|-----------|------|-------------|-------------|
| Core Memory | 64 × 5 = 320 floats | Saved/Loaded | Per-core memory buffer |
| Core Internal State | 64 × 5 = 320 floats | Saved/Loaded | Per-core state vector |
| Core Gate | 5 floats | Saved/Loaded | Per-core blend strength |
| SSM Hidden State (h) | 64 × 16 × 5 = 5,120 floats | Saved/Loaded | SSM recurrent state |
| SSM Parameters | ~27,000 floats | Saved/Loaded | A, B, C, delta, etc. |
| Field State | 64 floats | Saved/Loaded | Global field vector |
| Field Momentum | 64 floats | Saved/Loaded | Momentum buffer |
| Vocabulary | vocab_size × 64 floats | Saved/Loaded | Word embeddings |
| Learned Responses | variable | Saved/Loaded | Hash→text map |
| N-gram Patterns | variable | Saved/Loaded | Context→predictions |
| Knowledge Store | variable | Saved/Loaded | Concepts, relations, facts |

### 7.2 Memory Usage Estimate

| Configuration | Memory |
|---------------|--------|
| Base (dim=64, cores=5, d_state=16) | ~50 KB model params |
| With vocabulary (1000 words) | ~300 KB |
| With learned responses (10,000 examples) | ~5 MB |
| With knowledge store (10,000 concepts) | ~10 MB |
| **Total typical** | **~15-20 MB** |

This is orders of magnitude smaller than Transformer models (which are typically 1-100 GB).

---

## 8. SSM (State Space Model) Implementation

### 8.1 Mathematical Formulation

The SSM implements the Mamba-style selective scan:

```
h(t) = exp(Δ(t) · A) ⊙ h(t-1) + Δ(t) · B · x(t)
y(t) = C · h(t) + D · x(t)
```

Where:
- `Δ(t) = softplus(delta · x(t) + delta_bias)` — input-dependent step size
- `A ∈ ℝ^(d_inner × d_state)` — state transition (negative via parameterization)
- `B ∈ ℝ^(d_inner × d_state)` — input projection
- `C ∈ ℝ^(d_inner × d_state)` — output projection
- `D ∈ ℝ^(d_inner)` — skip connection
- `h ∈ ℝ^(d_inner × d_state)` — hidden state

### 8.2 Implementation Details

- **Flat memory layout:** All matrices stored as `Vec<f32>` with `[i * d_state + j]` indexing
- **Numerical stability:** A stored as `a_log` (log space), converted via `exp(a_log)` in `load_from_projection()`
- **Activation:** `softplus` for delta, `silu` for gating in channel mixing
- **Time mixing:** RWKV-style exponential moving average of input tokens
- **Channel mixing:** MLP with silu activation and gating

### 8.3 Key Functions

| Function | Description | Complexity |
|----------|-------------|------------|
| `selective_scan_step()` | Single step of SSM | O(d_inner × d_state) |
| `selective_scan_step_raw()` | Raw version (no bounds checking) | O(d_inner × d_state) |
| `selective_scan_sequence()` | Process sequence of inputs | O(seq_len × d_inner × d_state) |
| `time_mixing()` | RWKV-style temporal blending | O(d_inner) |
| `channel_mixing()` | MLP with gating | O(d_inner × 4) |
| `wkv_attention()` | RWKV-style WKV attention | O(d_inner × d_state) |
| `ssm_transform_pulse()` | Apply SSM to single pulse | O(d_inner × d_state) |
| `ssm_transform_pulses()` | Apply SSM to all pulses | O(n × d_inner × d_state) |

### 8.4 Comparison with Mamba

| Feature | Mamba | Nova SSM |
|---------|-------|----------|
| Selective scan | ✅ Yes | ✅ Yes |
| Flat memory layout | ✅ Yes | ✅ Yes |
| Input-dependent Δ | ✅ Yes | ✅ Yes |
| Learned parameters | ✅ Yes (gradient descent) | ❌ No (random, never updated) |
| Convolution mode | ✅ Yes | ❌ No |
| Hardware-aware algorithm | ✅ Yes (tile-based) | ❌ No (naive loops) |
| CUDA implementation | ✅ Yes (optimized) | ✅ Yes (basic) |

---

## 9. Field Dynamics

### 9.1 Physical Analogy

The field is inspired by classical field theory:
- **Pulses** are like particles moving through a field
- **Field** is a global potential that influences all pulses
- **Diffusion** spreads field information to pulses
- **Momentum** provides smooth temporal evolution

### 9.2 Mathematical Formulation

```
Field Average:
  μ[i] = Σ(w_j · p_j[i]) / Σ(w_j)

Momentum Update:
  m[i] = 0.9 · m[i] + η · (μ[i] - s[i])

State Update:
  s[i] = s[i] + m[i]
  s[i] = clamp(s[i], -1.0, 1.0)

Diffusion:
  p_j[i] = (1 - α) · p_j[i] + α · s[i]
```

Where:
- `μ` = weighted field average
- `w_j` = weight of pulse j
- `p_j` = content of pulse j
- `m` = momentum vector
- `s` = field state vector
- `η` = learning rate (default: 0.1)
- `α` = diffusion rate (default: 0.3)

### 9.3 SSM Integration

When SSM is enabled on the field:
```
ssm_output = selective_scan_step(ssm, μ)
blended = μ · (1 - ssm_gate) + ssm_output · ssm_gate
m[i] = 0.9 · m[i] + η · (blended[i] - s[i])
```

The SSM gate (default: 0.3) controls how much SSM output influences the field update.

### 9.4 Limitations

1. **Single global state:** The field compresses ALL pulses into ONE vector. This is a severe bottleneck for representing multiple simultaneous contexts.
2. **No positional encoding:** The field treats all pulses equally regardless of position. Position information is only in the pulse's `position` field, which is not used in field averaging.
3. **Linear diffusion:** The diffusion is uniform across all dimensions — no selective attention to specific features.

---

## 10. Pulse-Based Computation

### 10.1 Pulse Lifecycle

```
Creation (from_text)
    │
    ▼
┌─────────────────────┐
│ Initial Content     │ ← Deterministic byte-to-float mapping
│ Initial Weight: 1.0 │
│ Initial Entropy: 1.0│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Core Transforms     │ ← Applied iteratively by each core
│ • tanh(content)     │
│ • Amplify/attenuate │
│ • Memory R/W        │
│ • Pairwise diff     │
│ • Similarity detect │
│ • SSM transform     │
│ • Knowledge augment │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Field Diffusion     │ ← Field state blended into content
│ content *= (1-α)    │
│ content += state * α│
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Entropy Reduction   │ ← Each transform reduces entropy
│ entropy *= 0.97     │
│ entropy = max(0.01) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Decoding            │ ← Map back to text
│ Cosine similarity   │
│ or fallback list    │
└─────────────────────┘
```

### 10.2 Deterministic Encoding

The `from_text()` function uses a bijective byte-to-float mapping:

```rust
pub fn from_text(text: &str, dim: usize) -> Self {
    let bytes = text.as_bytes();
    let mut content = vec![0.0; dim];
    for i in 0..dim.min(bytes.len()) {
        content[i] = (bytes[i] as f32) / 255.0 * 2.0 - 1.0;  // Map [0,255] → [-1, 1]
    }
    // Pad remaining dimensions with hash-based values
    for i in bytes.len()..dim {
        let hash = (bytes.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
                    .wrapping_mul(31).wrapping_add(i as u64)) as f32;
        content[i] = (hash % 1000) as f32 / 1000.0 * 2.0 - 1.0;
    }
    NovaPulse { content, weight: 1.0, entropy: 1.0, position: 0, parent: None }
}
```

**Critical Implication:** This encoding preserves all information (it's bijective) but produces embeddings with NO semantic structure. "king" and "queen" are as different as "king" and "xyz". This is fundamentally different from learned embeddings where similar words have similar vectors.

### 10.3 Pulse Decoding Problem

The decoding process (`pulses_to_text()`) has a fundamental issue:
1. After multiple iterations of tanh transforms and field diffusion, the pulse content is heavily modified
2. Cosine similarity with vocabulary embeddings rarely exceeds 0.95 (the early exit threshold)
3. The fallback is a hardcoded list of 150 common English words indexed by pulse hash
4. This means most outputs come from the fallback list, not from actual learned associations

---

## 11. Core Transforms

### 11.1 Transform Details

#### Syntax Transform
```rust
fn syntax_transform(&self, pulse: &mut NovaPulse) {
    for x in pulse.content.iter_mut() {
        *x = x.tanh();  // Smooth nonlinearity, maps [-inf, inf] → [-1, 1]
    }
}
```

#### Semantic Transform
```rust
fn semantic_transform(&self, pulse: &mut NovaPulse) {
    for x in pulse.content.iter_mut() {
        if x.abs() > 0.5 {
            *x *= 1.5;  // Amplify strong signals
        } else {
            *x *= 0.5;  // Attenuate weak signals
        }
    }
}
```

#### Memory Transform
```rust
fn memory_transform(&self, pulse: &mut NovaPulse, core: &mut NovaCore) {
    // Read from memory
    for i in 0..pulse.content.len().min(core.memory.len()) {
        pulse.content[i] += core.memory[i] * 0.1;
    }
    // Write to memory (running average)
    for i in 0..pulse.content.len().min(core.memory.len()) {
        core.memory[i] = core.memory[i] * 0.9 + pulse.content[i] * 0.1;
    }
}
```

#### Reasoning Transform (O(n²))
```rust
fn reasoning_transform(&self, pulses: &mut [NovaPulse]) {
    for i in 0..pulses.len() {
        for j in (i+1)..pulses.len() {
            for k in 0..pulses[i].content.len().min(pulses[j].content.len()) {
                let diff = pulses[i].content[k] - pulses[j].content[k];
                pulses[i].content[k] -= diff * 0.1;  // Contrastive
                pulses[j].content[k] += diff * 0.1;
            }
        }
    }
}
```

#### Pattern Transform (O(n²))
```rust
fn pattern_transform(&self, pulses: &mut [NovaPulse]) {
    for i in 0..pulses.len() {
        for j in (i+1)..pulses.len() {
            let sim = pulses[i].similarity(&pulses[j]);
            if sim > 0.8 {
                // Merge similar pulses
                for k in 0..pulses[i].content.len() {
                    pulses[i].content[k] = (pulses[i].content[k] + pulses[j].content[k]) * 0.5;
                }
            }
        }
    }
}
```

#### SSM Transform
```rust
fn ssm_transform(&self, pulse: &mut NovaPulse, core: &mut NovaCore) {
    let original = pulse.content.clone();
    ssm_transform_pulse(&mut core.ssm, &mut pulse.content, core.use_time_mixing);
    let ssm_strength = core.gate * 0.5;
    for i in 0..pulse.content.len() {
        pulse.content[i] = original[i] * (1.0 - ssm_strength) + pulse.content[i] * ssm_strength;
        pulse.content[i] = pulse.content[i].clamp(-1.0, 1.0);
    }
}
```

### 11.2 Transform Order & Impact

The transforms are applied in a fixed order within each core iteration:

1. **Syntax** (tanh) — Smooths values, always reduces magnitude
2. **Semantic** — Amplifies/attenuates based on magnitude threshold
3. **Memory** — Reads from and writes to core memory buffer
4. **Reasoning** — Contrastive differences between pulse pairs
5. **Pattern** — Similarity-based merging
6. **Default** (tanh with scaling) — Additional smoothing
7. **SSM** — Selective scan with blending
8. **Knowledge** — Concept vector blending

After all transforms, entropy is reduced by 3% and weight is decayed slightly.

---

## 12. Benchmarking & Evaluation

### 12.1 Benchmark Suite Structure

```
NovaBenchmark
├── run_language_understanding()  → 3 tasks
├── run_reasoning_suite()         → 3 tasks
├── run_code_suite()              → 2 tasks
├── run_long_context()            → 2 tasks
├── run_efficiency_suite()        → 2 metrics
├── run_memory_suite()            → 2 tasks
└── generate_training_data()      → Auto-improvement
```

### 12.2 Evaluation Methodology

All tasks use simple string matching for evaluation:
```rust
evaluator: |answer, expected| {
    if answer.to_lowercase().contains(expected) { 1.0 } else { 0.0 }
}
```

**Critical Limitation:** This is NOT a rigorous evaluation. It only checks if the expected string appears anywhere in the output. A model that outputs "I don't know the answer is mortal but I'll try" would score 1.0 for the "mortal" task.

### 12.3 Benchmark Results Interpretation

Given the hash-based learning and deterministic embeddings, benchmark scores primarily measure:
1. Whether the input was memorized (exact hash match)
2. Whether a similar input was memorized (word overlap)
3. Whether the n-gram model happens to produce the right word
4. Random chance from the 150-word fallback list

The benchmark scores do NOT measure genuine language understanding or reasoning.

---

## 13. Utility Modules (Coding, Math, Tools)

### 13.1 Integration Status

| Module | File | Lines | Integrated into Inference? | Status |
|--------|------|-------|---------------------------|--------|
| CodingEngine | `coding.rs` | 756 | ❌ No | Standalone utility |
| MathEngine | `math.rs` | 650+ | ❌ No | Standalone utility |
| ToolEngine | `tools.rs` | 550+ | ❌ No | Standalone utility |

### 13.2 Why They're Not Integrated

These modules were added as part of the "implement all remaining phases" task but were never connected to the main inference pipeline. They are:
- Registered as modules in `main.rs`
- Importable by other modules
- Fully functional as standalone utilities
- **Not called** from `loom.rs`, `core.rs`, or `trainer.rs`

### 13.3 Integration Points (if desired)

| Module | Integration Point | What Would Need to Change |
|--------|-------------------|---------------------------|
| CodingEngine | `core.rs` transform | Add `code_transform()` that calls `CodingEngine` for code-related pulses |
| MathEngine | `core.rs` transform | Add `math_transform()` that evaluates mathematical expressions in pulses |
| ToolEngine | `loom.rs` process | Add tool invocation step after pulse decoding, before returning response |

---

## 14. Build System & Dependencies

### 14.1 Cargo.toml Dependencies

| Dependency | Version | Feature | Purpose |
|------------|---------|---------|---------|
| clap | 4.4 | derive, color | CLI argument parsing |
| colored | 2.0 | — | Terminal output coloring |
| rand | 0.8 | — | Random number generation |
| rayon | 1.7 | — | Parallel CPU processing |
| serde | 1.0 | derive | Serialization/deserialization |
| serde_json | 1.0 | — | JSON format support |
| anyhow | 1.0 | — | Error handling |
| thiserror | 1.0 | — | Error derive macros |
| regex-lite | 0.1 | — | Regex for dataset filtering |
| once_cell | 1.19 | — | Global accelerator singleton |
| cudarc | 0.19 | optional | CUDA GPU acceleration |
| ureq | 2.9 | optional | HTTP client for tools |

### 14.2 Feature Flags

| Feature | Enables | Default? |
|---------|---------|----------|
| `cuda` | CUDA GPU acceleration via cudarc | ❌ No |
| `gpu` | Alias for `cuda` | ❌ No |
| `hip` | AMD GPU support (future) | ❌ No |
| `http` | HTTP support for ToolEngine | ❌ No |

### 14.3 Build Script (build.rs)

The build script:
1. Detects CUDA toolkit path (from `CUDA_HOME` or `CUDA_PATH` env vars)
2. Compiles `kernels/ssm.cu` to PTX using `nvcc`
3. Targets `sm_75` (Turing) and optionally `sm_80` (Ampere)
4. Embeds PTX path via `SSM_KERNELS_PTX` environment variable
5. Only runs when `--features cuda` is enabled

### 14.4 Release Profile

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

For SIMD optimization: `RUSTFLAGS="-C target-cpu=native" cargo build --release`

---

## 15. Known Issues & Limitations

### 15.1 Critical Issues

| # | Issue | Impact | Location |
|---|-------|--------|----------|
| 1 | **SSM parameters never trained** | Core expressive component is random | `trainer.rs` — no SSM update code |
| 2 | **No gradient computation** | Training is heuristic, not gradient-based | `trainer.rs` — no autograd |
| 3 | **Deterministic embeddings** | No semantic structure in vocabulary | `pulse.rs` `from_text()` |
| 4 | **150-word fallback list** | Most outputs come from hardcoded list | `loom.rs` `pulses_to_text()` |
| 5 | **O(n²) transforms** | Reasoning and pattern are O(n²) | `core.rs` |
| 6 | **LongContextManager not integrated** | No long context support | `context.rs` — never called |
| 7 | **NovaOptimizer not used** | AdamW defined but never called | `optimizer.rs` — no integration |

### 15.2 Moderate Issues

| # | Issue | Impact | Location |
|---|-------|--------|----------|
| 8 | Single global field state | Cannot represent multiple contexts | `field.rs` |
| 9 | No positional encoding | Position information not used in field | `field.rs` |
| 10 | Benchmark evaluation is string matching | Scores are not meaningful | `benchmark/tasks.rs` |
| 11 | Coding/Math/Tools not integrated | Standalone utilities | `coding.rs`, `math.rs`, `tools.rs` |
| 12 | Neural path always runs | Wasted computation for exact matches | `loom.rs` `process()` |
| 13 | No proper tokenization | Whitespace splitting only | `loom.rs` `text_to_pulses()` |

### 15.3 Minor Issues

| # | Issue | Impact | Location |
|---|-------|--------|----------|
| 14 | Hardcoded convergence threshold (0.12) | Not adaptive to input | `loom.rs` |
| 15 | Fixed max_iterations (6) | Not adaptive to task complexity | `loom.rs` |
| 16 | No batch normalization | Training instability | `trainer.rs` |
| 17 | No dropout or regularization | Overfitting to training data | `trainer.rs` |
| 18 | GPU fallback tracking but no alerting | Silent CPU fallback | `cuda.rs` |
| 19 | chrono_now() is approximate | Date calculation is simplified | `model.rs` |
| 20 | No unit tests | No regression protection | All modules |

---

## 16. Roadmap & Recommendations

### 16.1 Critical Path to Competitiveness

To transform Nova from a prototype into a competitive LLM, the following are REQUIRED:

1. **Implement True Gradient-Based Training**
   - Replace heuristic updates with proper backpropagation
   - Compute gradients through SSM parameters (A, B, C, delta, delta_bias, D)
   - Use the existing `NovaOptimizer` (AdamW) for parameter updates
   - This requires either manual gradient computation or a framework like `candle`/`burn`

2. **Learn Vocabulary Embeddings**
   - Replace deterministic byte mapping with learned embeddings
   - Train embeddings jointly with the rest of the model
   - This enables semantic similarity between related words

3. **Fix Pulse Decoding**
   - Replace the 150-word fallback list with proper vocabulary projection
   - Add a learned output projection layer (d_inner → vocab_size)
   - Use softmax over vocabulary for word prediction

4. **Integrate LongContextManager**
   - Connect sliding window SSM to the main inference path
   - Enable processing of sequences longer than ~20 words
   - Implement hierarchical field for multi-scale context

### 16.2 Recommended Architecture Changes

1. **Replace O(n²) transforms** with O(n) alternatives
   - Use learned pairwise interactions instead of brute-force comparisons
   - Implement attention-like mechanisms that scale linearly (e.g., linear attention)

2. **Add proper tokenization** (BPE or SentencePiece)
   - Replace whitespace splitting with subword tokenization
   - This enables handling of out-of-vocabulary words and morphologically rich languages

3. **Implement batched training** with actual gradient accumulation
   - The current `train_batch()` doesn't accumulate gradients properly
   - Use the existing `NovaOptimizer` with proper gradient accumulation

4. **Add validation and early stopping**
   - Track validation loss during training
   - Stop training when validation loss stops improving
   - Save best model checkpoint

### 16.3 Quick Wins (Low Effort, High Impact)

1. **Connect NovaOptimizer to trainer.rs** — The optimizer is already implemented, just needs to be called
2. **Integrate Coding/Math/Tools** — Add transform calls in `core.rs` to invoke these engines
3. **Improve benchmark evaluation** — Use exact match or BLEU score instead of string contains
4. **Add unit tests** — Start with SSM and field dynamics tests

### 16.4 Long-Term Vision

```
Phase 1 (Current):  Functional prototype with hash-based learning
                    ↓
Phase 2:            Gradient-based training with learned embeddings
                    ↓
Phase 3:            Competitive performance on standard benchmarks
                    ↓
Phase 4:            Production-ready with proper tokenization, batching, and scaling
                    ↓
Phase 5:            Novel post-transformer architecture that rivals Transformers
```

---

## Appendix A: File Inventory

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `src/main.rs` | 1150 | ✅ Complete | CLI entry point |
| `src/loom.rs` | 1082 | ✅ Complete | Main orchestrator |
| `src/trainer.rs` | 1139 | ✅ Complete | Training system |
| `src/cuda.rs` | 900+ | ✅ Complete | GPU acceleration |
| `src/coding.rs` | 756 | ✅ Complete | Code analysis/generation |
| `src/ssm.rs` | 619 | ✅ Complete | State Space Model |
| `src/optimizer.rs` | 613 | ✅ Complete | AdamW optimizer |
| `src/tools.rs` | 550+ | ✅ Complete | Tool engine |
| `src/model.rs` | 583 | ✅ Complete | Model save/load |
| `src/knowledge.rs` | 448 | ✅ Complete | Knowledge store |
| `src/context.rs` | 395 | ✅ Complete | Long context manager |
| `src/core.rs` | 357 | ✅ Complete | Core transforms |
| `src/field.rs` | 274 | ✅ Complete | Field dynamics |
| `src/math.rs` | 650+ | ✅ Complete | Math engine |
| `src/dataset.rs` | 1022 | ✅ Complete | Dataset management |
| `src/pulse.rs` | 124 | ✅ Complete | Pulse computation |
| `kernels/ssm.cu` | 358 | ✅ Complete | CUDA kernels |
| `build.rs` | 67 | ✅ Complete | Build script |
| `Cargo.toml` | 69 | ✅ Complete | Dependencies |

## Appendix B: Key Metrics

| Metric | Value |
|--------|-------|
| Total Rust source lines | ~11,000 |
| Total CUDA kernel lines | 358 |
| Model parameters (trainable) | ~27,000 (SSM) + 640 (memory/state) |
| Model parameters (actually trained) | 640 (heuristic updates only) |
| Inference complexity (best case) | O(n × d) |
| Inference complexity (worst case) | O(n² × d) |
| Default dimension | 64 |
| Default cores | 5 |
| Default SSM state size | 16 |
| Default max iterations | 6 |
| Vocabulary size | Configurable (hash-based) |
| GPU support | CUDA (optional) |
| Training speed | ~1000 examples/sec (CPU) |

---

*End of Report*


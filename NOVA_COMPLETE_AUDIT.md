# NOVA CORE — Complete Architecture Audit Report
**Date:** July 5, 2026  
**Auditor:** Senior AI Research Engineer  
**Scope:** Full reverse engineering of Nova Core v0.1.0  
**Status:** No code modified. No files rewritten. Pure analysis.

---

## PHASE 1 — PROJECT OVERVIEW

### 1.1 Overall Architecture

Nova is a Rust-based "post-transformer" language model that explicitly rejects the three pillars of modern LLMs:

| Rejected Concept | Nova Replacement |
|---|---|
| **Tokens** (discrete vocabulary IDs) | **Pulses** (continuous vectors) |
| **Attention** (O(n²) pairwise) | **Field** (O(n) diffusion dynamics) |
| **Layers** (fixed depth stack) | **Cores** (adaptive depth, parallel graph) |

The architecture has **10 major modules** that together form a non-standard processing pipeline:

```
main.rs
  ├── clap CLI entry point
  └── dispatches to:

Module              Purpose                              Lines
──────────────────────────────────────────────────────────────
pulse.rs            Continuous vector "token"            143
field.rs            Global information field (O(n))      447
core.rs             Adaptive processing units            643
ssm.rs              State Space Model (Mamba-alike)      631
loom.rs             Main orchestration engine           2557
trainer.rs          Training pipeline                    1152
optimizer.rs        AdamW with finite-difference         613
knowledge.rs        Concept/relation/fact store          448
context.rs          Long context sliding window          395
model.rs            Save/load/serialize models           583
dataset.rs          HF dataset loading/filtering         1022
cuda.rs             GPU acceleration stub                188
coding.rs           Code analysis/generation             (pending)
math.rs             Math expression engine               (pending)
tools.rs            External tool invocation             (pending)
```

### 1.2 Execution Flow (High-Level)

```
CLI input string
    │
    ▼
NovaLoom::process(text)
    │
    ├── 1. Check learned_responses[hash(text)] ──► return cached answer
    ├── 2. Check partial hash matches ──► return cached answer
    ├── 3. Check if tool input ──► route to ToolEngine
    ├── 4. Check if math input ──► route to MathEngine
    ├── 5. Check if code input ──► route to CodingEngine
    │
    └── 6. Neural path (PRIMARY):
            text_to_pulses(text)    ──► Vec<NovaPulse>
            process_cores_parallel()  ──► cores transform pulses
            field.update()            ──► field diffuses info to pulses
            [iterate until convergence or max_iterations]
            apply_multi_core_semantic_consensus()
            map_pulses_to_vocab()     ──► cosine similarity nearest word
            ──► return output string
```

### 1.3 Folder Structure

```
nova_core_pro_V1/
├── Cargo.toml                    # Dependencies, features (cuda, http)
├── build.rs                      # PTX kernel compilation
├── kernels/
│   └── ssm.cu                    # CUDA SSM kernel (unused)
├── models/                       # Saved .nova model files
├── src/
│   ├── main.rs                   # CLI entry point (1139 lines)
│   ├── loom.rs                   # Orchestrator (2557 lines)
│   ├── pulse.rs                  # Continuous vectors (143 lines)
│   ├── field.rs                  # Field dynamics (447 lines)
│   ├── core.rs                   # Adaptive cores (643 lines)
│   ├── ssm.rs                    # State Space Model (631 lines)
│   ├── trainer.rs                # Training pipeline (1152 lines)
│   ├── optimizer.rs              # AdamW + finite diff (613 lines)
│   ├── knowledge.rs              # Knowledge store (448 lines)
│   ├── context.rs                # Long context (395 lines)
│   ├── model.rs                  # Serialization (583 lines)
│   ├── dataset.rs                # Data loading (1022 lines)
│   ├── cuda.rs                   # GPU stub (188 lines)
│   ├── coding.rs                 # Code engine (pending read)
│   ├── math.rs                   # Math engine (pending read)
│   ├── tools.rs                  # Tool engine (pending read)
│   └── benchmark/                # Benchmarking suite
├── convert_*.py                  # Python converters (HF → .nova)
└── *.py                          # Training/download scripts
```

---

## PHASE 2 — DATA FLOW (Complete Trace)

Trace: **"What is the capital of France?"**

### Stage 1: Input → CLI Parsing
```
Input string: "What is the capital of France?"
    │
    ▼
main.rs: Commands::Run { input: "What is the capital of France?" }
    │
    ├── nova.process("What is the capital of France?")
```

### Stage 2: Hash Check (Memoization)
```rust
// loom.rs line 1737
let input_hash = hash_text("What is the capital of France?");
// hash_text: simple byte folding: acc = acc.wrapping_mul(31).wrapping_add(b)
// Returns u64 hash
// 
// IF learned_responses contains this hash → RETURN cached answer immediately
// This is PURE MEMORIZATION. No computation.
// 
// IF partial match found via learned_inputs → RETURN cached answer
```

**If NOT memorized → proceed to neural path.**

### Stage 3: Tool/Math/Code Routing
```rust
// is_tool_input → checks for "read file", "http get", etc. → NO
// is_math_input → checks for numbers, operators, "solve", etc. → NO  
// is_code_input → checks for "fn ", "def ", "function", etc. → NO
```

### Stage 4: text_to_pulses() — "Tokenization"
```rust
// loom.rs line 138
fn text_to_pulses("What is the capital of France?") {
    text.split_whitespace()     // ["What", "is", "the", "capital", "of", "France?"]
        .enumerate()
        .map(|(pos, word)| NovaPulse::from_text(word, dim=64, pos))
}
```

For each word, `NovaPulse::from_text()`:
```rust
// pulse.rs line 57
fn from_text("France?", dim=64, pos=5) {
    let mut content = vec![0.0; 64];
    let bytes = "France?".as_bytes();  // [70, 114, 97, 110, 99, 101, 63]
    
    for (i, &b) in bytes.iter().enumerate() {
        if i < 64 {
            content[i] = (b as f32) / 255.0 * 2.0 - 1.0;  // Normalize byte to [-1, 1]
        }
    }
    content[0] += (7_f32 / 20.0).min(0.5);  // Word length signal
    
    NovaPulse {
        content,                // [0.549, -0.106, -0.239, -0.137, ...]
        semantic_content: content.clone(),
        weight: 0.466,          // word.len() / 15.0
        entropy: 0.3,           // 0.3 if word.len() >= 4
        position: 5,
        converged: false,
    }
}
```

**Critical insight:** Embeddings are byte-level deterministic encodings, NOT learned.  
`"France?"` and `"Germany?"` will have different first-5 bytes resulting in different vectors.  
`"France"` and `"France?"` will differ in the 7th position. There is NO semantic similarity encoded.

### Stage 5: Core Processing (The "Neural" Path)
```rust
// loom.rs line 1858
for iteration in 0..adaptive_max {  // adaptive_max ~2-20 based on entropy
    process_cores_parallel(&mut pulses);  // ALL cores transform pulses in parallel
    field.update(&mut pulses);            // Field diffuses info to pulses
    
    // Check convergence
    avg_entropy < convergence_threshold? (0.3) → break
    entropy_delta < 0.001? → break
    content_convergence > 0.85? → break
}
```

#### `process_cores_parallel()` calls EACH core's `process()` in parallel:

**Core 0 — Syntax** (256 memory):
```rust
fn syntax_transform(pulses, step) {
    // Applies tanh to all pulse content
    // Reduces entropy by factor 0.97 each step
    for x in pulse.content { *x = x.tanh() * factor; }
}
```

**Core 1 — Semantic** (256 memory):
```rust
fn semantic_transform(pulses, step) {
    // Amplifies values above 0.3 by 1.12x
    // Attenuates values below 0.3 by 0.95x
    // After step 2, reduces entropy by 0.85
}
```

**Core 2 — Memory** (512 memory):
```rust
fn memory_transform(pulses, step) {
    // Writes first dim of pulse content into memory array
    // Blends memory back into pulse content[0]
    // After 8 steps, does a slow blend back
    // THIS ONLY OPERATES ON content[0] — ignores 63 other dimensions
}
```

**Core 3 — Reasoning** (256 memory):
```rust
// PRIORITY 1: reasoning_transform_v2()
// Phase 1: Contradiction detection — if pulses have similarity < -0.3, attenuate both
// Phase 2: Implication propagation — strong pulses influence similar weak ones  
// Phase 3: Evidence accumulation — average all confident pulse directions
//
// FALLBACK (high entropy): differences between adjacent pulse[0] values
```

**Core 4 — Pattern** (128 memory):
```rust
fn pattern_transform(pulses, step) {
    // Detects repeating patterns at distance 3
    // If similarity > 0.7, boosts weights
}
```

Then each core applies **SSM Transform** (Mamba selective scan on pulse content).

### Stage 6: Field Update
```rust
// field.rs line 92
fn update(pulses) {
    // 1. Compute weighted average of ALL pulse content (O(n))
    // 2. Update field state with momentum: state += momentum * 0.9 + diff * lr
    // 3. Diffuse field state back to pulses: pulse = pulse * (1-diff) + state * diff
    // Reduces all pulse entropy by 0.98
}
```

### Stage 7: Multi-Core Semantic Consensus
```rust
fn apply_multi_core_semantic_consensus(pulses) {
    // Average all cores' internal_state weighted by gate
    // Blend 20% of this average into the LAST pulse
}
```

### Stage 8: map_pulses_to_vocab() — Output Generation
```rust
fn map_pulses_to_vocab(pulses) {
    // For each pulse:
    //   1. Compute cosine similarity against ALL vocabulary entries
    //   2. Pick the word with highest cosine similarity
    //   3. If best similarity < 0.1, pick RANDOM word from all_words
    //   4. Return joined string
}
```

### Stage 9: Return Output
```
"What is the capital of France?"
    ──► neural path runs 5-15 iterations
    ──► pulses transformed by 5 cores + field
    ──► cosine similarity picks closest vocabulary words
    ──► OUTPUT: "the capital the of the france" (garbage)
```

---

## PHASE 3 — TRAINING PIPELINE

### 3.1 Training Architecture

Nova has **THREE training modes**, only one of which has any resemblance to actual training:

| Mode | File | Method | Real Learning? |
|---|---|---|---|
| `train()` | trainer.rs:527 | Epoch-based, hash + n-gram | NO (memorization) |
| `train_neural()` | trainer.rs:578 | Finite-difference gradients + AdamW | PARTIAL |
| `train_one_pass()` | trainer.rs:876 | Hash associations only | NO |
| `train_one_pass_ultra()` | trainer.rs:984 | Hash associations, no cores | NO |

### 3.2 Dataset Loading

```
Dataset file (CSV/JSON/JSONL/TXT/HF)
    │
    ▼
NovaDataset::load_all()
    │   - Auto-detect columns (input/target patterns)
    │   - Parse rows into Vec<TrainingExample { input, target }>
    │
    ▼
train_val_split(0.1) → (train_data, val_data)
```

### 3.3 Vocabulary Creation

```rust
fn init_vocabulary(examples) {
    // Collect ALL unique words from inputs AND targets
    // For each word, create a DETERMINISTIC RANDOM embedding:
    let seed = word.bytes().fold(0u64, |acc, b| acc * 31 + b);
    let mut rng = StdRng::from_seed(seed_bytes);
    for _ in 0..dim { vec.push(rng.gen_range(-0.3..0.3)); }
    normalize(vec);  // Unit vector
}
```

**Critical:** Vocabulary embeddings are:
- **Deterministic** (same word → same embedding always)
- **Random** (no semantic structure — "cat" and "dog" are as similar as "cat" and "quantum")
- **Fixed** (never updated during training)
- **NOT learned** (no gradient flows into vocabulary)
- **Hash-based** (the "learning" is memorizing which hash → which word)

### 3.4 Neural Training (train_neural)

```
1. FORWARD PASS (COSTLY):
   - text_to_pulses(input)         → Vec<NovaPulse>
   - process_cores_parallel()      → ALL cores transform pulses
   - field.update()                → Field dynamics
   - [repeat until convergence or max_iter]
   
2. LOSS COMPUTATION:
   - For each target word, look up its embedding in vocab_forward
   - Compute MSE between pulse[i].content and target_word_embedding
   - Average MSE = loss
   
3. GRADIENT COMPUTATION (THE WEAKNESS):
   - Uses FINITE DIFFERENCES (NOT backpropagation):
   
   for each parameter p:
       original = p
       p += epsilon                    # eps = 0.001
       loss_up = compute_loss(model)   # FULL forward pass again!
       p = original - epsilon
       loss_down = compute_loss(model) # ANOTHER full forward pass!
       grad = (loss_up - loss_down) / (2 * epsilon)
       p = original
   
   SCALE: 2 forward passes per parameter
   Parameters: core_memory (256×5), internal_state (64×5), gate (5), 
               SSM params (64×16×3 per core)
   ≈ 15,700 parameters × 2 = 31,400 forward passes per example!
   
4. ADAMW UPDATE:
   - Accumulate gradients in GradientBuffer
   - Clip by global norm (threshold: 1.0)
   - Apply AdamW: m = β₁·m + (1-β₁)·g
                   v = β₂·v + (1-β₂)·g²
                   θ = θ - lr·(m/(√v+ε) + wd·θ)
```

### 3.5 Which Weights Learn vs. Don't

| Parameter | Learns? | How? | Notes |
|---|---|---|---|
| Core memory (Vec<f32>) | ✅ Finite-diff | 2 FP per param | Only first 64 of 256 dims |
| Core internal_state | ✅ Finite-diff | 2 FP per param | Only first 16 of 64 dims |
| Core gate (f32) | ✅ Finite-diff | 2 FP | Clamped to [0.1, 0.95] |
| Field state (Vec<f32>) | ✅ Finite-diff | Included | Direct AdamW update |
| Field momentum (Vec<f32>) | ✅ Finite-diff | Included | Direct AdamW update |
| SSM A_log | ✅ Finite-diff | 2 FP per param | Only first 32 of 1024 |
| SSM B | ✅ Finite-diff | 2 FP per param | Only first 32 of 1024 |
| SSM C | ❌ | Not computed | C gradients NOT in finite-diff code |
| SSM delta | ❌ | Not computed | D_biass not included |
| SSM delta_bias | ❌ | Not computed | Not included |
| Vocabulary | ❌ NEVER | N/A | Hash-based random, FIXED |
| learned_responses | ✅ Direct store | hash→string | This is actually memorization |
| ngram_patterns | ✅ Direct store | count frequency | This is n-gram statistics |

### 3.6 Where Gradients Flow

```
Gradient PATH (finite-difference):
    perturb parameter p
    ▼
    text_to_pulses(input) ─► Vec<NovaPulse>
    ▼
    process_cores_parallel() ─► modifies pulse content
    ▼
    field.update() ─► modifies pulse content
    ▼
    MSE(pulse.content, target_embedding) ─► loss (scalar)
    ▼
    (loss_up - loss_down) / 2ε ─► gradient estimate for p
    ▼
    AdamW update on p
```

### 3.7 Where Gradients STOP

```
Gradients STOP at:
1. Vocabulary embeddings — NEVER computed, NEVER updated
2. Text_to_pulses encoding — deterministic byte mapping, no parameters
3. SSM C, delta, delta_bias — omitted from finite-diff computation
4. Core memory beyond index 64 — truncated in finite-diff loop
5. Internal_state beyond index 16 — truncated in finite-diff loop
6. SSM parameters beyond index 32 — truncated
```

### 3.8 Does Semantic Learning Actually Occur?

**No.** Here's why:

1. **Finite difference is computationally intractable** for anything beyond toy scale. For 15,700 parameters × 2 forward passes each = 31,400 forward passes per example. At 0.1ms per forward pass, that's ~3 seconds per example. With batch_size=16, that's 48 seconds per batch. With accumulation_steps=4, that's 192 seconds per optimizer step.

2. **Gradient estimates are extremely noisy** — finite difference with ε=0.001 in a high-dimensional space with nonlinearities (tanh, sigmoid, softplus) produces gradients with near-zero signal-to-noise ratio.

3. **Vocabulary is fixed** — even if the model "learns" to produce better pulse vectors, the mapping from pulses to words is cosine similarity against random vectors. The words "France" and "paris" are no more similar than "France" and "the".

4. **The ACTUAL learning mechanism** is `learned_responses[hash(input)] = output`. This is a hash table lookup, not semantic learning.

### 3.9 "Training" = Hash-Based Memorization

All training modes ultimately do this:
```rust
// This is what actually makes Nova "learn"
let input_hash = hash_text("what color is the sky");
model.learned_responses.insert(input_hash, "blue");
model.learned_inputs.insert(input_hash, "what color is the sky");

// And this generates outputs:
// loom.rs line 1738
if let Some(response) = model.learned_responses.get(&input_hash) {
    return response.clone();  // PURE MEMORIZATION
}
```

---

## PHASE 4 — INFERENCE PIPELINE

### 4.1 Which Modules Run (in order)

```
NovaLoom::process(text)     [ALWAYS]
├── hash_text(text)         [ALWAYS - fast]
├── learned_responses check [ALWAYS - hash lookup]
├── partial hash match      [ALWAYS - iterates all learned inputs]
│
├── is_tool_input()         [IF tool_enabled]
│   └── handle_tool_request() [IF tool detected]
├── is_math_input()         [IF math_enabled]
│   └── solve_math_response() [IF math detected]
├── is_code_input()         [IF coding_enabled]
│   ├── debug_code_response() [IF debug detected]
│   └── generate_code_response() [IF code gen detected]
│
├── text_to_pulses()        [ALWAYS - neural path]
├── process_cores_parallel() [ALWAYS - all 5 cores]
│   ├── syntax_transform    [ALWAYS]
│   ├── semantic_transform  [ALWAYS] + semantic_refine [ALWAYS]
│   ├── memory_transform    [ALWAYS]
│   ├── reasoning_transform_v2 [ALWAYS]
│   ├── pattern_transform   [ALWAYS]
│   ├── ssm_transform (each core) [ALWAYS - unless use_ssm=false]
│   ├── multi-core comm bus [ALWAYS - broadcast + blend]
│   └── knowledge_transform [IF knowledge has concepts]
├── field.update()          [ALWAYS]
│   └── SSM-enhanced [IF use_ssm]
├── apply_multi_core_semantic_consensus() [ALWAYS]
├── map_pulses_to_vocab()   [IF vocabulary not empty]
│   └── cosine similarity loop [ALWAYS - O(vocab_size × dim)]
└── return String
```

### 4.2 Which Modules Are Skipped

- **CUDA kernels** — `is_kernels_ready()` returns `false` always. CPU fallback active.
- **LongContextManager** — only triggers if `pulses.len() > 2048` (very long inputs)
- **SlidingWindowSSM** — only for sequences exceeding max_seq_length
- **Hash + n-gram fallbacks** — commented out in PRIORITY 1 rewrite

### 4.3 Where Output Actually Comes From

The output comes from **one of four paths**, checked in order:

**Path 1: Learned Response (hash match)** → Most reliable output
```rust
// line 1738
if let Some(response) = learned_responses.get(&hash_text(text)) {
    if is_valid_response(response) { return response; }
}
```
_This is pure memorization. If trained on "what color is the sky" → "blue", this works._

**Path 2: Partial Hash Match** → Somewhat reliable
```rust
// line 1745
for (hash, response) in learned_responses {
    if text.contains(input_text) || input_text.contains(text) {
        return response;
    }
}
```

**Path 3: Specialized Engine** → Deterministic but keyword-gated
```rust
// Tool, Math, or Code engine
// Only triggers if specific keywords detected ("calculate", "fn ", etc.)
```

**Path 4: Neural Path** → Usually garbage
```rust
// line 1900
if !vocabulary.is_empty() {
    let neural_text = map_pulses_to_vocab(&pulses);
    // Cosine similarity against random word embeddings
    // Returns "the capital the of france" style output
}
```

### 4.4 How Words Are Selected

```
Pulse content vectors ([-1, 1] floats)
    │
    ▼
For each pulse, iterate ALL vocabulary entries (simplified):
    best_word = "the"
    best_sim = -1.0
    for each (word, vec) in vocabulary:
        sim = cosine_similarity(pulse.content, vec)
        if sim > best_sim:
            best_word = word
    
    if best_sim > 0.1:
        output.push(best_word)       // Best cosine similarity match
    else:
        output.push(random_word)     // Pseudo-random from all_words
```

**This is nearest-neighbor search against random vectors.** It has no notion of semantic similarity because the vocabulary was created with random seeds.

### 4.5 Text Generation (`generate_text`)

For longer outputs in SmartChat mode:

```
generate_text(prompt, max_words=30)
    │
    ├── Loop max_words times:
    │   ├── Loop detection (ABAB, AAA, ABCABC patterns)
    │   ├── IF pulse prediction: predict_next_word_via_pulses_excluding()
    │   │   ├── text_to_pulses(context)
    │   │   ├── process_cores_parallel()
    │   │   ├── field.update()
    │   │   ├── apply_multi_core_semantic_consensus()
    │   │   ├── find_closest_word_excluding() → cosine similarity
    │   │   └── Return best word (even if similarity < 0.2!)
    │   ├── ELSE n-gram backoff (legacy)
    │   └── Diverse word fallback
    │
    └── Join generated words (strip prompt)
```

---

## PHASE 5 — MEMORY SYSTEM

### 5.1 Hash Memory (learned_responses)

```rust
// Type: HashMap<u64, String>
// Key: hash_text(input)
// Value: output text

// Operation:
fn process(text) {
    let hash = hash_text(text);   // Simple byte-folding hash
    if learned_responses.contains(hash) {
        return learned_responses[hash];  // O(1) lookup
    }
}
```

- **Size:** Unbounded (grows with training data)
- **Persistence:** Saved/loaded in .nova files
- **Granularity:** Exact input → exact output mapping
- **Limitation:** No generalization. "What color is the sky" → "blue" is memorized, but "what color is the sky on a cloudy day" produces different hash → not found.

### 5.2 Knowledge Store

```rust
pub struct KnowledgeStore {
    concepts: HashMap<String, Concept>,           // word → embedding
    relations: HashMap<String, Vec<(rel, obj, conf)>>,  // subject → relations
    reverse_relations: HashMap<String, Vec<(rel, subj, conf)>>,
    facts: HashMap<u64, Fact>,
    facts_by_category: HashMap<String, Vec<u64>>,
}
```

- Concepts are created from words > 3 chars during training
- Relations are "followed_by" (adjacent words) and "predicts" (last input → first target)
- Facts are the full input→target pair stored as text
- Knowledge is **only used** to augment pulse content via `augment_pulse_with_knowledge()`:
  - Find closest concept by cosine similarity
  - Blend concept embedding into pulse content
- **Critical:** Concept embeddings are deterministic byte-mapped, NOT learned

### 5.3 Vocabulary

```rust
// Type: HashMap<String, Vec<f32>>
// Created: During training init_vocabulary()
// Size: Number of unique words in training data

// Each embedding:
let seed = hash_bytes(word);
let mut rng = SeededRng::from_seed(seed);
for _ in 0..DIM { vec.push(rng.gen_range(-0.3..0.3)); }
normalize_to_unit(vec);
```

- **Random** but deterministic
- **Fixed** — never updated during training
- **No semantic structure** — Euclidean distance between any two random unit vectors is approximately √2

### 5.4 Field Memory (NovaField)

```rust
state: Vec<f32>,        // dim=64, global information
momentum: Vec<f32>,     // momentum for smoother updates
convergence_history: Vec<Vec<f32>>,  // last 5 states for convergence detection
ssm: Option<StateSpace>,  // SSM-enhanced field (optional)
```

- Acts as a global **reservoir** that accumulates information from all pulses
- Updates: `state += momentum * 0.9 + (field_avg - state) * lr`
- Influences pulses: `pulse = pulse * (1-diffusion) + state * diffusion`
- Diffusion decays over time: `diffusion * 0.95^update_count`

### 5.5 Core Memory

```rust
// Each core has:
memory: Vec<f32>,      // Size varies: syntax=256, semantic=256, memory=512, reasoning=256, pattern=128
internal_state: Vec<f32>,  // dim=64
ssm: StateSpace,          // d_inner×d_state hidden state
prev_pulse_content: Vec<Vec<f32>>,  // For convergence detection
```

- `memory` stores only `pulse.content[0]` — the first dimension
- `internal_state` averages SSM hidden state across dimensions
- `prev_pulse_content` tracks pulse changes between iterations

### 5.6 N-Gram Patterns

```rust
ngram_patterns: HashMap<u64, Vec<(String, f32)>>,
// Key: hash of n-word context
// Value: list of (next_word, confidence) predictions

// Order: 3 (default)
// Training: learn_ngrams extracts order-2 and order-3 patterns
```

- **Order 2:** "the cat" → ["sat": 1.0, "ran": 2.0]
- **Order 3:** "the cat sat" → ["on": 1.0]
- Used as **fallback** generation when vocabulary exists but ngrams > 100K
- Confidence is incremented each time pattern is observed (max 10.0)

### 5.7 Conversation Memory

There is **no explicit conversation memory** or context window in the traditional sense. The LongContextManager is designed for very long single inputs (>2048 tokens), not for multi-turn conversation. Each `process()` call is stateless — the field and cores reset between calls unless explicitly trained.

---

## PHASE 6 — EMBEDDINGS

### 6.1 How Embeddings Are Created

There are **THREE separate embedding mechanisms** in Nova:

#### 6.1.1 Pulse Content (from text)
```rust
// pulse.rs:57
fn from_text(word, dim, pos) -> NovaPulse {
    let bytes = word.as_bytes();
    for i in 0..min(bytes.len(), dim) {
        content[i] = (bytes[i] / 255.0) * 2.0 - 1.0  // byte → [-1, 1]
    }
    content[0] += min(word.len() / 20.0, 0.5)  // length signal
}
```
- **Deterministic** — same word → same vector
- **Not random** — based on ASCII byte values
- **Semantically meaningless** — "cat" (99, 97, 116) and "car" (99, 97, 114) differ in only the last dimension

#### 6.1.2 Vocabulary Embeddings
```rust
// trainer.rs:152
let seed = hash(word.bytes());
let mut rng = SeededRng::from_seed(seed);
vec![rng.gen_range(-0.3..0.3); dim]
normalize(vec);
```
- **Deterministic** — seeded random based on word bytes  
- **Random** — no relationship between semantically similar words
- **Fixed** — never updated

#### 6.1.3 Concept Embeddings (Knowledge)
```rust
// knowledge.rs:287
let bytes = word.as_bytes();
for j in 0..min(bytes.len(), dim) {
    embedding[j] = (bytes[j] / 255.0) * 2.0 - 1.0
}
embedding[1] += sin(position / 100.0)
normalize(embedding)
```
- Same byte-mapping as pulses
- Different from vocabulary embeddings (different normalization)
- Also semantically meaningless

### 6.2 Are They Random? Deterministic? Learned? Fixed?

| Embedding Type | Random? | Deterministic? | Learned? | Fixed? |
|---|---|---|---|---|
| Pulse content | No | Yes (byte mapping) | No | Computed per call |
| Vocabulary | Pseudo-random (seeded) | Yes (same seed = same vec) | **No** | **Yes, FIXED FOREVER** |
| Concept (Knowledge) | No | Yes (byte mapping) | No | Updated via blending |
| Semantic content | No | Yes (derived from pulse) | No | Updated by transform |

### 6.3 How Similar Words are Represented

**They are NOT similar.** Cosine similarity between vocabulary embeddings for related words:

```
cos("france", "paris") ≈ random value in [-0.3, 0.3]
cos("france", "quantum") ≈ random value in [-0.3, 0.3]
cos("cat", "cat") = 1.0 (deterministic, same seed)
cos("cat", "dog") ≈ random value
```

The pulse content (byte-mapped) does have structure: words sharing the same bytes show similarity. But this encodes ASCII byte patterns, not semantics. "France" and "French" share the first 3 bytes "Fre" → their pulse vectors are more similar in the first 3 dimensions. But cosine similarity over all 64 dimensions is still dominated by random values.

---

## PHASE 7 — REASONING

### 7.1 Does Reasoning Actually Happen?

**No, reasoning does not happen in any meaningful sense.**

The system performs three operations that mimic reasoning:

#### 7.1.1 Pulse Transform as "Reasoning"

The reasoning core (`reasoning_transform_v2`) performs:
- **Contradiction detection:** If two pulses have cosine similarity < -0.3, attenuate both values
- **Implication propagation:** Strong pulses (high weight) influence similar weak pulses
- **Evidence accumulation:** Average all confident pulse directions

However, this operates on **byte-mapped pulse vectors** — the "contradictions" and "agreements" are in byte-value space, not semantic space. Two pulses representing "hot" and "cold" may or may not have negative cosine similarity in byte space — this is entirely coincidental.

#### 7.1.2 What Actually Happens

The chain is:
```
Input → deterministic byte vectors → random linear transforms → 
cosine similarity against random word vectors → output

This is NOT reasoning. It is nearest-neighbor search in a random projection space.
```

#### 7.1.3 Math and Tool "Reasoning"

When keyword-gated paths trigger:

```
Math: Keyword match ("solve", "calculate") → extract numbers → evaluate expression → format output
Tool: Keyword match ("read file", "http get") → extract path/url → invoke tool → format output
Code: Keyword match ("fn ", "function") → route to coding engine → format output
```

These are **rule-based routing**, not neural reasoning. They work because they're hardcoded logic, not learned behavior.

### 7.2 Pulse Propagation vs. Reasoning

Each iteration of the pulse propagation does:
1. **Cores transform pulses** — nonlinear scalar operations (tanh, amplify, blend memory)
2. **SSM transform** — state-space pass (linear ODE discretization)
3. **Field diffusion** — weighted average pulse → field → back to pulses
4. **Cross-core communication** — blend core states into pulses

This is most similar to **iterative message passing** or **diffusion on a complete graph** — but the messages have no semantic content because the initial embeddings are byte-mapped garbage.

---

## PHASE 8 — OUTPUT GENERATION

### 8.1 Why "What color is the sky" → "blue"

If the training data includes `("what color is the sky", "blue")`:

```
hash("what color is the sky") = 12345u64
learned_responses[12345] = "blue"

// On query:
hash("what color is the sky") = 12345u64
return learned_responses[12345] → "blue" ✓
```

**This works because of pure hash-table memorization, not understanding.**

### 8.2 Why "What is capital of France" → garbage

Scenarios:

**Scenario A: NOT in training data**
```
hash("what is the capital of france") = 67890u64
learned_responses[67890] = NOT FOUND

// Falls through to neural path:
pulses = text_to_pulses("what is the capital of france")
// process through cores + field (5 iterations)
// map_pulses_to_vocab:
//   pulse[0] → cosine similarity vs random vocab → "the" or random word
//   pulse[1] → cosine similarity vs random vocab → "capital" (might match by byte chance)
//   pulse[2] → some random word
// OUTPUT: "the capital the of france something" ✗
```

**Scenario B: In training data (trained version)**
```
hash("what is the capital of france") = 67890u64
learned_responses[67890] = "paris"

// On query → "paris" ✓
// But only because it's EXACTLY memorized
```

### 8.3 Why "What is the capital of France?" (with question mark) fails

```
hash("What is the capital of France?") != hash("what is the capital of france")
// Case sensitivity + punctuation → different hash → NOT FOUND
// Falls through to neural path → garbage
```

### 8.4 Complete Decision Chain for Unknown Input

```
User: "Explain quantum entanglement"

1. Hash check: hash("Explain quantum entanglement") → not in learned_responses → continue
2. Partial match: no stored input contains "quantum entanglement" → continue
3. Tool check: doesn't contain tool keywords → skip
4. Math check: doesn't contain math keywords → skip
5. Code check: doesn't contain code keywords → skip
6. Neural path:
   a. text_to_pulses → [pulse("Explain"), pulse("quantum"), pulse("entanglement")]
   b. Each pulse has 64 dimensions, first few filled with byte values
   c. 5-15 iterations through cores + field
   d. Cores apply tanh, amplify, memory, reasoning, pattern transforms
   e. SSM runs selective scan on each pulse
   f. Field averages pulses and diffuses back
   g. Multi-core consensus blends core states
   h. map_pulses_to_vocab: cosine similarity against ALL random word vectors
   i. Best matches: "the", "quantum" (might match by byte), "field" (random chance)
7. OUTPUT: "the quantum field the of the" ← plausible-sounding but meaningless
```

---

## PHASE 9 — BENCHMARK (vs. Modern LLMs)

Component-by-component comparison:

| Component | GPT-4 | DeepSeek | Llama 3 | Gemini | **Nova** |
|---|---|---|---|---|---|
| **Tokenizer** | BPE (100K vocab) | BPE (128K) | BPE (128K) | SentencePiece (256K) | **None. Byte → float mapping** |
| **Embeddings** | Learned 12800d | Learned 7168d | Learned 8192d | Learned | **Seeded random. Fixed.** |
| **Training** | BP + RLHF + SFT | BP + MoE + RL | BP + SFT + RLHF | BP + multimodal | **Hash memorization + finite-diff (≈NOT training)** |
| **Memory** | Transformer context (128K) | 128K context | 128K context | 2M context | **Hash table + n-gram statistics** |
| **Reasoning** | Chain-of-thought, multi-step | Multi-step, tool use | Multi-step | Multi-modal reasoning | **Keyword-gated rule dispatch + pulse diffusion (≈NO reasoning)** |
| **Generation** | Autoregressive, temperature | Autoregressive | Autoregressive | Autoregressive | **Nearest neighbor + cosine similarity** |
| **Learning** | Gradient descent + data | Gradient descent | Gradient descent | Gradient descent | **Hash insert + finite difference (≈NO learning)** |
| **Inference** | Efficient (TensorRT, vLLM) | Efficient | Efficient (llama.cpp) | Efficient | **CPU-only. 5 cores × iterations × cosine sim over all vocab** |
| **Optimization** | AdamW, LR schedule | AdamW, MoE routing | AdamW, QLoRA | AdamW, PA | **AdamW + finite difference (≈NOT used)** |
| **Context** | 128K tokens | 128K | 128K | 2M | **2048 "pulses" (≈50 words)** |
| **Knowledge** | Trained on internet-scale | Trained on internet-scale | Trained on internet-scale | Trained on internet-scale | **100-1000 hash associations from small training set** |
| **Coding** | Expert-level | Expert-level | Strong | Strong | **Keyword-gated: code generation is a hardcoded template** |
| **Tool use** | Function calling | Tool use built-in | Tool use via JSON | Extension API | **Keyword-gated: file read/write/http/search templates** |

### 9.1 Nova's Unique Properties (for good or bad)

| Property | Nova | Traditional LLM |
|---|---|---|
| Complexity per token | **O(n)** (field) | O(n²) (attention) |
| Complexity per inference | **O(cores × dim × iterations)** | O(1) (single forward pass) |
| Memory per token | **O(dim)** = 64 floats | O(dim) = 4096-12800 floats |
| Training speed | Fast (hash insert) | Slow (BP over billions of params) |
| Generalization | **None** (hash lookup only) | Strong (learned representations) |
| Parameter count | **~100K** (mostly SSM) | **7B-1.8T** |
| Knowledge capacity | **~1000 hash entries** | **Trillions of tokens** |

---

## PHASE 10 — BOTTLENECKS (Ranked)

### #1. VOCABULARY IS RANDOM AND FIXED — Highest Impact

**Problem:** Vocabulary embeddings are created with seeded random number generators and are NEVER updated during training. The entire "neural" output path relies on cosine similarity between pulse vectors and these random vectors. Since the vectors have no semantic structure, the output is effectively random word selection.

**Why:** `trainer.rs:152` creates vocab with `StdRng::from_seed(seed).gen_range(-0.3..0.3)` and never trains them. The cosine similarity between any two random unit vectors in 64D is ~0 with std ~0.125.

**Impact:** The neural path produces garbage for any input not exactly memorized.

### #2. FINITE DIFFERENCE GRADIENTS — Critical

**Problem:** Instead of backpropagation (which requires automatic differentiation or at least differentiable operations), Nova uses finite-difference gradient approximation:

```
grad ≈ (loss(x+ε) - loss(x-ε)) / 2ε
```

This requires **2 complete forward passes per parameter**. With ~15,700 parameters and ~100μs per forward pass, that's ~3 seconds per training example. For practical training (1000 examples), that's ~50 minutes — and each gradient is extremely noisy.

**Why:** Nova's operations (tanh, clamp, if-statements, hash lookups) are not differentiable. The entire pipeline was built without autograd in mind.

**Impact:** Training is impractically slow AND produces unusably noisy gradients. The model learns nothing from the gradient path.

### #3. BYTE-LEVEL PULSE ENCODING — Critical

**Problem:** `NovaPulse::from_text()` maps ASCII byte values directly to float vectors. This means:
- "cat" → [0.776, 0.525, 0.388, ...] (bytes 99, 97, 116)
- "car" → [0.776, 0.525, 0.478, ...] (bytes 99, 97, 114)
- "dog" → [0.580, 0.792, 0.710, ...] (bytes 100, 111, 103)

"cat" and "car" have cosine similarity ~0.95 (share first 2 bytes). "cat" and "dog" have similarity ~0.3. This is ASCII-level similarity, NOT semantic. Transformers learned embeddings that make "cat" closer to "kitten" than "car".

**Impact:** The core transforms operate on byte-level patterns, not meaning. "Reasoning" over byte values produces byte-level outputs.

### #4. HASH TABLE IS THE ONLY WORKING MEMORY — High Impact

**Problem:** The `learned_responses` HashMap is the only mechanism that produces correct outputs. It's a simple hash table:
- Input `"what color is the sky"` → hash → stored output `"blue"`
- Input `"what color is the sky today"` → different hash → NOT FOUND

There is NO generalization. Every possible input must be exactly memorized.

**Impact:** The model cannot answer ANY question it hasn't seen before. Zero-shot generalization = 0.

### #5. NO TOKENIZATION — High Impact

**Problem:** Traditional LLMs use subword tokenization (BPE, SentencePiece) that handles:
- Rare words (split into known subwords)
- Unknown words (split into subwords)
- Consistent vocabulary (OOV is minimal)

Nova's word-level splitting handles none of this:
- "France?" is a different "word" from "France"
- "running" and "run" are completely different
- Unknown words get byte mappings but no semantic handle

**Impact:** Punctuation, capitalization, and morphology all create different inputs. P50K different surface forms of 10K base words.

### #6. COSINE SIMILARITY IS O(∣VOCAB∣ × ∣PULSES∣ × DIM) — Medium Impact

**Problem:** Every inference step computes cosine similarity between EVERY pulse and EVERY vocabulary word. With V=1000 words, P=10 pulses, D=64: that's 640,000 multiply-adds per inference. For V=10,000: 6.4M operations. No indexing, no partitioning, no approximate nearest neighbor.

**Impact:** Scales linearly with vocabulary, which limits vocabulary size.

### #7. SSM PARAMETERS ARE MOSTLY UNTRAINED — Medium Impact

**Problem:** The SSM (State Space Model) has parameters (A_log, B, C, delta, delta_bias, D) that are:
- Initialized with small random/heuristic values  
- Only C, delta, and delta_bias gradients are NOT computed in finite-diff
- Only first 32 of 1024 parameters per matrix are perturbed
- Never exposed to real gradient descent

**Impact:** The SSM is running with essentially random parameters, making it a random linear transform rather than a learned selective scan.

### #8. ADAPTIVE ITERATION IS COSTLY FOR NO BENEFIT — Medium Impact

**Problem:** Each inference runs 2-20 iterations of the full core+field pipeline. Each iteration involves:
- 5 cores running in parallel (each with SSM transform)
- Field update (average all pulses)
- Cross-core communication (broadcast + blend)
- Knowledge augmentation (concept lookup + blend)

For 10 iterations: that's 50 core processes + 10 field updates + 10 comm cycles. With ~100μs per core process + ~50μs for field + ~20μs for comm = ~1.7ms per inference. For comparison, a 7B LLM generates ~30 tokens/ms on GPU.

**Impact:** Nova is slower than it needs to be for outputs that are still garbage.

### #9. CONVERGENCE IS MEASURED BY BYTE-STABILITY, NOT SEMANTIC — Medium Impact

**Problem:** Convergence detection checks if pulse content stops changing between iterations. But because pulses encode byte values, "convergence" means the byte values have stabilized — not that a semantic understanding has been reached.

**Impact:** Early exit happens when pulses stop changing, not when they're "right." This could happen in 2 iterations (bad) or never converge (max iterations).

### #10. NO GRADIENT FLOW TO VOCABULARY — Medium Impact

**Problem:** Even if the finite-difference path worked perfectly, the vocabulary embeddings are never updated. The model would need to learn to produce pulse vectors that exactly match the random initialization of each word embedding. This is like learning to output specific random numbers — the target is meaningless.

**Impact:** The loss function (MSE between pulse vectors and random vocabulary embeddings) is fundamentally ill-posed. Lowering this loss doesn't improve semantic output quality.

---

## PHASE 11 — SUMMARY

### What Nova Actually IS

Nova is a **hash-based memorization system** wrapped in an elegant but non-functional neural architecture. The correct output comes from exactly one place: `learned_responses[hash(input)]`. This is a hash table.

The pulse/field/core/SSM pipeline is **executed but does not contribute to correct outputs**. It produces plausible-sounding word sequences by:
1. Taking byte-level input encodings
2. Applying random-ish transforms
3. Finding the nearest random word vector
4. Returning the result

### What Makes It Look Like It Works

1. **Training on simple QA pairs** → hash table memorizes them → inference from hash → correct output
2. **Keyword-gated engines** → "calculate", "prime", "fn " trigger hardcoded paths
3. **Conversational overrides** (now removed) provided canned responses
4. **The vocabulary random model** by chance sometimes hits the right word

### The Fundamental Issue

The architecture has an **irresolvable conceptual flaw at its core**: the vocabulary embeddings are random and fixed. Even if gradient descent were perfect, the loss function (MSE between pulse vector and random embedding) is not aligned with output quality. The model could achieve zero loss by learning to output specific random numbers for each input — but this doesn't correspond to any semantic understanding.

For the neural path to work, the system needs AT MINIMUM:
1. **Learned embeddings** that encode semantic relationships
2. **Differentiable operations** for backpropagation
3. **Gradient flow into ALL parameters** including vocabulary
4. **A loss function that measures answer correctness**, not vector similarity

### What Would Need to Change (Directionally)

1. Replace byte-mapped pulse encoding with learned embeddings
2. Make ALL operations differentiable for proper backprop
3. Train vocabulary jointly with the rest of the model
4. Replace finite-difference with actual gradient computation
5. Use a proper autoregressive loss (next-word prediction)
6. Scale parameters to at least millions
7. Train on at least millions of tokens
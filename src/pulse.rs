//! Nova Pulse - Continuous meaning units with KV-cache for generation
//!
//! Pulses replace discrete tokens with continuous vector representations.
//! Each pulse carries content, semantic meaning, weight, and entropy.
//! The KV-cache stores SSM hidden states for O(1) per-step generation.

use crate::embedding::EMBED_DIM;
use rand::Rng;

/// A single pulse - the fundamental unit of meaning in Nova.
/// Replaces discrete tokens with continuous vector representations.
#[derive(Debug, Clone)]
pub struct NovaPulse {
    /// Main content vector (the pulse's meaning)
    pub content: Vec<f32>,
    /// Semantic content vector (refined meaning)
    pub semantic_content: Vec<f32>,
    /// Importance weight (0.0 = irrelevant, 1.0 = crucial)
    pub weight: f32,
    /// Entropy/uncertainty (0.0 = certain, 1.0 = chaotic)
    pub entropy: f32,
    /// Position in sequence
    pub position: usize,
    /// Parent pulse index (for hierarchical structure)
    pub parent: Option<usize>,
    /// Whether this pulse has converged (stable state)
    pub converged: bool,
}

impl NovaPulse {
    /// Create a new pulse from a content vector
    pub fn new(content: Vec<f32>, position: usize) -> Self {
        let dim = content.len();
        NovaPulse {
            semantic_content: content.clone(),
            content,
            weight: 1.0,
            entropy: 0.5,
            position,
            parent: None,
            converged: false,
        }
    }

    /// Create a pulse from token embedding
    pub fn from_embedding(embedding: &[f32], position: usize) -> Self {
        Self::new(embedding.to_vec(), position)
    }

    /// Create a zero-initialized pulse
    pub fn zeros(dim: usize, position: usize) -> Self {
        Self::new(vec![0.0; dim], position)
    }

    /// Cosine similarity between two pulses
    pub fn similarity(&self, other: &NovaPulse) -> f32 {
        let n = self.content.len().min(other.content.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..n {
            dot += self.content[i] * other.content[i];
            norm_a += self.content[i] * self.content[i];
            norm_b += other.content[i] * other.content[i];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
        (dot / denom).clamp(-1.0, 1.0)
    }

    /// Reduce entropy (make pulse more certain)
    pub fn reduce_entropy(&mut self, factor: f32) {
        self.entropy *= factor;
        self.entropy = self.entropy.clamp(0.0, 1.0);
    }

    /// Increase entropy (make pulse more uncertain)
    pub fn increase_entropy(&mut self, factor: f32) {
        self.entropy = self.entropy * (1.0 + factor);
        self.entropy = self.entropy.clamp(0.0, 1.0);
    }

    /// Get the dimension of this pulse's content
    pub fn dim(&self) -> usize {
        self.content.len()
    }
}

// ============================================================================
// KV-Cache for Autoregressive Generation
// ============================================================================

/// KV-cache stores per-layer SSM hidden states for O(1) per-step generation.
/// Instead of reprocessing the entire sequence at each step,
/// we cache the SSM's hidden state (h) and continue from there.
pub struct KvCache {
    /// Cached SSM hidden states per core per layer
    /// Shape: [num_cores][num_layers][d_inner * d_state]
    pub ssm_states: Vec<Vec<Vec<f32>>>,
    /// Number of cores
    pub num_cores: usize,
    /// Number of SSM layers per core
    pub num_layers: usize,
    /// SSM inner dimension
    pub d_inner: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Current sequence length (number of tokens generated so far)
    pub seq_len: usize,
}

impl KvCache {
    /// Create a new KV-cache
    pub fn new(num_cores: usize, num_layers: usize, d_inner: usize, d_state: usize) -> Self {
        let ssm_states = vec![
            vec![
                vec![0.0; d_inner * d_state];
                num_layers
            ];
            num_cores
        ];

        KvCache {
            ssm_states,
            num_cores,
            num_layers,
            d_inner,
            d_state,
            seq_len: 0,
        }
    }

    /// Store SSM hidden states for a core's layer
    pub fn store_layer_state(&mut self, core_idx: usize, layer_idx: usize, state: &[f32]) {
        if core_idx < self.num_cores && layer_idx < self.num_layers {
            let n = self.ssm_states[core_idx][layer_idx].len().min(state.len());
            self.ssm_states[core_idx][layer_idx][..n].copy_from_slice(&state[..n]);
        }
    }

    /// Load SSM hidden states for a core's layer
    pub fn load_layer_state(&self, core_idx: usize, layer_idx: usize) -> &[f32] {
        if core_idx < self.num_cores && layer_idx < self.num_layers {
            &self.ssm_states[core_idx][layer_idx]
        } else {
            &[] // Empty slice if out of bounds
        }
    }

    /// Load SSM hidden states mutably for a core's layer
    pub fn load_layer_state_mut(&mut self, core_idx: usize, layer_idx: usize) -> &mut [f32] {
        if core_idx < self.num_cores && layer_idx < self.num_layers {
            &mut self.ssm_states[core_idx][layer_idx]
        } else {
            &mut [] // Empty slice if out of bounds
        }
    }

    /// Increment sequence length
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Clear the cache for a new sequence
    pub fn reset(&mut self) {
        for core_states in self.ssm_states.iter_mut() {
            for layer_state in core_states.iter_mut() {
                layer_state.fill(0.0);
            }
        }
        self.seq_len = 0;
    }
}

/// Generate text using the KV-cache (autoregressive)
pub struct TextGenerator {
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-k sampling: only sample from top k tokens
    pub top_k: usize,
    /// Top-p (nucleus) sampling: only sample from tokens with cumulative probability p
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl TextGenerator {
    pub fn new() -> Self {
        TextGenerator {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_tokens: 512,
        }
    }

    /// Sample next token from logits
    pub fn sample(&self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 3; // <UNK>
        }

        // Apply temperature
        let scaled: Vec<f32> = if self.temperature > 0.0 {
            logits.iter().map(|&x| x / self.temperature).collect()
        } else {
            logits.to_vec()
        };

        // Apply softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let probabilities: Vec<f32> = scaled.iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect();

        // Top-k filtering
        let mut indices: Vec<usize> = (0..probabilities.len()).collect();
        indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
        let top_k_limit = self.top_k.min(probabilities.len());
        let top_k_indices = &indices[..top_k_limit];

        // Top-p (nucleus) filtering
        let mut cumulative = 0.0f32;
        let mut cutoff = top_k_limit;
        for &idx in top_k_indices.iter() {
            cumulative += probabilities[idx];
            if cumulative >= self.top_p {
                cutoff = cutoff.min(top_k_indices.iter().position(|&x| x == idx).unwrap_or(cutoff) + 1);
                break;
            }
        }

        // Sample from filtered distribution
        let filtered_indices = &top_k_indices[..cutoff];
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        for &idx in filtered_indices {
            cum += probabilities[idx];
            if r <= cum {
                return idx;
            }
        }

        // Fallback: return highest probability
        indices[0]
    }
}
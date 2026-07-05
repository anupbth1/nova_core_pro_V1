//! Nova Field - Global Information Field (O(n) State Aggregator)
//!
//! The Field serves as a global state aggregator that replaces attention.
//! Instead of O(n²) pairwise attention, it maintains a global field vector
//! that all pulses interact with in O(n) time.
//!
//! Architecture: Field aggregates core outputs → SSM processes field →
//! Field diffuses back to pulses. All O(n), no attention.

use crate::pulse::NovaPulse;
use crate::ssm::{StateSpace, layer_norm};

/// Global information field that replaces attention mechanism.
/// O(n) complexity: each pulse reads from and writes to the field once.
pub struct NovaField {
    /// The global field vector - shared state container
    pub content: Vec<f32>,
    /// Field momentum (exponential moving average)
    pub momentum: Vec<f32>,
    /// Momentum decay rate (0.0 = no momentum, 0.9 = heavy smoothing)
    pub momentum_rate: f32,
    /// Learning rate for field updates
    pub learning_rate: f32,
    /// SSM for processing field content
    pub ssm: StateSpace,
    /// Field dimension
    pub dim: usize,
    /// Whether the field has converged
    pub converged: bool,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Previous content for convergence detection
    pub prev_content: Vec<f32>,
}

impl NovaField {
    /// Create a new field with the given dimension
    pub fn new(dim: usize) -> Self {
        let ssm = StateSpace::new_with_glu(dim, 4);
        NovaField {
            content: vec![0.0; dim],
            momentum: vec![0.0; dim],
            momentum_rate: 0.9,
            learning_rate: 0.1,
            ssm,
            dim,
            converged: false,
            convergence_threshold: 0.001,
            prev_content: vec![0.0; dim],
        }
    }

    /// Process core outputs and update the global field.
    /// Each core contributes a weighted signal to the field.
    /// O(cores * dim) = O(1) relative to sequence length.
    pub fn process_core_outputs(&mut self, core_states: &[Vec<f32>], core_gates: &[f32]) {
        if core_states.is_empty() {
            return;
        }

        // Save previous content for convergence detection
        self.prev_content.copy_from_slice(&self.content);

        // 1. Weighted average of all core states into field
        let mut weighted_sum = vec![0.0; self.dim];
        let mut total_gate = 0.0f32;

        for (state, &gate) in core_states.iter().zip(core_gates.iter()) {
            let min_len = self.dim.min(state.len());
            for i in 0..min_len {
                weighted_sum[i] += state[i] * gate;
            }
            total_gate += gate;
        }

        if total_gate > 0.0 {
            for i in 0..self.dim {
                weighted_sum[i] /= total_gate;
            }
        }

        // 2. Update field with momentum
        let alpha = self.momentum_rate;
        let lr = self.learning_rate;
        for i in 0..self.dim {
            self.momentum[i] = alpha * self.momentum[i] + (1.0 - alpha) * weighted_sum[i];
            self.content[i] += lr * (self.momentum[i] - self.content[i]);
            self.content[i] = self.content[i].clamp(-1.0, 1.0);
        }

        // 3. Process field through SSM
        let mut buffer = vec![0.0; self.dim * 4];
        self.ssm.forward(&mut self.content, &mut buffer);

        // 4. Check convergence
        self.check_convergence();
    }

    /// Diffuse field content back to pulses.
    /// Each pulse is updated based on similarity to the field.
    /// O(n * dim) = O(n) linear in sequence length.
    pub fn diffuse_to_pulses(&self, pulses: &mut [NovaPulse], strength: f32) {
        if pulses.is_empty() || strength <= 0.0 {
            return;
        }

        for pulse in pulses.iter_mut() {
            if pulse.converged {
                continue;
            }

            let min_len = pulse.content.len().min(self.dim);

            // Compute similarity between pulse and field
            let similarity = self.compute_similarity(&pulse.content, &self.content);

            // Blend: more similar = more influence from field
            let blend = strength * (0.5 + 0.5 * similarity);
            for i in 0..min_len {
                pulse.content[i] = pulse.content[i] * (1.0 - blend) + self.content[i] * blend;
                pulse.content[i] = pulse.content[i].clamp(-1.0, 1.0);
            }

            // Update semantic content
            for i in 0..min_len.min(pulse.semantic_content.len()) {
                pulse.semantic_content[i] = pulse.semantic_content[i] * 0.9 + pulse.content[i] * 0.1;
            }
        }
    }

    /// Compute cosine similarity between two vectors
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..n {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
        (dot / denom).clamp(-1.0, 1.0)
    }

    /// Check if field has converged
    fn check_convergence(&mut self) {
        let mut max_delta = 0.0f32;
        for i in 0..self.dim {
            let delta = (self.content[i] - self.prev_content[i]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
        self.converged = max_delta < self.convergence_threshold;
    }

    /// Reset field state for new sequences
    pub fn reset(&mut self) {
        self.content.fill(0.0);
        self.momentum.fill(0.0);
        self.converged = false;
        self.ssm.reset();
    }

    /// Get a snapshot of the field
    pub fn snapshot(&self) -> Vec<f32> {
        self.content.clone()
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        self.ssm.num_params()
    }
}


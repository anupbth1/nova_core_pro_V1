//! Nova Field - Global information field (replaces attention mechanism)
//!
//! Instead of O(n²) pairwise attention, Nova uses O(n) field dynamics
//! where information diffuses through a continuous field.
//!
//! NEW: SSM-enhanced field dynamics integrate Mamba's selective scan
//! into the field diffusion process, giving the field state-space
//! memory for long-range dependencies.
//!
//! PRIORITY 1: Added content convergence tracking for adaptive early exit.

use crate::pulse::NovaPulse;
use crate::ssm::{self, StateSpace};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovaField {
    /// Dimension of the field (same as pulse dimension)
    dim: usize,
    
    /// Current field state (global information)
    state: Vec<f32>,
    
    /// Momentum for smoother updates
    momentum: Vec<f32>,
    
    /// Learning rate for field updates
    learning_rate: f32,
    
    /// Diffusion rate (how fast information spreads)
    diffusion: f32,
    
    /// Number of updates performed
    update_count: usize,
    
    /// NEW: SSM parameters for field-level selective scan
    /// The field itself has a StateSpace that processes the
    /// aggregated pulse information through a selective scan,
    /// giving it temporal memory across update steps.
    pub ssm: Option<StateSpace>,
    
    /// Whether to use SSM-enhanced field dynamics
    pub use_ssm: bool,
    
    /// SSM gate - how much SSM influences field state
    pub ssm_gate: f32,
    
    /// PRIORITY 1: History of field states for convergence detection.
    /// Stores the last N field states to measure stabilization.
    convergence_history: Vec<Vec<f32>>,
    
    /// PRIORITY 1: Maximum number of field states to keep in history.
    max_history: usize,
}


impl NovaField {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            state: vec![0.0; dim],
            momentum: vec![0.0; dim],
            learning_rate: 0.1,
            diffusion: 0.3,
            update_count: 0,
            // SSM initialized lazily — call enable_ssm() to activate
            ssm: None,
            use_ssm: false,
            ssm_gate: 0.3,
            convergence_history: Vec::with_capacity(5),
            max_history: 5,
        }
    }
    
    /// Enable SSM-enhanced field dynamics.
    /// This creates a StateSpace for the field with d_state=16.
    pub fn enable_ssm(&mut self) {
        if self.ssm.is_none() {
            self.ssm = Some(StateSpace::new(self.dim, 16));
            self.use_ssm = true;
        }
    }
    
    /// Disable SSM-enhanced field dynamics.
    pub fn disable_ssm(&mut self) {
        self.use_ssm = false;
    }
    
    /// Core field update - O(n) complexity
    /// This is the secret sauce: no pairwise attention!
    pub fn update(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() {
            return;
        }
        
        self.update_count += 1;
        
        // ========== Step 1: Compute weighted field average (O(n)) ==========
        let mut field_avg = vec![0.0; self.dim];
        let mut total_weight = 0.0;
        
        // Parallel accumulation using Rayon
        let contributions: Vec<(Vec<f32>, f32)> = pulses.par_iter()
            .map(|pulse| {
                let mut contrib = vec![0.0; self.dim];
                let w = pulse.weight;
                for i in 0..self.dim {
                    contrib[i] = pulse.content[i] * w;
                }
                (contrib, w)
            })
            .collect();
        
        for (contrib, w) in contributions {
            total_weight += w;
            for i in 0..self.dim {
                field_avg[i] += contrib[i];
            }
        }
        
        if total_weight > 0.0 {
            for i in 0..self.dim {
                field_avg[i] /= total_weight;
            }
        }
        
        // ========== Step 2: Update field state with momentum ==========
        // NEW: If SSM is enabled, run the field average through selective scan
        // before updating the field state. This gives the field temporal memory.
        let ssm_enhanced_avg = if self.use_ssm {
            if let Some(ref mut ssm) = self.ssm {
                // Run the field average through SSM selective scan
                // This processes the aggregated pulse info through state-space dynamics
                let ssm_output = ssm::selective_scan_step(ssm, &field_avg, true);
                
                // Blend original field average with SSM output
                let blend = self.ssm_gate;
                let mut blended = vec![0.0; self.dim];
                for i in 0..self.dim {
                    blended[i] = field_avg[i] * (1.0 - blend) + ssm_output[i] * blend;
                }
                blended
            } else {
                field_avg.clone()
            }
        } else {
            field_avg.clone()
        };
        
        // Update field state with momentum (using SSM-enhanced average if enabled)
        for i in 0..self.dim {
            let diff = ssm_enhanced_avg[i] - self.state[i];
            self.momentum[i] = self.momentum[i] * 0.9 + diff * self.learning_rate;
            self.state[i] += self.momentum[i];
            self.state[i] = self.state[i].clamp(-1.0, 1.0);
        }
        
        // ========== Step 3: Diffuse field information to pulses (O(n)) ==========
        let diffusion_factor = self.diffusion * (0.95_f32).powf(self.update_count as f32);
        
        pulses.par_iter_mut().for_each(|pulse| {
            for i in 0..self.dim {
                // Field influences pulses (attention replacement)
                pulse.content[i] = pulse.content[i] * (1.0 - diffusion_factor) 
                                 + self.state[i] * diffusion_factor;
            }
            // Pulses become more certain as field stabilizes
            pulse.reduce_entropy(0.98);
        });
        
        // PRIORITY 1: Store field state in convergence history
        self.convergence_history.push(self.state.clone());
        if self.convergence_history.len() > self.max_history {
            self.convergence_history.remove(0);
        }
    }
    
    /// PRIORITY 1: Compute content convergence score (0.0 = no convergence, 1.0 = fully converged).
    /// Measures how much the field state has stabilized over recent updates.
    /// Returns 1.0 when the field state stops changing significantly.
    pub fn content_convergence(&self) -> f32 {
        if self.convergence_history.len() < 2 {
            return 0.0;
        }
        
        // Compare the last two field states
        let last = &self.convergence_history[self.convergence_history.len() - 1];
        let prev = &self.convergence_history[self.convergence_history.len() - 2];
        
        let mut total_delta = 0.0f32;
        let min_len = last.len().min(prev.len());
        for i in 0..min_len {
            total_delta += (last[i] - prev[i]).abs();
        }
        let avg_delta = total_delta / min_len as f32;
        
        // Convert delta to convergence score: 0 delta = 1.0 convergence
        1.0 / (1.0 + avg_delta * 50.0)
    }
    
    /// Get current field state (for debugging)
    pub fn state(&self) -> &[f32] {
        &self.state
    }
    
    /// Set field state (for model loading)
    pub fn set_state(&mut self, state: &[f32]) {
        let len = self.state.len().min(state.len());
        self.state[..len].copy_from_slice(&state[..len]);
    }
    
    /// Set field momentum (for model loading)
    pub fn set_momentum(&mut self, momentum: &[f32]) {
        let len = self.momentum.len().min(momentum.len());
        self.momentum[..len].copy_from_slice(&momentum[..len]);
    }
    
    /// Set update count
    pub fn set_update_count(&mut self, count: usize) {
        self.update_count = count;
    }

    /// Get current field momentum (for context compression)
    pub fn momentum(&self) -> &[f32] {
        &self.momentum
    }

    /// Get mutable reference to field state (for training updates)
    pub fn state_mut(&mut self) -> &mut [f32] {
        &mut self.state
    }

    /// Get mutable reference to field momentum (for training updates)
    pub fn momentum_mut(&mut self) -> &mut [f32] {
        &mut self.momentum
    }

    /// OPTIMIZED: Get mutable references to both state and momentum simultaneously.
    /// This avoids the borrow checker issue when both are needed at once.
    pub fn state_and_momentum_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (&mut self.state, &mut self.momentum)
    }

    /// Get the learning rate (for GPU accelerator)
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get the diffusion rate (for GPU accelerator)
    pub fn diffusion(&self) -> f32 {
        self.diffusion
    }

    /// Get field energy (measure of information content)
    pub fn energy(&self) -> f32 {
        self.state.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
    
    /// Get the current diffusion rate.
    pub fn get_diffusion_rate(&self) -> f32 {
        self.diffusion
    }
    
    /// Set the diffusion rate.
    pub fn set_diffusion_rate(&mut self, rate: f32) {
        self.diffusion = rate;
    }
    
    /// Reset field (for new contexts)
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.momentum.fill(0.0);
        self.update_count = 0;
        self.convergence_history.clear();
        // Also reset SSM state if enabled
        if let Some(ref mut ssm) = self.ssm {
            ssm.reset();
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_update() {
        let mut field = NovaField::new(32);
        let mut pulses = vec![
            NovaPulse::from_text("hello", 32, 0),
            NovaPulse::from_text("world", 32, 1),
        ];
        
        let original_first = pulses[0].content.clone();
        field.update(&mut pulses);
        
        // Pulses should have changed due to field influence
        assert_ne!(pulses[0].content, original_first);
        println!("✅ Field update works! Energy: {:.3}", field.energy());
    }
    
    #[test]
    fn test_o_n_complexity() {
        let mut field = NovaField::new(128);
        
        // Test with 10 pulses
        let start = std::time::Instant::now();
        let mut pulses = (0..10).map(|i| NovaPulse::new(128, i)).collect::<Vec<_>>();
        field.update(&mut pulses);
        let time_10 = start.elapsed();
        
        // Test with 100 pulses (should be ~10x slower, not 100x like attention)
        let start = std::time::Instant::now();
        let mut pulses = (0..100).map(|i| NovaPulse::new(128, i)).collect::<Vec<_>>();
        field.update(&mut pulses);
        let time_100 = start.elapsed();
        
        // O(n) means 100/10 = 10x, not 100x
        let ratio = time_100.as_secs_f32() / time_10.as_secs_f32();
        println!("✅ O(n) check: 10x slower would be linear, got {:.1}x", ratio);
        assert!(ratio < 15.0, "Should be roughly linear, got {:.1}x", ratio);
    }
    
    #[test]
    fn test_content_convergence() {
        let mut field = NovaField::new(32);
        let mut pulses = vec![
            NovaPulse::from_text("test", 32, 0),
        ];
        
        // First update: no convergence yet
        let conv1 = field.content_convergence();
        assert_eq!(conv1, 0.0);
        
        // Multiple updates should increase convergence
        for _ in 0..5 {
            field.update(&mut pulses);
        }
        
        let conv2 = field.content_convergence();
        println!("✅ Field convergence after 5 updates: {:.4}", conv2);
        assert!(conv2 > 0.0);
    }
}

// ============================================================================
// CPU Fallback Functions for CUDA Module
// ============================================================================

/// CPU fallback implementation for field update operation.
/// This is used by the CUDA module when GPU acceleration is not available.
pub fn field_update_raw(
    pulses_content: &[Vec<f32>],
    pulses_weight: &[f32],
    field_state: &mut [f32],
    field_momentum: &mut [f32],
    learning_rate: f32,
    diffusion: f32,
) {
    if pulses_content.is_empty() || pulses_weight.is_empty() {
        return;
    }
    
    let dim = field_state.len();
    let mut field_avg = vec![0.0; dim];
    let mut total_weight = 0.0;
    
    for (content, &weight) in pulses_content.iter().zip(pulses_weight.iter()) {
        total_weight += weight;
        let len = content.len().min(dim);
        for i in 0..len {
            field_avg[i] += content[i] * weight;
        }
    }
    
    if total_weight > 0.0 {
        for i in 0..dim {
            field_avg[i] /= total_weight;
        }
    }
    
    // Update field state with momentum
    for i in 0..dim {
        let diff = field_avg[i] - field_state[i];
        field_momentum[i] = field_momentum[i] * 0.9 + diff * learning_rate;
        field_state[i] += field_momentum[i];
        field_state[i] = field_state[i].clamp(-1.0, 1.0);
    }
}

/// CPU fallback implementation for field diffuse operation.
/// This is used by the CUDA module when GPU acceleration is not available.
pub fn field_diffuse_raw(
    pulses_content: &mut [Vec<f32>],
    field_state: &[f32],
    diffusion_factor: f32,
) {
    if pulses_content.is_empty() {
        return;
    }
    
    let dim = field_state.len();
    
    for content in pulses_content.iter_mut() {
        let len = content.len().min(dim);
        for i in 0..len {
            // Field influences pulses (attention replacement)
            content[i] = content[i] * (1.0 - diffusion_factor) 
                       + field_state[i] * diffusion_factor;
        }
    }
}

/// CPU fallback implementation for cosine similarity operation.
/// This is used by the CUDA module when GPU acceleration is not available.
pub fn cosine_similarity_raw(
    query: &[f32],
    vocabulary: &[Vec<f32>],
    vocab_norms: &[f32],
    similarities: &mut [f32],
) {
    let dim = query.len();
    let vocab_size = vocabulary.len().min(similarities.len());
    
    for i in 0..vocab_size {
        let word = &vocabulary[i];
        let mut dot = 0.0;
        let len = word.len().min(dim);
        
        for j in 0..len {
            dot += query[j] * word[j];
        }
        
        let query_norm = query.iter().take(dim).map(|&x| x * x).sum::<f32>().sqrt();
        let word_norm = vocab_norms.get(i).copied().unwrap_or_else(|| {
            word.iter().take(dim).map(|&x| x * x).sum::<f32>().sqrt()
        });
        
        if query_norm > 0.0 && word_norm > 0.0 {
            similarities[i] = dot / (query_norm * word_norm);
        } else {
            similarities[i] = 0.0;
        }
    }
}

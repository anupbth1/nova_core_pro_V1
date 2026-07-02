//! Nova Field - Global information field (replaces attention mechanism)
//!
//! Instead of O(n²) pairwise attention, Nova uses O(n) field dynamics
//! where information diffuses through a continuous field.
//!
//! NEW: SSM-enhanced field dynamics integrate Mamba's selective scan
//! into the field diffusion process, giving the field state-space
//! memory for long-range dependencies.

use crate::pulse::NovaPulse;
use crate::ssm::{self, StateSpace};
use rayon::prelude::*;

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

    /// Get mutable reference to field state (for training updates)
    pub fn state_mut(&mut self) -> &mut [f32] {
        &mut self.state
    }

    /// Get mutable reference to field momentum (for training updates)
    pub fn momentum_mut(&mut self) -> &mut [f32] {
        &mut self.momentum
    }

    /// Get field energy (measure of information content)
    pub fn energy(&self) -> f32 {
        self.state.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
    
    /// Reset field (for new contexts)
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.momentum.fill(0.0);
        self.update_count = 0;
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
}
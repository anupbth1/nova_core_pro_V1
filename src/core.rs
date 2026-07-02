//! Nova Core - Adaptive processing units
//!
//! Each core now has:
//! - Original Nova transforms (syntax, semantic, memory, reasoning, pattern)
//! - NEW: SSM (State Space Model) transform from Mamba/RWKV
//! - StateSpace parameters for selective scan

use crate::pulse::NovaPulse;
use crate::ssm::{self, StateSpace};

#[derive(Debug, Clone)]
pub struct NovaCore {
    pub id: usize,
    pub name: String,
    pub memory: Vec<f32>,
    pub adaptive_depth: usize,
    pub internal_state: Vec<f32>,
    pub gate: f32,
    /// NEW: State Space Model parameters for selective scan
    pub ssm: StateSpace,
    /// Whether to use SSM transform in this core
    pub use_ssm: bool,
    /// Whether to use RWKV time mixing before SSM
    pub use_time_mixing: bool,
}

impl NovaCore {
    pub fn new(id: usize, name: &str, memory_size: usize, dim: usize) -> Self {
        // SSM dimensions: d_inner = dim (Nova pulse dimension), d_state = 16 (standard)
        let ssm = StateSpace::new(dim, 16);
        
        // Determine which cores use SSM based on their role
        let use_ssm = match name {
            "syntax" => true,      // SSM helps with sequential byte encoding
            "semantic" => true,    // SSM helps with meaning propagation
            "memory" => true,      // SSM IS a memory mechanism!
            "reasoning" => true,   // SSM helps with step-by-step reasoning
            "pattern" => true,     // SSM helps with pattern recognition
            _ => true,             // All specialized cores also use SSM
        };
        
        // Time mixing is useful for cores that need temporal context
        let use_time_mixing = match name {
            "memory" => true,      // Memory needs temporal mixing
            "reasoning" => true,   // Reasoning needs step-by-step context
            "context_window" => true, // Context window is all about temporal
            _ => false,            // Other cores use pure SSM
        };
        
        Self {
            id,
            name: name.to_string(),
            memory: vec![0.0; memory_size],
            adaptive_depth: 1,
            internal_state: vec![0.0; dim],
            gate: 0.8,
            ssm,
            use_ssm,
            use_time_mixing,
        }
    }
    
    pub fn process(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() { return; }
        
        let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
        let avg_weight: f32 = pulses.iter().map(|p| p.weight).sum::<f32>() / pulses.len() as f32;
        
        self.adaptive_depth = 1 + (avg_entropy * 6.0) as usize + (avg_weight * 2.0) as usize;
        self.adaptive_depth = self.adaptive_depth.clamp(1, 12);
        
        for step in 0..self.adaptive_depth {
            match self.name.as_str() {
                "syntax" => {
                    self.syntax_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "semantic" => {
                    self.semantic_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "memory" => {
                    self.memory_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "reasoning" => {
                    self.reasoning_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "pattern" => {
                    self.pattern_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                _ => {
                    self.default_transform(pulses);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
            }
            self.update_internal_state(pulses);
        }
    }
    
    fn syntax_transform(&mut self, pulses: &mut [NovaPulse], step: usize) {
        let factor = 1.0 - (step as f32 * 0.03).min(0.5);
        for pulse in pulses.iter_mut() {
            for x in pulse.content.iter_mut() {
                *x = x.tanh() * factor;
            }
            pulse.reduce_entropy(0.97);
        }
    }
    
    fn semantic_transform(&mut self, pulses: &mut [NovaPulse], step: usize) {
        for pulse in pulses.iter_mut() {
            for x in pulse.content.iter_mut() {
                if x.abs() > 0.3 {
                    *x = (*x * 1.12).clamp(-1.0, 1.0);
                } else {
                    *x *= 0.95;
                }
            }
            if step > 2 {
                pulse.reduce_entropy(0.85);
            }
        }
    }
    
    fn memory_transform(&mut self, pulses: &mut [NovaPulse], step: usize) {
        for (i, pulse) in pulses.iter_mut().enumerate() {
            if i < self.memory.len() && pulse.weight > 0.5 {
                self.memory[i] = self.memory[i] * 0.85 + pulse.content[0] * 0.15;
            }
        }
        for (i, pulse) in pulses.iter_mut().enumerate() {
            if i < self.memory.len() {
                let blend = if step < 3 { 0.3 } else { 0.6 };
                pulse.content[0] = pulse.content[0] * (1.0 - blend) + self.memory[i] * blend;
            }
        }
        if step > 7 {
            for i in 0..self.memory.len().min(pulses.len()) {
                self.memory[i] = self.memory[i] * 0.99 + pulses[i].content[0] * 0.01;
            }
        }
    }
    
    fn reasoning_transform(&mut self, pulses: &mut [NovaPulse], step: usize) {
        if pulses.len() < 2 { return; }
        for i in 1..pulses.len() {
            let diff = pulses[i].content[0] - pulses[i-1].content[0];
            if step % 2 == 0 {
                pulses[i].content[0] += diff * 0.25;
            } else {
                pulses[i-1].content[0] -= diff * 0.15;
            }
        }
    }
    
    fn pattern_transform(&mut self, pulses: &mut [NovaPulse], _step: usize) {
        if pulses.len() < 3 { return; }
        let pattern_len = 3;
        for i in 0..pulses.len() - pattern_len {
            let similarity = pulses[i].similarity(&pulses[i + pattern_len]);
            if similarity > 0.7 {
                for j in 0..pattern_len {
                    if i + j < pulses.len() && i + pattern_len + j < pulses.len() {
                        pulses[i + pattern_len + j].weight += pulses[i + j].weight * 0.1;
                    }
                }
            }
        }
    }
    
    fn default_transform(&mut self, pulses: &mut [NovaPulse]) {
        for pulse in pulses.iter_mut() {
            for x in pulse.content.iter_mut() {
                *x = x.tanh();
            }
        }
    }
    
    /// NEW: SSM Transform - Applies Mamba's selective scan to pulses.
    ///
    /// This is the key innovation: each pulse's content is processed through
    /// the State Space Model, which maintains a hidden state that evolves
    /// over time. This gives Nova the ability to handle long-range dependencies
    /// like Transformers, but with O(n) complexity.
    ///
    /// The SSM transform:
    /// 1. Takes each pulse's content as input x(t)
    /// 2. Updates hidden state: h(t) = exp(Δ*A) * h(t-1) + Δ*B*x(t)
    /// 3. Produces output: y(t) = C*h(t) + D*x(t)
    /// 4. Optionally applies RWKV time mixing for temporal context
    fn ssm_transform(&mut self, pulses: &mut [NovaPulse], _step: usize) {
        if pulses.is_empty() { return; }
        
        // Extract content vectors from pulses
        let mut contents: Vec<Vec<f32>> = pulses.iter()
            .map(|p| p.content.clone())
            .collect();
        
        // Apply SSM transform to all pulses
        // This processes each pulse through the selective scan,
        // updating the SSM hidden state with each pulse
        for content in contents.iter_mut() {
            ssm::ssm_transform_pulse(&mut self.ssm, content, self.use_time_mixing);
        }
        
        // Write back to pulses (blend with original based on gate)
        let ssm_strength = self.gate * 0.5; // SSM contributes up to 50%
        for (i, pulse) in pulses.iter_mut().enumerate() {
            if i < contents.len() {
                for j in 0..pulse.content.len().min(contents[i].len()) {
                    // Blend: original * (1 - ssm_strength) + SSM_output * ssm_strength
                    pulse.content[j] = pulse.content[j] * (1.0 - ssm_strength) 
                                     + contents[i][j] * ssm_strength;
                }
                // Clamp to [-1, 1] range
                for x in pulse.content.iter_mut() {
                    *x = x.clamp(-1.0, 1.0);
                }
            }
        }
    }
    
    fn update_internal_state(&mut self, pulses: &[NovaPulse]) {
        if pulses.is_empty() || self.internal_state.is_empty() { return; }
        
        // Enhanced internal state update using SSM-aware averaging
        // Instead of just tracking first element, use SSM hidden state
        if self.use_ssm {
            // Aggregate SSM hidden state into internal_state
            let d_inner = self.ssm.d_inner;
            let d_state = self.ssm.d_state;
            let state_len = self.internal_state.len().min(d_inner);
            
            for i in 0..state_len {
                let mut h_sum = 0.0;
                for j in 0..d_state.min(4) { // Use first 4 state dims for efficiency
                    h_sum += self.ssm.h[i][j];
                }
                self.internal_state[i] = self.internal_state[i] * 0.9 + (h_sum / 4.0) * 0.1;
            }
        } else {
            // Original behavior
            let avg_content: f32 = pulses.iter()
                .map(|p| p.content.first().copied().unwrap_or(0.0))
                .sum::<f32>() / pulses.len() as f32;
            self.internal_state[0] = self.internal_state[0] * 0.9 + avg_content * 0.1;
        }
    }
    
    /// Reset SSM state (for new sequences)
    pub fn reset_ssm(&mut self) {
        self.ssm.reset();
    }
}

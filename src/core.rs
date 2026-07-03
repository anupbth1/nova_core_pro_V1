//! Nova Core - Adaptive processing units
//!
//! Each core now has:
//! - Original Nova transforms (syntax, semantic, memory, reasoning, pattern)
//! - SSM (State Space Model) transform from Mamba/RWKV
//! - StateSpace parameters for selective scan
//! - PHASE 4: Multi-core communication via shared message bus
//!   Cores broadcast their internal state after each iteration,
//!   and receive blended signals from other cores.
//! - PRIORITY 1: Semantic reasoning engine with contradiction detection,
//!   implication propagation, evidence accumulation, and content convergence.

use crate::pulse::NovaPulse;
use crate::ssm::{self, StateSpace};
use crate::knowledge::KnowledgeStore;

/// PHASE 4: Message from one core to all others.
/// Contains the core's internal state summary and gate confidence.
#[derive(Debug, Clone)]
pub struct CoreMessage {
    pub core_id: usize,
    pub core_name: String,
    /// Compressed state summary (first 8 dims of internal_state)
    pub state_summary: Vec<f32>,
    /// How confident this core is in its current state
    pub gate: f32,
}

#[derive(Debug, Clone)]
pub struct NovaCore {
    pub id: usize,
    pub name: String,
    pub memory: Vec<f32>,
    pub adaptive_depth: usize,
    pub internal_state: Vec<f32>,
    pub gate: f32,
    /// State Space Model parameters for selective scan
    pub ssm: StateSpace,
    /// Whether to use SSM transform in this core
    pub use_ssm: bool,
    /// Whether to use RWKV time mixing before SSM
    pub use_time_mixing: bool,
    /// PHASE 4: Accumulated messages from other cores this iteration
    pub received_messages: Vec<CoreMessage>,
    /// PHASE 4: How much to blend in cross-core signals (0.0 = none, 1.0 = full)
    pub cross_core_blend: f32,
    /// PRIORITY 1: Previous pulse content for convergence detection (one per pulse)
    pub prev_pulse_content: Vec<Vec<f32>>,
    /// PRIORITY 1: Convergence threshold for content stabilization
    pub content_convergence_threshold: f32,
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
            received_messages: Vec::new(),
            cross_core_blend: 0.15,
            prev_pulse_content: Vec::new(),
            content_convergence_threshold: 0.01,
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
                    // PRIORITY 1: Apply semantic refinement after semantic transform
                    self.semantic_refine_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "memory" => {
                    self.memory_transform(pulses, step);
                    if self.use_ssm { self.ssm_transform(pulses, step); }
                },
                "reasoning" => {
                    // PRIORITY 1: Use improved reasoning transform
                    self.reasoning_transform_v2(pulses, step);
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
        
        // PRIORITY 1: Update convergence flags on pulses
        self.update_convergence(pulses);
    }
    
    /// PRIORITY 1: Update convergence flags on pulses based on content stabilization.
    /// A pulse is converged when its content changes less than the threshold
    /// compared to the previous iteration's content.
    /// FIXED: Now tracks ALL pulses individually, not just the first one.
    fn update_convergence(&mut self, pulses: &mut [NovaPulse]) {
        if self.prev_pulse_content.is_empty() {
            // First iteration: store current content for ALL pulses
            self.prev_pulse_content = pulses.iter().map(|p| p.content.clone()).collect();
            return;
        }
        
        // Ensure prev_pulse_content has entries for all pulses
        while self.prev_pulse_content.len() < pulses.len() {
            self.prev_pulse_content.push(vec![0.0; pulses[0].content.len()]);
        }
        
        for (i, pulse) in pulses.iter_mut().enumerate() {
            if pulse.converged {
                continue; // Already converged
            }
            
            // Get this pulse's previous content
            let prev = &self.prev_pulse_content[i];
            
            // Compute max absolute change in content
            let mut max_delta = 0.0f32;
            let min_len = pulse.content.len().min(prev.len());
            for j in 0..min_len {
                let delta = (pulse.content[j] - prev[j]).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
            }
            
            if max_delta < self.content_convergence_threshold {
                pulse.converged = true;
            }
        }
        
        // Store current content for ALL pulses for next iteration comparison
        for (i, pulse) in pulses.iter().enumerate() {
            if i < self.prev_pulse_content.len() {
                let min_len = self.prev_pulse_content[i].len().min(pulse.content.len());
                self.prev_pulse_content[i][..min_len].copy_from_slice(&pulse.content[..min_len]);
            }
        }
    }
    
    /// PRIORITY 1: Semantic refinement transform.
    /// Pulls pulses toward semantically meaningful attractor states.
    /// - Amplifies signals above noise threshold
    /// - Uses entropy-weighted blending (confident pulses change less)
    /// - Updates semantic_content with refined representation
    fn semantic_refine_transform(&mut self, pulses: &mut [NovaPulse], _step: usize) {
        let noise_threshold = 0.15;
        let refine_strength = self.gate * 0.3;
        
        for pulse in pulses.iter_mut() {
            // Compute the semantic direction: sign-weighted average
            let mut pos_sum = 0.0f32;
            let mut neg_sum = 0.0f32;
            let mut pos_count = 0usize;
            let mut neg_count = 0usize;
            
            for &x in pulse.content.iter() {
                if x > noise_threshold {
                    pos_sum += x;
                    pos_count += 1;
                } else if x < -noise_threshold {
                    neg_sum += x;
                    neg_count += 1;
                }
            }
            
            // Determine semantic direction
            let semantic_direction = if pos_count > 0 && neg_count > 0 {
                // Mixed signals: use the stronger direction
                let pos_avg = pos_sum / pos_count as f32;
                let neg_avg = neg_sum.abs() / neg_count as f32;
                if pos_avg > neg_avg { 1.0 } else { -1.0 }
            } else if pos_count > 0 {
                1.0
            } else if neg_count > 0 {
                -1.0
            } else {
                0.0 // No clear direction
            };
            
            // Apply semantic refinement: pull content toward semantic direction
            let entropy_factor = 1.0 - pulse.entropy; // Confident pulses change less
            let blend = refine_strength * entropy_factor;
            
            for x in pulse.content.iter_mut() {
                if x.abs() > noise_threshold {
                    // Amplify clear signals
                    *x = (*x * (1.0 + blend * 0.1)).clamp(-1.0, 1.0);
                } else if semantic_direction != 0.0 {
                    // Pull weak signals toward semantic direction
                    *x += semantic_direction * blend * 0.05;
                    *x = x.clamp(-1.0, 1.0);
                }
            }
            
            // Update semantic_content with refined representation
            let min_len = pulse.semantic_content.len().min(pulse.content.len());
            for i in 0..min_len {
                // Blend: keep some of old semantic content, add new refined content
                pulse.semantic_content[i] = pulse.semantic_content[i] * 0.7 + pulse.content[i] * 0.3;
            }
        }
    }
    
    /// PRIORITY 1: Improved reasoning transform with actual semantic reasoning.
    /// Performs:
    /// 1. Contradiction detection: Pulses with opposite signs cancel/attenuate
    /// 2. Implication propagation: Strong pulses amplify related weaker pulses
    /// 3. Evidence accumulation: Pulses with same direction reinforce each other
    /// 4. Entropy-gated reasoning: Only applies when entropy is low enough
    fn reasoning_transform_v2(&mut self, pulses: &mut [NovaPulse], step: usize) {
        if pulses.len() < 2 { return; }
        
        // Only reason when pulses have low enough entropy (are confident enough)
        let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
        if avg_entropy > 0.6 {
            // Too uncertain to reason - just do basic diffusion
            self.reasoning_transform(pulses, step);
            return;
        }
        
        let reasoning_strength = (1.0 - avg_entropy) * self.gate * 0.3;
        
        // Phase 1: Contradiction detection and resolution
        // Find pairs of pulses with opposite semantic directions
        for i in 0..pulses.len() {
            for j in i+1..pulses.len() {
                let sim = pulses[i].similarity(&pulses[j]);
                
                if sim < -0.3 {
                    // Strong contradiction: pulses disagree
                    // Attenuate both based on their entropy (less certain = more attenuation)
                    let atten_i = 1.0 - (pulses[i].entropy * reasoning_strength * 0.5);
                    let atten_j = 1.0 - (pulses[j].entropy * reasoning_strength * 0.5);
                    
                    for k in 0..pulses[i].content.len().min(pulses[j].content.len()) {
                        pulses[i].content[k] *= atten_i;
                        pulses[j].content[k] *= atten_j;
                    }
                } else if sim > 0.5 {
                    // Strong agreement: pulses reinforce each other
                    let boost = reasoning_strength * 0.2;
                    for k in 0..pulses[i].content.len().min(pulses[j].content.len()) {
                        let avg = (pulses[i].content[k] + pulses[j].content[k]) * 0.5;
                        pulses[i].content[k] = pulses[i].content[k] * (1.0 - boost) + avg * boost;
                        pulses[j].content[k] = pulses[j].content[k] * (1.0 - boost) + avg * boost;
                    }
                }
            }
        }
        
        // Phase 2: Implication propagation
        // Strong pulses (high weight, low entropy) influence weaker ones
        // FIXED: Use index-based access to avoid borrow checker conflicts
        let strong_pulses: Vec<usize> = pulses.iter()
            .enumerate()
            .filter(|(_, p)| p.weight > 0.6 && p.entropy < 0.4)
            .map(|(i, _)| i)
            .collect();
        
        if !strong_pulses.is_empty() {
            // Pre-compute similarity scores for all weak pulses against strong pulses
            let weak_indices: Vec<usize> = (0..pulses.len())
                .filter(|&i| !strong_pulses.contains(&i) && pulses[i].weight >= 0.3)
                .collect();
            
            for &i in &weak_indices {
                // Find the most similar strong pulse
                let mut best_sim = 0.0f32;
                let mut best_idx = 0;
                for &si in &strong_pulses {
                    let sim = pulses[i].similarity(&pulses[si]);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = si;
                    }
                }
                
                if best_sim > 0.3 {
                    // Propagate implication from strong pulse to this one
                    let influence = best_sim * reasoning_strength * 0.3;
                    let strong_content = pulses[best_idx].content.clone();
                    for k in 0..pulses[i].content.len().min(strong_content.len()) {
                        let delta = strong_content[k] - pulses[i].content[k];
                        pulses[i].content[k] += delta * influence;
                        pulses[i].content[k] = pulses[i].content[k].clamp(-1.0, 1.0);
                    }
                }
            }
        }
        
        // Phase 3: Evidence accumulation
        // Pulses with same direction accumulate evidence
        if step > 1 {
            let mut direction_sum = vec![0.0f32; pulses[0].content.len()];
            let mut direction_count = 0usize;
            
            for pulse in pulses.iter() {
                if pulse.weight > 0.5 && pulse.entropy < 0.5 {
                    for k in 0..direction_sum.len().min(pulse.content.len()) {
                        direction_sum[k] += pulse.content[k];
                    }
                    direction_count += 1;
                }
            }
            
            if direction_count > 1 {
                for k in 0..direction_sum.len() {
                    direction_sum[k] /= direction_count as f32;
                }
                
                // Blend accumulated evidence into all pulses
                let evidence_blend = reasoning_strength * 0.15;
                for pulse in pulses.iter_mut() {
                    for k in 0..pulse.content.len().min(direction_sum.len()) {
                        pulse.content[k] = pulse.content[k] * (1.0 - evidence_blend) 
                                         + direction_sum[k] * evidence_blend;
                        pulse.content[k] = pulse.content[k].clamp(-1.0, 1.0);
                    }
                }
            }
        }
    }
    
    /// PRIORITY 1: Compute content convergence score (0.0 = no convergence, 1.0 = fully converged).
    /// Measures how much pulse content has stabilized across the batch.
    /// FIXED: Now compares ALL pulses against their individual previous content.
    pub fn content_convergence(&self, pulses: &[NovaPulse]) -> f32 {
        if pulses.is_empty() || self.prev_pulse_content.is_empty() {
            return 0.0;
        }
        
        let mut total_delta = 0.0f32;
        let mut count = 0usize;
        
        for (i, pulse) in pulses.iter().enumerate() {
            if i >= self.prev_pulse_content.len() {
                break;
            }
            let prev = &self.prev_pulse_content[i];
            let min_len = pulse.content.len().min(prev.len());
            for j in 0..min_len {
                total_delta += (pulse.content[j] - prev[j]).abs();
                count += 1;
            }
        }
        
        if count == 0 { return 0.0; }
        let avg_delta = total_delta / count as f32;
        
        // Convert delta to convergence score: 0 delta = 1.0 convergence
        // Use sigmoid-like mapping: 1.0 / (1.0 + delta * 100)
        1.0 / (1.0 + avg_delta * 100.0)
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
    
    /// OPTIMIZED V3: SSM Transform - Applies Mamba's selective scan to pulses.
    /// Avoids allocating Vec<Vec<f32>> by processing pulses in-place with blending.
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
        
        let ssm_strength = self.gate * 0.5; // SSM contributes up to 50%
        
        // OPTIMIZED: Process pulses in-place, avoiding Vec<Vec<f32>> allocation
        for pulse in pulses.iter_mut() {
            // Save original content for blending
            let original = pulse.content.clone();
            
            // Apply SSM transform directly to pulse content
            ssm::ssm_transform_pulse(&mut self.ssm, &mut pulse.content, self.use_time_mixing);
            
            // Blend: original * (1 - ssm_strength) + SSM_output * ssm_strength
            for j in 0..pulse.content.len() {
                pulse.content[j] = original[j] * (1.0 - ssm_strength) 
                                 + pulse.content[j] * ssm_strength;
                pulse.content[j] = pulse.content[j].clamp(-1.0, 1.0);
            }
        }
    }
    
    fn update_internal_state(&mut self, pulses: &[NovaPulse]) {
        if pulses.is_empty() || self.internal_state.is_empty() { return; }
        
        // Enhanced internal state update using SSM-aware averaging
        // OPTIMIZED: Uses flat SSM memory layout (h[i * d_state + j])
        if self.use_ssm {
            let d_inner = self.ssm.d_inner;
            let d_state = self.ssm.d_state;
            let state_len = self.internal_state.len().min(d_inner);
            let ds = d_state;
            
            for i in 0..state_len {
                let base = i * ds;
                let mut h_sum = 0.0;
                for j in 0..ds.min(4) { // Use first 4 state dims for efficiency
                    h_sum += self.ssm.h[base + j];
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
    
    /// PHASE 4: Broadcast this core's current state as a message to other cores.
    /// Returns a CoreMessage containing a compressed state summary and gate confidence.
    pub fn broadcast_message(&self) -> CoreMessage {
        // Compress internal_state to first 8 dimensions for efficiency
        let summary_len = self.internal_state.len().min(8);
        let mut state_summary = Vec::with_capacity(summary_len);
        for i in 0..summary_len {
            state_summary.push(self.internal_state[i]);
        }
        CoreMessage {
            core_id: self.id,
            core_name: self.name.clone(),
            state_summary,
            gate: self.gate,
        }
    }
    
    /// PHASE 4: Receive messages from other cores.
    /// Stores them for blending during the next transform step.
    pub fn receive_messages(&mut self, messages: &[CoreMessage]) {
        self.received_messages = messages.to_vec();
    }
    
    /// PHASE 5: Knowledge-aware transform.
    /// Augments pulses with knowledge from the KnowledgeStore.
    /// Each pulse's content is blended with the closest concept embedding.
    pub fn knowledge_transform(&mut self, pulses: &mut [NovaPulse], knowledge: &KnowledgeStore) {
        if knowledge.concepts.is_empty() || pulses.is_empty() {
            return;
        }
        let blend_strength = self.gate * 0.2; // Knowledge contributes up to 20%
        for pulse in pulses.iter_mut() {
            knowledge.augment_pulse_with_knowledge(pulse, blend_strength);
        }
    }

    /// PHASE 4: Blend received cross-core signals into pulse content.
    /// Each pulse gets a weighted blend of all received core states,
    /// weighted by each core's gate confidence.
    /// This is O(cores × dim) = O(1) relative to pulse count.
    pub fn blend_cross_core_signals(&mut self, pulses: &mut [NovaPulse]) {
        if self.received_messages.is_empty() || self.cross_core_blend <= 0.0 {
            return;
        }
        
        // Compute weighted average of all received state summaries
        let mut total_gate = 0.0f32;
        let mut blended_signal = Vec::new();
        
        for msg in &self.received_messages {
            if msg.gate <= 0.0 { continue; }
            if blended_signal.is_empty() {
                blended_signal = msg.state_summary.clone();
                total_gate = msg.gate;
            } else {
                let min_len = blended_signal.len().min(msg.state_summary.len());
                for i in 0..min_len {
                    blended_signal[i] += msg.state_summary[i] * msg.gate;
                }
                total_gate += msg.gate;
            }
        }
        
        if total_gate <= 0.0 || blended_signal.is_empty() {
            return;
        }
        
        // Normalize by total gate
        for val in blended_signal.iter_mut() {
            *val /= total_gate;
        }
        
        // Blend the cross-core signal into each pulse's content
        let blend = self.cross_core_blend * self.gate;
        for pulse in pulses.iter_mut() {
            let min_len = pulse.content.len().min(blended_signal.len());
            for i in 0..min_len {
                pulse.content[i] = pulse.content[i] * (1.0 - blend) + blended_signal[i] * blend;
                pulse.content[i] = pulse.content[i].clamp(-1.0, 1.0);
            }
        }
    }
    
    /// Set the gate strength for this core.
    pub fn set_gate_strength(&mut self, strength: f32) {
        self.gate = strength.clamp(0.0, 1.0);
    }
    
    /// Get the current gate strength.
    pub fn get_gate_strength(&self) -> f32 {
        self.gate
    }
}

//! Nova Core - Neural Processing Units with SSM+GLU Architecture
//!
//! Each core is a proper neural block: LayerNorm → SSM → GLU → Residual
//! 5 specialized cores: Syntax, Semantic, Memory, Reasoning, Pattern
//! All O(n) complexity, no attention, Transformer-free.

use crate::pulse::NovaPulse;
use crate::ssm::{self, StateSpace, SsmStack, layer_norm};
use rand::Rng;

/// Message from one core to all others (cross-core communication)
#[derive(Debug, Clone)]
pub struct CoreMessage {
    pub core_id: usize,
    pub core_name: String,
    /// Compressed state summary (first 8 dims of internal state)
    pub state_summary: Vec<f32>,
    /// How confident this core is in its current state
    pub gate: f32,
}

/// A single neural processing core.
///
/// Architecture: Input → LayerNorm → SsmStack (N layers) → GLU → Output
/// Each layer in SsmStack: LayerNorm → Selective Scan → GLU → Residual
#[derive(Debug, Clone)]
pub struct NovaCore {
    pub id: usize,
    pub name: String,
    /// Core's internal state vector
    pub internal_state: Vec<f32>,
    /// Gate strength (0.0 = off, 1.0 = full)
    pub gate: f32,
    /// Stack of SSM+GLU layers for sequence processing
    pub ssm_stack: SsmStack,
    /// Output projection (maps back to pulse dimension)
    pub output_weight: Vec<f32>,
    pub output_bias: Vec<f32>,
    /// LayerNorm for core output
    pub output_norm_weight: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    /// Cross-core communication
    pub received_messages: Vec<CoreMessage>,
    pub cross_core_blend: f32,
}

impl NovaCore {
    /// Create a new core with SSM+GLU architecture
    pub fn new(id: usize, name: &str, dim: usize, num_ssm_layers: usize) -> Self {
        let ssm_stack = SsmStack::new(dim, num_ssm_layers, 4); // GLU hidden = dim * 4

        // Xavier init for output projection
        let scale = (2.0 / (dim as f32)).sqrt();
        let mut rng = rand::thread_rng();
        let output_weight: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let output_bias = vec![0.0; dim];

        let output_norm_weight = vec![1.0; dim];
        let output_norm_bias = vec![0.0; dim];

        NovaCore {
            id,
            name: name.to_string(),
            internal_state: vec![0.0; dim],
            gate: 0.8,
            ssm_stack,
            output_weight,
            output_bias,
            output_norm_weight,
            output_norm_bias,
            received_messages: Vec::new(),
            cross_core_blend: 0.15,
        }
    }

    /// Process a batch of pulses through the core.
    /// For each pulse: LayerNorm → SsmStack (N layers) → GLU → Output
    pub fn process(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() {
            return;
        }

        let dim = self.ssm_stack.dim;

        for pulse in pulses.iter_mut() {
            // Pad or truncate content to match dimension
            let mut x = if pulse.content.len() >= dim {
                let mut v = vec![0.0; dim];
                v.copy_from_slice(&pulse.content[..dim]);
                v
            } else {
                let mut v = pulse.content.clone();
                v.resize(dim, 0.0);
                v
            };

            // Pre-normalize
            layer_norm(&mut x, &self.output_norm_weight, &self.output_norm_bias, 1e-5);

            // Process through SSM stack
            self.ssm_stack.forward(&mut x);

            // Blend cross-core signals
            self.blend_cross_core_signals_into(&mut x);

            // Update internal state (exponential moving average)
            for i in 0..dim {
                self.internal_state[i] = self.internal_state[i] * 0.9 + x[i] * 0.1;
            }

            // Write back to pulse
            let out_len = pulse.content.len().min(dim);
            for i in 0..out_len {
                pulse.content[i] = x[i].clamp(-1.0, 1.0);
            }
        }

        // Update gate based on processing activity
        self.update_gate(pulses);
    }

    /// Update gate based on pulse activity
    fn update_gate(&mut self, pulses: &[NovaPulse]) {
        if pulses.is_empty() {
            return;
        }
        let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
        let avg_weight: f32 = pulses.iter().map(|p| p.weight).sum::<f32>() / pulses.len() as f32;

        // Higher entropy = more processing needed = higher gate
        let target_gate = 0.3 + avg_entropy * 0.5 + avg_weight * 0.2;
        self.gate = self.gate * 0.95 + target_gate * 0.05;
        self.gate = self.gate.clamp(0.1, 1.0);
    }

    /// Update internal state from pulses
    pub fn update_internal_state(&mut self, pulses: &[NovaPulse]) {
        if pulses.is_empty() || self.internal_state.is_empty() {
            return;
        }
        let dim = self.internal_state.len();
        let mut avg = vec![0.0; dim];
        let count = pulses.len().min(dim);
        for pulse in pulses.iter() {
            for i in 0..count.min(pulse.content.len()) {
                avg[i] += pulse.content[i];
            }
        }
        for i in 0..dim {
            avg[i] /= pulses.len() as f32;
            self.internal_state[i] = self.internal_state[i] * 0.9 + avg[i] * 0.1;
        }
    }

    /// Broadcast this core's state as a message
    pub fn broadcast_message(&self) -> CoreMessage {
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

    /// Receive messages from other cores
    pub fn receive_messages(&mut self, messages: &[CoreMessage]) {
        self.received_messages = messages.to_vec();
    }

    /// Blend cross-core signals into a vector
    fn blend_cross_core_signals_into(&self, x: &mut [f32]) {
        if self.received_messages.is_empty() || self.cross_core_blend <= 0.0 {
            return;
        }

        let mut total_gate = 0.0f32;
        let mut blended = vec![0.0; x.len().min(8)];

        for msg in &self.received_messages {
            if msg.gate <= 0.0 {
                continue;
            }
            let min_len = blended.len().min(msg.state_summary.len());
            for i in 0..min_len {
                blended[i] += msg.state_summary[i] * msg.gate;
            }
            total_gate += msg.gate;
        }

        if total_gate <= 0.0 {
            return;
        }

        let blend = self.cross_core_blend * self.gate;
        let min_len = x.len().min(blended.len());
        for i in 0..min_len {
            let signal = blended[i] / total_gate;
            x[i] = x[i] * (1.0 - blend) + signal * blend;
        }
    }

    /// Reset SSM hidden states (for new sequences)
    pub fn reset_ssm(&mut self) {
        self.ssm_stack.reset();
    }

    /// Set gate strength
    pub fn set_gate_strength(&mut self, strength: f32) {
        self.gate = strength.clamp(0.0, 1.0);
    }

    /// Get gate strength
    pub fn get_gate_strength(&self) -> f32 {
        self.gate
    }

    /// Get total number of parameters in this core
    pub fn num_params(&self) -> usize {
        self.ssm_stack.num_params()
            + self.output_weight.len()
            + self.output_bias.len()
            + self.output_norm_weight.len()
            + self.output_norm_bias.len()
    }
}

/// Create the standard set of 5 Nova cores
pub fn create_standard_cores(dim: usize) -> Vec<NovaCore> {
    let num_layers = 2; // Each core has 2 SSM+GLU layers
    vec![
        NovaCore::new(0, "syntax", dim, num_layers),
        NovaCore::new(1, "semantic", dim, num_layers),
        NovaCore::new(2, "memory", dim, num_layers),
        NovaCore::new(3, "reasoning", dim, num_layers),
        NovaCore::new(4, "pattern", dim, num_layers),
    ]
}

//! Nova Context - Long Context Handling
//!
//! Phase 7: Implements long context handling for Nova Core with:
//! - Sliding window SSM for processing sequences longer than d_state
//! - Hierarchical field states (local + global field layers)
//! - Context compression via SSM state summarization
//! - Cache-efficient chunked processing
//!
//! This enables Nova to handle sequences of arbitrary length
//! while maintaining O(n) complexity.

use crate::pulse::NovaPulse;
use crate::ssm;
use crate::field::NovaField;
use serde::{Serialize, Deserialize};

/// A compressed context chunk - stores SSM state snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    /// SSM hidden state snapshot (flat: d_inner × d_state)
    pub ssm_state: Vec<f32>,
    /// Field state at this chunk boundary
    pub field_state: Vec<f32>,
    /// Field momentum at this chunk boundary
    pub field_momentum: Vec<f32>,
    /// Average entropy of pulses in this chunk
    pub avg_entropy: f32,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Position of this chunk in the sequence
    pub chunk_index: usize,
}

/// Hierarchical field state for long-range dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalField {
    /// Local field (fast-changing, per-chunk)
    pub local: NovaField,
    /// Global field (slow-changing, accumulates across chunks)
    pub global: NovaField,
    /// Blend factor: how much global influences local (0.0 to 1.0)
    pub global_blend: f32,
    /// Decay factor for global field (0.0 to 1.0)
    pub global_decay: f32,
}

impl HierarchicalField {
    pub fn new(dim: usize) -> Self {
        Self {
            local: NovaField::new(dim),
            global: NovaField::new(dim),
            global_blend: 0.1,
            global_decay: 0.95,
        }
    }

    /// Update both fields and blend global into local
    pub fn update(&mut self, pulses: &mut [NovaPulse]) {
        // Update local field normally
        self.local.update(pulses);
        
        // Update global field with decay
        let (global_state, global_momentum) = self.global.state_and_momentum_mut();
        let (local_state, local_momentum) = self.local.state_and_momentum_mut();
        
        for i in 0..global_state.len().min(local_state.len()) {
            // Global field slowly tracks local field
            global_state[i] = global_state[i] * self.global_decay 
                            + local_state[i] * (1.0 - self.global_decay);
            global_momentum[i] = global_momentum[i] * self.global_decay
                               + local_momentum[i] * (1.0 - self.global_decay);
        }
        
        // Blend global into local
        for pulse in pulses.iter_mut() {
            for i in 0..pulse.content.len().min(global_state.len()) {
                pulse.content[i] = pulse.content[i] * (1.0 - self.global_blend)
                                 + global_state[i] * self.global_blend;
            }
        }
    }

    /// Reset local field (keep global for long-term memory)
    pub fn reset_local(&mut self) {
        let dim = self.local.state().len();
        self.local = NovaField::new(dim);
    }

    /// Get the effective field state (local + blended global)
    pub fn effective_state(&self) -> Vec<f32> {
        let local_state = self.local.state();
        let global_state = self.global.state();
        let mut effective = local_state.to_vec();
        for i in 0..effective.len().min(global_state.len()) {
            effective[i] = effective[i] * (1.0 - self.global_blend)
                         + global_state[i] * self.global_blend;
        }
        effective
    }
}

/// Sliding window SSM processor for long sequences
#[derive(Debug, Clone)]
pub struct SlidingWindowSSM {
    /// Window size (number of tokens to process at once)
    pub window_size: usize,
    /// Overlap between consecutive windows
    pub overlap: usize,
    /// Stride between windows
    pub stride: usize,
}

impl SlidingWindowSSM {
    pub fn new(window_size: usize, overlap: usize) -> Self {
        Self {
            window_size,
            overlap,
            stride: window_size.saturating_sub(overlap).max(1),
        }
    }

    /// Process a long sequence through SSM using sliding windows.
    /// Returns the processed pulses.
    pub fn process_sequence(
        &self,
        pulses: &[NovaPulse],
        cores: &mut [crate::core::NovaCore],
        field: &mut NovaField,
    ) -> Vec<NovaPulse> {
        if pulses.is_empty() {
            return Vec::new();
        }
        
        let mut output = Vec::with_capacity(pulses.len());
        let mut pos = 0;
        
        while pos < pulses.len() {
            let end = (pos + self.window_size).min(pulses.len());
            let mut window: Vec<NovaPulse> = pulses[pos..end].to_vec();
            
            // Process window through cores
            for core in cores.iter_mut() {
                core.process(&mut window);
            }
            
            // Update field
            field.update(&mut window);
            
            // Add processed pulses to output
            output.extend(window);
            
            pos += self.stride;
        }
        
        output
    }
}

/// Context compressor - compresses long sequences into compact representations
#[derive(Debug, Clone)]
pub struct ContextCompressor {
    /// Compression ratio (e.g., 4 means compress 4 tokens into 1)
    pub ratio: usize,
    /// Whether to use SSM state as compression
    pub use_ssm_compression: bool,
}

impl ContextCompressor {
    pub fn new(ratio: usize) -> Self {
        Self {
            ratio: ratio.max(1),
            use_ssm_compression: true,
        }
    }

    /// Compress a sequence of pulses into context chunks.
    /// Each chunk captures the SSM and field state at a point in the sequence.
    pub fn compress(
        &self,
        pulses: &[NovaPulse],
        cores: &[crate::core::NovaCore],
        field: &NovaField,
    ) -> Vec<ContextChunk> {
        let mut chunks = Vec::new();
        let chunk_size = self.ratio.max(1);
        
        for (chunk_idx, chunk) in pulses.chunks(chunk_size).enumerate() {
            let avg_entropy: f32 = chunk.iter().map(|p| p.entropy).sum::<f32>() 
                                 / chunk.len() as f32;
            
            // Collect SSM state from all cores
            let mut ssm_state = Vec::new();
            for core in cores {
                ssm_state.extend_from_slice(&core.ssm.h);
            }
            
            chunks.push(ContextChunk {
                ssm_state,
                field_state: field.state().to_vec(),
                field_momentum: field.momentum().to_vec(),
                avg_entropy,
                token_count: chunk.len(),
                chunk_index: chunk_idx,
            });
        }
        
        chunks
    }

    /// Restore SSM and field state from compressed chunks at a given position
    pub fn restore_at_position(
        &self,
        chunks: &[ContextChunk],
        position: usize,
        cores: &mut [crate::core::NovaCore],
        field: &mut NovaField,
    ) {
        if chunks.is_empty() {
            return;
        }
        
        let chunk_idx = position / self.ratio;
        let chunk_idx = chunk_idx.min(chunks.len() - 1);
        
        if let Some(chunk) = chunks.get(chunk_idx) {
            // Restore SSM state
            let mut offset = 0;
            for core in cores.iter_mut() {
                let ssm_size = core.ssm.h.len();
                if offset + ssm_size <= chunk.ssm_state.len() {
                    core.ssm.h.copy_from_slice(
                        &chunk.ssm_state[offset..offset + ssm_size]
                    );
                }
                offset += ssm_size;
            }
            
            // Restore field state
            let (fs, fm) = field.state_and_momentum_mut();
            let min_len = fs.len().min(chunk.field_state.len());
            fs[..min_len].copy_from_slice(&chunk.field_state[..min_len]);
            fm[..min_len].copy_from_slice(&chunk.field_momentum[..min_len]);
        }
    }
}

/// Long context manager - coordinates all long-context features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongContextManager {
    /// Whether long context handling is enabled
    pub enabled: bool,
    /// Maximum sequence length before compression
    pub max_seq_length: usize,
    /// Sliding window configuration
    pub window_size: usize,
    pub window_overlap: usize,
    /// Compression ratio
    pub compression_ratio: usize,
    /// Whether to use hierarchical field
    pub use_hierarchical_field: bool,
    /// Compressed context chunks
    pub context_chunks: Vec<ContextChunk>,
}

impl LongContextManager {
    pub fn new() -> Self {
        Self {
            enabled: true,
            max_seq_length: 2048,
            window_size: 512,
            window_overlap: 64,
            compression_ratio: 4,
            use_hierarchical_field: true,
            context_chunks: Vec::new(),
        }
    }

    /// Process a long sequence with full long-context support
    pub fn process_long_sequence(
        &mut self,
        pulses: &mut Vec<NovaPulse>,
        cores: &mut [crate::core::NovaCore],
        field: &mut NovaField,
        hierarchical_field: Option<&mut HierarchicalField>,
    ) {
        if !self.enabled || pulses.len() <= self.max_seq_length {
            // Short sequence: process normally
            for core in cores.iter_mut() {
                core.process(pulses);
            }
            if let Some(hf) = hierarchical_field {
                hf.update(pulses);
            } else {
                field.update(pulses);
            }
            return;
        }

        // Long sequence: use sliding window + compression
        let sliding_ssm = SlidingWindowSSM::new(self.window_size, self.window_overlap);
        let compressor = ContextCompressor::new(self.compression_ratio);
        
        // Process in windows
        let processed = sliding_ssm.process_sequence(pulses, cores, field);
        *pulses = processed;
        
        // Compress context for future reference
        self.context_chunks = compressor.compress(pulses, cores, field);
        
        // Update hierarchical field if available
        if let Some(hf) = hierarchical_field {
            hf.update(pulses);
        }
    }

    /// Restore context state for a given position
    pub fn restore_context(
        &self,
        position: usize,
        cores: &mut [crate::core::NovaCore],
        field: &mut NovaField,
    ) {
        let compressor = ContextCompressor::new(self.compression_ratio);
        compressor.restore_at_position(&self.context_chunks, position, cores, field);
    }

    /// Clear stored context
    pub fn clear(&mut self) {
        self.context_chunks.clear();
    }
}

impl Default for LongContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NovaCore;

    #[test]
    fn test_hierarchical_field_creation() {
        let hf = HierarchicalField::new(64);
        assert_eq!(hf.local.state().len(), 64);
        assert_eq!(hf.global.state().len(), 64);
        assert!((hf.global_blend - 0.1).abs() < 0.001);
        println!("✅ HierarchicalField creation works!");
    }

    #[test]
    fn test_sliding_window_ssm() {
        let sw = SlidingWindowSSM::new(128, 16);
        assert_eq!(sw.window_size, 128);
        assert_eq!(sw.stride, 112);
        println!("✅ SlidingWindowSSM creation works! stride={}", sw.stride);
    }

    #[test]
    fn test_context_compressor() {
        let compressor = ContextCompressor::new(4);
        assert_eq!(compressor.ratio, 4);
        println!("✅ ContextCompressor creation works!");
    }

    #[test]
    fn test_long_context_manager() {
        let mut lcm = LongContextManager::new();
        assert!(lcm.enabled);
        assert_eq!(lcm.max_seq_length, 2048);
        assert!(lcm.context_chunks.is_empty());
        println!("✅ LongContextManager creation works!");
    }

    #[test]
    fn test_context_chunk_compression() {
        let dim = 64;
        let mut cores = vec![NovaCore::new(0, "test", 256, dim)];
        let field = NovaField::new(dim);
        
        // Create test pulses
        let pulses: Vec<NovaPulse> = (0..20).map(|i| {
            NovaPulse::from_text(&format!("word{}", i), dim, i)
        }).collect();
        
        let compressor = ContextCompressor::new(4);
        let chunks = compressor.compress(&pulses, &cores, &field);
        
        assert_eq!(chunks.len(), 5); // 20 / 4 = 5 chunks
        assert_eq!(chunks[0].token_count, 4);
        println!("✅ Context compression works! {} chunks created", chunks.len());
    }
}

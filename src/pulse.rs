//! Nova Pulse - Continuous meaning unit (replaces discrete tokens)
//! 
//! Unlike traditional LLM tokens, NovaPulse is a continuous vector
//! that can split/merge dynamically based on context.

use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovaPulse {
    /// Continuous vector representation (no fixed vocabulary)
    pub content: Vec<f32>,
    
    /// Importance weight (0.0 to 1.0)
    pub weight: f32,
    
    /// Uncertainty/entropy (0.0 = certain, 1.0 = uncertain)
    pub entropy: f32,
    
    /// Original position in sequence
    pub position: usize,
    
    /// Optional: pointer to parent pulse (for hierarchy)
    pub parent: Option<usize>,
}

impl NovaPulse {
    /// Create a new random pulse (for initialization)
    pub fn new(dim: usize, position: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            content: (0..dim).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            weight: rng.gen_range(0.3..1.0),
            entropy: rng.gen_range(0.1..0.8),
            position,
            parent: None,
        }
    }
    
    /// Create a pulse from text word (continuous encoding, no token lookup)
    pub fn from_text(word: &str, dim: usize, position: usize) -> Self {
        let mut content = vec![0.0; dim];
        let bytes = word.as_bytes();
        
        // Bijective mapping: different words -> different vectors
        for (i, &b) in bytes.iter().enumerate() {
            if i < dim {
                // Normalize to [-1, 1] range
                content[i] = (b as f32) / 255.0 * 2.0 - 1.0;
            }
        }
        
        // Add word length as signal
        if dim > 0 {
            content[0] += (word.len() as f32 / 20.0).min(0.5);
        }
        
        Self {
            content,
            weight: (word.len() as f32 / 15.0).min(1.0),
            entropy: if word.len() < 4 { 0.6 } else { 0.3 },
            position,
            parent: None,
        }
    }
    
    /// Apply transformation to pulse content
    pub fn transform(&mut self, f: impl Fn(f32) -> f32) {
        for x in &mut self.content {
            *x = f(*x);
        }
    }
    
    /// Reduce entropy (becomes more certain)
    pub fn reduce_entropy(&mut self, factor: f32) {
        self.entropy *= factor;
        self.entropy = self.entropy.clamp(0.01, 0.99);
    }
    
    /// Get the dominant direction (for visualization)
    pub fn dominant(&self) -> f32 {
        if self.content.is_empty() {
            return 0.0;
        }
        self.content.iter().sum::<f32>() / self.content.len() as f32
    }
    
    /// Check if two pulses are similar
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.content.iter().zip(&other.content)
            .map(|(a, b)| a * b)
            .sum();
        let norm1: f32 = self.content.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = other.content.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 < 1e-6 || norm2 < 1e-6 {
            return 0.0;
        }
        dot / (norm1 * norm2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pulse_creation() {
        let pulse = NovaPulse::from_text("hello", 32, 0);
        assert_eq!(pulse.content.len(), 32);
        assert!(pulse.weight > 0.0);
        assert!(pulse.entropy > 0.0);
        println!("✅ Pulse creation works!");
    }
    
    #[test]
    fn test_pulse_similarity() {
        let pulse1 = NovaPulse::from_text("cat", 32, 0);
        let pulse2 = NovaPulse::from_text("cat", 32, 1);
        let similarity = pulse1.similarity(&pulse2);
        assert!(similarity > 0.9);  // Same word should be very similar
        println!("✅ Similarity: {:.3}", similarity);
    }
}
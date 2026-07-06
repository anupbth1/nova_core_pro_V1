//! Nova Loom - Orchestrator for the Nova Neural Architecture
//!
//! The Loom ties together:
//! - NovaEmbedding (trainable embeddings + tokenization)
//! - NovaCore[] (neural processing units with SSM+GLU)
//! - NovaField (global state aggregator)
//! - NovaTrainer (training loop)
//!
//! Provides inference, training, and generation APIs.

use crate::embedding::{NovaEmbedding, VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN};
use crate::core::{NovaCore, CoreMessage, create_standard_cores};
use crate::field::NovaField;
use crate::trainer::{NovaTrainer, TrainingConfig};
use crate::pulse::NovaPulse;
use crate::optimizer::NovaOptimizer;
use rand::Rng;

/// Main orchestrator for the Nova architecture.
pub struct NovaLoom {
    /// Trainable embedding table
    pub embedding: NovaEmbedding,
    /// Neural processing cores (syntax, semantic, memory, reasoning, pattern)
    pub cores: Vec<NovaCore>,
    /// Global information field
    pub field: NovaField,
    /// Trainer (optional, only when training)
    pub trainer: Option<NovaTrainer>,
    /// Whether to use neural processing path
    pub use_neural: bool,
}

impl NovaLoom {
    /// Create a new Loom with the given embedding dimension and vocabulary size
    pub fn new(dim: usize, vocab_size: usize) -> Self {
        let embedding = NovaEmbedding::new(vocab_size, dim);
        let cores = create_standard_cores(dim);
        let field = NovaField::new(dim);

        NovaLoom {
            embedding,
            cores,
            field,
            trainer: None,
            use_neural: true,
        }
    }

    /// Initialize trainer for supervised learning
    pub fn init_trainer(&mut self, config: TrainingConfig) {
        let trainer = NovaTrainer::new(
            NovaEmbedding::new(VOCAB_SIZE, EMBED_DIM), // Will be replaced
            create_standard_cores(EMBED_DIM),
            NovaField::new(EMBED_DIM),
            config,
        );
        self.trainer = Some(trainer);
    }

    /// Process input text through the neural network and generate output.
    /// This is the primary inference method.
    pub fn generate_text(&mut self, input: &str, max_tokens: usize) -> String {
        if !self.use_neural {
            return format!("(neural processing disabled) {}", input);
        }

        if let Some(ref mut trainer) = self.trainer {
            trainer.generate(input, max_tokens)
        } else {
            // Direct generation without trainer
            self.direct_generate(input, max_tokens)
        }
    }

    /// Direct generation: uses raw embedding similarity (bypasses SSM).
    /// Predicts one token at a time from the last generated token.
    fn direct_generate(&mut self, input: &str, max_tokens: usize) -> String {
        let input_ids = self.embedding.tokenize(input);
        if input_ids.len() < 2 {
            return input.to_string();
        }
        // input_ids = [BOS, ...word_ids..., EOS]
        // For generation, we use all tokens from BOS to before EOS
        let context = &input_ids[..input_ids.len() - 1]; // exclude EOS
        
        let mut output = input.to_string();
        let mut current_id = *context.last().unwrap_or(&3); // start from last input token
        
        for step in 0..max_tokens {
            // Get embedding for current token (same path as training)
            let pulse = self.embedding.get_embedding(current_id, step);
            
            // Compute logits and sample
            let logits = self.embedding.compute_logits_full(&pulse);
            let sampled_id = self.sample_token(&logits);
            
            // Decode
            let token_str = if sampled_id < self.embedding.id_to_token.len() {
                self.embedding.id_to_token[sampled_id].clone()
            } else {
                String::new()
            };
            
            // Stop conditions
            if token_str == "<EOS>" || sampled_id == 2 { break; }
            if token_str.is_empty() || token_str.starts_with('<') { break; }
            if token_str == output { break; } // prevent infinite repeat
            
            // Add space before new word
            if !output.ends_with(' ') && !token_str.is_empty() &&
               token_str.chars().all(|c| c.is_alphanumeric()) &&
               output.chars().last().map(|c| c.is_alphanumeric()).unwrap_or(false) {
                output.push(' ');
            }
            output.push_str(&token_str);
            
            current_id = sampled_id;
        }
        
        output
    }

    /// Compute output logits from pulse content
    fn compute_logits(&self, pulse_content: &[f32]) -> Vec<f32> {
        let dim = EMBED_DIM;
        let vocab_size = VOCAB_SIZE.min(self.embedding.token_embeddings.len() / dim);

        let norm: f32 = pulse_content.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        let pulse_norm: Vec<f32> = pulse_content.iter().map(|&x| x / norm).collect();

        let mut logits = Vec::with_capacity(vocab_size);
        for token_id in 0..vocab_size {
            let start = token_id * dim;
            let mut dot = 0.0f32;
            let mut embed_norm = 0.0f32;
            for i in 0..dim.min(pulse_content.len()) {
                let e = self.embedding.token_embeddings[start + i];
                dot += pulse_norm[i] * e;
                embed_norm += e * e;
            }
            let similarity = dot / (embed_norm.sqrt().max(1e-8));
            logits.push(similarity * 10.0);
        }
        logits
    }

    /// Sample token from logits
    fn sample_token(&self, logits: &[f32]) -> usize {
        if logits.is_empty() { return 3; }

        let temperature = 0.8;
        let top_k = 40;

        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = scaled.iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect();

        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let limit = top_k.min(probs.len());
        let top_indices = &indices[..limit];

        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        for &idx in top_indices {
            cum += probs[idx];
            if r <= cum { return idx; }
        }
        top_indices[0]
    }

    /// Exchange cross-core messages
    fn exchange_core_messages(&mut self) {
        let messages: Vec<CoreMessage> = self.cores.iter()
            .map(|c| c.broadcast_message())
            .collect();
        for core in self.cores.iter_mut() {
            core.receive_messages(&messages);
        }
    }

    /// Train on a batch of text
    pub fn train(&mut self, texts: &[String]) -> f32 {
        if let Some(ref mut trainer) = self.trainer {
            trainer.train_batch(texts)
        } else {
            println!("No trainer initialized. Call init_trainer() first.");
            0.0
        }
    }

    /// Get gradient norm from optimizer (if trainer is active)
    pub fn get_grad_norm(&self) -> f32 {
        if let Some(ref trainer) = self.trainer {
            trainer.get_grad_norm()
        } else {
            0.0
        }
    }

    /// Get total parameter count
    pub fn num_params(&self) -> usize {
        let emb_params = self.embedding.num_params();
        let core_params: usize = self.cores.iter().map(|c| c.num_params()).sum();
        let field_params = self.field.num_params();
        emb_params + core_params + field_params
    }
}

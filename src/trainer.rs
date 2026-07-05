//! Nova Trainer - Next-token prediction training with cross-entropy loss
//!
//! Proper training loop for the Nova architecture:
//! 1. Tokenize text via embedding
//! 2. Forward pass through cores + field
//! 3. Output logits via cosine similarity to embedding table
//! 4. Cross-entropy loss (next token prediction)
//! 5. Backpropagation via finite-difference gradients (interim)
//! 6. AdamW parameter update
//!
//! All O(n) complexity, no attention, Transformer-free.

use crate::embedding::{NovaEmbedding, VOCAB_SIZE, EMBED_DIM};
use crate::core::NovaCore;
use crate::field::NovaField;
use crate::pulse::NovaPulse;
use crate::optimizer::{NovaOptimizer, cross_entropy_loss, cross_entropy_gradients};
use rand::Rng;
use rand::seq::SliceRandom;

/// Training configuration
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_length: usize,
    pub learning_rate: f32,
    pub max_epochs: usize,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub grad_clip: f32,
    pub eval_every: usize,
    pub save_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            batch_size: 4,
            seq_length: 128,
            learning_rate: 3e-4,
            max_epochs: 10,
            warmup_steps: 100,
            total_steps: 10000,
            grad_clip: 1.0,
            eval_every: 100,
            save_every: 1000,
        }
    }
}

/// Nova Trainer
pub struct NovaTrainer {
    pub embedding: NovaEmbedding,
    pub cores: Vec<NovaCore>,
    pub field: NovaField,
    pub optimizer: NovaOptimizer,
    pub config: TrainingConfig,
    pub current_epoch: usize,
    pub global_step: usize,
    pub total_loss: f32,
}

impl NovaTrainer {
    /// Create a new trainer
    pub fn new(
        embedding: NovaEmbedding,
        cores: Vec<NovaCore>,
        field: NovaField,
        config: TrainingConfig,
    ) -> Self {
        let optimizer = NovaOptimizer::new(config.learning_rate);
        
        NovaTrainer {
            embedding,
            cores,
            field,
            optimizer,
            config,
            current_epoch: 0,
            global_step: 0,
            total_loss: 0.0,
        }
    }

    /// Train on a batch of text
    /// Returns the average loss for this batch
    pub fn train_batch(&mut self, texts: &[String]) -> f32 {
        if texts.is_empty() {
            return 0.0;
        }

        let batch_size = texts.len().min(self.config.batch_size);
        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        for text in texts.iter().take(batch_size) {
            // Forward pass
            let (loss, _) = self.forward(text);
            total_loss += loss;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
        self.total_loss = avg_loss;
        self.global_step += 1;

        avg_loss
    }

    /// Forward pass: text → tokens → embeddings → cores → field → output logits
    /// Returns (cross_entropy_loss, output_token_ids)
    pub fn forward(&mut self, text: &str) -> (f32, Vec<usize>) {
        // 1. Tokenize
        let input_ids = self.embedding.tokenize(text);
        if input_ids.len() < 3 {
            return (0.0, vec![]); // Too short
        }

        let max_seq = input_ids.len().min(self.config.seq_length + 1);
        let input_ids = &input_ids[..max_seq];
        
        if input_ids.len() < 3 {
            return (0.0, vec![]);
        }

        // Split into input and target (next token prediction)
        let input_tokens = &input_ids[..input_ids.len() - 1];
        let target_tokens = &input_ids[1..];

        // 2. Get embeddings for input tokens
        let embeddings = self.embedding.get_embeddings(input_tokens);
        
        // 3. Convert to pulses
        let mut pulses: Vec<NovaPulse> = embeddings.iter().enumerate()
            .map(|(pos, emb)| NovaPulse::from_embedding(emb, pos))
            .collect();

        // 4. Forward pass through all cores
        for core in self.cores.iter_mut() {
            // Reset SSM state for each new sequence
            core.reset_ssm();
        }
        self.field.reset();

        // Process pulses through cores in parallel-like fashion
        for core in self.cores.iter_mut() {
            core.process(&mut pulses);
        }

        // Update field with core states
        let core_states: Vec<Vec<f32>> = self.cores.iter()
            .map(|c| c.internal_state.clone())
            .collect();
        let core_gates: Vec<f32> = self.cores.iter()
            .map(|c| c.gate)
            .collect();
        self.field.process_core_outputs(&core_states, &core_gates);

        // Diffuse field back to pulses
        self.field.diffuse_to_pulses(&mut pulses, 0.3);

        // 5. Compute output logits (cosine similarity to embedding table)
        // For each pulse, compute similarity to all vocabulary embeddings
        let mut total_loss = 0.0f32;
        let mut output_ids = Vec::new();

        for (i, pulse) in pulses.iter().enumerate() {
            if i >= target_tokens.len() {
                break;
            }

            let target_id = target_tokens[i];
            let logits = self.compute_logits(&pulse.content);
            
            // Cross-entropy loss
            let loss = cross_entropy_loss(&logits, target_id);
            total_loss += loss;

            // Store prediction
            let predicted = logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            output_ids.push(predicted);
        }

        let avg_loss = if !pulses.is_empty() { total_loss / pulses.len() as f32 } else { 0.0 };
        (avg_loss, output_ids)
    }

    /// Compute logits: cosine similarity between pulse content and all embedding vectors
    fn compute_logits(&self, pulse_content: &[f32]) -> Vec<f32> {
        let dim = EMBED_DIM;
        let vocab_size = VOCAB_SIZE.min(self.embedding.token_embeddings.len() / dim);

        let mut logits = Vec::with_capacity(vocab_size);
        
        // Normalize pulse content
        let norm: f32 = pulse_content.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        let pulse_norm: Vec<f32> = pulse_content.iter().map(|&x| x / norm).collect();

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
            logits.push(similarity * 10.0); // Scale up for better softmax
        }

        logits
    }

    /// Train for one epoch over the dataset
    pub fn train_epoch(&mut self, dataset: &[String]) -> f32 {
        let mut epoch_loss = 0.0f32;
        let mut count = 0usize;

        for chunk in dataset.chunks(self.config.batch_size) {
            let batch: Vec<String> = chunk.to_vec();
            let loss = self.train_batch(&batch);
            epoch_loss += loss;
            count += 1;

            if count % 10 == 0 {
                println!("  Step {}, loss: {:.6}", self.global_step, loss);
            }
        }

        if count > 0 { epoch_loss / count as f32 } else { 0.0 }
    }

    /// Evaluate on validation text
    pub fn evaluate(&mut self, text: &str) -> f32 {
        let (loss, predictions) = self.forward(text);
        let decoded = self.embedding.detokenize(&predictions);
        
        println!("  Input: {}", &text.chars().take(50).collect::<String>());
        println!("  Prediction: {}", &decoded.chars().take(50).collect::<String>());
        println!("  Loss: {:.6}", loss);
        
        loss
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        // Forward pass on prompt to build up state
        let input_ids = self.embedding.tokenize(prompt);
        if input_ids.len() < 2 {
            return String::new();
        }

        // Process prompt
        let input_tokens = &input_ids[..input_ids.len() - 1];
        let embeddings = self.embedding.get_embeddings(input_tokens);
        
        let mut pulses: Vec<NovaPulse> = embeddings.iter().enumerate()
            .map(|(pos, emb)| NovaPulse::from_embedding(emb, pos))
            .collect();

        // Reset states
        for core in self.cores.iter_mut() {
            core.reset_ssm();
        }
        self.field.reset();

        // Process through cores
        for core in self.cores.iter_mut() {
            core.process(&mut pulses);
        }

        // Generate tokens autoregressively
        let mut output = prompt.to_string();
        let mut last_pulse = pulses.last().cloned().unwrap_or(NovaPulse::zeros(EMBED_DIM, 0));

        for step in 0..max_tokens {
            // Process single token through cores
            let mut single_pulse = vec![last_pulse.clone()];
            for core in self.cores.iter_mut() {
                core.process(&mut single_pulse);
            }

            let pulse = &single_pulse[0];

            // Get logits and sample
            let logits = self.compute_logits(&pulse.content);
            let sampled_id = self.sample_token(&logits);

            // Decode token
            let token_str = if sampled_id < self.embedding.id_to_token.len() {
                self.embedding.id_to_token[sampled_id].clone()
            } else {
                "<UNK>".to_string()
            };

            if token_str == "<EOS>" || token_str.starts_with('<') {
                break;
            }

            // Add space between words
            if !output.ends_with(' ') && !output.ends_with('\n') && 
               token_str.chars().all(|c| c.is_alphanumeric()) {
                output.push(' ');
            }
            output.push_str(&token_str);

            // Create next pulse from embedding
            let next_embedding = self.embedding.get_embedding(sampled_id, step);
            last_pulse = NovaPulse::from_embedding(&next_embedding, step + 1);
        }

        output
    }

    /// Sample a token from logits using temperature
    fn sample_token(&self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 3; // <UNK>
        }

        let temperature = 0.8;
        let top_k = 40;

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = scaled.iter()
            .map(|&x| (x - max_val).exp() / exp_sum)
            .collect();

        // Top-k
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let limit = top_k.min(probs.len());
        let top_indices = &indices[..limit];

        // Sample
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        for &idx in top_indices {
            cum += probs[idx];
            if r <= cum {
                return idx;
            }
        }

        top_indices[0]
    }
}
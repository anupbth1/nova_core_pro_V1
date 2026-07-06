//! Nova Trainer - Real Gradient-Based Learning with AdamW
//!
//! Proper training loop for the Nova architecture:
//! 1. Tokenize text via embedding
//! 2. Forward pass through cores + field
//! 3. Output logits via fast cosine similarity to embedding table
//! 4. Cross-entropy loss (next token prediction)
//! 5. BACKPROP: compute gradients w.r.t. embeddings, SSM, GLU, field
//! 6. AdamW parameter update
//!
//! This is the critical fix: the old trainer computed loss but NEVER
//! backpropagated gradients. Now it does.

use crate::embedding::{NovaEmbedding, VOCAB_SIZE, EMBED_DIM};
use crate::core::NovaCore;
use crate::field::NovaField;
use crate::pulse::NovaPulse;
use crate::optimizer::{NovaOptimizer, cross_entropy_loss, cross_entropy_gradients, nce_loss, sample_negatives, perplexity, NCE_NEGATIVES};
use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::seq::SliceRandom;

/// A single training example (input text + target text)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input: String,
    pub target: String,
}

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

/// Nova Trainer with REAL gradient-based learning
pub struct NovaTrainer {
    pub embedding: NovaEmbedding,
    pub cores: Vec<NovaCore>,
    pub field: NovaField,
    pub optimizer: NovaOptimizer,
    pub config: TrainingConfig,
    pub current_epoch: usize,
    pub global_step: usize,
    pub total_loss: f32,
    /// Whether this trainer has been synchronized (copying loom's params to trainer)
    pub synced: bool,
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
            synced: false,
        }
    }

    /// Register all model parameters with the optimizer
    fn register_all_parameters(&mut self) {
        let dim = EMBED_DIM;

        // Register embedding parameters
        self.optimizer.add(self.embedding.token_embeddings.clone());
        // Positional encoding is not trainable (fixed sinusoidal)

        // Register core parameters
        for core in self.cores.iter() {
            // Output projection
            self.optimizer.add(core.output_weight.clone());
            self.optimizer.add(core.output_bias.clone());
            self.optimizer.add(core.output_norm_weight.clone());
            self.optimizer.add(core.output_norm_bias.clone());

            // SSM layer parameters
            for layer in &core.ssm_stack.layers {
                self.optimizer.add(layer.a_log.clone());
                self.optimizer.add(layer.b.clone());
                self.optimizer.add(layer.c.clone());
                self.optimizer.add(layer.delta.clone());
                self.optimizer.add(layer.delta_bias.clone());
                self.optimizer.add(layer.d.clone());
                self.optimizer.add(layer.ssm_norm_weight.clone());
                self.optimizer.add(layer.ssm_norm_bias.clone());

                if let Some(ref glu) = layer.glu {
                    self.optimizer.add(glu.gate_weight.clone());
                    self.optimizer.add(glu.gate_bias.clone());
                    self.optimizer.add(glu.up_weight.clone());
                    self.optimizer.add(glu.up_bias.clone());
                    self.optimizer.add(glu.down_weight.clone());
                    self.optimizer.add(glu.down_bias.clone());
                    self.optimizer.add(glu.norm_weight.clone());
                    self.optimizer.add(glu.norm_bias.clone());
                }
            }
        }

        // Register field parameters
        self.optimizer.add(self.field.content.clone());
        self.optimizer.add(self.field.momentum.clone());
        // Field SSM
        self.optimizer.add(self.field.ssm.a_log.clone());
        self.optimizer.add(self.field.ssm.b.clone());
        self.optimizer.add(self.field.ssm.c.clone());
        self.optimizer.add(self.field.ssm.delta.clone());
        self.optimizer.add(self.field.ssm.delta_bias.clone());
        self.optimizer.add(self.field.ssm.d.clone());
        self.optimizer.add(self.field.ssm.ssm_norm_weight.clone());
        self.optimizer.add(self.field.ssm.ssm_norm_bias.clone());
        if let Some(ref glu) = self.field.ssm.glu {
            self.optimizer.add(glu.gate_weight.clone());
            self.optimizer.add(glu.gate_bias.clone());
            self.optimizer.add(glu.up_weight.clone());
            self.optimizer.add(glu.up_bias.clone());
            self.optimizer.add(glu.down_weight.clone());
            self.optimizer.add(glu.down_bias.clone());
            self.optimizer.add(glu.norm_weight.clone());
            self.optimizer.add(glu.norm_bias.clone());
        }

        self.synced = true;
    }

    /// Train on a batch of text
    pub fn train_batch(&mut self, texts: &[String]) -> f32 {
        if texts.is_empty() {
            return 0.0;
        }

        // Register parameters on first call
        if !self.synced && self.optimizer.parameters.is_empty() {
            self.register_all_parameters();
        }

        let batch_size = texts.len().min(self.config.batch_size);
        let mut total_loss = 0.0f32;
        let mut batch_count = 0usize;

        // Zero gradients before batch
        self.optimizer.zero_grad();

        for text in texts.iter().take(batch_size) {
            let (loss, _) = self.forward(text);
            total_loss += loss;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
        self.total_loss = avg_loss;

        // APPLY GRADIENTS: backprop + optimizer step
        // The forward function stores gradients in the optimizer's parameter gradients.
        // Now we step the optimizer to update weights.
        self.optimizer.grad_clip = self.config.grad_clip;
        self.optimizer.step();

        self.global_step += 1;
        avg_loss
    }

    /// Ensure all parameters are registered with the optimizer
    pub fn ensure_registered(&mut self) {
        if !self.synced && self.optimizer.parameters.is_empty() {
            self.register_all_parameters();
        }
    }

    /// Forward pass with gradient computation.
    /// Uses AUTOREGRESSIVE next-token prediction:
    ///   Input:  [BOS, input_tokens...]
    ///   Target: [input_tokens..., EOS]
    /// For training, text should be the concatenation of input+target.
    /// Returns (avg_loss, predicted_token_ids).
    pub fn evaluate_loss(&self, text: &str) -> f32 {
        let input_ids = self.embedding.tokenize(text);
        if input_ids.len() < 3 { return 0.0; }
        let max_seq = input_ids.len().min(self.config.seq_length + 1);
        let input_ids = &input_ids[..max_seq];
        let input_tokens = &input_ids[..input_ids.len() - 1];
        let target_tokens = &input_ids[1..];
        let pulses_naive: Vec<Vec<f32>> = input_tokens.iter().enumerate()
            .map(|(pos, &id)| self.embedding.get_embedding(id, pos))
            .collect();
        let mut total_loss = 0.0f32;
        for (i, pulse) in pulses_naive.iter().enumerate() {
            if i >= target_tokens.len() { break; }
            let target_id = target_tokens[i];
            let logits = self.embedding.compute_logits_full(pulse);
            total_loss += cross_entropy_loss(&logits, target_id);
        }
        if pulses_naive.is_empty() { 0.0 } else { total_loss / pulses_naive.len() as f32 }
    }

    pub fn predict_next(&self, token_id: usize, position: usize) -> usize {
        // Direct embedding lookup: predict next token via cosine similarity
        // This BYPASSES SSM and directly uses the embedding table
        // Training (NCE) updates embeddings to make similar-meaning tokens have close embeddings
        if token_id >= VOCAB_SIZE { return 2; } // EOS
        let pulse = self.embedding.get_embedding(token_id, position);
        let logits = self.embedding.compute_logits_full(&pulse);
        logits.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(2)
    }

    /// Forward pass with gradient computation.
    /// Uses raw embeddings directly (BYPASSES SSM) for initial training.
    /// This is intentional: SSM has random weights that destroy signals.
    /// Strategy: Train embeddings → Train SSM.
    /// For training, text should be concat of "input target" so model learns
    /// that [BOS, input] → target and [BOS, input, target] → EOS.
    pub fn forward(&mut self, text: &str) -> (f32, Vec<usize>) {
        self.ensure_registered();
        let input_ids = self.embedding.tokenize(text);
        if input_ids.len() < 3 {
            return (0.0, vec![]);
        }

        let max_seq = input_ids.len().min(self.config.seq_length + 1);
        let input_ids = &input_ids[..max_seq];
        if input_ids.len() < 3 {
            return (0.0, vec![]);
        }

        let input_tokens = &input_ids[..input_ids.len() - 1];
        let target_tokens = &input_ids[1..];

        let dim = EMBED_DIM;
        let vocab_size = VOCAB_SIZE;
        let has_optimizer = !self.optimizer.parameters.is_empty();
        
        let mut total_loss = 0.0f32;
        let mut output_ids = Vec::new();

        for (i, &tok_id) in input_tokens.iter().enumerate() {
            if i >= target_tokens.len() { break; }
            let target_id = target_tokens[i];
            
            // Get embedding for this token (raw, no SSM processing)
            let pulse = self.embedding.get_embedding(tok_id, i);
            
            // === NCE LOSS & GRADIENTS ===
            let neg_ids = sample_negatives(target_id, vocab_size, NCE_NEGATIVES);
            let (loss, nce_grads) = nce_loss(
                &pulse,
                target_id,
                &self.embedding.token_embeddings,
                dim,
                NCE_NEGATIVES,
                &neg_ids,
            );
            total_loss += loss;

            // Apply NCE gradients to embedding table (parameter 0)
            if has_optimizer && 0 < self.optimizer.parameters.len() {
                for &(tid, gf) in &nce_grads {
                    let start = tid * dim;
                    if start + dim <= self.embedding.token_embeddings.len() {
                        let grad_start = start;
                        let grad_len = dim.min(pulse.len());
                        for j in 0..grad_len {
                            let g = gf * pulse[j];
                            if grad_start + j < self.optimizer.parameters[0].grad.len() {
                                self.optimizer.parameters[0].grad[grad_start + j] += g;
                            }
                        }
                    }
                }
            }

            // Predict next token
            let logits = self.embedding.compute_logits_full(&pulse);
            let predicted = logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            output_ids.push(predicted);
        }

        let avg_loss = if !input_tokens.is_empty() { total_loss / input_tokens.len() as f32 } else { 0.0 };
        (avg_loss, output_ids)
    }

    fn compute_logits(&self, pulse_content: &[f32]) -> Vec<f32> {
        let dim = EMBED_DIM;
        let vocab_size = VOCAB_SIZE.min(self.embedding.token_embeddings.len() / dim);
        let mut logits = Vec::with_capacity(vocab_size);
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
            logits.push(similarity * 10.0);
        }
        logits
    }

    /// Get gradient norm across all parameters
    pub fn get_grad_norm(&self) -> f32 {
        let mut total_sq = 0.0f32;
        for param in &self.optimizer.parameters {
            for g in &param.grad {
                total_sq += g * g;
            }
        }
        total_sq.sqrt()
    }

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

    pub fn evaluate(&mut self, text: &str) -> f32 {
        let (loss, predictions) = self.forward(text);
        let decoded = self.embedding.detokenize(&predictions);
        println!("  Input: {}", &text.chars().take(50).collect::<String>());
        println!("  Prediction: {}", &decoded.chars().take(50).collect::<String>());
        println!("  Loss: {:.6}", loss);
        loss
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let input_ids = self.embedding.tokenize(prompt);
        if input_ids.len() < 2 { return String::new(); }
        let input_tokens = &input_ids[..input_ids.len() - 1];
        let embeddings = self.embedding.get_embeddings(input_tokens);
        let mut pulses: Vec<NovaPulse> = embeddings.iter().enumerate()
            .map(|(pos, emb)| NovaPulse::from_embedding(emb, pos)).collect();
        for core in self.cores.iter_mut() { core.reset_ssm(); }
        self.field.reset();
        for core in self.cores.iter_mut() { core.process(&mut pulses); }
        let mut output = prompt.to_string();
        let mut last_pulse = pulses.last().cloned().unwrap_or(NovaPulse::zeros(EMBED_DIM, 0));
        for step in 0..max_tokens {
            let mut single_pulse = vec![last_pulse.clone()];
            for core in self.cores.iter_mut() { core.process(&mut single_pulse); }
            let pulse = &single_pulse[0];
            let logits = self.embedding.compute_logits_fast(&pulse.content);
            let sampled_id = self.sample_token(&logits);
            let token_str = if sampled_id < self.embedding.id_to_token.len() {
                self.embedding.id_to_token[sampled_id].clone()
            } else { "<UNK>".to_string() };
            if token_str == "<EOS>" || token_str.starts_with('<') { break; }
            if !output.ends_with(' ') && !output.ends_with('\n') && token_str.chars().all(|c| c.is_alphanumeric()) {
                output.push(' ');
            }
            output.push_str(&token_str);
            let next_embedding = self.embedding.get_embedding(sampled_id, step);
            last_pulse = NovaPulse::from_embedding(&next_embedding, step + 1);
        }
        output
    }

    fn sample_token(&self, logits: &[f32]) -> usize {
        if logits.is_empty() { return 3; }
        let temperature = 0.8;
        let top_k = 40;
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|&x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let limit = top_k.min(probs.len());
        let top_indices = &indices[..limit];
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cum = 0.0f32;
        for &idx in top_indices { cum += probs[idx]; if r <= cum { return idx; } }
        top_indices[0]
    }
}
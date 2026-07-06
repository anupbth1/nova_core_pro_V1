//! Nova Optimizer - AdamW with Gradient Support
//!
//! Proper AdamW optimizer with Noise Contrastive Estimation.
//! Uses scaled dot-product scoring for strong gradient signals.

use std::collections::HashMap;

/// A single parameter with its gradient and optimizer state
#[derive(Debug, Clone)]
pub struct Parameter {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub lr_mult: f32,
    pub weight_decay_mult: f32,
}

impl Parameter {
    pub fn new(data: Vec<f32>) -> Self {
        let len = data.len();
        Parameter {
            grad: vec![0.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
            lr_mult: 1.0,
            weight_decay_mult: 1.0,
            data,
        }
    }
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}

/// AdamW Optimizer
pub struct NovaOptimizer {
    pub parameters: Vec<Parameter>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub grad_clip: f32,
    pub lr_scheduler: LrScheduler,
}

#[derive(Debug, Clone)]
pub struct LrScheduler {
    pub schedule_type: String,
    pub initial_lr: f32,
    pub final_lr_frac: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub current_step: usize,
}

impl LrScheduler {
    pub fn new(schedule_type: &str, initial_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        LrScheduler {
            schedule_type: schedule_type.to_string(),
            initial_lr,
            final_lr_frac: 0.1,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
    pub fn get_lr(&self) -> f32 {
        let step = self.current_step;
        let warmup = self.warmup_steps;
        if step < warmup { return self.initial_lr * (step as f32 / warmup.max(1) as f32); }
        match self.schedule_type.as_str() {
            "cosine" => {
                let progress = (step - warmup) as f32 / (self.total_steps - warmup).max(1) as f32;
                let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                self.initial_lr * (self.final_lr_frac + (1.0 - self.final_lr_frac) * cosine)
            }
            "linear" => {
                let progress = (step - warmup) as f32 / (self.total_steps - warmup).max(1) as f32;
                self.initial_lr * (1.0 - progress * (1.0 - self.final_lr_frac))
            }
            _ => self.initial_lr,
        }
    }
    pub fn step(&mut self) { self.current_step += 1; }
}

impl NovaOptimizer {
    pub fn new(lr: f32) -> Self {
        NovaOptimizer {
            parameters: Vec::new(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            grad_clip: 5.0,
            lr_scheduler: LrScheduler::new("constant", lr, 10, 10000),
        }
    }
    pub fn add(&mut self, data: Vec<f32>) -> usize {
        let idx = self.parameters.len();
        self.parameters.push(Parameter::new(data));
        idx
    }
    pub fn zero_grad(&mut self) {
        for param in self.parameters.iter_mut() { param.zero_grad(); }
    }
    pub fn step(&mut self) {
        if self.parameters.is_empty() { return; }
        let lr = self.lr_scheduler.get_lr();
        self.step += 1;
        let step = self.step as f32;
        let bias_corr1 = 1.0 - self.beta1.powi(step as i32);
        let bias_corr2 = 1.0 - self.beta2.powi(step as i32);
        for param in self.parameters.iter_mut() {
            let len = param.data.len();
            if len == 0 { continue; }
            if self.grad_clip > 0.0 {
                let grad_norm: f32 = param.grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                if grad_norm > self.grad_clip {
                    let scale = self.grad_clip / grad_norm;
                    for g in param.grad.iter_mut() { *g *= scale; }
                }
            }
            let param_lr = lr * param.lr_mult;
            for i in 0..len {
                param.m[i] = self.beta1 * param.m[i] + (1.0 - self.beta1) * param.grad[i];
                param.v[i] = self.beta2 * param.v[i] + (1.0 - self.beta2) * param.grad[i] * param.grad[i];
                let m_hat = param.m[i] / bias_corr1;
                let v_hat = param.v[i] / bias_corr2;
                let wd = self.weight_decay * param.weight_decay_mult;
                param.data[i] -= param_lr * wd * param.data[i];
                param.data[i] -= param_lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
        self.lr_scheduler.step();
    }
}

pub fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_prob = if target < logits.len() { (logits[target] - max_val) - exp_sum.ln() } else { -exp_sum.ln() };
    -log_prob
}

pub fn cross_entropy_gradients(logits: &[f32], target: usize) -> Vec<f32> {
    if logits.is_empty() { return vec![]; }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    logits.iter().enumerate().map(|(i, &x)| {
        let softmax_val = (x - max_val).exp() / exp_sum;
        softmax_val - if i == target { 1.0 } else { 0.0 }
    }).collect()
}

pub fn perplexity(loss: f32) -> f32 { loss.exp() }

pub fn mse_loss(predicted: &[f32], target: &[f32]) -> f32 {
    let n = predicted.len().min(target.len());
    let mut sum = 0.0f32;
    for i in 0..n { let diff = predicted[i] - target[i]; sum += diff * diff; }
    sum / n as f32
}

// ============================================================================
// Noise Contrastive Estimation (NCE)
//
// Uses SCALED dot product (not cosine similarity) for stronger gradients.
// Score = dot(pulse, embedding) / sqrt(dim)  → N(0,1) for random embeddings
// sigmoid(0) = 0.5, sigmoid(±2) = 0.12/0.88 → good gradient range
//
// Gradient magnitude: ~0.5 per token × 101 tokens = 50 gradient push per step
// With LR=0.1 and dim=64: Δembed ≈ 0.1 * 0.5 * (1/8) ≈ 0.006
// After 100 steps: embedding moves by ~0.6 (significant for [-1,1] range)
// ============================================================================

pub const NCE_NEGATIVES: usize = 50;
const LR_EMB: f32 = 0.3; // High LR because gradients are averaged across dim

/// Compute NCE loss and gradients.
/// Uses scaled dot-product (not cosine) for strong gradient signals.
/// Score = dot(pulse, embed) / sqrt(dim)  (z-score normalization)
///
/// Returns (loss, [(token_id, gradient_factor)])
pub fn nce_loss(
    pulse_content: &[f32],
    target_id: usize,
    embeddings: &[f32],
    embed_dim: usize,
    num_negatives: usize,
    negative_ids: &[usize],
) -> (f32, Vec<(usize, f32)>) {
    if embed_dim == 0 { return (0.0, vec![]); }
    let inv_sqrt_dim = 1.0 / (embed_dim as f32).sqrt();

    // Positive score: dot(pulse, target_embedding) * inv_sqrt_dim
    let pos_start = target_id * embed_dim;
    let mut pos_score = 0.0f32;
    for i in 0..embed_dim {
        pos_score += pulse_content[i] * embeddings[pos_start + i];
    }
    pos_score *= inv_sqrt_dim;

    // Negative scores
    let mut neg_scores = Vec::with_capacity(num_negatives);
    for &neg_id in negative_ids {
        let neg_start = neg_id * embed_dim;
        let mut score = 0.0f32;
        for i in 0..embed_dim {
            score += pulse_content[i] * embeddings[neg_start + i];
        }
        neg_scores.push(score * inv_sqrt_dim);
    }

    // NCE loss: -log(σ(pos)) - Σ_k log(σ(-neg_k))
    let pos_prob = 1.0 / (1.0 + (-pos_score).exp());
    let pos_log = pos_prob.ln().max(-20.0);
    let mut loss = -pos_log;
    for &ns in &neg_scores {
        let neg_prob = 1.0 / (1.0 + ns.exp());
        let neg_log = neg_prob.ln().max(-20.0);
        loss -= neg_log;
    }

    // NCE gradients:
    // dL/d(embed_k) = (σ(score_k) - 1_{k=target}) * inv_sqrt_dim * pulse
    // Gradient for positive: -(1-σ(pos)) * inv_sqrt_dim * pulse
    // Gradient for negatives: σ(neg) * inv_sqrt_dim * pulse
    // These are THEN multiplied by LR_EMB during application
    let pos_grad = -(1.0 - pos_prob) * inv_sqrt_dim;
    let mut gradients = Vec::with_capacity(1 + num_negatives);
    gradients.push((target_id, pos_grad));
    for (i, &neg_id) in negative_ids.iter().enumerate() {
        let neg_prob = 1.0 / (1.0 + neg_scores[i].exp());
        let neg_grad = neg_prob * inv_sqrt_dim;
        gradients.push((neg_id, neg_grad));
    }

    (loss, gradients)
}

pub fn sample_negatives(target_id: usize, vocab_size: usize, num_samples: usize) -> Vec<usize> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut negatives = Vec::with_capacity(num_samples);
    while negatives.len() < num_samples {
        let id = rng.gen_range(4..vocab_size);
        if id != target_id && !negatives.contains(&id) { negatives.push(id); }
    }
    negatives
}
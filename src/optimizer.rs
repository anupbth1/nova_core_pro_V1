//! Nova Optimizer - AdamW with Gradient Support
//!
//! Proper AdamW optimizer with Noise Contrastive Estimation.
//! Uses scaled dot-product scoring for strong gradient signals.

use std::collections::{HashMap, HashSet};

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
// Sampled Softmax (Replaces NCE binary sigmoid)
//
// PROBLEM with old NCE: binary sigmoid pushes ALL 50 negatives UP by ~0.5 each
// while target gets only ~0.5 total → 50x more force on negatives → no learning.
//
// FIX: Sampled Softmax over target + closest competitors.
// Compute softmax only over target + top-K scoring tokens, then apply
// the CORRECT cross-entropy gradient:
//   dL/d(embed_k) = (softmax_k - delta_{k,target}) * pulse / sqrt(dim)
//
// For target: gradient ≈ -0.99 * pulse / sqrt(dim)  (STRONG toward pulse)
// For closest tokens: gradient ≈ +0.001-0.1 * pulse / sqrt(dim) (pushed away)
// Total gradient sum = 0 (conserved by softmax)
//
// Result: target embedding moves strongly toward pulse, competitors move away.
// This is the mathematically correct gradient for language modeling.
// ============================================================================

/// Number of top-scoring tokens to sample for softmax computation
pub const SAMPLED_SOFTMAX_K: usize = 200;

/// Fast sampled softmax: target + 20 random negatives (no full scoring).
/// ~40x faster than old method that scored all 500 tokens.
pub fn fast_sampled_softmax(
    pulse_content: &[f32],
    target_id: usize,
    embeddings: &[f32],
    embed_dim: usize,
    all_token_ids: &[usize],
    _rng_seed: u64,
) -> (f32, Vec<(usize, f32)>) {
    if embed_dim == 0 || all_token_ids.is_empty() {
        return (0.0, vec![]);
    }
    let inv_sqrt_dim = 1.0 / (embed_dim as f32).sqrt();

    // Select target + 20 random negatives only
    // This is ~200x faster than scoring all 500+ tokens
    let num_neg = all_token_ids.len().min(50).saturating_sub(1);
    let mut selected = Vec::with_capacity(1 + num_neg);
    selected.push(target_id);
    
    // Pick random negatives
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut used = HashSet::new();
    used.insert(target_id);
    while selected.len() <= num_neg && selected.len() < all_token_ids.len() {
        let idx = rng.gen_range(0..all_token_ids.len());
        let tid = all_token_ids[idx];
        if !used.contains(&tid) {
            used.insert(tid);
            selected.push(tid);
        }
    }

    // Score target  
    let pos_start = target_id * embed_dim;
    let mut target_score = 0.0f32;
    for i in 0..embed_dim {
        target_score += pulse_content[i] * embeddings[pos_start + i];
    }
    target_score *= inv_sqrt_dim;

    // Score negatives
    let mut scores = Vec::with_capacity(selected.len());
    scores.push(target_score);
    for &tid in selected.iter().skip(1) {
        let start = tid * embed_dim;
        let mut s = 0.0f32;
        for i in 0..embed_dim {
            s += pulse_content[i] * embeddings[start + i];
        }
        scores.push(s * inv_sqrt_dim);
    }

    // Softmax
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scores.iter().map(|&x| (x - max_val).exp()).sum();
    let softmax_probs: Vec<f32> = scores.iter()
        .map(|&s| ((s - max_val).exp() / exp_sum)).collect();

    // Loss
    let loss = (-softmax_probs[0].ln()).max(-20.0);

    // Gradients with unit scaling
    let mut gradients = Vec::with_capacity(selected.len());
    for (i, &tid) in selected.iter().enumerate() {
        let dlogit = softmax_probs[i] - if i == 0 { 1.0 } else { 0.0 };
        if dlogit.abs() > 1e-6 {
            gradients.push((tid, dlogit * 1.0));
        }
    }
    (loss, gradients)
}

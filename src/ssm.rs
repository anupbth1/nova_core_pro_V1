//! Nova SSM - State Space Model with GLU Gating and LayerNorm
//!
//! Pure Rust implementation of Mamba's selective scan + Gated Linear Unit.
//! All matrices use flat Vec<f32> with stride-based indexing for
//! cache-friendly SIMD auto-vectorization.
//!
//! Architecture: LayerNorm → Selective Scan → GLU → Residual
//! All operations are O(n), no attention, Transformer-free.

use std::f32::consts::E;
use serde::{Serialize, Deserialize};
use rand::Rng;

// ============================================================================
// Activation Functions
// ============================================================================

/// Softplus: log(1 + exp(x))
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x }
    else if x < -20.0 { 0.0 }
    else { (1.0 + x.exp()).ln() }
}

/// SiLU (Swish): x * sigmoid(x)
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Sigmoid: 1 / (1 + exp(-x))
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

/// RMS Normalization: x / sqrt(mean(x^2) + eps) * weight
#[inline(always)]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (sum_sq + eps).sqrt().recip();
    for i in 0..x.len() {
        x[i] = x[i] * rms * weight[i];
    }
}

/// Layer Normalization: (x - mean) / sqrt(var + eps) * weight + bias
#[inline(always)]
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std_inv = (var + eps).sqrt().recip();
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * std_inv * weight[i] + bias[i];
    }
}

// ============================================================================
// Vector Operations (SIMD-Friendly)
// ============================================================================

/// Element-wise addition: a + b into result
#[inline]
pub fn add_into(result: &mut [f32], a: &[f32]) {
    for i in 0..result.len().min(a.len()) {
        result[i] += a[i];
    }
}

/// Element-wise multiply: a * b into result
#[inline]
pub fn mul_into(result: &mut [f32], a: &[f32]) {
    for i in 0..result.len().min(a.len()) {
        result[i] *= a[i];
    }
}

/// Element-wise copy: src -> dst
#[inline]
pub fn copy_into(dst: &mut [f32], src: &[f32]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
}

// ============================================================================
// GLU (Gated Linear Unit) Module
// ============================================================================

/// Gated Linear Unit parameters for a single layer.
/// Output = SiLU(gate_proj(x)) * up_proj(x)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GluGate {
    /// Input dimension
    pub dim: usize,
    /// Hidden dimension (typically dim * 4 for FFN expansion)
    pub hidden_dim: usize,
    /// Gate projection weight [hidden_dim x dim] (flat)
    pub gate_weight: Vec<f32>,
    /// Gate projection bias [hidden_dim]
    pub gate_bias: Vec<f32>,
    /// Up projection weight [hidden_dim x dim] (flat)
    pub up_weight: Vec<f32>,
    /// Up projection bias [hidden_dim]
    pub up_bias: Vec<f32>,
    /// Down projection weight [dim x hidden_dim] (flat)
    pub down_weight: Vec<f32>,
    /// Down projection bias [dim]
    pub down_bias: Vec<f32>,
    /// LayerNorm weight for pre-normalization
    pub norm_weight: Vec<f32>,
    /// LayerNorm bias for pre-normalization
    pub norm_bias: Vec<f32>,
}

impl GluGate {
    /// Create new GLU with Xavier initialization
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        let scale_gate = (2.0 / (dim as f32)).sqrt();
        let scale_down = (2.0 / (hidden_dim as f32)).sqrt();
        let mut rng = rand::thread_rng();

        let gate_weight: Vec<f32> = (0..hidden_dim * dim)
            .map(|_| rng.gen_range(-scale_gate..scale_gate))
            .collect();
        let gate_bias = vec![0.0; hidden_dim];

        let up_weight: Vec<f32> = (0..hidden_dim * dim)
            .map(|_| rng.gen_range(-scale_gate..scale_gate))
            .collect();
        let up_bias = vec![0.0; hidden_dim];

        let down_weight: Vec<f32> = (0..dim * hidden_dim)
            .map(|_| rng.gen_range(-scale_down..scale_down))
            .collect();
        let down_bias = vec![0.0; dim];

        let norm_weight = vec![1.0; dim];
        let norm_bias = vec![0.0; dim];

        GluGate {
            dim,
            hidden_dim,
            gate_weight,
            gate_bias,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
            norm_weight,
            norm_bias,
        }
    }

    /// Forward pass: LayerNorm → SiLU(W_g * x) * (W_u * x) → W_d * output
    /// Input and output are both [dim] vectors
    pub fn forward(&self, x: &mut [f32], temp_buffer: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert!(temp_buffer.len() >= self.hidden_dim);

        // 1. Pre-normalization
        layer_norm(x, &self.norm_weight, &self.norm_bias, 1e-5);

        let hidden = &mut temp_buffer[..self.hidden_dim];

        // 2. Gate projection: gate = SiLU(W_g * x + b_g)
        for i in 0..self.hidden_dim {
            let mut sum = self.gate_bias[i];
            for j in 0..self.dim {
                sum += self.gate_weight[i * self.dim + j] * x[j];
            }
            hidden[i] = silu(sum);
        }

        // 3. Up projection: up = W_u * x + b_u
        for i in 0..self.hidden_dim {
            let mut sum = self.up_bias[i];
            for j in 0..self.dim {
                sum += self.up_weight[i * self.dim + j] * x[j];
            }
            hidden[i] *= sum; // gate * up (SiLU * linear)
        }

        // 4. Down projection: x = W_d * gate_up + b_d
        let mut output = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut sum = self.down_bias[i];
            for j in 0..self.hidden_dim {
                sum += self.down_weight[i * self.hidden_dim + j] * hidden[j];
            }
            output[i] = sum;
        }

        // 5. Residual connection
        for i in 0..self.dim {
            x[i] = x[i] + output[i];
        }
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.gate_weight.len() + self.gate_bias.len()
            + self.up_weight.len() + self.up_bias.len()
            + self.down_weight.len() + self.down_bias.len()
            + self.norm_weight.len() + self.norm_bias.len()
    }
}

// ============================================================================
// StateSpace Parameters (Flat Memory Layout)
// ============================================================================

/// State Space Model parameters for selective scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpace {
    /// State dimension (N in Mamba paper, typically 16)
    pub d_state: usize,
    /// Inner dimension (d_inner = dim, typically same as embedding dim)
    pub d_inner: usize,

    // ===== Flat matrices (d_inner × d_state) stored as [d_inner * d_state] =====
    pub a_log: Vec<f32>,    // A_log = log(diag(-exp(A_log)))
    pub b: Vec<f32>,        // B projection
    pub c: Vec<f32>,        // C projection
    pub h: Vec<f32>,        // Hidden state h(t) — the SSM memory
    pub output_buf: Vec<f32>, // Pre-allocated output buffer

    // ===== Vectors (d_inner,) =====
    pub delta: Vec<f32>,        // Δ projection weights
    pub delta_bias: Vec<f32>,   // Δ bias
    pub d: Vec<f32>,            // D vector - skip connection

    // ===== LayerNorm for SSM input =====
    pub ssm_norm_weight: Vec<f32>,
    pub ssm_norm_bias: Vec<f32>,

    // ===== GLU module (applied after SSM) =====
    pub glu: Option<GluGate>,
}

impl StateSpace {
    /// Create a new StateSpace with given dimensions
    pub fn new(dim: usize) -> Self {
        let d_state = 16; // Standard Mamba state dimension
        let d_inner = dim;
        let total = d_inner * d_state;

        // A_log = log(arange(1, d_state+1)) repeated for d_inner
        let mut a_log = Vec::with_capacity(total);
        for _ in 0..d_inner {
            for j in 0..d_state {
                a_log.push((j as f32 + 1.0).ln());
            }
        }

        // B and C initialized to small values
        let b = vec![0.01; total];
        let c = vec![0.01; total];

        // D = ones (skip connection)
        let d = vec![1.0; d_inner];

        // Delta initialized small positive
        let delta = vec![0.1; d_inner];
        let delta_bias = vec![0.0; d_inner];

        // Hidden state = zeros
        let h = vec![0.0; total];

        // Pre-allocate output buffer
        let output_buf = vec![0.0; d_inner];

        // SSM LayerNorm
        let ssm_norm_weight = vec![1.0; d_inner];
        let ssm_norm_bias = vec![0.0; d_inner];

        // GLU module (optional, initially None)
        let glu = None;

        Self {
            d_state,
            d_inner,
            a_log,
            b,
            c,
            h,
            output_buf,
            delta,
            delta_bias,
            d,
            ssm_norm_weight,
            ssm_norm_bias,
            glu,
        }
    }

    /// Create with GLU enabled (SSM → GLU stack)
    pub fn new_with_glu(dim: usize, glu_hidden_mult: usize) -> Self {
        let mut ssm = Self::new(dim);
        ssm.glu = Some(GluGate::new(dim, dim * glu_hidden_mult));
        ssm
    }

    /// Reset hidden state (for new sequences)
    pub fn reset(&mut self) {
        self.h.fill(0.0);
    }

    /// Full forward pass: LayerNorm → Selective Scan → GLU → Residual
    /// x: input vector of length dim
    /// buffer: temporary workspace (must have length >= hidden_dim of GLU)
    pub fn forward(&mut self, x: &mut [f32], buffer: &mut [f32]) {
        let dim = self.d_inner;

        // Save input for residual connection
        let residual = x.to_vec();

        // 1. Selective scan step (includes internal LayerNorm)
        let output = self.selective_scan_step(x, true);
        for i in 0..dim.min(output.len()) {
            x[i] = output[i];
        }

        // 2. GLU (if enabled) — includes internal LayerNorm and residual
        if let Some(ref glu) = self.glu {
            glu.forward(x, buffer);
        } else {
            // Without GLU, just add residual
            for i in 0..dim {
                x[i] = x[i] + residual[i];
            }
        }
    }

    /// Selective scan step (Mamba-style discretized SSM).
    /// Applies: y = C * h(t) + D * x(t)
    /// where h(t) = exp(Δ*A) * h(t-1) + Δ*B*x(t)
    pub fn selective_scan_step(&mut self, x: &[f32], normalize: bool) -> &[f32] {
        let d_inner = self.d_inner;
        let d_state = self.d_state;

        if x.len() < d_inner {
            let min_len = x.len().min(d_inner);
            for i in 0..min_len {
                self.output_buf[i] = x[i];
            }
            return &self.output_buf[..d_inner];
        }

        // Pre-normalize input if requested
        let mut normalized = x.to_vec();
        if normalize {
            layer_norm(&mut normalized, &self.ssm_norm_weight, &self.ssm_norm_bias, 1e-5);
        }

        let x_ref = if normalize { &normalized } else { x };

        // Compute Δ = softplus(W_delta * x + delta_bias)
        // Simplified: use delta * x[i] + delta_bias[i]
        let delta_soft: Vec<f32> = self.delta.iter()
            .zip(self.delta_bias.iter())
            .map(|(&d, &b)| softplus(d * x_ref[0] + b))
            .collect();

        // For each channel in d_inner:
        for i in 0..d_inner {
            let base = i * d_state;

            // ΔA = exp(Δ * A) discretization
            // A = -exp(A_log), so ΔA = exp(Δ * (-exp(A_log)))
            let mut h_out = [0.0f32; 16]; // Max d_state
            let _ds = d_state.min(16);

            for j in 0..d_state {
                let delta_a = (-self.a_log[base + j].exp() * delta_soft[i]).exp();
                let delta_b = delta_soft[i] * self.b[base + j];
                // h(t) = exp(Δ*A) * h(t-1) + Δ*B*x(t)
                h_out[j] = delta_a * self.h[base + j] + delta_b * x_ref[i];
            }

            // y = C * h(t) + D * x(t)
            let mut y = 0.0f32;
            for j in 0..d_state {
                y += self.c[base + j] * h_out[j];
            }
            y += self.d[i] * x_ref[i];

            self.output_buf[i] = y;

            // Update hidden state
            for j in 0..d_state {
                self.h[base + j] = h_out[j];
            }
        }

        &self.output_buf[..d_inner]
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let mut n = self.a_log.len() + self.b.len() + self.c.len()
            + self.delta.len() + self.delta_bias.len() + self.d.len()
            + self.ssm_norm_weight.len() + self.ssm_norm_bias.len();
        if let Some(ref glu) = self.glu {
            n += glu.num_params();
        }
        n
    }
}

// ============================================================================
// Simplified SSM Transform (Backward Compatible API)
// ============================================================================

/// Apply SSM transform to a pulse's content vector (backward compatible).
pub fn ssm_transform_pulse(
    ssm: &mut StateSpace,
    pulse_content: &mut [f32],
    _use_time_mixing: bool,
) -> Vec<f32> {
    let d = pulse_content.len();
    let d_inner = ssm.d_inner;

    let mut x: Vec<f32> = if d == d_inner {
        pulse_content.to_vec()
    } else if d < d_inner {
        let mut padded = vec![0.0; d_inner];
        padded[..d].copy_from_slice(pulse_content);
        padded
    } else {
        pulse_content[..d_inner].to_vec()
    };

    // Use forward pass which includes GLU
    let mut buffer = vec![0.0; d_inner * 4]; // Enough for GLU hidden
    ssm.forward(&mut x, &mut buffer);

    let out_len = x.len().min(d);
    for i in 0..out_len {
        pulse_content[i] = x[i];
    }

    x
}

/// Apply SSM transform to multiple pulses (batch).
pub fn ssm_transform_pulses(
    ssm: &mut StateSpace,
    pulses_content: &mut [Vec<f32>],
    _use_time_mixing: bool,
) {
    let mut buffer = vec![0.0; ssm.d_inner * 4];
    for content in pulses_content.iter_mut() {
        let d = content.len();
        let mut x = if d == ssm.d_inner {
            content.clone()
        } else {
            let mut padded = vec![0.0; ssm.d_inner];
            padded[..d.min(ssm.d_inner)].copy_from_slice(&content[..d.min(ssm.d_inner)]);
            padded
        };
        ssm.forward(&mut x, &mut buffer);
        let out_len = x.len().min(d);
        for i in 0..out_len {
            content[i] = x[i];
        }
    }
}

// ============================================================================
// Multi-Layer SSM Stack
// ============================================================================

/// A stack of SSM layers with GLU gating.
/// Each layer: LayerNorm → SSM → GLU → Residual
#[derive(Debug, Clone)]
pub struct SsmStack {
    pub layers: Vec<StateSpace>,
    pub dim: usize,
    pub num_layers: usize,
    pub glu_hidden_mult: usize,
}

impl SsmStack {
    /// Create a stack of SSM layers
    pub fn new(dim: usize, num_layers: usize, glu_hidden_mult: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(StateSpace::new_with_glu(dim, glu_hidden_mult));
        }
        SsmStack {
            layers,
            dim,
            num_layers,
            glu_hidden_mult,
        }
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, x: &mut [f32]) {
        let mut buffer = vec![0.0; self.dim * self.glu_hidden_mult];
        for layer in self.layers.iter_mut() {
            layer.forward(x, &mut buffer);
        }
    }

    /// Reset all layers' hidden states
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.6931).abs() < 0.001);
        assert!((softplus(10.0) - 10.0).abs() < 0.001);
        assert!((softplus(-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0)).abs() < 0.001);
        assert!((silu(1.0) - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0)).abs() < 0.001);
        assert!((gelu(1.0) - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        rms_norm(&mut x, &weight, 1e-5);
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        assert!((sum_sq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        layer_norm(&mut x, &weight, &bias, 1e-5);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.001);
        let var: f32 = x.iter().map(|&v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_glu_creation() {
        let glu = GluGate::new(64, 256);
        assert_eq!(glu.dim, 64);
        assert_eq!(glu.hidden_dim, 256);
        assert_eq!(glu.gate_weight.len(), 256 * 64);
        assert_eq!(glu.gate_bias.len(), 256);
    }

    #[test]
    fn test_glu_forward() {
        let glu = GluGate::new(64, 256);
        let mut x = vec![0.5; 64];
        let mut buffer = vec![0.0; 256];
        glu.forward(&mut x, &mut buffer);
        assert_eq!(x.len(), 64);
        // Should produce non-zero output
        let sum: f32 = x.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_state_space_creation() {
        let ssm = StateSpace::new(64);
        assert_eq!(ssm.d_inner, 64);
        assert_eq!(ssm.d_state, 16);
        assert_eq!(ssm.h.len(), 64 * 16);
    }

    #[test]
    fn test_state_space_with_glu() {
        let mut ssm = StateSpace::new_with_glu(64, 4);
        assert!(ssm.glu.is_some());
        let mut x = vec![0.5; 64];
        let mut buffer = vec![0.0; 256];
        ssm.forward(&mut x, &mut buffer);
        assert_eq!(x.len(), 64);
        let sum: f32 = x.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_selective_scan_step() {
        let mut ssm = StateSpace::new(64);
        let x = vec![0.5; 64];
        let y = ssm.selective_scan_step(&x, true);
        let sum: f32 = y.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_ssm_stack() {
        let mut stack = SsmStack::new(64, 4, 4);
        assert_eq!(stack.num_layers, 4);
        let mut x = vec![0.5; 64];
        stack.forward(&mut x);
        assert_eq!(x.len(), 64);
        let sum: f32 = x.iter().sum();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_ssm_transform_pulse() {
        let mut ssm = StateSpace::new_with_glu(64, 4);
        let mut content = vec![0.3; 64];
        let original = content.clone();
        let output = ssm_transform_pulse(&mut ssm, &mut content, false);
        assert_ne!(content, original);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_ssm_transform_sequence() {
        let mut ssm = StateSpace::new_with_glu(64, 4);
        let mut contents = vec![
            vec![0.3; 64],
            vec![0.5; 64],
            vec![0.7; 64],
        ];
        let original = contents.clone();
        ssm_transform_pulses(&mut ssm, &mut contents, false);
        // Each output should differ from input
        for i in 0..contents.len() {
            assert_ne!(contents[i], original[i]);
        }
        // Sequential outputs should differ due to evolving hidden state
        assert_ne!(contents[0], contents[2]);
    }
}
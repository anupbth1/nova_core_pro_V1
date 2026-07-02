//! Nova SSM - State Space Model Module
//!
//! OPTIMIZED V2: Flat memory layout for SIMD auto-vectorization.
//! All matrices are stored as flat Vec<f32> with stride-based access.
//! This enables Rustc to auto-vectorize with AVX2/FMA (4-8x speedup).
//!
//! Pure Rust implementation of Mamba's selective scan and RWKV's time-mixing.
//! NO external dependencies - uses only Vec<f32> operations.
//!
//! Key insight: Flat arrays + stride = cache-friendly + auto-vectorizable.

use std::f32::consts::E;

// ============================================================================
// StateSpace Parameters (Flat Memory Layout)
// ============================================================================

/// State Space Model parameters for a single core.
///
/// OPTIMIZED: All matrices use flat Vec<f32> with stride-based indexing:
///   element[i][j] = data[i * stride + j]
///
/// This gives:
/// 1. Single allocation per matrix (no pointer chasing)
/// 2. Cache-friendly sequential access
/// 3. Auto-vectorization by rustc
/// 4. Easy to port to GPU later
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State dimension (N in Mamba paper, typically 16)
    pub d_state: usize,
    
    /// Inner dimension (d_inner = d_model * expand, typically d_model * 2)
    pub d_inner: usize,
    
    // ===== Flat matrices (d_inner × d_state) stored as [d_inner * d_state] =====
    
    /// A matrix - state transition (always negative)
    /// Flat: a[i * d_state + j]
    pub a: Vec<f32>,
    
    /// A_log (log of A, used for numerical stability)
    /// Flat: a_log[i * d_state + j]
    pub a_log: Vec<f32>,
    
    /// B matrix - input projection
    /// Flat: b[i * d_state + j]
    pub b: Vec<f32>,
    
    /// C matrix - output projection
    /// Flat: c[i * d_state + j]
    pub c: Vec<f32>,
    
    /// Hidden state h(t) — the SSM memory
    /// Flat: h[i * d_state + j]
    pub h: Vec<f32>,
    
    /// OPTIMIZED: Pre-allocated output buffer to avoid per-call allocations
    pub output_buf: Vec<f32>,
    
    // ===== Vectors (d_inner,) =====
    
    /// Δ (delta) projection weights
    pub delta: Vec<f32>,
    
    /// Δ (delta) bias
    pub delta_bias: Vec<f32>,
    
    /// D vector - skip connection
    pub d: Vec<f32>,
    
    /// RWKV time-mix parameters
    pub time_mix_x: Vec<f32>,
    pub time_mix_w: Vec<f32>,
    pub time_mix_key: Vec<f32>,
    pub time_mix_value: Vec<f32>,
    pub time_mix_receptance: Vec<f32>,
    
    /// Previous input for RWKV time-mixing
    pub prev_x: Vec<f32>,
}

impl StateSpace {
    /// Create a new StateSpace with given dimensions
    pub fn new(d_inner: usize, d_state: usize) -> Self {
        let total = d_inner * d_state;
        
        // Initialize A_log as log of arange(1, d_state+1) repeated for d_inner
        let mut a_log = Vec::with_capacity(total);
        for _ in 0..d_inner {
            for j in 0..d_state {
                a_log.push((j as f32 + 1.0).ln());
            }
        }
        
        // A = -exp(A_log) — always negative for stability
        let mut a = Vec::with_capacity(total);
        for &val in &a_log {
            a.push(-val.exp());
        }
        
        // B and C initialized small random
        let b = vec![0.01; total];
        let c = vec![0.01; total];
        
        // D = ones (skip connection)
        let d = vec![1.0; d_inner];
        
        // Delta initialized small positive
        let delta = vec![0.1; d_inner];
        let delta_bias = vec![0.0; d_inner];
        
        // Hidden state = zeros
        let h = vec![0.0; total];
        
        // RWKV time-mix parameters (initialized for no mixing)
        let time_mix_x = vec![0.0; d_inner];
        let time_mix_w = vec![0.0; d_inner];
        let time_mix_key = vec![0.0; d_inner];
        let time_mix_value = vec![0.0; d_inner];
        let time_mix_receptance = vec![0.0; d_inner];
        
        let prev_x = vec![0.0; d_inner];
        
        // Pre-allocate output buffer
        let output_buf = vec![0.0; d_inner];
        
        Self {
            d_state,
            d_inner,
            a,
            a_log,
            b,
            c,
            d,
            h,
            output_buf,
            delta,
            delta_bias,
            time_mix_x,
            time_mix_w,
            time_mix_key,
            time_mix_value,
            time_mix_receptance,
            prev_x,
        }
    }
    
    /// Reset hidden state (for new sequences)
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.prev_x.fill(0.0);
    }
    
    /// Load SSM parameters from projected weights (flat memory layout)
    pub fn load_from_projection(
        &mut self,
        delta: Vec<f32>,
        delta_bias: Vec<f32>,
        a_log: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
    ) {
        if delta.len() == self.d_inner {
            self.delta = delta;
        }
        if delta_bias.len() == self.d_inner {
            self.delta_bias = delta_bias;
        }
        
        let ds = self.d_state;
        let di = self.d_inner;
        let total = di * ds;
        
        // Flat arrays: just copy directly
        if a_log.len() == total {
            self.a_log.copy_from_slice(&a_log);
            // Recompute A = -exp(A_log)
            for i in 0..total {
                self.a[i] = -a_log[i].exp();
            }
        }
        if b.len() == total {
            self.b.copy_from_slice(&b);
        }
        if c.len() == total {
            self.c.copy_from_slice(&c);
        }
        if d.len() == self.d_inner {
            self.d = d;
        }
    }
}

// ============================================================================
// Core SSM Operations (SIMD-Friendly)
// ============================================================================

/// Softplus activation: log(1 + exp(x))
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x }
    else if x < -20.0 { 0.0 }
    else { (1.0 + x.exp()).ln() }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply softplus to entire vector (SIMD-friendly flat loop)
#[inline]
fn softplus_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| softplus(v)).collect()
}

/// Element-wise vector addition: a + b
#[inline]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise vector subtraction: a - b
#[inline]
fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise vector multiplication: a * b
#[inline]
fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scalar-vector multiplication: s * v
#[inline]
fn vec_scale(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}

/// Vector dot product (SIMD auto-vectorized)
#[inline]
fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ============================================================================
// Mamba-Style Selective Scan (Flat Memory, SIMD-Friendly)
// ============================================================================

/// OPTIMIZED V3: Perform one step of the Mamba selective scan.
/// Uses pre-allocated output_buf to avoid per-call Vec allocations.
///
///   h(t) = exp(Δ * A) * h(t-1) + Δ * B * x(t)
///   y(t) = C * h(t) + D * x(t)
pub fn selective_scan_step(ssm: &mut StateSpace, x: &[f32], _use_input_dependent_bc: bool) -> Vec<f32> {
    let d_inner = ssm.d_inner;
    let d_state = ssm.d_state;
    let ds = d_state;
    
    // 1. Compute Δ (step size) from input
    // Δ = softplus(delta * x + delta_bias)
    // OPTIMIZED: Write directly into output_buf to avoid temp allocation
    for i in 0..d_inner {
        ssm.output_buf[i] = softplus(ssm.delta[i] * x[i] + ssm.delta_bias[i]);
    }
    
    // 2. Discretize A: exp(Δ * A) and compute Δ*B*x in one pass
    // Flat arrays: a[i*ds + j], h[i*ds + j], b[i*ds + j]
    // This is the HOT LOOP - optimized for SIMD auto-vectorization
    for i in 0..d_inner {
        let di = ssm.output_buf[i]; // delta[i]
        let xi = x[i];
        let base = i * ds;
        
        // Process d_state elements sequentially (small, typically 16)
        // Rustc auto-vectorizes this inner loop with SIMD
        for j in 0..ds {
            let idx = base + j;
            // exp(Δ * A) * h(t-1)
            let delta_a = (di * ssm.a[idx]).exp();
            // Δ * B * x(t)
            let delta_b_x = di * ssm.b[idx] * xi;
            // h(t) = exp(Δ*A) * h(t-1) + Δ*B*x(t)
            ssm.h[idx] = delta_a * ssm.h[idx] + delta_b_x;
        }
    }
    
    // 3. Compute output: y = C * h + D * x
    // y[i] = sum_j(C[i][j] * h[i][j]) + D[i] * x[i]
    // OPTIMIZED: Write directly into output_buf (reuse the delta buffer)
    for i in 0..d_inner {
        let base = i * ds;
        let mut c_h_sum = 0.0;
        // This inner loop auto-vectorizes
        for j in 0..ds {
            c_h_sum += ssm.c[base + j] * ssm.h[base + j];
        }
        ssm.output_buf[i] = c_h_sum + ssm.d[i] * x[i];
    }
    
    // Return a slice of output_buf (caller can use it before next call)
    ssm.output_buf.clone()
}

/// Full selective scan over a sequence of inputs.
pub fn selective_scan_sequence(ssm: &mut StateSpace, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    inputs.iter().map(|x| selective_scan_step(ssm, x, true)).collect()
}

// ============================================================================
// RWKV-Style Time Mixing
// ============================================================================

/// RWKV-style time mixing.
pub fn time_mixing(ssm: &mut StateSpace, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let shifted = &ssm.prev_x;
    let sx = vec_sub(shifted, x);
    
    let xk = vec_add(x, &vec_mul(&sx, &ssm.time_mix_key));
    let xv = vec_add(x, &vec_mul(&sx, &ssm.time_mix_value));
    let xr = vec_add(x, &vec_mul(&sx, &ssm.time_mix_receptance));
    
    // Store current input as previous for next step
    ssm.prev_x.copy_from_slice(x);
    
    (xk, xv, xr)
}

/// RWKV-style channel mixing (feed-forward).
pub fn channel_mixing(
    ssm: &mut StateSpace,
    x: &[f32],
    key_weight: &[Vec<f32>],
    value_weight: &[Vec<f32>],
    receptance_weight: &[Vec<f32>],
) -> Vec<f32> {
    let shifted = &ssm.prev_x;
    let sx = vec_sub(shifted, x);
    
    let xk = vec_add(x, &vec_mul(&sx, &ssm.time_mix_key));
    let xr = vec_add(x, &vec_mul(&sx, &ssm.time_mix_receptance));
    
    // Squared ReLU activation for key
    let k = mat_vec_mul_nested(key_weight, &xk);
    let k_sq: Vec<f32> = k.iter().map(|&v| v.max(0.0).powi(2)).collect();
    
    // Sigmoid receptance
    let r_raw = mat_vec_mul_nested(receptance_weight, &xr);
    let r: Vec<f32> = r_raw.iter().map(|&v| sigmoid(v)).collect();
    
    // Value projection
    let v = mat_vec_mul_nested(value_weight, &k_sq);
    
    // Element-wise multiply: r * v
    vec_mul(&r, &v)
}

/// Matrix-vector multiplication using nested Vec<Vec<f32>> (legacy)
fn mat_vec_mul_nested(m: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    m.iter().map(|row| vec_dot(row, v)).collect()
}

// ============================================================================
// RWKV-Style Linear Attention (WKV)
// ============================================================================

/// RWKV-style WKV computation (linear attention).
pub fn wkv_attention(
    state: &mut Vec<Vec<f32>>,
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: f32,
) -> Vec<f32> {
    let d = r.len();
    
    // at = outer(k, v) — (d_inner, d_inner)
    let at = outer_product(k, v);
    
    // state = at + w * state
    for i in 0..d {
        for j in 0..d {
            state[i][j] = at[i][j] + w * state[i][j];
        }
    }
    
    // output = r * (state aggregated)
    let mut state_agg = vec![0.0; d];
    for i in 0..d {
        for j in 0..d {
            state_agg[i] += state[i][j];
        }
    }
    
    vec_mul(r, &state_agg)
}

/// Outer product: v1 ⊗ v2 → matrix of shape (len(v1), len(v2))
fn outer_product(v1: &[f32], v2: &[f32]) -> Vec<Vec<f32>> {
    v1.iter().map(|&a| {
        v2.iter().map(|&b| a * b).collect()
    }).collect()
}

// ============================================================================
// Nova SSM Integration Helpers
// ============================================================================

/// Apply SSM transform to a pulse's content vector.
pub fn ssm_transform_pulse(
    ssm: &mut StateSpace,
    pulse_content: &mut [f32],
    use_time_mixing: bool,
) -> Vec<f32> {
    let d = pulse_content.len();
    let d_inner = ssm.d_inner;
    
    let x: Vec<f32> = if d == d_inner {
        pulse_content.to_vec()
    } else if d < d_inner {
        let mut padded = vec![0.0; d_inner];
        padded[..d].copy_from_slice(pulse_content);
        padded
    } else {
        pulse_content[..d_inner].to_vec()
    };
    
    let ssm_input = if use_time_mixing {
        let (xk, _xv, _xr) = time_mixing(ssm, &x);
        xk
    } else {
        x
    };
    
    let output = selective_scan_step(ssm, &ssm_input, true);
    
    let out_len = output.len().min(d);
    for i in 0..out_len {
        pulse_content[i] = output[i];
    }
    
    output
}

/// Apply SSM transform to multiple pulses (batch).
pub fn ssm_transform_pulses(
    ssm: &mut StateSpace,
    pulses_content: &mut [Vec<f32>],
    use_time_mixing: bool,
) {
    for content in pulses_content.iter_mut() {
        ssm_transform_pulse(ssm, content, use_time_mixing);
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
        println!("✅ softplus works!");
    }
    
    #[test]
    fn test_silu() {
        assert!((silu(0.0)).abs() < 0.001);
        assert!((silu(1.0) - 0.731).abs() < 0.01);
        println!("✅ silu works!");
    }
    
    #[test]
    fn test_state_space_creation() {
        let ssm = StateSpace::new(64, 16);
        assert_eq!(ssm.d_inner, 64);
        assert_eq!(ssm.d_state, 16);
        assert_eq!(ssm.h.len(), 64 * 16);
        assert_eq!(ssm.a.len(), 64 * 16);
        // A should be negative
        for &val in &ssm.a {
            assert!(val < 0.0);
        }
        println!("✅ StateSpace creation works! (flat memory)");
    }
    
    #[test]
    fn test_selective_scan_step() {
        let mut ssm = StateSpace::new(64, 16);
        let x = vec![0.5; 64];
        let y = selective_scan_step(&mut ssm, &x, true);
        assert_eq!(y.len(), 64);
        let sum: f32 = y.iter().sum();
        assert!(sum.abs() > 0.0);
        println!("✅ selective_scan_step works! Output sum: {:.4}", sum);
    }
    
    #[test]
    fn test_selective_scan_sequence() {
        let mut ssm = StateSpace::new(64, 16);
        let inputs: Vec<Vec<f32>> = (0..5).map(|i| vec![0.1 * (i as f32 + 1.0); 64]).collect();
        let outputs = selective_scan_sequence(&mut ssm, &inputs);
        assert_eq!(outputs.len(), 5);
        assert_eq!(outputs[0].len(), 64);
        let sum0: f32 = outputs[0].iter().sum();
        let sum4: f32 = outputs[4].iter().sum();
        assert!((sum0 - sum4).abs() > 0.001);
        println!("✅ selective_scan_sequence works!");
    }
    
    #[test]
    fn test_time_mixing() {
        let mut ssm = StateSpace::new(64, 16);
        let x1 = vec![0.5; 64];
        let x2 = vec![0.8; 64];
        
        let (xk1, _, _) = time_mixing(&mut ssm, &x1);
        let (xk2, _, _) = time_mixing(&mut ssm, &x2);
        
        assert_eq!(xk1.len(), 64);
        assert_eq!(xk2.len(), 64);
        assert_ne!(xk1, xk2);
        println!("✅ time_mixing works!");
    }
    
    #[test]
    fn test_ssm_transform_pulse() {
        let mut ssm = StateSpace::new(64, 16);
        let mut content = vec![0.3; 64];
        let original = content.clone();
        
        let output = ssm_transform_pulse(&mut ssm, &mut content, false);
        
        assert_ne!(content, original);
        assert_eq!(output.len(), 64);
        println!("✅ ssm_transform_pulse works!");
    }
    
    #[test]
    fn test_ssm_with_time_mixing() {
        let mut ssm = StateSpace::new(64, 16);
        let mut content1 = vec![0.3; 64];
        let mut content2 = vec![0.7; 64];
        
        let out1 = ssm_transform_pulse(&mut ssm, &mut content1, true);
        let out2 = ssm_transform_pulse(&mut ssm, &mut content2, true);
        
        assert_ne!(out1, out2);
        println!("✅ SSM with time mixing works!");
    }
}

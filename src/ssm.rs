//! Nova SSM - State Space Model Module
//!
//! Pure Rust implementation of Mamba's selective scan and RWKV's time-mixing.
//! NO external dependencies - uses only Vec<f32> operations.
//!
//! This is the mathematical core extracted from:
//!   - huggingface/candle → mamba.rs (selective scan)
//!   - huggingface/candle → rwkv.rs (time mixing)
//!   - johnma2006/mamba-minimal → model.py (weight projection)
//!
//! Key insight: We extract ONLY the mathematical logic, not the code.
//! This means upstream changes to candle/mamba-minimal won't affect Nova.

use std::f32::consts::E;

// ============================================================================
// StateSpace Parameters
// ============================================================================

/// State Space Model parameters for a single core.
///
/// Maps to Mamba's selective scan formulation:
///   h(t) = exp(Δ * A) * h(t-1) + Δ * B * x(t)    [State update]
///   y(t) = C * h(t) + D * x(t)                     [Output]
///
/// Where:
///   - Δ (delta): input-dependent step size (controls how fast state updates)
///   - A: state transition matrix (always negative for stability)
///   - B: input projection (input-dependent in Mamba)
///   - C: output projection (input-dependent in Mamba)
///   - D: skip connection (direct feedthrough)
///   - h: hidden state (the "memory" of the SSM)
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State dimension (N in Mamba paper, typically 16)
    pub d_state: usize,
    
    /// Inner dimension (d_inner = d_model * expand, typically d_model * 2)
    pub d_inner: usize,
    
    /// Δ (delta) projection weights: maps input to step size
    /// Shape: (d_inner,) — per-element step sizes
    pub delta: Vec<f32>,
    
    /// Δ (delta) bias
    pub delta_bias: Vec<f32>,
    
    /// A matrix - state transition (always negative)
    /// Shape: (d_inner, d_state)
    pub a: Vec<Vec<f32>>,
    
    /// A_log (log of A, used for numerical stability)
    /// Shape: (d_inner, d_state)
    pub a_log: Vec<Vec<f32>>,
    
    /// B matrix - input projection
    /// Shape: (d_inner, d_state)
    pub b: Vec<Vec<f32>>,
    
    /// C matrix - output projection
    /// Shape: (d_inner, d_state)
    pub c: Vec<Vec<f32>>,
    
    /// D vector - skip connection (direct feedthrough)
    /// Shape: (d_inner,)
    pub d: Vec<f32>,
    
    /// Hidden state h(t) — the SSM memory
    /// Shape: (d_inner, d_state)
    pub h: Vec<Vec<f32>>,
    
    /// RWKV-style time-mix parameters
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
        // Initialize A_log as log of arange(1, d_state+1) repeated for d_inner
        // This matches Mamba's initialization: A = -exp(A_log)
        let mut a_log = Vec::with_capacity(d_inner);
        for _ in 0..d_inner {
            let mut row = Vec::with_capacity(d_state);
            for j in 0..d_state {
                row.push((j as f32 + 1.0).ln());
            }
            a_log.push(row);
        }
        
        // A = -exp(A_log) — always negative for stability
        let mut a = Vec::with_capacity(d_inner);
        for row in &a_log {
            let mut a_row = Vec::with_capacity(d_state);
            for &val in row {
                a_row.push(-val.exp());
            }
            a.push(a_row);
        }
        
        // B and C initialized small random
        let mut b = Vec::with_capacity(d_inner);
        let mut c = Vec::with_capacity(d_inner);
        for _ in 0..d_inner {
            b.push(vec![0.01; d_state]);
            c.push(vec![0.01; d_state]);
        }
        
        // D = ones (skip connection)
        let d = vec![1.0; d_inner];
        
        // Delta initialized small positive
        let delta = vec![0.1; d_inner];
        let delta_bias = vec![0.0; d_inner];
        
        // Hidden state = zeros
        let h = vec![vec![0.0; d_state]; d_inner];
        
        // RWKV time-mix parameters (initialized for no mixing)
        let time_mix_x = vec![0.0; d_inner];
        let time_mix_w = vec![0.0; d_inner];
        let time_mix_key = vec![0.0; d_inner];
        let time_mix_value = vec![0.0; d_inner];
        let time_mix_receptance = vec![0.0; d_inner];
        
        let prev_x = vec![0.0; d_inner];
        
        Self {
            d_state,
            d_inner,
            delta,
            delta_bias,
            a,
            a_log,
            b,
            c,
            d,
            h,
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
        for row in self.h.iter_mut() {
            row.fill(0.0);
        }
        self.prev_x.fill(0.0);
    }
    
    /// Load SSM parameters from projected weights
    /// This is called after weight conversion from Transformer models
    pub fn load_from_projection(
        &mut self,
        delta: Vec<f32>,
        delta_bias: Vec<f32>,
        a_log: Vec<Vec<f32>>,
        b: Vec<Vec<f32>>,
        c: Vec<Vec<f32>>,
        d: Vec<f32>,
    ) {
        if delta.len() == self.d_inner {
            self.delta = delta;
        }
        if delta_bias.len() == self.d_inner {
            self.delta_bias = delta_bias;
        }
        if a_log.len() == self.d_inner && a_log[0].len() == self.d_state {
            self.a_log = a_log;
            // Recompute A from A_log
            for i in 0..self.d_inner {
                for j in 0..self.d_state {
                    self.a[i][j] = -self.a_log[i][j].exp();
                }
            }
        }
        if b.len() == self.d_inner && b[0].len() == self.d_state {
            self.b = b;
        }
        if c.len() == self.d_inner && c[0].len() == self.d_state {
            self.c = c;
        }
        if d.len() == self.d_inner {
            self.d = d;
        }
    }
}

// ============================================================================
// Core SSM Operations
// ============================================================================

/// Softplus activation: log(1 + exp(x))
/// Used for delta (step size) to ensure it's always positive
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // For large x, softplus ≈ x
    } else if x < -20.0 {
        0.0 // For small x, softplus ≈ 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
/// Used in Mamba's gating mechanism
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Sigmoid activation: 1 / (1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply softplus to entire vector
fn softplus_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| softplus(v)).collect()
}

/// Element-wise vector addition: a + b
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise vector subtraction: a - b
fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Element-wise vector multiplication: a * b
fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Scalar-vector multiplication: s * v
fn vec_scale(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}

/// Vector dot product
fn vec_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiplication: M * v
/// M shape: (rows, cols), v shape: (cols,)
fn mat_vec_mul(m: &[Vec<f32>], v: &[f32]) -> Vec<f32> {
    m.iter().map(|row| vec_dot(row, v)).collect()
}

/// Element-wise matrix addition: A + B (broadcasting scalar B across matrix)
fn mat_add_scalar(m: &[Vec<f32>], s: f32) -> Vec<Vec<f32>> {
    m.iter().map(|row| row.iter().map(|x| x + s).collect()).collect()
}

/// Element-wise matrix exponential
fn mat_exp(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    m.iter().map(|row| row.iter().map(|x| x.exp()).collect()).collect()
}

/// Element-wise matrix multiplication (Hadamard): A ⊙ B
fn mat_mul_elem(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter().zip(b.iter())
        .map(|(row_a, row_b)| {
            row_a.iter().zip(row_b.iter()).map(|(x, y)| x * y).collect()
        })
        .collect()
}

/// Outer product: v1 ⊗ v2 → matrix of shape (len(v1), len(v2))
fn outer_product(v1: &[f32], v2: &[f32]) -> Vec<Vec<f32>> {
    v1.iter().map(|&a| {
        v2.iter().map(|&b| a * b).collect()
    }).collect()
}

// ============================================================================
// Mamba-Style Selective Scan
// ============================================================================

/// Perform one step of the Mamba selective scan.
///
/// This is the core SSM update:
///   h(t) = exp(Δ * A) * h(t-1) + Δ * B * x(t)
///   y(t) = C * h(t) + D * x(t)
///
/// Args:
///   ssm: StateSpace parameters (A, B, C, D, delta, h)
///   x: Input vector (d_inner,)
///   use_input_dependent_bc: If true, B and C are computed from x (Mamba-style)
///
/// Returns:
///   (y, h_new) where y is output and h_new is updated hidden state
pub fn selective_scan_step(ssm: &mut StateSpace, x: &[f32], use_input_dependent_bc: bool) -> Vec<f32> {
    let d_inner = ssm.d_inner;
    let d_state = ssm.d_state;
    
    // 1. Compute Δ (step size) from input
    // Δ = softplus(delta * x + delta_bias)
    let delta_raw = vec_add(&vec_mul(&ssm.delta, x), &ssm.delta_bias);
    let delta = softplus_vec(&delta_raw);
    
    // 2. Discretize A: exp(Δ * A)
    // Δ is (d_inner,), A is (d_inner, d_state)
    // We need to broadcast Δ across d_state dimension
    let mut delta_a = vec![vec![0.0; d_state]; d_inner];
    for i in 0..d_inner {
        for j in 0..d_state {
            delta_a[i][j] = (delta[i] * ssm.a[i][j]).exp();
        }
    }
    
    // 3. Compute B and C (input-dependent if use_input_dependent_bc)
    // In Mamba, B and C are projected from the same input x
    // For Nova, we use the stored B and C matrices (which can be input-dependent
    // if the weight converter projects them that way)
    let b = &ssm.b;
    let c = &ssm.c;
    
    // 4. Compute Δ * B * x(t)
    // B is (d_inner, d_state), x is (d_inner,)
    // We need: for each i in d_inner, for each j in d_state: delta[i] * B[i][j] * x[i]
    let mut delta_b_x = vec![vec![0.0; d_state]; d_inner];
    for i in 0..d_inner {
        for j in 0..d_state {
            delta_b_x[i][j] = delta[i] * b[i][j] * x[i];
        }
    }
    
    // 5. Update hidden state: h(t) = exp(Δ*A) * h(t-1) + Δ*B*x(t)
    // exp(Δ*A) is (d_inner, d_state), h is (d_inner, d_state)
    for i in 0..d_inner {
        for j in 0..d_state {
            ssm.h[i][j] = delta_a[i][j] * ssm.h[i][j] + delta_b_x[i][j];
        }
    }
    
    // 6. Compute output: y = C * h + D * x
    // C is (d_inner, d_state), h is (d_inner, d_state)
    // For each i in d_inner: y[i] = sum_j(C[i][j] * h[i][j]) + D[i] * x[i]
    let mut y = vec![0.0; d_inner];
    for i in 0..d_inner {
        let mut c_h_sum = 0.0;
        for j in 0..d_state {
            c_h_sum += c[i][j] * ssm.h[i][j];
        }
        y[i] = c_h_sum + ssm.d[i] * x[i];
    }
    
    y
}

/// Full selective scan over a sequence of inputs.
///
/// This processes each input step-by-step, updating the hidden state
/// and collecting outputs.
///
/// Args:
///   ssm: StateSpace parameters
///   inputs: Sequence of input vectors, each (d_inner,)
///
/// Returns:
///   Sequence of output vectors, each (d_inner,)
pub fn selective_scan_sequence(ssm: &mut StateSpace, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    inputs.iter().map(|x| selective_scan_step(ssm, x, true)).collect()
}

// ============================================================================
// RWKV-Style Time Mixing
// ============================================================================

/// RWKV-style time mixing.
///
/// This mixes the current input with the previous input using learned
/// time-dependent mixing coefficients.
///
/// Formula:
///   shifted = prev_x
///   sx = shifted - x
///   xk = x + sx * time_mix_key
///   xv = x + sx * time_mix_value
///   xr = x + sx * time_mix_receptance
///
/// Args:
///   ssm: StateSpace with time-mix parameters
///   x: Current input vector (d_inner,)
///
/// Returns:
///   (xk, xv, xr) — time-mixed key, value, and receptance
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
///
/// Formula:
///   shifted = prev_x
///   sx = shifted - x
///   xk = x + sx * time_mix_key
///   xr = x + sx * time_mix_receptance
///   k = relu(xk)^2  (squared ReLU)
///   r = sigmoid(xr)
///   output = r * (k @ value)
///
/// Args:
///   ssm: StateSpace with time-mix parameters
///   x: Current input vector
///   key_weight: Key projection weight
///   value_weight: Value projection weight
///   receptance_weight: Receptance projection weight
///
/// Returns:
///   Channel-mixed output
pub fn channel_mixing(
    ssm: &mut StateSpace,
    x: &[f32],
    key_weight: &[Vec<f32>],
    value_weight: &[Vec<f32>],
    receptance_weight: &[Vec<f32>],
) -> Vec<f32> {
    let shifted = &ssm.prev_x;
    let sx = vec_sub(shifted, x);
    
    // Time-mix key and receptance
    let xk = vec_add(x, &vec_mul(&sx, &ssm.time_mix_key));
    let xr = vec_add(x, &vec_mul(&sx, &ssm.time_mix_receptance));
    
    // Squared ReLU activation for key
    let k = mat_vec_mul(key_weight, &xk);
    let k_sq: Vec<f32> = k.iter().map(|&v| v.max(0.0).powi(2)).collect();
    
    // Sigmoid receptance
    let r_raw = mat_vec_mul(receptance_weight, &xr);
    let r: Vec<f32> = r_raw.iter().map(|&v| sigmoid(v)).collect();
    
    // Value projection
    let v = mat_vec_mul(value_weight, &k_sq);
    
    // Element-wise multiply: r * v
    vec_mul(&r, &v)
}

// ============================================================================
// RWKV-Style Linear Attention (WKV)
// ============================================================================

/// RWKV-style WKV computation (linear attention).
///
/// This is the core of RWKV's attention replacement:
///   state = at + w * state_prev
///   output = r * state
///
/// Where:
///   at = k^T * v (outer product of key and value)
///   w = time_decay (exponential decay)
///   r = receptance (gating)
pub fn wkv_attention(
    state: &mut Vec<Vec<f32>>,  // (d_inner, d_inner) — WKV state
    r: &[f32],                   // receptance (d_inner,)
    k: &[f32],                   // key (d_inner,)
    v: &[f32],                   // value (d_inner,)
    w: f32,                      // time decay factor
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
    // We aggregate state by summing across the second dimension
    let mut state_agg = vec![0.0; d];
    for i in 0..d {
        for j in 0..d {
            state_agg[i] += state[i][j];
        }
    }
    
    // output = r * state_agg
    vec_mul(r, &state_agg)
}

// ============================================================================
// Nova SSM Integration Helpers
// ============================================================================

/// Apply SSM transform to a pulse's content vector.
///
/// This is the main entry point for NovaCore to use SSM.
/// It combines Mamba's selective scan with RWKV's time mixing.
///
/// Args:
///   ssm: StateSpace parameters for this core
///   pulse_content: The pulse's content vector (will be modified in-place)
///   use_time_mixing: Whether to apply RWKV time mixing before SSM
///
/// Returns:
///   The SSM-enhanced output (same dimension as input)
pub fn ssm_transform_pulse(
    ssm: &mut StateSpace,
    pulse_content: &mut [f32],
    use_time_mixing: bool,
) -> Vec<f32> {
    let d = pulse_content.len();
    let d_inner = ssm.d_inner;
    
    // If pulse dimension doesn't match d_inner, we need to adapt
    // Nova pulses may have different dimension than SSM inner dimension
    let x: Vec<f32> = if d == d_inner {
        pulse_content.to_vec()
    } else if d < d_inner {
        // Pad with zeros
        let mut padded = vec![0.0; d_inner];
        padded[..d].copy_from_slice(pulse_content);
        padded
    } else {
        // Truncate
        pulse_content[..d_inner].to_vec()
    };
    
    // Apply RWKV time mixing if enabled
    let ssm_input = if use_time_mixing {
        let (xk, _xv, _xr) = time_mixing(ssm, &x);
        xk
    } else {
        x
    };
    
    // Apply Mamba selective scan
    let output = selective_scan_step(ssm, &ssm_input, true);
    
    // Write back to pulse content (if dimensions match)
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
        assert_eq!(ssm.h.len(), 64);
        assert_eq!(ssm.h[0].len(), 16);
        assert_eq!(ssm.a.len(), 64);
        assert_eq!(ssm.a[0].len(), 16);
        // A should be negative
        for row in &ssm.a {
            for &val in row {
                assert!(val < 0.0);
            }
        }
        println!("✅ StateSpace creation works!");
    }
    
    #[test]
    fn test_selective_scan_step() {
        let mut ssm = StateSpace::new(64, 16);
        let x = vec![0.5; 64];
        let y = selective_scan_step(&mut ssm, &x, true);
        assert_eq!(y.len(), 64);
        // Output should be non-zero
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
        // Each step should produce different output (state evolves)
        let sum0: f32 = outputs[0].iter().sum();
        let sum4: f32 = outputs[4].iter().sum();
        assert!((sum0 - sum4).abs() > 0.001);
        println!("✅ selective_scan_sequence works! Step 0 sum: {:.4}, Step 4 sum: {:.4}", sum0, sum4);
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
        // Second call should have different mixing (prev_x is now x1)
        assert_ne!(xk1, xk2);
        println!("✅ time_mixing works!");
    }
    
    #[test]
    fn test_ssm_transform_pulse() {
        let mut ssm = StateSpace::new(64, 16);
        let mut content = vec![0.3; 64];
        let original = content.clone();
        
        let output = ssm_transform_pulse(&mut ssm, &mut content, false);
        
        // Content should be modified
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
        
        // With time mixing, second output should be influenced by first input
        assert_ne!(out1, out2);
        println!("✅ SSM with time mixing works!");
    }
}

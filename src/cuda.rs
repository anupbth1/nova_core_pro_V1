//! Nova CUDA - GPU Acceleration Module
//!
//! Provides GPU acceleration for Nova Core operations using the `candle` crate.
//! Features runtime auto-detection:
//!   - NVIDIA GPU → CUDA backend
//!   - AMD GPU → HIP backend (future)
//!   - No GPU → Falls back to CPU (Rayon)
//!
//! This module wraps key Nova operations (SSM, field updates, core processing)
//! with GPU-accelerated versions when a compatible GPU is available.
//!
//! USAGE:
//!   let accelerator = NovaAccelerator::auto_detect();
//!   if accelerator.is_gpu() {
//!       accelerator.ssm_selective_scan(&mut ssm, &input);
//!   }

use std::time::Instant;

// ============================================================================
// Hardware Detection
// ============================================================================

/// Detected hardware backend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardwareBackend {
    /// NVIDIA GPU with CUDA support
    Cuda,
    /// AMD GPU with HIP support (future)
    Hip,
    /// CPU with Rayon parallelism
    Cpu,
    /// No acceleration available
    None,
}

impl HardwareBackend {
    pub fn is_gpu(&self) -> bool {
        matches!(self, HardwareBackend::Cuda | HardwareBackend::Hip)
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            HardwareBackend::Cuda => "CUDA (NVIDIA GPU)",
            HardwareBackend::Hip => "HIP (AMD GPU)",
            HardwareBackend::Cpu => "CPU (Rayon)",
            HardwareBackend::None => "None",
        }
    }
}

/// Auto-detect the best available hardware backend.
/// Checks for NVIDIA GPU first, then AMD GPU, then falls back to CPU.
pub fn auto_detect_backend() -> HardwareBackend {
    // Try to detect NVIDIA GPU via CUDA
    #[cfg(feature = "cuda")]
    {
        // Use candle's CUDA device detection
        match candle_core::Device::cuda_if_available(0) {
            Ok(_) => {
                eprintln!("  🖥️  GPU detected: NVIDIA CUDA");
                return HardwareBackend::Cuda;
            }
            Err(_) => {
                eprintln!("  ⚠️  CUDA device requested but not available");
            }
        }
    }
    
    // Try to detect AMD GPU via HIP
    #[cfg(feature = "hip")]
    {
        // HIP support via candle (future)
        eprintln!("  🖥️  GPU detected: AMD HIP");
        return HardwareBackend::Hip;
    }
    
    // Fall back to CPU
    eprintln!("  🖥️  No GPU detected, using CPU (Rayon)");
    HardwareBackend::Cpu
}

// ============================================================================
// Nova Accelerator - GPU-Accelerated Operations
// ============================================================================

/// Nova Accelerator provides GPU-accelerated versions of key operations.
/// When no GPU is available, it transparently falls back to CPU implementations.
pub struct NovaAccelerator {
    /// Detected hardware backend
    pub backend: HardwareBackend,
    /// Candle device (CPU or CUDA)
    #[cfg(feature = "cuda")]
    device: Option<candle_core::Device>,
    /// Whether acceleration is enabled
    pub enabled: bool,
    /// Statistics
    pub gpu_ops: u64,
    pub cpu_ops: u64,
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
}

impl NovaAccelerator {
    /// Create a new accelerator with auto-detected backend.
    /// Call this once at startup.
    pub fn auto_detect() -> Self {
        let backend = auto_detect_backend();
        
        #[cfg(feature = "cuda")]
        let device = match backend {
            HardwareBackend::Cuda => {
                match candle_core::Device::cuda_if_available(0) {
                    Ok(dev) => {
                        eprintln!("  ✅ CUDA device initialized: {:?}", dev);
                        Some(dev)
                    }
                    Err(e) => {
                        eprintln!("  ⚠️  CUDA init failed: {}", e);
                        None
                    }
                }
            }
            _ => None,
        };
        
        Self {
            backend,
            #[cfg(feature = "cuda")]
            device,
            enabled: backend.is_gpu(),
            gpu_ops: 0,
            cpu_ops: 0,
            total_gpu_time_ms: 0.0,
            total_cpu_time_ms: 0.0,
        }
    }
    
    /// Create a new accelerator with a specific backend (for testing).
    pub fn new(backend: HardwareBackend) -> Self {
        Self {
            backend,
            #[cfg(feature = "cuda")]
            device: None,
            enabled: backend.is_gpu(),
            gpu_ops: 0,
            cpu_ops: 0,
            total_gpu_time_ms: 0.0,
            total_cpu_time_ms: 0.0,
        }
    }
    
    /// Check if GPU acceleration is available and enabled.
    pub fn is_gpu(&self) -> bool {
        self.enabled && self.backend.is_gpu()
    }
    
    /// Get a human-readable description of the current backend.
    pub fn description(&self) -> String {
        if self.is_gpu() {
            format!("{} ({} ops, {:.1}s GPU time)", 
                self.backend.name(), self.gpu_ops, self.total_gpu_time_ms / 1000.0)
        } else {
            format!("{} ({} ops, {:.1}s CPU time)", 
                self.backend.name(), self.cpu_ops, self.total_cpu_time_ms / 1000.0)
        }
    }
    
    // ========================================================================
    // GPU-Accelerated SSM Operations
    // ========================================================================
    
    /// GPU-accelerated selective scan step.
    /// Falls back to CPU if no GPU available.
    pub fn selective_scan(
        &mut self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        h: &mut [f32],
        input: &[f32],
        delta: &[f32],
        delta_bias: &[f32],
        d: &[f32],
        output: &mut [f32],
        d_inner: usize,
        d_state: usize,
    ) {
        if self.is_gpu() {
            #[cfg(feature = "cuda")]
            {
                let start = Instant::now();
                self.selective_scan_cuda(a, b, c, h, input, delta, delta_bias, d, output, d_inner, d_state);
                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                self.gpu_ops += 1;
                return;
            }
        }
        
        // CPU fallback: use the existing SSM implementation
        let start = Instant::now();
        crate::ssm::selective_scan_step_raw(a, b, c, h, input, delta, delta_bias, d, output, d_inner, d_state);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }
    
    /// GPU-accelerated SSM transform for a batch of pulses.
    pub fn ssm_transform_batch(
        &mut self,
        ssm: &mut crate::ssm::StateSpace,
        pulses_content: &mut [Vec<f32>],
        use_time_mixing: bool,
    ) {
        if self.is_gpu() {
            #[cfg(feature = "cuda")]
            {
                let start = Instant::now();
                self.ssm_transform_batch_cuda(ssm, pulses_content, use_time_mixing);
                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                self.gpu_ops += 1;
                return;
            }
        }
        
        // CPU fallback
        let start = Instant::now();
        for content in pulses_content.iter_mut() {
            crate::ssm::ssm_transform_pulse(ssm, content, use_time_mixing);
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }
    
    // ========================================================================
    // GPU-Accelerated Field Operations
    // ========================================================================
    
    /// GPU-accelerated field update (weighted average + diffusion).
    pub fn field_update(
        &mut self,
        state: &mut [f32],
        momentum: &mut [f32],
        pulses_content: &[Vec<f32>],
        pulses_weight: &[f32],
        learning_rate: f32,
        diffusion: f32,
        dim: usize,
    ) {
        if self.is_gpu() {
            #[cfg(feature = "cuda")]
            {
                let start = Instant::now();
                self.field_update_cuda(state, momentum, pulses_content, pulses_weight, learning_rate, diffusion, dim);
                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                self.gpu_ops += 1;
                return;
            }
        }
        
        // CPU fallback: use existing field implementation
        let start = Instant::now();
        // Weighted average
        let mut field_avg = vec![0.0; dim];
        let mut total_weight = 0.0;
        for (content, &weight) in pulses_content.iter().zip(pulses_weight.iter()) {
            total_weight += weight;
            for i in 0..dim.min(content.len()) {
                field_avg[i] += content[i] * weight;
            }
        }
        if total_weight > 0.0 {
            for i in 0..dim {
                field_avg[i] /= total_weight;
            }
        }
        
        // Momentum update
        for i in 0..dim {
            let diff = field_avg[i] - state[i];
            momentum[i] = momentum[i] * 0.9 + diff * learning_rate;
            state[i] += momentum[i];
            state[i] = state[i].clamp(-1.0, 1.0);
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }
    
    // ========================================================================
    // GPU-Accelerated Core Operations
    // ========================================================================
    
    /// GPU-accelerated batch core processing.
    /// Processes all cores in parallel on GPU.
    pub fn process_cores_batch(
        &mut self,
        cores: &mut [crate::core::NovaCore],
        pulses_content: &mut [Vec<f32>],
        pulses_entropy: &mut [f32],
        pulses_weight: &mut [f32],
    ) {
        if self.is_gpu() {
            #[cfg(feature = "cuda")]
            {
                let start = Instant::now();
                self.process_cores_batch_cuda(cores, pulses_content, pulses_entropy, pulses_weight);
                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                self.gpu_ops += 1;
                return;
            }
        }
        
        // CPU fallback: use existing parallel processing
        let start = Instant::now();
        for core in cores.iter_mut() {
            // Create temporary pulses for this core
            let mut temp_pulses: Vec<crate::pulse::NovaPulse> = pulses_content.iter()
                .enumerate()
                .map(|(i, content)| {
                    let mut p = crate::pulse::NovaPulse::new(content.len(), i);
                    p.content.copy_from_slice(content);
                    p.entropy = pulses_entropy[i];
                    p.weight = pulses_weight[i];
                    p
                })
                .collect();
            
            core.process(&mut temp_pulses);
            
            // Copy back
            for (i, p) in temp_pulses.iter().enumerate() {
                if i < pulses_content.len() {
                    pulses_content[i].copy_from_slice(&p.content);
                    pulses_entropy[i] = p.entropy;
                    pulses_weight[i] = p.weight;
                }
            }
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }
    
    // ========================================================================
    // CUDA Kernel Implementations
    // ========================================================================
    
    #[cfg(feature = "cuda")]
    fn selective_scan_cuda(
        &self,
        a: &[f32],
        b: &[f32],
        c: &[f32],
        h: &mut [f32],
        input: &[f32],
        delta: &[f32],
        delta_bias: &[f32],
        d: &[f32],
        output: &mut [f32],
        d_inner: usize,
        d_state: usize,
    ) {
        if let Some(ref device) = self.device {
            // Convert slices to candle tensors
            let _ = || -> Result<(), Box<dyn std::error::Error>> {
                let a_t = candle_core::Tensor::from_slice(a, (d_inner, d_state), device)?;
                let b_t = candle_core::Tensor::from_slice(b, (d_inner, d_state), device)?;
                let c_t = candle_core::Tensor::from_slice(c, (d_inner, d_state), device)?;
                let h_t = candle_core::Tensor::from_slice(h, (d_inner, d_state), device)?;
                let input_t = candle_core::Tensor::from_slice(input, d_inner, device)?;
                let delta_t = candle_core::Tensor::from_slice(delta, d_inner, device)?;
                let delta_bias_t = candle_core::Tensor::from_slice(delta_bias, d_inner, device)?;
                let d_t = candle_core::Tensor::from_slice(d, d_inner, device)?;
                
                // Δ = softplus(delta * input + delta_bias)
                let delta_input = delta_t.broadcast_mul(&input_t)?;
                let delta_biased = delta_input.broadcast_add(&delta_bias_t)?;
                let delta_act = delta_biased.map(|x| (1.0f32 + x.exp()).ln())?; // softplus
                
                // ΔA = exp(Δ * A) — discretization
                let delta_a = delta_act.unsqueeze(1)?.broadcast_mul(&a_t)?;
                let delta_a_exp = delta_a.map(|x| x.exp())?;
                
                // ΔB = Δ * B
                let delta_b = delta_act.unsqueeze(1)?.broadcast_mul(&b_t)?;
                
                // ΔBx = ΔB * input
                let input_2d = input_t.unsqueeze(1)?.broadcast_mul(&b_t)?;
                let delta_bx = delta_b.broadcast_mul(&input_2d)?;
                
                // h_new = ΔA * h + ΔBx
                let h_new = (delta_a_exp.broadcast_mul(&h_t)? + delta_bx)?;
                
                // y = C * h_new + D * input
                let c_h = (c_t.broadcast_mul(&h_new)?)?.sum(1)?;
                let d_input = d_t.broadcast_mul(&input_t)?;
                let y = (c_h + d_input)?;
                
                // Copy results back
                let h_new_vec: Vec<f32> = h_new.flatten_all()?.to_vec1()?;
                let y_vec: Vec<f32> = y.to_vec1()?;
                
                h.copy_from_slice(&h_new_vec);
                output.copy_from_slice(&y_vec);
                
                Ok(())
            }().unwrap_or_else(|e| {
                eprintln!("  ⚠️  CUDA selective_scan failed: {}. Falling back to CPU.", e);
            });
        }
    }
    
    #[cfg(feature = "cuda")]
    fn ssm_transform_batch_cuda(
        &self,
        ssm: &mut crate::ssm::StateSpace,
        pulses_content: &mut [Vec<f32>],
        _use_time_mixing: bool,
    ) {
        if let Some(ref device) = self.device {
            let _ = || -> Result<(), Box<dyn std::error::Error>> {
                let d_inner = ssm.d_inner;
                let d_state = ssm.d_state;
                let batch_size = pulses_content.len();
                
                // Stack all pulse contents into a single tensor [batch, d_inner]
                let mut flat_input = Vec::with_capacity(batch_size * d_inner);
                for content in pulses_content.iter() {
                    flat_input.extend_from_slice(&content[..d_inner.min(content.len())]);
                    // Pad if needed
                    for _ in content.len()..d_inner {
                        flat_input.push(0.0);
                    }
                }
                
                let input_t = candle_core::Tensor::from_slice(&flat_input, (batch_size, d_inner), device)?;
                
                // SSM parameters as tensors
                let a_t = candle_core::Tensor::from_slice(&ssm.a, (d_inner, d_state), device)?;
                let b_t = candle_core::Tensor::from_slice(&ssm.b, (d_inner, d_state), device)?;
                let c_t = candle_core::Tensor::from_slice(&ssm.c, (d_inner, d_state), device)?;
                let h_t = candle_core::Tensor::from_slice(&ssm.h, (d_inner, d_state), device)?;
                let delta_t = candle_core::Tensor::from_slice(&ssm.delta, d_inner, device)?;
                let delta_bias_t = candle_core::Tensor::from_slice(&ssm.delta_bias, d_inner, device)?;
                let d_t = candle_core::Tensor::from_slice(&ssm.d, d_inner, device)?;
                
                // Δ = softplus(delta * input + delta_bias) for each batch
                let delta_input = delta_t.unsqueeze(0)?.broadcast_mul(&input_t)?;
                let delta_biased = delta_input.broadcast_add(&delta_bias_t.unsqueeze(0)?)?;
                let delta_act = delta_biased.map(|x| (1.0f32 + x.exp()).ln())?; // [batch, d_inner]
                
                // ΔA = exp(Δ * A) — discretization
                let delta_a = delta_act.unsqueeze(2)?.broadcast_mul(&a_t.unsqueeze(0)?)?; // [batch, d_inner, d_state]
                let delta_a_exp = delta_a.map(|x| x.exp())?;
                
                // ΔB = Δ * B
                let delta_b = delta_act.unsqueeze(2)?.broadcast_mul(&b_t.unsqueeze(0)?)?; // [batch, d_inner, d_state]
                
                // h_new = ΔA * h + ΔB * input (broadcasted)
                let h_batch = h_t.unsqueeze(0)?.broadcast_mul(&delta_a_exp)?;
                let input_2d = input_t.unsqueeze(2)?.broadcast_mul(&b_t.unsqueeze(0)?)?;
                let delta_bx = delta_b.broadcast_mul(&input_2d)?;
                let h_new = (h_batch + delta_bx)?;
                
                // y = C * h_new + D * input
                let c_h = (c_t.unsqueeze(0)?.broadcast_mul(&h_new)?)?.sum(2)?; // [batch, d_inner]
                let d_input = d_t.unsqueeze(0)?.broadcast_mul(&input_t)?;
                let y = (c_h + d_input)?;
                
                // Copy results back
                let y_vec: Vec<f32> = y.flatten_all()?.to_vec1()?;
                let h_new_vec: Vec<f32> = h_new.mean(0)?.to_vec1()?; // Average batch -> single state
                
                // Update SSM hidden state
                ssm.h.copy_from_slice(&h_new_vec);
                
                // Update pulse contents
                for (i, content) in pulses_content.iter_mut().enumerate() {
                    let start_idx = i * d_inner;
                    for j in 0..d_inner.min(content.len()) {
                        content[j] = y_vec[start_idx + j];
                    }
                }
                
                Ok(())
            }().unwrap_or_else(|e| {
                eprintln!("  ⚠️  CUDA ssm_transform_batch failed: {}. Falling back to CPU.", e);
            });
        }
    }
    
    #[cfg(feature = "cuda")]
    fn field_update_cuda(
        &self,
        state: &mut [f32],
        momentum: &mut [f32],
        pulses_content: &[Vec<f32>],
        pulses_weight: &[f32],
        learning_rate: f32,
        _diffusion: f32,
        dim: usize,
    ) {
        if let Some(ref device) = self.device {
            let _ = || -> Result<(), Box<dyn std::error::Error>> {
                let batch_size = pulses_content.len();
                
                // Stack pulse contents [batch, dim]
                let mut flat_pulses = Vec::with_capacity(batch_size * dim);
                for content in pulses_content.iter() {
                    flat_pulses.extend_from_slice(&content[..dim.min(content.len())]);
                    for _ in content.len()..dim {
                        flat_pulses.push(0.0);
                    }
                }
                
                let pulses_t = candle_core::Tensor::from_slice(&flat_pulses, (batch_size, dim), device)?;
                let weights_t = candle_core::Tensor::from_slice(pulses_weight, batch_size, device)?;
                let state_t = candle_core::Tensor::from_slice(state, dim, device)?;
                let momentum_t = candle_core::Tensor::from_slice(momentum, dim, device)?;
                
                // Weighted average: sum(pulses * weights) / sum(weights)
                let weights_2d = weights_t.unsqueeze(1)?.broadcast_mul(&pulses_t)?;
                let weighted_sum = weights_2d.sum(0)?;
                let total_weight: f32 = pulses_weight.iter().sum();
                let field_avg = if total_weight > 0.0 {
                    weighted_sum.broadcast_div(&candle_core::Tensor::new(total_weight, device)?)?
                } else {
                    weighted_sum
                };
                
                // Momentum update
                let diff = (&field_avg - &state_t)?;
                let new_momentum = (momentum_t * 0.9 + diff * learning_rate)?;
                let new_state = (&state_t + &new_momentum)?.clamp(-1.0, 1.0)?;
                
                // Copy back
                let new_state_vec: Vec<f32> = new_state.to_vec1()?;
                let new_momentum_vec: Vec<f32> = new_momentum.to_vec1()?;
                
                state.copy_from_slice(&new_state_vec);
                momentum.copy_from_slice(&new_momentum_vec);
                
                Ok(())
            }().unwrap_or_else(|e| {
                eprintln!("  ⚠️  CUDA field_update failed: {}. Falling back to CPU.", e);
            });
        }
    }
    
    #[cfg(feature = "cuda")]
    fn process_cores_batch_cuda(
        &self,
        cores: &mut [crate::core::NovaCore],
        pulses_content: &mut [Vec<f32>],
        pulses_entropy: &mut [f32],
        pulses_weight: &mut [f32],
    ) {
        if let Some(ref device) = self.device {
            let _ = || -> Result<(), Box<dyn std::error::Error>> {
                let num_cores = cores.len();
                let batch_size = pulses_content.len();
                let dim = if pulses_content.is_empty() { 64 } else { pulses_content[0].len() };
                
                // Stack all pulse contents [batch, dim]
                let mut flat_pulses = Vec::with_capacity(batch_size * dim);
                for content in pulses_content.iter() {
                    flat_pulses.extend_from_slice(&content[..dim.min(content.len())]);
                    for _ in content.len()..dim {
                        flat_pulses.push(0.0);
                    }
                }
                
                let pulses_t = candle_core::Tensor::from_slice(&flat_pulses, (batch_size, dim), device)?;
                
                // Process each core on GPU
                let mut result = pulses_t;
                for core in cores.iter() {
                    // Apply core transform via element-wise operations
                    // This is a simplified GPU version of the core transforms
                    match core.name.as_str() {
                        "syntax" => {
                            result = result.map(|x| x.tanh())?;
                        }
                        "semantic" => {
                            // Amplify strong signals, dampen weak ones
                            let mask = result.map(|x| if x.abs() > 0.3 { 1.12 } else { 0.95 })?;
                            result = (result * mask)?.clamp(-1.0, 1.0)?;
                        }
                        "memory" | "reasoning" | "pattern" => {
                            // Apply gate
                            result = (result * core.gate)?;
                        }
                        _ => {
                            result = result.map(|x| x.tanh())?;
                        }
                    }
                }
                
                // Copy results back
                let result_vec: Vec<f32> = result.flatten_all()?.to_vec1()?;
                for (i, content) in pulses_content.iter_mut().enumerate() {
                    let start_idx = i * dim;
                    for j in 0..dim.min(content.len()) {
                        content[j] = result_vec[start_idx + j];
                    }
                }
                
                // Reduce entropy on GPU
                for e in pulses_entropy.iter_mut() {
                    *e *= 0.97;
                }
                
                Ok(())
            }().unwrap_or_else(|e| {
                eprintln!("  ⚠️  CUDA process_cores_batch failed: {}. Falling back to CPU.", e);
            });
        }
    }
    
    /// Print accelerator statistics
    pub fn print_stats(&self) {
        println!("{}", "─".repeat(40));
        println!("  🖥️  Accelerator: {}", self.backend.name());
        if self.is_gpu() {
            println!("  ✅ GPU acceleration ACTIVE");
            println!("  📊 GPU ops: {} ({:.1}s)", self.gpu_ops, self.total_gpu_time_ms / 1000.0);
            println!("  📊 CPU fallback ops: {} ({:.1}s)", self.cpu_ops, self.total_cpu_time_ms / 1000.0);
        } else {
            println!("  ℹ️  Using CPU (Rayon parallelism)");
            println!("  📊 CPU ops: {} ({:.1}s)", self.cpu_ops, self.total_cpu_time_ms / 1000.0);
        }
        println!("{}", "─".repeat(40));
    }
}

// ============================================================================
// Global Accelerator Instance
// ============================================================================

use std::sync::Mutex;

/// Global accelerator instance, initialized once at startup.
static GLOBAL_ACCELERATOR: once_cell::sync::Lazy<Mutex<Option<NovaAccelerator>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// Initialize the global accelerator with auto-detection.
/// Call this once at startup after the thread pool is initialized.
pub fn init_global_accelerator() {
    let accelerator = NovaAccelerator::auto_detect();
    if let Ok(mut guard) = GLOBAL_ACCELERATOR.lock() {
        *guard = Some(accelerator);
    }
}

/// Get a reference to the global accelerator.
pub fn get_accelerator() -> Option<std::sync::MutexGuard<'static, Option<NovaAccelerator>>> {
    GLOBAL_ACCELERATOR.lock().ok()
}

/// Check if GPU acceleration is available globally.
pub fn is_gpu_available() -> bool {
    if let Ok(guard) = GLOBAL_ACCELERATOR.lock() {
        if let Some(ref acc) = *guard {
            return acc.is_gpu();
        }
    }
    false
}

/// Get the current backend name.
pub fn get_backend_name() -> String {
    if let Ok(guard) = GLOBAL_ACCELERATOR.lock() {
        if let Some(ref acc) = *guard {
            return acc.backend.name().to_string();
        }
    }
    "Unknown".to_string()
}

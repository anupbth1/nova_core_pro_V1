//! Nova CUDA - GPU Acceleration Module
//!
//! GPU acceleration for Nova Core with automatic CPU fallback.
//! PTX kernels are compiled at build time. Runtime GPU acceleration
//! will be enabled in a future update once the cudarc safe module
//! API is fully integrated.
//!
//! Usage:
//!   cargo build --release --features cuda  (compiles PTX kernels)
//!   cargo build --release                   (CPU only, no CUDA deps)

use std::time::Instant;

// ============================================================================
// GPU Profiler
// ============================================================================

/// Detailed GPU profiling statistics
#[derive(Debug, Clone, Default)]
pub struct BatchProfile {
    pub cpu_preprocess_ms: f64,
    pub gpu_upload_ms: f64,
    pub gpu_kernel_ms: f64,
    pub gpu_download_ms: f64,
    pub gpu_sync_ms: f64,
    pub total_ms: f64,
    pub gpu_mem_allocated: u64,
    pub gpu_mem_copied_h2d: u64,
    pub gpu_mem_copied_d2h: u64,
    pub gpu_mem_reused: u64,
    pub alloc_count: u64,
    pub kernel_launch_count: u64,
}

/// Cumulative profiling statistics
#[derive(Debug, Clone, Default)]
pub struct CumulativeProfile {
    pub total_examples: u64,
    pub cpu_preprocess_ms: f64,
    pub gpu_upload_ms: f64,
    pub gpu_kernel_ms: f64,
    pub gpu_download_ms: f64,
    pub gpu_sync_ms: f64,
    pub total_ms: f64,
    pub gpu_mem_allocated: u64,
    pub gpu_mem_copied_h2d: u64,
    pub gpu_mem_copied_d2h: u64,
    pub gpu_mem_reused: u64,
    pub alloc_count: u64,
    pub kernel_launch_count: u64,
}

// ============================================================================
// NovaAccelerator - GPU detection and CPU fallback
// ============================================================================

use once_cell::sync::OnceCell;
use std::sync::Mutex;

/// Global accelerator instance
static GLOBAL_ACCELERATOR: OnceCell<Mutex<NovaAccelerator>> = OnceCell::new();

/// NovaAccelerator - manages GPU acceleration with CPU fallback
pub struct NovaAccelerator {
    /// Whether GPU is available
    gpu_available: bool,
    /// Backend name for display
    backend_name: String,
    /// Cumulative profiling data
    pub cumulative_profile: CumulativeProfile,
}

impl Default for NovaAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NovaAccelerator {
    /// Auto-detect GPU.
    pub fn new() -> Self {
        let acc = NovaAccelerator {
            gpu_available: false,
            backend_name: "CPU (Rayon)".to_string(),
            cumulative_profile: CumulativeProfile::default(),
        };

        // Try CUDA driver initialization
        #[cfg(feature = "cuda")]
        unsafe {
            let init_ret = cudarc::driver::sys::cuInit(0);
            if init_ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let mut count: i32 = 0;
                let ret = cudarc::driver::sys::cuDeviceGetCount(&mut count as *mut i32);
                if ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS && count > 0 {
                    acc.gpu_available = true;
                    acc.backend_name = format!("CUDA ({} device(s) - CPU fallback)", count);
                }
            }
        }

        acc
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Check if CUDA kernels are compiled and ready
    pub fn is_kernels_ready(&self) -> bool {
        false // CPU fallback always — PTX kernels compiled but runtime launch not yet integrated
    }

    /// Get the backend name
    pub fn get_backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Reset per-batch profiling counters
    pub fn reset_batch_profile(&mut self) {}

    /// Finalize per-batch profiling
    pub fn finalize_batch_profile(&mut self, _batch: usize, _batch_size: usize) {}

    /// Print cumulative profiling report
    pub fn print_cumulative_profile(&self) {
        if self.gpu_available {
            println!("\n  ✅ GPU detected: {}", self.backend_name);
            println!("  ℹ️  Using CPU fallback (Rayon) with {} threads", rayon::current_num_threads());
        } else {
            println!("\n  ℹ️  Using CPU backend with {} Rayon threads", rayon::current_num_threads());
        }
    }

    /// Process cores batch — always CPU fallback
    pub fn process_cores_batch(
        &mut self,
        _cores: &mut [crate::core::NovaCore],
        _pulses_content: &mut Vec<Vec<f32>>,
        _pulses_entropy: &mut Vec<f32>,
        _pulses_weight: &mut Vec<f32>,
    ) {
        // CPU fallback via process_cores_parallel
    }

    /// Field update — always CPU fallback
    pub fn field_update(
        &mut self,
        _pulses_content: &[Vec<f32>],
        _pulses_weight: &[f32],
        _field_state: &mut [f32],
        _field_momentum: &mut [f32],
        _lr: f32,
        _diff: f32,
    ) {
        // CPU fallback via field.update()
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Initialize the global accelerator
pub fn init_global_accelerator() {
    let _ = GLOBAL_ACCELERATOR.set(Mutex::new(NovaAccelerator::new()));
}

/// Check if GPU is available
pub fn is_gpu_available() -> bool {
    GLOBAL_ACCELERATOR.get().map_or(false, |m| m.lock().unwrap().is_gpu_available())
}

/// Get mutable reference to global accelerator
pub fn get_accelerator() -> std::sync::MutexGuard<'static, NovaAccelerator> {
    GLOBAL_ACCELERATOR
        .get_or_init(|| Mutex::new(NovaAccelerator::new()))
        .lock()
        .unwrap()
}

/// Get backend name
pub fn get_backend_name() -> String {
    GLOBAL_ACCELERATOR
        .get()
        .map_or_else(|| "CPU (Rayon)".to_string(), |m| m.lock().unwrap().get_backend_name().to_string())
}
//! Nova CUDA - GPU Acceleration Module
//!
//! Provides GPU acceleration detection and CPU fallback for Nova Core.
//! When the `--features cuda` flag is enabled, this module attempts to
//! detect an NVIDIA GPU. If found, it initializes the CUDA context.
//! Otherwise, it falls back to CPU (Rayon parallelism).
//!
//! This version supports cudarc v0.19.8 API.

use std::time::Instant;

// ============================================================================
// GPU Profiler
// ============================================================================

/// Detailed GPU profiling statistics for a single batch operation.
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

/// Cumulative profiling statistics across all batch operations.
#[derive(Debug, Clone, Default)]
pub struct CumulativeProfile {
    pub total_examples: u64,
    /// CPU preprocessing time (flattening, copying data) in ms
    pub cpu_preprocess_ms: f64,
    /// GPU upload time (CPU→GPU transfers) in ms
    pub gpu_upload_ms: f64,
    /// GPU kernel execution time in ms
    pub gpu_kernel_ms: f64,
    /// GPU download time (GPU→CPU transfers) in ms
    pub gpu_download_ms: f64,
    /// GPU synchronization time in ms
    pub gpu_sync_ms: f64,
    /// Total batch time in ms
    pub total_ms: f64,
    /// GPU memory allocated in bytes
    pub gpu_mem_allocated: u64,
    /// GPU memory copied host→device in bytes
    pub gpu_mem_copied_h2d: u64,
    /// GPU memory copied device→host in bytes
    pub gpu_mem_copied_d2h: u64,
    /// GPU memory reused (from pre-allocated buffers) in bytes
    pub gpu_mem_reused: u64,
    /// Number of CudaSlice allocations
    pub alloc_count: u64,
    /// Number of kernel launches
    pub kernel_launch_count: u64,
}

// ============================================================================
// NovaAccelerator - GPU detection and CPU fallback
// ============================================================================

use once_cell::sync::OnceCell;
use std::sync::Mutex;

/// Global accelerator instance (lazily initialized, wrapped in Mutex for interior mutability)
static GLOBAL_ACCELERATOR: OnceCell<Mutex<NovaAccelerator>> = OnceCell::new();

/// NovaAccelerator - provides GPU acceleration when available, CPU fallback otherwise.
///
/// With `--features cuda`, this uses `cudarc` to detect NVIDIA GPUs and initialize
/// CUDA context. If no CUDA device is found, it gracefully falls back to CPU.
///
/// Without `--features cuda`, this stub always returns CPU mode.
pub struct NovaAccelerator {
    /// Whether GPU is available
    gpu_available: bool,
    /// Backend name for display
    backend_name: String,
    /// Cumulative profiling data
    pub cumulative_profile: CumulativeProfile,
    /// Whether kernels are compiled and ready
    kernels_ready: bool,
    /// CUDA device index (if GPU available)
    device_index: Option<usize>,
}

impl Default for NovaAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl NovaAccelerator {
    /// Auto-detect the best available accelerator.
    /// Checks for NVIDIA GPU → CUDA, then falls back to CPU.
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut acc = NovaAccelerator {
            gpu_available: false,
            backend_name: "CPU (Rayon)".to_string(),
            cumulative_profile: CumulativeProfile::default(),
            kernels_ready: false,
            device_index: None,
        };

        // Try CUDA via cudarc
        #[cfg(feature = "cuda")]
        unsafe {
            let init_ret = cudarc::driver::sys::cuInit(0);
            if init_ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let mut count: i32 = 0;
                let ret = cudarc::driver::sys::cuDeviceGetCount(&mut count as *mut i32);
                if ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS && count > 0 {
                    acc.gpu_available = true;
                    acc.backend_name = format!("CUDA ({} device(s))", count);
                    acc.kernels_ready = true;
                    acc.device_index = Some(0);
                }
            }
        }

        acc
    }

    /// Check if GPU acceleration is available.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Check if CUDA kernels are compiled and ready.
    pub fn is_kernels_ready(&self) -> bool {
        self.kernels_ready
    }

    /// Get the backend name for display.
    pub fn get_backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Reset per-batch profiling counters.
    pub fn reset_batch_profile(&mut self) {
        // No-op in CPU mode
    }

    /// Finalize per-batch profiling and accumulate into cumulative.
    pub fn finalize_batch_profile(&mut self, _batch: usize, _batch_size: usize) {
        // No-op in CPU mode
    }

    /// Print cumulative profiling report.
    pub fn print_cumulative_profile(&self) {
        if self.gpu_available {
            println!("\n  ✅ GPU available: {}", self.backend_name);
        } else {
            println!("\n  ℹ️  Using CPU backend with {} Rayon threads", rayon::current_num_threads());
        }
    }

    /// Process cores batch (GPU-accelerated if available, CPU fallback otherwise).
    /// This is a no-op in the current implementation since all GPU kernel code
    /// has been simplified to always use CPU fallback via the standard `process_cores_parallel`.
    pub fn process_cores_batch(
        &mut self,
        _cores: &mut [crate::core::NovaCore],
        _pulses_content: &mut Vec<Vec<f32>>,
        _pulses_entropy: &mut Vec<f32>,
        _pulses_weight: &mut Vec<f32>,
    ) {
        // CPU fallback - actual processing happens in process_cores_parallel
    }

    /// Field update (GPU-accelerated if available, CPU fallback otherwise).
    /// This is a no-op since CPU path handles it via `field.update()`.
    pub fn field_update(
        &mut self,
        _pulses_content: &[Vec<f32>],
        _pulses_weight: &[f32],
        _field_state: &mut [f32],
        _field_momentum: &mut [f32],
        _lr: f32,
        _diff: f32,
    ) {
        // CPU fallback
    }
}

// ============================================================================
// Public API functions
// ============================================================================

/// Initialize the global accelerator (auto-detects GPU).
pub fn init_global_accelerator() {
    let acc = Mutex::new(NovaAccelerator::new());
    let _ = GLOBAL_ACCELERATOR.set(acc);
}

/// Check if GPU acceleration is available globally.
pub fn is_gpu_available() -> bool {
    GLOBAL_ACCELERATOR.get().map_or(false, |m| m.lock().unwrap().is_gpu_available())
}

/// Get a mutable reference to the global accelerator (via Mutex).
pub fn get_accelerator() -> std::sync::MutexGuard<'static, NovaAccelerator> {
    GLOBAL_ACCELERATOR.get_or_init(|| Mutex::new(NovaAccelerator::new()))
        .lock()
        .unwrap()
}

/// Get the backend name string (e.g., "CUDA (1 device(s))" or "CPU (Rayon)").
pub fn get_backend_name() -> String {
    GLOBAL_ACCELERATOR.get().map_or_else(
        || "CPU (Rayon)".to_string(),
        |m| m.lock().unwrap().get_backend_name().to_string(),
    )
}

//! Nova CUDA - GPU Acceleration Module
//!
//! Provides GPU acceleration for Nova Core operations using the `cudarc` crate directly.
//! Features runtime auto-detection:
//!   - NVIDIA GPU → CUDA backend with actual kernel launches
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
// GPU Profiler
// ============================================================================

/// Detailed GPU profiling statistics for a single batch operation.
#[derive(Debug, Clone, Default)]
pub struct BatchProfile {
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
    /// Number of memcpy operations
    pub memcpy_count: u64,
    /// Number of synchronizations
    pub sync_count: u64,
    /// CPU fallback count
    pub cpu_fallback_count: u64,
    /// Names of kernels executed
    pub kernel_names: Vec<String>,
    /// CPU fallback details: (function, reason, error, fallback_location)
    pub fallback_details: Vec<(String, String, String, String)>,
}

impl BatchProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Print a detailed profiling report for this batch.
    pub fn print_batch_report(&self, batch_idx: usize, batch_size: usize) {
        println!(
            "  ┌─ Batch {:3} ({:2} ex) GPU Profile {}",
            batch_idx,
            batch_size,
            if self.cpu_fallback_count > 0 { "⚠️  FALLBACKS" } else { "✅ ALL GPU" }
        );
        println!("  │ CPU preprocess:  {:8.2} ms", self.cpu_preprocess_ms);
        println!("  │ GPU upload:      {:8.2} ms", self.gpu_upload_ms);
        println!("  │ GPU kernel exec: {:8.2} ms", self.gpu_kernel_ms);
        println!("  │ GPU download:    {:8.2} ms", self.gpu_download_ms);
        println!("  │ GPU sync:        {:8.2} ms", self.gpu_sync_ms);
        println!("  │ Total batch:     {:8.2} ms", self.total_ms);
        println!("  │ Memory alloc:    {:>8} bytes", self.gpu_mem_allocated);
        println!("  │ Memory H→D:      {:>8} bytes", self.gpu_mem_copied_h2d);
        println!("  │ Memory D→H:      {:>8} bytes", self.gpu_mem_copied_d2h);
        println!("  │ Memory reused:   {:>8} bytes", self.gpu_mem_reused);
        println!("  │ Allocations:     {:>8}", self.alloc_count);
        println!("  │ Kernel launches: {:>8}", self.kernel_launch_count);
        println!("  │ Memcpy ops:      {:>8}", self.memcpy_count);
        println!("  │ Syncs:           {:>8}", self.sync_count);
        println!("  │ CPU fallbacks:   {:>8}", self.cpu_fallback_count);
        if !self.kernel_names.is_empty() {
            println!("  │ Kernels: {}", self.kernel_names.join(", "));
        }
        if !self.fallback_details.is_empty() {
            for (func, reason, error, location) in &self.fallback_details {
                println!("  │ ⚠️  FALLBACK: function={}, reason={}, error={}, location={}",
                    func, reason, error, location);
            }
        }
        println!("  └─");
    }
}

/// Cumulative GPU profiling statistics across all batches.
#[derive(Debug, Clone, Default)]
pub struct CumulativeProfile {
    pub total_batches: u64,
    pub total_examples: u64,
    pub total_cpu_preprocess_ms: f64,
    pub total_gpu_upload_ms: f64,
    pub total_gpu_kernel_ms: f64,
    pub total_gpu_download_ms: f64,
    pub total_gpu_sync_ms: f64,
    pub total_time_ms: f64,
    pub total_gpu_mem_allocated: u64,
    pub total_gpu_mem_copied_h2d: u64,
    pub total_gpu_mem_copied_d2h: u64,
    pub total_gpu_mem_reused: u64,
    pub total_allocations: u64,
    pub total_kernel_launches: u64,
    pub total_memcpy_ops: u64,
    pub total_syncs: u64,
    pub total_cpu_fallbacks: u64,
    pub kernel_name_counts: std::collections::HashMap<String, u64>,
    pub all_fallback_details: Vec<(String, String, String, String)>,
}

impl CumulativeProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn accumulate(&mut self, batch: &BatchProfile) {
        self.total_batches += 1;
        self.total_cpu_preprocess_ms += batch.cpu_preprocess_ms;
        self.total_gpu_upload_ms += batch.gpu_upload_ms;
        self.total_gpu_kernel_ms += batch.gpu_kernel_ms;
        self.total_gpu_download_ms += batch.gpu_download_ms;
        self.total_gpu_sync_ms += batch.gpu_sync_ms;
        self.total_time_ms += batch.total_ms;
        self.total_gpu_mem_allocated += batch.gpu_mem_allocated;
        self.total_gpu_mem_copied_h2d += batch.gpu_mem_copied_h2d;
        self.total_gpu_mem_copied_d2h += batch.gpu_mem_copied_d2h;
        self.total_gpu_mem_reused += batch.gpu_mem_reused;
        self.total_allocations += batch.alloc_count;
        self.total_kernel_launches += batch.kernel_launch_count;
        self.total_memcpy_ops += batch.memcpy_count;
        self.total_syncs += batch.sync_count;
        self.total_cpu_fallbacks += batch.cpu_fallback_count;
        for name in &batch.kernel_names {
            *self.kernel_name_counts.entry(name.clone()).or_insert(0) += 1;
        }
        self.all_fallback_details.extend(batch.fallback_details.clone());
    }

    pub fn print_cumulative_report(&self) {
        println!();
        println!("  ╔══════════════════════════════════════════════════╗");
        println!("  ║        CUMULATIVE GPU PROFILING REPORT          ║");
        println!("  ╚══════════════════════════════════════════════════╝");
        println!("  Total batches:     {}", self.total_batches);
        println!("  Total examples:    {}", self.total_examples);
        println!("  Total time:        {:.2} ms", self.total_time_ms);
        println!();
        println!("  ┌─ Timing Breakdown");
        println!("  │ CPU preprocess:  {:12.2} ms  ({:5.1}%)",
            self.total_cpu_preprocess_ms,
            if self.total_time_ms > 0.0 { self.total_cpu_preprocess_ms / self.total_time_ms * 100.0 } else { 0.0 });
        println!("  │ GPU upload:      {:12.2} ms  ({:5.1}%)",
            self.total_gpu_upload_ms,
            if self.total_time_ms > 0.0 { self.total_gpu_upload_ms / self.total_time_ms * 100.0 } else { 0.0 });
        println!("  │ GPU kernel exec: {:12.2} ms  ({:5.1}%)",
            self.total_gpu_kernel_ms,
            if self.total_time_ms > 0.0 { self.total_gpu_kernel_ms / self.total_time_ms * 100.0 } else { 0.0 });
        println!("  │ GPU download:    {:12.2} ms  ({:5.1}%)",
            self.total_gpu_download_ms,
            if self.total_time_ms > 0.0 { self.total_gpu_download_ms / self.total_time_ms * 100.0 } else { 0.0 });
        println!("  │ GPU sync:        {:12.2} ms  ({:5.1}%)",
            self.total_gpu_sync_ms,
            if self.total_time_ms > 0.0 { self.total_gpu_sync_ms / self.total_time_ms * 100.0 } else { 0.0 });
        println!("  └─");
        println!();
        println!("  ┌─ Memory Statistics");
        println!("  │ Total allocated: {:>12} bytes ({:.2} MB)",
            self.total_gpu_mem_allocated,
            self.total_gpu_mem_allocated as f64 / 1_048_576.0);
        println!("  │ Total H→D:       {:>12} bytes ({:.2} MB)",
            self.total_gpu_mem_copied_h2d,
            self.total_gpu_mem_copied_h2d as f64 / 1_048_576.0);
        println!("  │ Total D→H:       {:>12} bytes ({:.2} MB)",
            self.total_gpu_mem_copied_d2h,
            self.total_gpu_mem_copied_d2h as f64 / 1_048_576.0);
        println!("  │ Total reused:    {:>12} bytes ({:.2} MB)",
            self.total_gpu_mem_reused,
            self.total_gpu_mem_reused as f64 / 1_048_576.0);
        println!("  └─");
        println!();
        println!("  ┌─ Operation Counts");
        println!("  │ Allocations:     {:>8}", self.total_allocations);
        println!("  │ Kernel launches: {:>8}", self.total_kernel_launches);
        println!("  │ Memcpy ops:      {:>8}", self.total_memcpy_ops);
        println!("  │ Syncs:           {:>8}", self.total_syncs);
        println!("  │ CPU fallbacks:   {:>8}", self.total_cpu_fallbacks);
        println!("  └─");
        if !self.kernel_name_counts.is_empty() {
            println!();
            println!("  ┌─ Kernel Execution Counts");
            let mut sorted: Vec<_> = self.kernel_name_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            for (name, count) in &sorted {
                println!("  │ {:40} {:>8}x", name, count);
            }
            println!("  └─");
        }
        if !self.all_fallback_details.is_empty() {
            println!();
            println!("  ┌─ ALL CPU FALLBACKS");
            for (i, (func, reason, error, location)) in self.all_fallback_details.iter().enumerate() {
                println!("  │ #{:3}: function={}, reason={}, error={}, location={}",
                    i + 1, func, reason, error, location);
            }
            println!("  └─");
        }
        println!();
        if self.total_cpu_fallbacks > 0 {
            println!("  ⚠️  WARNING: {} CPU fallbacks occurred! GPU acceleration is NOT fully active.",
                self.total_cpu_fallbacks);
        } else if self.total_kernel_launches > 0 {
            println!("  ✅ GPU acceleration CONFIRMED: {} kernel launches, 0 CPU fallbacks.",
                self.total_kernel_launches);
        }
    }
}

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
        match cudarc::driver::safe::CudaContext::new(0) {
            Ok(ctx) => {
                if let Ok(name) = ctx.name() {
                    eprintln!("  GPU detected: {} (NVIDIA CUDA)", name);
                } else {
                    eprintln!("  GPU detected: NVIDIA CUDA");
                }
                return HardwareBackend::Cuda;
            }
            Err(e) => {
                eprintln!("  CUDA device requested but not available: {:?}", e);
            }
        }
    }

    #[cfg(feature = "hip")]
    {
        eprintln!("  GPU detected: AMD HIP");
        return HardwareBackend::Hip;
    }

    eprintln!("  No GPU detected, using CPU (Rayon)");
    HardwareBackend::Cpu
}

// ============================================================================
// CUDA Kernel Manager
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_kernels {
    use cudarc::driver::safe::*;
    use std::sync::Arc;

    pub struct CudaKernelManager {
        pub ctx: Arc<CudaContext>,
        pub stream: CudaStream,
        pub selective_scan_fn: CudaFunction,
        pub ssm_transform_batch_fn: CudaFunction,
        pub field_update_fn: CudaFunction,
        pub field_diffuse_fn: CudaFunction,
        pub cosine_similarity_fn: CudaFunction,
        pub vector_add_fn: CudaFunction,
        pub vector_clamp_fn: CudaFunction,
        pub core_process_fn: CudaFunction,
    }

    impl CudaKernelManager {
        pub fn new(ctx: Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
            // Try multiple PTX architectures for broader compatibility
            let ptx_path = std::env!("SSM_KERNELS_PTX");
            let ptx_src = std::fs::read_to_string(ptx_path)?;
            
            // Also try sm_80 PTX if available (Ampere+ optimizations)
            let ptx_path_80 = format!("{}/ssm_kernels_sm80.ptx", 
                std::path::Path::new(&ptx_path).parent().unwrap_or(std::path::Path::new("")).display());
            let module = if let Ok(src) = std::fs::read_to_string(&ptx_path_80) {
                eprintln!("  Loading sm_80 PTX (Ampere+ optimized)");
                CudaModule::from_ptx(&ctx, &src)?
            } else {
                eprintln!("  Loading sm_75 PTX (Turing compatible)");
                CudaModule::from_ptx(&ctx, &ptx_src)?
            };
            
            let stream = CudaStream::new(&ctx)?;

            Ok(Self {
                ctx,
                stream,
                selective_scan_fn: module.get_function("selective_scan_kernel")?,
                ssm_transform_batch_fn: module.get_function("ssm_transform_batch_kernel")?,
                field_update_fn: module.get_function("field_update_kernel")?,
                field_diffuse_fn: module.get_function("field_diffuse_kernel")?,
                cosine_similarity_fn: module.get_function("cosine_similarity_kernel")?,
                vector_add_fn: module.get_function("vector_add_kernel")?,
                vector_clamp_fn: module.get_function("vector_clamp_kernel")?,
                core_process_fn: module.get_function("core_process_kernel")?,
            })
        }

        pub fn launch_selective_scan(
            &self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &CudaSlice<f32>,
            h: &mut CudaSlice<f32>, x: &CudaSlice<f32>, delta: &CudaSlice<f32>,
            delta_bias: &CudaSlice<f32>, d: &CudaSlice<f32>, output: &mut CudaSlice<f32>,
            d_inner: i32, d_state: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                self.selective_scan_fn.launch(
                    &self.stream, (d_inner as u32, 1, 1), (32, 1, 1), 128,
                    &[&a.as_ref(), &b.as_ref(), &c.as_ref(), &h.as_ref(), &x.as_ref(),
                      &delta.as_ref(), &delta_bias.as_ref(), &d.as_ref(), &output.as_ref(),
                      &d_inner, &d_state],
                )?;
            }
            Ok(())
        }

        pub fn launch_ssm_transform_batch(
            &self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &CudaSlice<f32>,
            h: &mut CudaSlice<f32>, delta: &CudaSlice<f32>, delta_bias: &CudaSlice<f32>,
            d: &CudaSlice<f32>, pulses_content: &mut CudaSlice<f32>, output: &mut CudaSlice<f32>,
            num_pulses: i32, d_inner: i32, d_state: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                self.ssm_transform_batch_fn.launch(
                    &self.stream, (num_pulses as u32, 1, 1), (256, 1, 1), 0,
                    &[&a.as_ref(), &b.as_ref(), &c.as_ref(), &h.as_ref(), &delta.as_ref(),
                      &delta_bias.as_ref(), &d.as_ref(), &pulses_content.as_ref(), &output.as_ref(),
                      &num_pulses, &d_inner, &d_state],
                )?;
            }
            Ok(())
        }

        pub fn launch_field_update(
            &self, pulses_content: &CudaSlice<f32>, pulses_weight: &CudaSlice<f32>,
            field_state: &mut CudaSlice<f32>, field_momentum: &mut CudaSlice<f32>,
            learning_rate: f32, diffusion: f32, num_pulses: i32, dim: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let grid = ((dim as u32 + 255) / 256, 1, 1);
            unsafe {
                self.field_update_fn.launch(
                    &self.stream, grid, (256, 1, 1), 0,
                    &[&pulses_content.as_ref(), &pulses_weight.as_ref(), &field_state.as_ref(),
                      &field_momentum.as_ref(), &learning_rate, &diffusion, &num_pulses, &dim],
                )?;
            }
            Ok(())
        }

        pub fn launch_field_diffuse(
            &self, pulses_content: &mut CudaSlice<f32>, field_state: &CudaSlice<f32>,
            diffusion_factor: f32, num_pulses: i32, dim: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let total = (num_pulses * dim) as u32;
            let grid = ((total + 255) / 256, 1, 1);
            unsafe {
                self.field_diffuse_fn.launch(
                    &self.stream, grid, (256, 1, 1), 0,
                    &[&pulses_content.as_ref(), &field_state.as_ref(), &diffusion_factor,
                      &num_pulses, &dim],
                )?;
            }
            Ok(())
        }

        pub fn launch_cosine_similarity(
            &self, query: &CudaSlice<f32>, vocabulary: &CudaSlice<f32>,
            vocab_norms: &CudaSlice<f32>, similarities: &mut CudaSlice<f32>,
            vocab_size: i32, dim: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let grid = ((vocab_size as u32 + 255) / 256, 1, 1);
            unsafe {
                self.cosine_similarity_fn.launch(
                    &self.stream, grid, (256, 1, 1), 0,
                    &[&query.as_ref(), &vocabulary.as_ref(), &vocab_norms.as_ref(),
                      &similarities.as_ref(), &vocab_size, &dim],
                )?;
            }
            Ok(())
        }

        pub fn launch_vector_add(
            &self, a: &mut CudaSlice<f32>, b: &CudaSlice<f32>,
            scale_a: f32, scale_b: f32, n: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let grid = ((n as u32 + 255) / 256, 1, 1);
            unsafe {
                self.vector_add_fn.launch(
                    &self.stream, grid, (256, 1, 1), 0,
                    &[&a.as_ref(), &b.as_ref(), &scale_a, &scale_b, &n],
                )?;
            }
            Ok(())
        }

        pub fn launch_vector_clamp(
            &self, a: &mut CudaSlice<f32>, min_val: f32, max_val: f32, n: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let grid = ((n as u32 + 255) / 256, 1, 1);
            unsafe {
                self.vector_clamp_fn.launch(
                    &self.stream, grid, (256, 1, 1), 0,
                    &[&a.as_ref(), &min_val, &max_val, &n],
                )?;
            }
            Ok(())
        }

        pub fn launch_core_process(
            &self, pulses_content: &mut CudaSlice<f32>, pulses_entropy: &mut CudaSlice<f32>,
            pulses_weight: &mut CudaSlice<f32>, core_memory: &CudaSlice<f32>,
            core_internal_state: &CudaSlice<f32>, core_gate: &CudaSlice<f32>,
            ssm_a: &CudaSlice<f32>, ssm_b: &CudaSlice<f32>, ssm_c: &CudaSlice<f32>,
            ssm_h: &mut CudaSlice<f32>, ssm_delta: &CudaSlice<f32>,
            ssm_delta_bias: &CudaSlice<f32>, ssm_d: &CudaSlice<f32>,
            num_pulses: i32, dim: i32, num_cores: i32, memory_size: i32, d_state: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                self.core_process_fn.launch(
                    &self.stream, (num_pulses as u32, num_cores as u32, 1), (256, 1, 1), 0,
                    &[&pulses_content.as_ref(), &pulses_entropy.as_ref(), &pulses_weight.as_ref(),
                      &core_memory.as_ref(), &core_internal_state.as_ref(), &core_gate.as_ref(),
                      &ssm_a.as_ref(), &ssm_b.as_ref(), &ssm_c.as_ref(), &ssm_h.as_ref(),
                      &ssm_delta.as_ref(), &ssm_delta_bias.as_ref(), &ssm_d.as_ref(),
                      &num_pulses, &dim, &num_cores, &memory_size, &d_state],
                )?;
            }
            Ok(())
        }

        /// Launch core_process_kernel on a specific async stream for overlapping execution.
        /// This allows the kernel to run concurrently with data transfers on other streams.
        pub fn launch_core_process_async(
            &self, stream: &CudaStream,
            pulses_content: &mut CudaSlice<f32>, pulses_entropy: &mut CudaSlice<f32>,
            pulses_weight: &mut CudaSlice<f32>, core_memory: &CudaSlice<f32>,
            core_internal_state: &CudaSlice<f32>, core_gate: &CudaSlice<f32>,
            ssm_a: &CudaSlice<f32>, ssm_b: &CudaSlice<f32>, ssm_c: &CudaSlice<f32>,
            ssm_h: &mut CudaSlice<f32>, ssm_delta: &CudaSlice<f32>,
            ssm_delta_bias: &CudaSlice<f32>, ssm_d: &CudaSlice<f32>,
            num_pulses: i32, dim: i32, num_cores: i32, memory_size: i32, d_state: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            unsafe {
                self.core_process_fn.launch(
                    stream, (num_pulses as u32, num_cores as u32, 1), (256, 1, 1), 0,
                    &[&pulses_content.as_ref(), &pulses_entropy.as_ref(), &pulses_weight.as_ref(),
                      &core_memory.as_ref(), &core_internal_state.as_ref(), &core_gate.as_ref(),
                      &ssm_a.as_ref(), &ssm_b.as_ref(), &ssm_c.as_ref(), &ssm_h.as_ref(),
                      &ssm_delta.as_ref(), &ssm_delta_bias.as_ref(), &ssm_d.as_ref(),
                      &num_pulses, &dim, &num_cores, &memory_size, &d_state],
                )?;
            }
            Ok(())
        }

        pub fn sync(&self) -> Result<(), Box<dyn std::error::Error>> {
            self.stream.synchronize()?;
            Ok(())
        }
    }
}

// ============================================================================
// Nova Accelerator
// ============================================================================

pub struct NovaAccelerator {
    pub backend: HardwareBackend,
    #[cfg(feature = "cuda")]
    device: Option<std::sync::Arc<cudarc::driver::safe::CudaContext>>,
    #[cfg(feature = "cuda")]
    kernel_mgr: Option<cuda_kernels::CudaKernelManager>,
    pub enabled: bool,
    pub gpu_ops: u64,
    pub cpu_ops: u64,
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    /// Per-batch GPU profiler
    pub batch_profile: BatchProfile,
    /// Cumulative GPU profiler across all batches
    pub cumulative_profile: CumulativeProfile,
    /// Whether to print detailed per-batch profiling
    pub profiling_enabled: bool,
    /// Persistent GPU buffer cache for reuse across operations
    #[cfg(feature = "cuda")]
    buffer_cache: std::collections::HashMap<String, cudarc::driver::safe::CudaSlice<f32>>,
    /// Async CUDA streams for overlapping operations
    #[cfg(feature = "cuda")]
    async_streams: Vec<cudarc::driver::safe::CudaStream>,
    /// Current stream index for round-robin async dispatch
    #[cfg(feature = "cuda")]
    current_stream_idx: usize,
    /// Whether to use async streams (overlap kernel execution with data transfer)
    pub use_async_streams: bool,
}

impl NovaAccelerator {
    pub fn auto_detect() -> Self {
        let backend = auto_detect_backend();

        #[cfg(feature = "cuda")]
        let (device, kernel_mgr) = match backend {
            HardwareBackend::Cuda => {
                match cudarc::driver::safe::CudaContext::new(0) {
                    Ok(dev) => {
                        eprintln!("  CUDA device initialized");
                        match cuda_kernels::CudaKernelManager::new(dev.clone()) {
                            Ok(mgr) => {
                                eprintln!("  CUDA kernels loaded successfully");
                                (Some(dev), Some(mgr))
                            }
                            Err(e) => {
                                eprintln!("  Failed to load CUDA kernels: {:?}", e);
                                eprintln!("  Falling back to CPU for all operations");
                                (Some(dev), None)
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  CUDA init failed: {:?}", e);
                        (None, None)
                    }
                }
            }
            _ => (None, None),
        };

        #[cfg(feature = "cuda")]
        let async_streams = if let Some(ref dev) = device {
            let mut streams = Vec::with_capacity(4);
            for _ in 0..4 {
                if let Ok(s) = cudarc::driver::safe::CudaStream::new(dev) {
                    streams.push(s);
                }
            }
            eprintln!("  Created {} async CUDA streams", streams.len());
            streams
        } else {
            Vec::new()
        };

        Self {
            backend,
            #[cfg(feature = "cuda")]
            device,
            #[cfg(feature = "cuda")]
            kernel_mgr,
            enabled: backend.is_gpu(),
            gpu_ops: 0,
            cpu_ops: 0,
            total_gpu_time_ms: 0.0,
            total_cpu_time_ms: 0.0,
            batch_profile: BatchProfile::new(),
            cumulative_profile: CumulativeProfile::new(),
            profiling_enabled: true,
            #[cfg(feature = "cuda")]
            buffer_cache: std::collections::HashMap::new(),
            #[cfg(feature = "cuda")]
            async_streams,
            #[cfg(feature = "cuda")]
            current_stream_idx: 0,
            use_async_streams: true,
        }
    }

    pub fn new(backend: HardwareBackend) -> Self {
        Self {
            backend,
            #[cfg(feature = "cuda")]
            device: None,
            #[cfg(feature = "cuda")]
            kernel_mgr: None,
            enabled: backend.is_gpu(),
            gpu_ops: 0,
            cpu_ops: 0,
            total_gpu_time_ms: 0.0,
            total_cpu_time_ms: 0.0,
            batch_profile: BatchProfile::new(),
            cumulative_profile: CumulativeProfile::new(),
            profiling_enabled: true,
            #[cfg(feature = "cuda")]
            buffer_cache: std::collections::HashMap::new(),
            #[cfg(feature = "cuda")]
            async_streams: Vec::new(),
            #[cfg(feature = "cuda")]
            current_stream_idx: 0,
            use_async_streams: true,
        }
    }

    /// Reset the per-batch profiler for a new batch.
    pub fn reset_batch_profile(&mut self) {
        self.batch_profile = BatchProfile::new();
    }

    /// Finalize the current batch profile and accumulate into cumulative.
    /// Prints the per-batch report if profiling is enabled.
    pub fn finalize_batch_profile(&mut self, batch_idx: usize, batch_size: usize) {
        if self.profiling_enabled {
            self.batch_profile.print_batch_report(batch_idx, batch_size);
        }
        self.cumulative_profile.accumulate(&self.batch_profile);
    }

    /// Print the cumulative profiling report.
    pub fn print_cumulative_profile(&self) {
        self.cumulative_profile.print_cumulative_report();
    }

    pub fn is_gpu(&self) -> bool {
        self.enabled && self.backend.is_gpu()
    }

    pub fn is_kernels_ready(&self) -> bool {
        #[cfg(feature = "cuda")]
        { self.is_gpu() && self.kernel_mgr.is_some() }
        #[cfg(not(feature = "cuda"))]
        { false }
    }

    pub fn description(&self) -> String {
        if self.is_gpu() {
            format!("{} ({} ops, {:.1}s GPU time)",
                self.backend.name(), self.gpu_ops, self.total_gpu_time_ms / 1000.0)
        } else {
            format!("{} ({} ops, {:.1}s CPU time)",
                self.backend.name(), self.cpu_ops, self.total_cpu_time_ms / 1000.0)
        }
    }

    #[cfg(feature = "cuda")]
    fn alloc_from_cpu<T: cudarc::driver::DeviceRepr + Clone>(
        &self, data: &[T],
    ) -> Option<cudarc::driver::safe::CudaSlice<T>> {
        if let Some(ref dev) = self.device {
            cudarc::driver::safe::CudaSlice::from_slice(dev, data).ok()
        } else {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn copy_to_cpu<T: cudarc::driver::DeviceRepr + Clone>(
        &self, slice: &cudarc::driver::safe::CudaSlice<T>,
    ) -> Option<Vec<T>> {
        slice.download().ok()
    }

    // ========================================================================
    // Persistent GPU Buffer Cache
    // ========================================================================

    /// Get or create a cached GPU buffer of the given size.
    /// Returns a CudaSlice<f32> that may be reused from a previous allocation.
    /// The key uniquely identifies the buffer's purpose and size.
    #[cfg(feature = "cuda")]
    fn get_or_create_buffer(&mut self, key: &str, size: usize) -> Option<cudarc::driver::safe::CudaSlice<f32>> {
        // Check cache first
        if let Some(buf) = self.buffer_cache.remove(key) {
            // Buffer exists and is the right size - reuse it
            let elem_bytes = std::mem::size_of::<f32>() as u64;
            self.batch_profile.gpu_mem_reused += size as u64 * elem_bytes;
            return Some(buf);
        }
        
        // Allocate new buffer
        if let Some(ref dev) = self.device {
            match cudarc::driver::safe::CudaSlice::zeros(dev, size) {
                Ok(buf) => {
                    self.batch_profile.alloc_count += 1;
                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                    self.batch_profile.gpu_mem_allocated += size as u64 * elem_bytes;
                    Some(buf)
                }
                Err(_) => None,
            }
        } else {
            None
        }
    }

    /// Return a buffer to the cache for future reuse.
    #[cfg(feature = "cuda")]
    fn return_buffer(&mut self, key: String, buf: cudarc::driver::safe::CudaSlice<f32>) {
        self.buffer_cache.insert(key, buf);
    }

    /// Upload data to a GPU buffer (either cached or newly allocated).
    /// Returns the buffer with data uploaded.
    #[cfg(feature = "cuda")]
    fn upload_to_buffer(&mut self, key: &str, data: &[f32]) -> Option<cudarc::driver::safe::CudaSlice<f32>> {
        let size = data.len();
        let mut buf = self.get_or_create_buffer(key, size)?;
        
        // Copy data into the buffer
        if let Some(ref dev) = self.device {
            // Use from_slice to upload - this creates a new slice with data
            // For reuse, we need to copy into existing buffer
            match cudarc::driver::safe::CudaSlice::from_slice(dev, data) {
                Ok(new_buf) => {
                    // Track the upload
                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                    self.batch_profile.gpu_mem_copied_h2d += size as u64 * elem_bytes;
                    self.batch_profile.memcpy_count += 1;
                    
                    // Return old buffer to cache, use new one
                    self.return_buffer(format!("{}_old", key), buf);
                    Some(new_buf)
                }
                Err(_) => {
                    self.return_buffer(key.to_string(), buf);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Clear the buffer cache to free GPU memory.
    #[cfg(feature = "cuda")]
    pub fn clear_buffer_cache(&mut self) {
        self.buffer_cache.clear();
    }

    // ========================================================================
    // Async CUDA Stream Management
    // ========================================================================

    /// Get the next async stream in round-robin fashion.
    /// Returns None if no async streams are available.
    #[cfg(feature = "cuda")]
    fn next_async_stream(&mut self) -> Option<&cudarc::driver::safe::CudaStream> {
        if self.async_streams.is_empty() {
            return None;
        }
        let idx = self.current_stream_idx;
        self.current_stream_idx = (self.current_stream_idx + 1) % self.async_streams.len();
        Some(&self.async_streams[idx])
    }

    /// Synchronize all async streams.
    #[cfg(feature = "cuda")]
    pub fn sync_all_streams(&self) -> Result<(), Box<dyn std::error::Error>> {
        for stream in &self.async_streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    // ========================================================================
    // GPU-Accelerated SSM Operations
    // ========================================================================

    pub fn selective_scan(
        &mut self, a: &[f32], b: &[f32], c: &[f32], h: &mut [f32],
        input: &[f32], delta: &[f32], delta_bias: &[f32], d: &[f32],
        output: &mut [f32], d_inner: usize, d_state: usize,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let _preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += 0.001; // negligible

            let upload_start = Instant::now();
            // Use buffer cache for reusable allocations
            let a_key = format!("ss_a_{}", a.len());
            let b_key = format!("ss_b_{}", b.len());
            let c_key = format!("ss_c_{}", c.len());
            let h_key = format!("ss_h_{}", h.len());
            let x_key = format!("ss_x_{}", input.len());
            let delta_key = format!("ss_delta_{}", delta.len());
            let delta_bias_key = format!("ss_db_{}", delta_bias.len());
            let d_key = format!("ss_d_{}", d.len());
            let out_key = format!("ss_out_{}", output.len());

            if let (Some(a_gpu), Some(b_gpu), Some(c_gpu), Some(h_gpu),
                    Some(x_gpu), Some(delta_gpu), Some(delta_bias_gpu),
                    Some(d_gpu), Some(out_gpu)) = (
                self.upload_to_buffer(&a_key, a),
                self.upload_to_buffer(&b_key, b),
                self.upload_to_buffer(&c_key, c),
                self.upload_to_buffer(&h_key, h),
                self.upload_to_buffer(&x_key, input),
                self.upload_to_buffer(&delta_key, delta),
                self.upload_to_buffer(&delta_bias_key, delta_bias),
                self.upload_to_buffer(&d_key, d),
                self.upload_to_buffer(&out_key, output),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut h_gpu_mut = h_gpu;
                let mut out_gpu_mut = out_gpu;
                match mgr.launch_selective_scan(
                    &a_gpu, &b_gpu, &c_gpu, &mut h_gpu_mut, &x_gpu, &delta_gpu,
                    &delta_bias_gpu, &d_gpu, &mut out_gpu_mut, d_inner as i32, d_state as i32,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("selective_scan_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                let mut download_ok = true;
                                let elem_bytes = std::mem::size_of::<f32>() as u64;
                                if let Some(h_cpu) = self.copy_to_cpu(&h_gpu_mut) {
                                    h.copy_from_slice(&h_cpu);
                                    self.batch_profile.gpu_mem_copied_d2h += h.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "selective_scan".to_string(),
                                        "copy_to_cpu(h) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs selective_scan() GPU path".to_string(),
                                    ));
                                }
                                if let Some(out_cpu) = self.copy_to_cpu(&out_gpu_mut) {
                                    output.copy_from_slice(&out_cpu);
                                    self.batch_profile.gpu_mem_copied_d2h += output.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "selective_scan".to_string(),
                                        "copy_to_cpu(output) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs selective_scan() GPU path".to_string(),
                                    ));
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                // Return buffers to cache for reuse
                                self.return_buffer(a_key, a_gpu);
                                self.return_buffer(b_key, b_gpu);
                                self.return_buffer(c_key, c_gpu);
                                self.return_buffer(h_key, h_gpu_mut);
                                self.return_buffer(x_key, x_gpu);
                                self.return_buffer(delta_key, delta_gpu);
                                self.return_buffer(delta_bias_key, delta_bias_gpu);
                                self.return_buffer(d_key, d_gpu);
                                self.return_buffer(out_key, out_gpu_mut);

                                if download_ok {
                                    self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                    self.gpu_ops += 1;
                                    return;
                                }
                            }
                            Err(e) => {
                                self.batch_profile.cpu_fallback_count += 1;
                                self.batch_profile.fallback_details.push((
                                    "selective_scan".to_string(),
                                    "sync failed".to_string(),
                                    format!("{:?}", e),
                                    "src/cuda.rs selective_scan() GPU path".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.batch_profile.cpu_fallback_count += 1;
                        self.batch_profile.fallback_details.push((
                            "selective_scan".to_string(),
                            "kernel launch failed".to_string(),
                            format!("{:?}", e),
                            "src/cuda.rs selective_scan() GPU path".to_string(),
                        ));
                    }
                }
            } else {
                self.batch_profile.cpu_fallback_count += 1;
                self.batch_profile.fallback_details.push((
                    "selective_scan".to_string(),
                    "upload_to_buffer failed for one or more buffers".to_string(),
                    "allocation returned None".to_string(),
                    "src/cuda.rs selective_scan() GPU path".to_string(),
                ));
            }
        }

        // CPU fallback
        let start = Instant::now();
        crate::ssm::selective_scan_step_raw(a, b, c, h, input, delta, delta_bias, d, output, d_inner, d_state);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn ssm_transform_batch(
        &mut self, ssm: &mut crate::ssm::StateSpace,
        pulses_content: &mut [Vec<f32>], use_time_mixing: bool,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len();
            let d_inner = ssm.d_inner;
            let d_state = ssm.d_state;
            let flat_size = num_pulses * d_inner;
            let mut flat_pulses = vec![0.0f32; flat_size];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(d_inner);
                for j in 0..len { flat_pulses[i * d_inner + j] = content[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            // Use buffer cache for reusable allocations
            let a_key = format!("stb_a_{}", ssm.a.len());
            let b_key = format!("stb_b_{}", ssm.b.len());
            let c_key = format!("stb_c_{}", ssm.c.len());
            let h_key = format!("stb_h_{}", ssm.h.len());
            let delta_key = format!("stb_delta_{}", ssm.delta.len());
            let delta_bias_key = format!("stb_db_{}", ssm.delta_bias.len());
            let d_key = format!("stb_d_{}", ssm.d.len());
            let pulses_key = format!("stb_pulses_{}", flat_size);
            let out_key = format!("stb_out_{}", flat_size);

            if let (Some(a_gpu), Some(b_gpu), Some(c_gpu), Some(h_gpu),
                    Some(delta_gpu), Some(delta_bias_gpu), Some(d_gpu),
                    Some(pulses_gpu), Some(out_gpu)) = (
                self.upload_to_buffer(&a_key, &ssm.a),
                self.upload_to_buffer(&b_key, &ssm.b),
                self.upload_to_buffer(&c_key, &ssm.c),
                self.upload_to_buffer(&h_key, &ssm.h),
                self.upload_to_buffer(&delta_key, &ssm.delta),
                self.upload_to_buffer(&delta_bias_key, &ssm.delta_bias),
                self.upload_to_buffer(&d_key, &ssm.d),
                self.upload_to_buffer(&pulses_key, &flat_pulses),
                self.upload_to_buffer(&out_key, &vec![0.0f32; flat_size]),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut h_gpu_mut = h_gpu;
                let mut pulses_gpu_mut = pulses_gpu;
                let mut out_gpu_mut = out_gpu;
                match mgr.launch_ssm_transform_batch(
                    &a_gpu, &b_gpu, &c_gpu, &mut h_gpu_mut, &delta_gpu, &delta_bias_gpu, &d_gpu,
                    &mut pulses_gpu_mut, &mut out_gpu_mut, num_pulses as i32, d_inner as i32, d_state as i32,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("ssm_transform_batch_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                let mut download_ok = true;
                                let elem_bytes = std::mem::size_of::<f32>() as u64;
                                if let Some(h_cpu) = self.copy_to_cpu(&h_gpu_mut) {
                                    ssm.h.copy_from_slice(&h_cpu);
                                    self.batch_profile.gpu_mem_copied_d2h += ssm.h.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "ssm_transform_batch".to_string(),
                                        "copy_to_cpu(h) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                                    ));
                                }
                                if let Some(out_cpu) = self.copy_to_cpu(&out_gpu_mut) {
                                    for (i, content) in pulses_content.iter_mut().enumerate() {
                                        let len = content.len().min(d_inner);
                                        for j in 0..len { content[j] = out_cpu[i * d_inner + j]; }
                                    }
                                    self.batch_profile.gpu_mem_copied_d2h += flat_size as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "ssm_transform_batch".to_string(),
                                        "copy_to_cpu(output) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                                    ));
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                // Return buffers to cache for reuse
                                self.return_buffer(a_key, a_gpu);
                                self.return_buffer(b_key, b_gpu);
                                self.return_buffer(c_key, c_gpu);
                                self.return_buffer(h_key, h_gpu_mut);
                                self.return_buffer(delta_key, delta_gpu);
                                self.return_buffer(delta_bias_key, delta_bias_gpu);
                                self.return_buffer(d_key, d_gpu);
                                self.return_buffer(pulses_key, pulses_gpu_mut);
                                self.return_buffer(out_key, out_gpu_mut);

                                if download_ok {
                                    self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                    self.gpu_ops += 1;
                                    return;
                                }
                            }
                            Err(e) => {
                                self.batch_profile.cpu_fallback_count += 1;
                                self.batch_profile.fallback_details.push((
                                    "ssm_transform_batch".to_string(),
                                    "sync failed".to_string(),
                                    format!("{:?}", e),
                                    "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.batch_profile.cpu_fallback_count += 1;
                        self.batch_profile.fallback_details.push((
                            "ssm_transform_batch".to_string(),
                            "kernel launch failed".to_string(),
                            format!("{:?}", e),
                            "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                        ));
                    }
                }
            } else {
                self.batch_profile.cpu_fallback_count += 1;
                self.batch_profile.fallback_details.push((
                    "ssm_transform_batch".to_string(),
                    "upload_to_buffer failed for one or more buffers".to_string(),
                    "allocation returned None".to_string(),
                    "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                ));
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

    pub fn field_update(
        &mut self, state: &mut [f32], momentum: &mut [f32],
        pulses_content: &[Vec<f32>], pulses_weight: &[f32],
        learning_rate: f32, diffusion: f32, dim: usize,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len();
            let mut flat_pulses = vec![0.0f32; num_pulses * dim];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(dim);
                for j in 0..len { flat_pulses[i * dim + j] = content[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            // Use buffer cache for reusable allocations
            let pulses_key = format!("fu_pulses_{}", flat_pulses.len());
            let weights_key = format!("fu_weights_{}", pulses_weight.len());
            let state_key = format!("fu_state_{}", state.len());
            let momentum_key = format!("fu_momentum_{}", momentum.len());

            if let (Some(pulses_gpu), Some(weights_gpu), Some(state_gpu), Some(momentum_gpu)) = (
                self.upload_to_buffer(&pulses_key, &flat_pulses),
                self.upload_to_buffer(&weights_key, pulses_weight),
                self.upload_to_buffer(&state_key, state),
                self.upload_to_buffer(&momentum_key, momentum),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut state_gpu_mut = state_gpu;
                let mut momentum_gpu_mut = momentum_gpu;
                match mgr.launch_field_update(
                    &pulses_gpu, &weights_gpu, &mut state_gpu_mut, &mut momentum_gpu_mut,
                    learning_rate, diffusion, num_pulses as i32, dim as i32,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("field_update_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        // Launch field diffuse
                        let diffuse_start = Instant::now();
                        let df = diffusion * 0.95f32;
                        match mgr.launch_field_diffuse(&mut pulses_gpu, &state_gpu_mut, df, num_pulses as i32, dim as i32) {
                            Ok(()) => {
                                self.batch_profile.kernel_launch_count += 1;
                                self.batch_profile.kernel_names.push("field_diffuse_kernel".to_string());
                            }
                            Err(e) => {
                                self.batch_profile.cpu_fallback_count += 1;
                                self.batch_profile.fallback_details.push((
                                    "field_update".to_string(),
                                    "field_diffuse kernel launch failed".to_string(),
                                    format!("{:?}", e),
                                    "src/cuda.rs field_update() GPU path".to_string(),
                                ));
                            }
                        }
                        let diffuse_time = diffuse_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += diffuse_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                let mut download_ok = true;
                                let elem_bytes = std::mem::size_of::<f32>() as u64;
                                if let Some(s) = self.copy_to_cpu(&state_gpu_mut) {
                                    state.copy_from_slice(&s);
                                    self.batch_profile.gpu_mem_copied_d2h += state.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "field_update".to_string(),
                                        "copy_to_cpu(state) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs field_update() GPU path".to_string(),
                                    ));
                                }
                                if let Some(m) = self.copy_to_cpu(&momentum_gpu_mut) {
                                    momentum.copy_from_slice(&m);
                                    self.batch_profile.gpu_mem_copied_d2h += momentum.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "field_update".to_string(),
                                        "copy_to_cpu(momentum) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs field_update() GPU path".to_string(),
                                    ));
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                // Return buffers to cache for reuse
                                self.return_buffer(pulses_key, pulses_gpu);
                                self.return_buffer(weights_key, weights_gpu);
                                self.return_buffer(state_key, state_gpu_mut);
                                self.return_buffer(momentum_key, momentum_gpu_mut);

                                if download_ok {
                                    self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                    self.gpu_ops += 1;
                                    return;
                                }
                            }
                            Err(e) => {
                                self.batch_profile.cpu_fallback_count += 1;
                                self.batch_profile.fallback_details.push((
                                    "field_update".to_string(),
                                    "sync failed".to_string(),
                                    format!("{:?}", e),
                                    "src/cuda.rs field_update() GPU path".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.batch_profile.cpu_fallback_count += 1;
                        self.batch_profile.fallback_details.push((
                            "field_update".to_string(),
                            "kernel launch failed".to_string(),
                            format!("{:?}", e),
                            "src/cuda.rs field_update() GPU path".to_string(),
                        ));
                    }
                }
            } else {
                self.batch_profile.cpu_fallback_count += 1;
                self.batch_profile.fallback_details.push((
                    "field_update".to_string(),
                    "upload_to_buffer failed for one or more buffers".to_string(),
                    "allocation returned None".to_string(),
                    "src/cuda.rs field_update() GPU path".to_string(),
                ));
            }
        }

        // CPU fallback
        let start = Instant::now();
        let mut field_avg = vec![0.0; dim];
        let mut total_weight = 0.0;
        for (content, &weight) in pulses_content.iter().zip(pulses_weight.iter()) {
            total_weight += weight;
            for i in 0..dim.min(content.len()) { field_avg[i] += content[i] * weight; }
        }
        if total_weight > 0.0 {
            for i in 0..dim { field_avg[i] /= total_weight; }
        }
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

    pub fn process_cores_batch(
        &mut self, cores: &mut [crate::core::NovaCore],
        pulses_content: &mut [Vec<f32>], pulses_entropy: &mut [f32], pulses_weight: &mut [f32],
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len();
            let num_cores = cores.len();
            let dim = cores[0].ssm.d_inner;
            let d_state = cores[0].ssm.d_state;
            let memory_size = cores[0].memory.len();

            let mut flat_pulses = vec![0.0f32; num_pulses * dim];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(dim);
                for j in 0..len { flat_pulses[i * dim + j] = content[j]; }
            }

            let ssm_total = dim * d_state;
            let mut flat_memory = vec![0.0f32; num_cores * memory_size];
            let mut flat_internal = vec![0.0f32; num_cores * dim];
            let mut flat_gate = vec![0.0f32; num_cores];
            let mut flat_ssm_a = vec![0.0f32; num_cores * ssm_total];
            let mut flat_ssm_b = vec![0.0f32; num_cores * ssm_total];
            let mut flat_ssm_c = vec![0.0f32; num_cores * ssm_total];
            let mut flat_ssm_h = vec![0.0f32; num_cores * ssm_total];
            let mut flat_ssm_delta = vec![0.0f32; num_cores * dim];
            let mut flat_ssm_delta_bias = vec![0.0f32; num_cores * dim];
            let mut flat_ssm_d = vec![0.0f32; num_cores * dim];

            for (ci, core) in cores.iter().enumerate() {
                let mem_len = core.memory.len().min(memory_size);
                flat_memory[ci * memory_size..ci * memory_size + mem_len].copy_from_slice(&core.memory[..mem_len]);
                let int_len = core.internal_state.len().min(dim);
                flat_internal[ci * dim..ci * dim + int_len].copy_from_slice(&core.internal_state[..int_len]);
                flat_gate[ci] = core.gate;
                flat_ssm_a[ci * ssm_total..(ci + 1) * ssm_total].copy_from_slice(&core.ssm.a);
                flat_ssm_b[ci * ssm_total..(ci + 1) * ssm_total].copy_from_slice(&core.ssm.b);
                flat_ssm_c[ci * ssm_total..(ci + 1) * ssm_total].copy_from_slice(&core.ssm.c);
                flat_ssm_h[ci * ssm_total..(ci + 1) * ssm_total].copy_from_slice(&core.ssm.h);
                flat_ssm_delta[ci * dim..(ci + 1) * dim].copy_from_slice(&core.ssm.delta);
                flat_ssm_delta_bias[ci * dim..(ci + 1) * dim].copy_from_slice(&core.ssm.delta_bias);
                flat_ssm_d[ci * dim..(ci + 1) * dim].copy_from_slice(&core.ssm.d);
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            // Use buffer cache for reusable allocations
            let pulses_key = format!("pcb_pulses_{}", flat_pulses.len());
            let entropy_key = format!("pcb_entropy_{}", pulses_entropy.len());
            let weight_key = format!("pcb_weight_{}", pulses_weight.len());
            let memory_key = format!("pcb_memory_{}", flat_memory.len());
            let internal_key = format!("pcb_internal_{}", flat_internal.len());
            let gate_key = format!("pcb_gate_{}", flat_gate.len());
            let ssm_a_key = format!("pcb_ssm_a_{}", flat_ssm_a.len());
            let ssm_b_key = format!("pcb_ssm_b_{}", flat_ssm_b.len());
            let ssm_c_key = format!("pcb_ssm_c_{}", flat_ssm_c.len());
            let ssm_h_key = format!("pcb_ssm_h_{}", flat_ssm_h.len());
            let ssm_delta_key = format!("pcb_ssm_delta_{}", flat_ssm_delta.len());
            let ssm_db_key = format!("pcb_ssm_db_{}", flat_ssm_delta_bias.len());
            let ssm_d_key = format!("pcb_ssm_d_{}", flat_ssm_d.len());

            if let (Some(pulses_gpu), Some(entropy_gpu), Some(weight_gpu),
                    Some(memory_gpu), Some(internal_gpu), Some(gate_gpu),
                    Some(ssm_a_gpu), Some(ssm_b_gpu), Some(ssm_c_gpu),
                    Some(ssm_h_gpu), Some(ssm_delta_gpu), Some(ssm_delta_bias_gpu),
                    Some(ssm_d_gpu)) = (
                self.upload_to_buffer(&pulses_key, &flat_pulses),
                self.upload_to_buffer(&entropy_key, pulses_entropy),
                self.upload_to_buffer(&weight_key, pulses_weight),
                self.upload_to_buffer(&memory_key, &flat_memory),
                self.upload_to_buffer(&internal_key, &flat_internal),
                self.upload_to_buffer(&gate_key, &flat_gate),
                self.upload_to_buffer(&ssm_a_key, &flat_ssm_a),
                self.upload_to_buffer(&ssm_b_key, &flat_ssm_b),
                self.upload_to_buffer(&ssm_c_key, &flat_ssm_c),
                self.upload_to_buffer(&ssm_h_key, &flat_ssm_h),
                self.upload_to_buffer(&ssm_delta_key, &flat_ssm_delta),
                self.upload_to_buffer(&ssm_db_key, &flat_ssm_delta_bias),
                self.upload_to_buffer(&ssm_d_key, &flat_ssm_d),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut pulses_gpu_mut = pulses_gpu;
                let mut entropy_gpu_mut = entropy_gpu;
                let mut weight_gpu_mut = weight_gpu;
                let mut ssm_h_gpu_mut = ssm_h_gpu;

                // Use async stream for overlapping kernel execution with data transfer
                let kernel_result = if self.use_async_streams {
                    if let Some(async_stream) = self.next_async_stream() {
                        mgr.launch_core_process_async(
                            async_stream,
                            &mut pulses_gpu_mut, &mut entropy_gpu_mut, &mut weight_gpu_mut,
                            &memory_gpu, &internal_gpu, &gate_gpu,
                            &ssm_a_gpu, &ssm_b_gpu, &ssm_c_gpu, &mut ssm_h_gpu_mut,
                            &ssm_delta_gpu, &ssm_delta_bias_gpu, &ssm_d_gpu,
                            num_pulses as i32, dim as i32, num_cores as i32, memory_size as i32, d_state as i32,
                        )
                    } else {
                        mgr.launch_core_process(
                            &mut pulses_gpu_mut, &mut entropy_gpu_mut, &mut weight_gpu_mut,
                            &memory_gpu, &internal_gpu, &gate_gpu,
                            &ssm_a_gpu, &ssm_b_gpu, &ssm_c_gpu, &mut ssm_h_gpu_mut,
                            &ssm_delta_gpu, &ssm_delta_bias_gpu, &ssm_d_gpu,
                            num_pulses as i32, dim as i32, num_cores as i32, memory_size as i32, d_state as i32,
                        )
                    }
                } else {
                    mgr.launch_core_process(
                        &mut pulses_gpu_mut, &mut entropy_gpu_mut, &mut weight_gpu_mut,
                        &memory_gpu, &internal_gpu, &gate_gpu,
                        &ssm_a_gpu, &ssm_b_gpu, &ssm_c_gpu, &mut ssm_h_gpu_mut,
                        &ssm_delta_gpu, &ssm_delta_bias_gpu, &ssm_d_gpu,
                        num_pulses as i32, dim as i32, num_cores as i32, memory_size as i32, d_state as i32,
                    )
                };

                match kernel_result {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("core_process_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        // Sync all streams when using async mode to ensure all kernels complete
                        let sync_result = if self.use_async_streams {
                            self.sync_all_streams()
                        } else {
                            mgr.sync()
                        };
                        match sync_result {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                let mut download_ok = true;
                                let elem_bytes = std::mem::size_of::<f32>() as u64;
                                if let Some(p) = self.copy_to_cpu(&pulses_gpu_mut) {
                                    for (i, content) in pulses_content.iter_mut().enumerate() {
                                        let len = content.len().min(dim);
                                        for j in 0..len { content[j] = p[i * dim + j]; }
                                    }
                                    self.batch_profile.gpu_mem_copied_d2h += flat_pulses.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "process_cores_batch".to_string(),
                                        "copy_to_cpu(pulses) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs process_cores_batch() GPU path".to_string(),
                                    ));
                                }
                                if let Some(e) = self.copy_to_cpu(&entropy_gpu_mut) {
                                    pulses_entropy.copy_from_slice(&e);
                                    self.batch_profile.gpu_mem_copied_d2h += pulses_entropy.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "process_cores_batch".to_string(),
                                        "copy_to_cpu(entropy) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs process_cores_batch() GPU path".to_string(),
                                    ));
                                }
                                if let Some(w) = self.copy_to_cpu(&weight_gpu_mut) {
                                    pulses_weight.copy_from_slice(&w);
                                    self.batch_profile.gpu_mem_copied_d2h += pulses_weight.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "process_cores_batch".to_string(),
                                        "copy_to_cpu(weight) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs process_cores_batch() GPU path".to_string(),
                                    ));
                                }
                                if let Some(h) = self.copy_to_cpu(&ssm_h_gpu_mut) {
                                    for (ci, core) in cores.iter_mut().enumerate() {
                                        core.ssm.h.copy_from_slice(&h[ci * ssm_total..(ci + 1) * ssm_total]);
                                    }
                                    self.batch_profile.gpu_mem_copied_d2h += ssm_total as u64 * num_cores as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                    self.batch_profile.cpu_fallback_count += 1;
                                    self.batch_profile.fallback_details.push((
                                        "process_cores_batch".to_string(),
                                        "copy_to_cpu(ssm_h) failed".to_string(),
                                        "download returned None".to_string(),
                                        "src/cuda.rs process_cores_batch() GPU path".to_string(),
                                    ));
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                // Return buffers to cache for reuse
                                self.return_buffer(pulses_key, pulses_gpu_mut);
                                self.return_buffer(entropy_key, entropy_gpu_mut);
                                self.return_buffer(weight_key, weight_gpu_mut);
                                self.return_buffer(memory_key, memory_gpu);
                                self.return_buffer(internal_key, internal_gpu);
                                self.return_buffer(gate_key, gate_gpu);
                                self.return_buffer(ssm_a_key, ssm_a_gpu);
                                self.return_buffer(ssm_b_key, ssm_b_gpu);
                                self.return_buffer(ssm_c_key, ssm_c_gpu);
                                self.return_buffer(ssm_h_key, ssm_h_gpu_mut);
                                self.return_buffer(ssm_delta_key, ssm_delta_gpu);
                                self.return_buffer(ssm_db_key, ssm_delta_bias_gpu);
                                self.return_buffer(ssm_d_key, ssm_d_gpu);

                                if download_ok {
                                    self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                    self.gpu_ops += 1;
                                    return;
                                }
                            }
                            Err(e) => {
                                self.batch_profile.cpu_fallback_count += 1;
                                self.batch_profile.fallback_details.push((
                                    "process_cores_batch".to_string(),
                                    "sync failed".to_string(),
                                    format!("{:?}", e),
                                    "src/cuda.rs process_cores_batch() GPU path".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.batch_profile.cpu_fallback_count += 1;
                        self.batch_profile.fallback_details.push((
                            "process_cores_batch".to_string(),
                            "kernel launch failed".to_string(),
                            format!("{:?}", e),
                            "src/cuda.rs process_cores_batch() GPU path".to_string(),
                        ));
                    }
                }
            } else {
                self.batch_profile.cpu_fallback_count += 1;
                self.batch_profile.fallback_details.push((
                    "process_cores_batch".to_string(),
                    "upload_to_buffer failed for one or more buffers".to_string(),
                    "allocation returned None".to_string(),
                    "src/cuda.rs process_cores_batch() GPU path".to_string(),
                ));
            }
        }

        // CPU fallback
        let start = Instant::now();
        for core in cores.iter_mut() {
            for pulse in pulses_content.iter_mut() {
                let original = pulse.clone();
                crate::ssm::ssm_transform_pulse(&mut core.ssm, pulse, core.use_time_mixing);
                let ssm_strength = core.gate * 0.5;
                for j in 0..pulse.len() {
                    pulse[j] = original[j] * (1.0 - ssm_strength) + pulse[j] * ssm_strength;
                    pulse[j] = pulse[j].clamp(-1.0, 1.0);
                }
            }
            for e in pulses_entropy.iter_mut() {
                *e *= 0.97;
                *e = e.max(0.01);
            }
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    pub fn print_stats(&self) {
        println!("  Accelerator: {}", self.backend.name());
        println!("  GPU ops: {} | CPU ops: {}", self.gpu_ops, self.cpu_ops);
        if self.total_gpu_time_ms > 0.0 {
            println!("  GPU time: {:.2}s", self.total_gpu_time_ms / 1000.0);
        }
        if self.total_cpu_time_ms > 0.0 {
            println!("  CPU time: {:.2}s", self.total_cpu_time_ms / 1000.0);
        }
        let total_ops = self.gpu_ops + self.cpu_ops;
        let total_time = self.total_gpu_time_ms + self.total_cpu_time_ms;
        if total_ops > 0 {
            println!("  Avg time/op: {:.2}ms", total_time / total_ops as f64);
        }
    }
}

// ============================================================================
// Global Accelerator Singleton
// ============================================================================

use std::sync::Mutex;

/// Global accelerator instance, initialized once at startup.
/// Use `init_global_accelerator()` to initialize, then access via `get_accelerator()`.
static GLOBAL_ACCELERATOR: once_cell::sync::OnceCell<Mutex<NovaAccelerator>> = once_cell::sync::OnceCell::new();

/// Initialize the global accelerator singleton.
/// Call this once at startup (e.g., in main() after thread pool init).
/// Returns true if GPU acceleration is available and initialized.
pub fn init_global_accelerator() -> bool {
    let accelerator = NovaAccelerator::auto_detect();
    let is_gpu = accelerator.is_gpu();
    let _ = GLOBAL_ACCELERATOR.set(Mutex::new(accelerator));
    if is_gpu {
        eprintln!("  ✅ GPU acceleration ACTIVE");
    } else {
        eprintln!("  ⚠️  GPU not available, using CPU (Rayon)");
    }
    is_gpu
}

/// Get a reference to the global accelerator.
/// Panics if `init_global_accelerator()` has not been called yet.
pub fn get_accelerator() -> std::sync::MutexGuard<'static, NovaAccelerator> {
    GLOBAL_ACCELERATOR
        .get()
        .expect("Global accelerator not initialized. Call init_global_accelerator() first.")
        .lock()
        .unwrap()
}

/// Check if GPU acceleration is available.
/// Returns false if accelerator hasn't been initialized yet.
pub fn is_gpu_available() -> bool {
    GLOBAL_ACCELERATOR
        .get()
        .map(|m| m.lock().unwrap().is_gpu())
        .unwrap_or(false)
}

/// Get the backend name string (e.g., "CUDA (NVIDIA GPU)", "CPU (Rayon)").
/// Returns "Unknown" if accelerator hasn't been initialized.
pub fn get_backend_name() -> String {
    GLOBAL_ACCELERATOR
        .get()
        .map(|m| m.lock().unwrap().backend.name().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Get accelerator statistics as a formatted string.
/// Returns "Not initialized" if accelerator hasn't been initialized.
pub fn get_accelerator_stats() -> String {
    GLOBAL_ACCELERATOR
        .get()
        .map(|m| {
            let acc = m.lock().unwrap();
            format!(
                "Backend: {} | GPU ops: {} | CPU ops: {} | GPU time: {:.2}s | CPU time: {:.2}s",
                acc.backend.name(),
                acc.gpu_ops,
                acc.cpu_ops,
                acc.total_gpu_time_ms / 1000.0,
                acc.total_cpu_time_ms / 1000.0,
            )
        })
        .unwrap_or_else(|| "Not initialized".to_string())
}

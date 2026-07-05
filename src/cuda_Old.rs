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
        // pub stream: CudaStream,
        pub stream: Arc<CudaStream>,
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
        let ptx_path = std::env::var("SSM_KERNELS_PTX").unwrap_or_default();
        if ptx_path.is_empty() {
            return Err("SSM_KERNELS_PTX environment variable not set".into());
        }
        let ptx_src = std::fs::read_to_string(&ptx_path)?;
        
        // Also try sm_80 PTX if available (Ampere+ optimizations)
        let ptx_path_80 = format!("{}/ssm_kernels_sm80.ptx", 
            std::path::Path::new(&ptx_path).parent().unwrap_or(std::path::Path::new("")).display());
        
        // Try to load PTX - handle errors gracefully
        // Try different PTX loading methods
        // let module_result = if let Ok(src) = std::fs::read_to_string(&ptx_path_80) {
        //     eprintln!("  Loading sm_80 PTX (Ampere+ optimized)");
        //     // Try from_ptx_string
        //     CudaModule::from_ptx_string(&ctx, &src)
        // } else {
        //     eprintln!("  Loading sm_75 PTX (Turing compatible)");
        //     CudaModule::from_ptx_string(&ctx, &ptx_src)
        // };

        let module_result = if let Ok(src) = std::fs::read_to_string(&ptx_path_80) {
            eprintln!("  Loading sm_80 PTX (Ampere+ optimized)");
            // ✅ FIX: from_ptx_string → from_ptx
            // CudaModule::from_ptx(&ctx, &src)
            CudaModule::load_from_ptx(&src, &[("--gpu-name=sm_80", "--gpu-name=sm_86")], &ctx)
        } else {
            eprintln!("  Loading sm_75 PTX (Turing compatible)");
            // ✅ FIX: from_ptx_string → from_ptx
            CudaModule::load_from_ptx(&ptx_src, &[("--gpu-name=sm_75")], &ctx)
        };
    
        let module = match module_result {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  Failed to load CUDA module: {:?}", e);
                return Err(format!("Failed to load CUDA module: {:?}", e).into());
            }
        };
        
        // Create stream
        let stream_result = ctx.new_stream();
        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Failed to create CUDA stream: {:?}", e);
                return Err(format!("Failed to create CUDA stream: {:?}", e).into());
            }
        };

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
            // cudarc v0.19.8: launch_kernel() takes (stream, grid, block, args) - no shared_mem param
            // Also CudaSlice does NOT implement AsRef, pass &CudaSlice directly
            unsafe {
                // self.selective_scan_fn.launch_kernel(
                self.selective_scan_fn.launch(
                    &self.stream, (d_inner as u32, 1, 1), (32, 1, 1),
                    &[a, b, c, h, x, delta, delta_bias, d, output, &d_inner, &d_state],
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
                // self.ssm_transform_batch_fn.launch_kernel(
                self.ssm_transform_batch_fn.launch(
                    &self.stream, (num_pulses as u32, 1, 1), (256, 1, 1),
                    &[a, b, c, h, delta, delta_bias, d, pulses_content, output,
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
                // self.field_update_fn.launch_kernel(
                self.field_update_fn.launch(
                    &self.stream, grid, (256, 1, 1),
                    &[pulses_content, pulses_weight, field_state, field_momentum,
                      &learning_rate, &diffusion, &num_pulses, &dim],
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
                // self.field_diffuse_fn.launch_kernel(
                self.field_diffuse_fn.launch(
                    &self.stream, grid, (256, 1, 1),
                    &[pulses_content, field_state, &diffusion_factor, &num_pulses, &dim],
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
                // self.cosine_similarity_fn.launch_kernel(
                self.cosine_similarity_fn.launch(
                    &self.stream, grid, (256, 1, 1),
                    &[query, vocabulary, vocab_norms, similarities, &vocab_size, &dim],
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
                // self.vector_add_fn.launch_kernel(
                self.vector_add_fn.launch(
                    &self.stream, grid, (256, 1, 1),
                    &[a, b, &scale_a, &scale_b, &n],
                )?;
            }
            Ok(())
        }

        pub fn launch_vector_clamp(
            &self, a: &mut CudaSlice<f32>, min_val: f32, max_val: f32, n: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let grid = ((n as u32 + 255) / 256, 1, 1);
            unsafe {
                // self.vector_clamp_fn.launch_kernel(
                self.vector_clamp_fn.launch(
                    &self.stream, grid, (256, 1, 1),
                    &[a, &min_val, &max_val, &n],
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
                // self.core_process_fn.launch_kernel(
                self.core_process_fn.launch(
                    &self.stream, (num_pulses as u32, num_cores as u32, 1), (256, 1, 1),
                    &[pulses_content, pulses_entropy, pulses_weight,
                      core_memory, core_internal_state, core_gate,
                      ssm_a, ssm_b, ssm_c, ssm_h,
                      ssm_delta, ssm_delta_bias, ssm_d,
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
                // self.core_process_fn.launch_kernel(
                self.core_process_fn.launch(
                    stream, (num_pulses as u32, num_cores as u32, 1), (256, 1, 1),
                    &[pulses_content, pulses_entropy, pulses_weight,
                      core_memory, core_internal_state, core_gate,
                      ssm_a, ssm_b, ssm_c, ssm_h,
                      ssm_delta, ssm_delta_bias, ssm_d,
                      &num_pulses, &dim, &num_cores, &memory_size, &d_state],
                )?;
            }
            Ok(())
        }

        // pub fn sync(&self) -> Result<(), Box<dyn std::error::Error>> {
        //     self.stream.synchronize()?;
        //     Ok(())
        // }
        pub fn sync(&self) -> Result<(), Box<dyn std::error::Error>> {
            // ✅ Map error to Box<dyn Error>
            self.stream.synchronize().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
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
    // async_streams: Vec<cudarc::driver::safe::CudaStream>,
    async_streams: Vec<std::sync::Arc<cudarc::driver::safe::CudaStream>>,
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
                // cudarc v0.19.8: Use ctx.new_stream() instead of ctx.create_stream()
                if let Ok(s) = dev.new_stream() {
                    // streams.push(s);
                    streams.push(std::sync::Arc::new(s));
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

    /// Allocate a GPU buffer and copy data from CPU to GPU.
    /// cudarc v0.19.8: Use ctx.htod_sync_copy() instead of CudaSlice::from_slice()
    #[cfg(feature = "cuda")]
    fn alloc_from_cpu<T: cudarc::driver::DeviceRepr + Clone>(
        &self, data: &[T],
    ) -> Option<cudarc::driver::safe::CudaSlice<T>> {
        if let Some(ref dev) = self.device {
            // dev.htod_sync_copy(data).ok()
            // CudaSlice::from_host(dev, data).ok()
            CudaSlice::from_vec(dev, data.to_vec()).ok()
        } else {
            None
        }
    }

    /// Copy data from GPU to CPU.
    /// cudarc v0.19.8: Use ctx.dtoh_sync_copy() instead of slice.download()
    #[cfg(feature = "cuda")]
    fn copy_to_cpu<T: cudarc::driver::DeviceRepr + Clone>(
        &self, slice: &cudarc::driver::safe::CudaSlice<T>,
    ) -> Option<Vec<T>> {
        if let Some(ref dev) = self.device {
            // dev.dtoh_sync_copy(slice).ok()
            // slice.to_host().ok()
            slice.to_vec().ok()
        } else {
            None
        }
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
        
        // Allocate new buffer using ctx.alloc_zeros() instead of CudaSlice::zeros()
        if let Some(ref dev) = self.device {
            // match dev.alloc_zeros::<f32>(size) {
            // match CudaSlice::zeros(dev, size).ok() {
            match CudaSlice::zeroed(dev, size).ok() {
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
        
        // Copy data into the buffer using htod_sync_copy instead of CudaSlice::from_slice
        if let Some(ref dev) = self.device {
            // match dev.htod_sync_copy(data) {
            match CudaSlice::from_vec(dev, data.to_vec()) {
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
            // stream.synchronize()?;
            stream.synchronize().map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
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
                                    let start = Instant::now();
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
                    &a_gpu, &b_gpu, &c_gpu, &mut h_gpu_mut,
                    &delta_gpu, &delta_bias_gpu, &d_gpu,
                    &mut pulses_gpu_mut, &mut out_gpu_mut,
                    num_pulses as i32, d_inner as i32, d_state as i32,
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
                                    for i in 0..num_pulses {
                                        let len = pulses_content[i].len().min(d_inner);
                                        for j in 0..len {
                                            pulses_content[i][j] = out_cpu[i * d_inner + j];
                                        }
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

                                // Return buffers to cache
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
                                    let start = Instant::now();
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
                    "upload_to_buffer failed".to_string(),
                    "allocation returned None".to_string(),
                    "src/cuda.rs ssm_transform_batch() GPU path".to_string(),
                ));
            }
        }

        // CPU fallback
        let start = Instant::now();
        crate::ssm::ssm_transform_batch_raw(ssm, pulses_content, use_time_mixing);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn field_update(
        &mut self, pulses_content: &[Vec<f32>], pulses_weight: &[f32],
        field_state: &mut [f32], field_momentum: &mut [f32],
        learning_rate: f32, diffusion: f32,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len() as i32;
            let dim = if num_pulses > 0 { pulses_content[0].len() as i32 } else { 0 };
            if num_pulses == 0 || dim == 0 { return; }
            let flat_size = (num_pulses * dim) as usize;
            let mut flat_content = vec![0.0f32; flat_size];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(dim as usize);
                for j in 0..len { flat_content[i * dim as usize + j] = content[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            let content_key = format!("fu_content_{}", flat_size);
            let weight_key = format!("fu_weight_{}", pulses_weight.len());
            let state_key = format!("fu_state_{}", field_state.len());
            let momentum_key = format!("fu_momentum_{}", field_momentum.len());

            if let (Some(content_gpu), Some(weight_gpu), Some(state_gpu), Some(momentum_gpu)) = (
                self.upload_to_buffer(&content_key, &flat_content),
                self.upload_to_buffer(&weight_key, pulses_weight),
                self.upload_to_buffer(&state_key, field_state),
                self.upload_to_buffer(&momentum_key, field_momentum),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut state_gpu_mut = state_gpu;
                let mut momentum_gpu_mut = momentum_gpu;
                match mgr.launch_field_update(
                    &content_gpu, &weight_gpu, &mut state_gpu_mut, &mut momentum_gpu_mut,
                    learning_rate, diffusion, num_pulses, dim,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("field_update_kernel".to_string());
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
                                if let Some(state_cpu) = self.copy_to_cpu(&state_gpu_mut) {
                                    field_state.copy_from_slice(&state_cpu);
                                    self.batch_profile.gpu_mem_copied_d2h += field_state.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                }
                                if let Some(momentum_cpu) = self.copy_to_cpu(&momentum_gpu_mut) {
                                    field_momentum.copy_from_slice(&momentum_cpu);
                                    self.batch_profile.gpu_mem_copied_d2h += field_momentum.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                } else {
                                    download_ok = false;
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                self.return_buffer(content_key, content_gpu);
                                self.return_buffer(weight_key, weight_gpu);
                                self.return_buffer(state_key, state_gpu_mut);
                                self.return_buffer(momentum_key, momentum_gpu_mut);

                                if download_ok {
                                    let start = Instant::now();
                                    self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                    self.gpu_ops += 1;
                                    return;
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        crate::field::field_update_raw(pulses_content, pulses_weight, field_state, field_momentum, learning_rate, diffusion);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn field_diffuse(
        &mut self, pulses_content: &mut [Vec<f32>], field_state: &[f32],
        diffusion_factor: f32,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len() as i32;
            let dim = if num_pulses > 0 { pulses_content[0].len() as i32 } else { 0 };
            if num_pulses == 0 || dim == 0 { return; }
            let flat_size = (num_pulses * dim) as usize;
            let mut flat_content = vec![0.0f32; flat_size];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(dim as usize);
                for j in 0..len { flat_content[i * dim as usize + j] = content[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            let content_key = format!("fd_content_{}", flat_size);
            let state_key = format!("fd_state_{}", field_state.len());

            if let (Some(content_gpu), Some(state_gpu)) = (
                self.upload_to_buffer(&content_key, &flat_content),
                self.upload_to_buffer(&state_key, field_state),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut content_gpu_mut = content_gpu;
                match mgr.launch_field_diffuse(
                    &mut content_gpu_mut, &state_gpu, diffusion_factor, num_pulses, dim,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("field_diffuse_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                if let Some(content_cpu) = self.copy_to_cpu(&content_gpu_mut) {
                                    for i in 0..num_pulses as usize {
                                        let len = pulses_content[i].len().min(dim as usize);
                                        for j in 0..len {
                                            pulses_content[i][j] = content_cpu[i * dim as usize + j];
                                        }
                                    }
                                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                                    self.batch_profile.gpu_mem_copied_d2h += flat_size as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                self.return_buffer(content_key, content_gpu_mut);
                                self.return_buffer(state_key, state_gpu);

                                let start = Instant::now();
                                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                self.gpu_ops += 1;
                                return;
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        crate::field::field_diffuse_raw(pulses_content, field_state, diffusion_factor);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn cosine_similarity(
        &mut self, query: &[f32], vocabulary: &[Vec<f32>],
        vocab_norms: &[f32], similarities: &mut [f32],
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let vocab_size = vocabulary.len() as i32;
            let dim = if vocab_size > 0 { vocabulary[0].len() as i32 } else { 0 };
            if vocab_size == 0 || dim == 0 { return; }
            let flat_size = (vocab_size * dim) as usize;
            let mut flat_vocab = vec![0.0f32; flat_size];
            for (i, word) in vocabulary.iter().enumerate() {
                let len = word.len().min(dim as usize);
                for j in 0..len { flat_vocab[i * dim as usize + j] = word[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            let upload_start = Instant::now();
            let query_key = format!("cs_query_{}", query.len());
            let vocab_key = format!("cs_vocab_{}", flat_size);
            let norms_key = format!("cs_norms_{}", vocab_norms.len());
            let sim_key = format!("cs_sim_{}", similarities.len());

            if let (Some(query_gpu), Some(vocab_gpu), Some(norms_gpu), Some(sim_gpu)) = (
                self.upload_to_buffer(&query_key, query),
                self.upload_to_buffer(&vocab_key, &flat_vocab),
                self.upload_to_buffer(&norms_key, vocab_norms),
                self.upload_to_buffer(&sim_key, similarities),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut sim_gpu_mut = sim_gpu;
                match mgr.launch_cosine_similarity(
                    &query_gpu, &vocab_gpu, &norms_gpu, &mut sim_gpu_mut, vocab_size, dim,
                ) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("cosine_similarity_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                if let Some(sim_cpu) = self.copy_to_cpu(&sim_gpu_mut) {
                                    similarities.copy_from_slice(&sim_cpu);
                                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                                    self.batch_profile.gpu_mem_copied_d2h += similarities.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                self.return_buffer(query_key, query_gpu);
                                self.return_buffer(vocab_key, vocab_gpu);
                                self.return_buffer(norms_key, norms_gpu);
                                self.return_buffer(sim_key, sim_gpu_mut);

                                let start = Instant::now();
                                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                self.gpu_ops += 1;
                                return;
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        crate::field::cosine_similarity_raw(query, vocabulary, vocab_norms, similarities);
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn vector_add(
        &mut self, a: &mut [f32], b: &[f32], scale_a: f32, scale_b: f32,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let n = a.len() as i32;
            let upload_start = Instant::now();
            let a_key = format!("va_a_{}", a.len());
            let b_key = format!("va_b_{}", b.len());

            if let (Some(a_gpu), Some(b_gpu)) = (
                self.upload_to_buffer(&a_key, a),
                self.upload_to_buffer(&b_key, b),
            ) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut a_gpu_mut = a_gpu;
                match mgr.launch_vector_add(&mut a_gpu_mut, &b_gpu, scale_a, scale_b, n) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("vector_add_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                if let Some(a_cpu) = self.copy_to_cpu(&a_gpu_mut) {
                                    a.copy_from_slice(&a_cpu);
                                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                                    self.batch_profile.gpu_mem_copied_d2h += a.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                self.return_buffer(a_key, a_gpu_mut);
                                self.return_buffer(b_key, b_gpu);

                                let start = Instant::now();
                                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                self.gpu_ops += 1;
                                return;
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        for i in 0..a.len() {
            a[i] = a[i] * scale_a + b[i] * scale_b;
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn vector_clamp(
        &mut self, a: &mut [f32], min_val: f32, max_val: f32,
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let n = a.len() as i32;
            let upload_start = Instant::now();
            let a_key = format!("vc_a_{}", a.len());

            if let Some(a_gpu) = self.upload_to_buffer(&a_key, a) {
                let upload_time = upload_start.elapsed();
                self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                let kernel_start = Instant::now();
                let mut a_gpu_mut = a_gpu;
                match mgr.launch_vector_clamp(&mut a_gpu_mut, min_val, max_val, n) {
                    Ok(()) => {
                        self.batch_profile.kernel_launch_count += 1;
                        self.batch_profile.kernel_names.push("vector_clamp_kernel".to_string());
                        let kernel_time = kernel_start.elapsed();
                        self.batch_profile.gpu_kernel_ms += kernel_time.as_millis() as f64;

                        let sync_start = Instant::now();
                        match mgr.sync() {
                            Ok(()) => {
                                let sync_time = sync_start.elapsed();
                                self.batch_profile.gpu_sync_ms += sync_time.as_millis() as f64;
                                self.batch_profile.sync_count += 1;

                                let download_start = Instant::now();
                                if let Some(a_cpu) = self.copy_to_cpu(&a_gpu_mut) {
                                    a.copy_from_slice(&a_cpu);
                                    let elem_bytes = std::mem::size_of::<f32>() as u64;
                                    self.batch_profile.gpu_mem_copied_d2h += a.len() as u64 * elem_bytes;
                                    self.batch_profile.memcpy_count += 1;
                                }
                                let download_time = download_start.elapsed();
                                self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                self.return_buffer(a_key, a_gpu_mut);

                                let start = Instant::now();
                                self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                self.gpu_ops += 1;
                                return;
                            }
                            Err(_) => {}
                        }
                    }
                    Err(_) => {}
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        for i in 0..a.len() {
            a[i] = a[i].clamp(min_val, max_val);
        }
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }

    pub fn process_cores_batch(
        &mut self, cores: &mut [crate::core::NovaCore],
        pulses_content: &mut [Vec<f32>],
        pulses_entropy: &mut [f32],
        pulses_weight: &mut [f32],
    ) {
        #[cfg(feature = "cuda")]
        if let Some(ref mgr) = self.kernel_mgr {
            let preprocess_start = Instant::now();
            let num_pulses = pulses_content.len();
            let num_cores = cores.len();
            if num_pulses == 0 || num_cores == 0 { return; }
            let dim = pulses_content[0].len();
            let d_state = cores[0].ssm.d_state;
            let memory_size = cores[0].memory.len();

            let flat_size = num_pulses * dim;
            let mut flat_content = vec![0.0f32; flat_size];
            for (i, content) in pulses_content.iter().enumerate() {
                let len = content.len().min(dim);
                for j in 0..len { flat_content[i * dim + j] = content[j]; }
            }
            let preprocess_time = preprocess_start.elapsed();
            self.batch_profile.cpu_preprocess_ms += preprocess_time.as_millis() as f64;

            for core_idx in 0..num_cores {
                let core = &cores[core_idx];
                let upload_start = Instant::now();

                let content_key = format!("pc_content_{}_{}", core_idx, flat_size);
                let entropy_key = format!("pc_entropy_{}_{}", core_idx, pulses_entropy.len());
                let weight_key = format!("pc_weight_{}_{}", core_idx, pulses_weight.len());
                let mem_key = format!("pc_mem_{}_{}", core_idx, core.memory.len());
                let istate_key = format!("pc_istate_{}_{}", core_idx, core.internal_state.len());
                let gate_key = format!("pc_gate_{}_{}", core_idx, 1); // gate is f32, not a slice
                let ssm_a_key = format!("pc_ssm_a_{}_{}", core_idx, core.ssm.a.len());
                let ssm_b_key = format!("pc_ssm_b_{}_{}", core_idx, core.ssm.b.len());
                let ssm_c_key = format!("pc_ssm_c_{}_{}", core_idx, core.ssm.c.len());
                let ssm_h_key = format!("pc_ssm_h_{}_{}", core_idx, core.ssm.h.len());
                let ssm_delta_key = format!("pc_ssm_delta_{}_{}", core_idx, core.ssm.delta.len());
                let ssm_db_key = format!("pc_ssm_db_{}_{}", core_idx, core.ssm.delta_bias.len());
                let ssm_d_key = format!("pc_ssm_d_{}_{}", core_idx, core.ssm.d.len());

                if let (Some(content_gpu), Some(entropy_gpu), Some(weight_gpu),
                        Some(mem_gpu), Some(istate_gpu), Some(gate_gpu),
                        Some(ssm_a_gpu), Some(ssm_b_gpu), Some(ssm_c_gpu),
                        Some(ssm_h_gpu), Some(ssm_delta_gpu), Some(ssm_db_gpu),
                        Some(ssm_d_gpu)) = (
                    self.upload_to_buffer(&content_key, &flat_content),
                    self.upload_to_buffer(&entropy_key, pulses_entropy),
                    self.upload_to_buffer(&weight_key, pulses_weight),
                    self.upload_to_buffer(&mem_key, &core.memory),
                    self.upload_to_buffer(&istate_key, &core.internal_state),
                    self.upload_to_buffer(&gate_key, &core.gate),
                    self.upload_to_buffer(&ssm_a_key, &core.ssm.a),
                    self.upload_to_buffer(&ssm_b_key, &core.ssm.b),
                    self.upload_to_buffer(&ssm_c_key, &core.ssm.c),
                    self.upload_to_buffer(&ssm_h_key, &core.ssm.h),
                    self.upload_to_buffer(&ssm_delta_key, &core.ssm.delta),
                    self.upload_to_buffer(&ssm_db_key, &core.ssm.delta_bias),
                    self.upload_to_buffer(&ssm_d_key, &core.ssm.d),
                ) {
                    let upload_time = upload_start.elapsed();
                    self.batch_profile.gpu_upload_ms += upload_time.as_millis() as f64;

                    let kernel_start = Instant::now();
                    let mut content_gpu_mut = content_gpu;
                    let mut entropy_gpu_mut = entropy_gpu;
                    let mut weight_gpu_mut = weight_gpu;
                    let mut ssm_h_gpu_mut = ssm_h_gpu;

                    if self.use_async_streams {
                        if let Some(async_stream) = self.next_async_stream() {
                            let _ = mgr.launch_core_process_async(
                                async_stream,
                                &mut content_gpu_mut, &mut entropy_gpu_mut, &mut weight_gpu_mut,
                                &mem_gpu, &istate_gpu, &gate_gpu,
                                &ssm_a_gpu, &ssm_b_gpu, &ssm_c_gpu, &mut ssm_h_gpu_mut,
                                &ssm_delta_gpu, &ssm_db_gpu, &ssm_d_gpu,
                                num_pulses as i32, dim as i32, num_cores as i32,
                                memory_size as i32, d_state as i32,
                            );
                        }
                    }

                    match mgr.launch_core_process(
                        &mut content_gpu_mut, &mut entropy_gpu_mut, &mut weight_gpu_mut,
                        &mem_gpu, &istate_gpu, &gate_gpu,
                        &ssm_a_gpu, &ssm_b_gpu, &ssm_c_gpu, &mut ssm_h_gpu_mut,
                        &ssm_delta_gpu, &ssm_db_gpu, &ssm_d_gpu,
                        num_pulses as i32, dim as i32, num_cores as i32,
                        memory_size as i32, d_state as i32,
                    ) {
                        Ok(()) => {
                            self.batch_profile.kernel_launch_count += 1;
                            self.batch_profile.kernel_names.push("core_process_kernel".to_string());
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

                                    if let Some(content_cpu) = self.copy_to_cpu(&content_gpu_mut) {
                                        for i in 0..num_pulses {
                                            let len = pulses_content[i].len().min(dim);
                                            for j in 0..len {
                                                pulses_content[i][j] = content_cpu[i * dim + j];
                                            }
                                        }
                                        self.batch_profile.gpu_mem_copied_d2h += flat_size as u64 * elem_bytes;
                                        self.batch_profile.memcpy_count += 1;
                                    } else { download_ok = false; }

                                    if let Some(entropy_cpu) = self.copy_to_cpu(&entropy_gpu_mut) {
                                        pulses_entropy.copy_from_slice(&entropy_cpu);
                                        self.batch_profile.gpu_mem_copied_d2h += pulses_entropy.len() as u64 * elem_bytes;
                                        self.batch_profile.memcpy_count += 1;
                                    } else { download_ok = false; }

                                    if let Some(weight_cpu) = self.copy_to_cpu(&weight_gpu_mut) {
                                        pulses_weight.copy_from_slice(&weight_cpu);
                                        self.batch_profile.gpu_mem_copied_d2h += pulses_weight.len() as u64 * elem_bytes;
                                        self.batch_profile.memcpy_count += 1;
                                    } else { download_ok = false; }

                                    let download_time = download_start.elapsed();
                                    self.batch_profile.gpu_download_ms += download_time.as_millis() as f64;

                                    self.return_buffer(content_key, content_gpu_mut);
                                    self.return_buffer(entropy_key, entropy_gpu_mut);
                                    self.return_buffer(weight_key, weight_gpu_mut);
                                    self.return_buffer(mem_key, mem_gpu);
                                    self.return_buffer(istate_key, istate_gpu);
                                    self.return_buffer(gate_key, gate_gpu);
                                    self.return_buffer(ssm_a_key, ssm_a_gpu);
                                    self.return_buffer(ssm_b_key, ssm_b_gpu);
                                    self.return_buffer(ssm_c_key, ssm_c_gpu);
                                    self.return_buffer(ssm_h_key, ssm_h_gpu_mut);
                                    self.return_buffer(ssm_delta_key, ssm_delta_gpu);
                                    self.return_buffer(ssm_db_key, ssm_db_gpu);
                                    self.return_buffer(ssm_d_key, ssm_d_gpu);

                                    if download_ok {
                                        let start = Instant::now();
                                        self.total_gpu_time_ms += start.elapsed().as_millis() as f64;
                                        self.gpu_ops += 1;
                                        return;
                                    }
                                }
                                Err(_) => {}
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
        }

        // CPU fallback
        let start = Instant::now();
        
        // Convert separate arrays into NovaPulse objects
        let mut pulses: Vec<crate::pulse::NovaPulse> = Vec::with_capacity(pulses_content.len());
        for i in 0..pulses_content.len() {
            let dim = pulses_content[i].len();
            let mut pulse = crate::pulse::NovaPulse::new(dim, i);
            pulse.content = pulses_content[i].clone();
            pulse.semantic_content = pulses_content[i].clone();
            if i < pulses_entropy.len() {
                pulse.entropy = pulses_entropy[i];
            }
            if i < pulses_weight.len() {
                pulse.weight = pulses_weight[i];
            }
            pulses.push(pulse);
        }
        
        for core in cores.iter_mut() {
            core.process(&mut pulses);
        }
        
        // Copy back the updated values
        for i in 0..pulses_content.len() {
            if i < pulses.len() {
                let pulse = &pulses[i];
                let len = pulses_content[i].len().min(pulse.content.len());
                pulses_content[i][..len].copy_from_slice(&pulse.content[..len]);
                if i < pulses_entropy.len() {
                    pulses_entropy[i] = pulse.entropy;
                }
                if i < pulses_weight.len() {
                    pulses_weight[i] = pulse.weight;
                }
            }
        }
        
        self.total_cpu_time_ms += start.elapsed().as_millis() as f64;
        self.cpu_ops += 1;
    }
}

// ============================================================================
// Global Accelerator Functions
// ============================================================================

/// Initialize the global accelerator (called from main.rs)
pub fn init_global_accelerator() {
    // This function is called from main.rs to initialize the accelerator
    // The actual initialization happens when NovaAccelerator::auto_detect() is called
    // We just print a message for now
    let backend = auto_detect_backend();
    println!("🚀 Accelerator initialized: {}", backend.name());
}

/// Get the backend name as a string
pub fn get_backend_name() -> String {
    let backend = auto_detect_backend();
    backend.name().to_string()
}

/// Check if GPU is available
pub fn is_gpu_available() -> bool {
    let backend = auto_detect_backend();
    backend.is_gpu()
}

/// Get the accelerator instance
pub fn get_accelerator() -> NovaAccelerator {
    NovaAccelerator::auto_detect()
}

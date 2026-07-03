// Nova Core Build Script
// Compiles CUDA kernels to PTX at build time when the "cuda" feature is enabled

fn main() {
    // Only compile CUDA kernels when the "cuda" feature is enabled
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=kernels/ssm.cu");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        
        // Determine CUDA toolkit path
        let cuda_path = std::env::var("CUDA_HOME")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| {
                // Default paths for different platforms
                if cfg!(target_os = "windows") {
                    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0".to_string()
                } else {
                    "/usr/local/cuda".to_string()
                }
            });
        
        let nvcc_path = format!("{}/bin/nvcc", cuda_path);
        
        // Create output directory for compiled PTX
        let out_dir = std::env::var("OUT_DIR").unwrap();
        
        // Common nvcc flags for all architectures
        let common_flags = &["--ptx", "-O3", "--use_fast_math", "--ftz=true", "--prec-div=false", "--prec-sqrt=false"];
        
        // Architecture list with names for logging
        let architectures = [
            ("sm_75", "Turing (T4, RTX 20xx)"),
            ("sm_80", "Ampere (A100, RTX 30xx)"),
            ("sm_86", "Ampere (RTX 30xx, GA10x)"),
            ("sm_89", "Ada Lovelace (RTX 40xx)"),
            ("sm_90", "Hopper (H100, H200)"),
        ];
        
        let mut compiled_count = 0;
        let mut primary_ptx = String::new();
        
        for (arch, name) in &architectures {
            let ptx_output = format!("{}/ssm_kernels_{}.ptx", out_dir, arch);
            let status = std::process::Command::new(&nvcc_path)
                .args(common_flags)
                .args(&["-arch", arch, "-o", &ptx_output, "kernels/ssm.cu"])
                .status();
            
            // If nvcc command couldn't be launched (e.g., nvcc not found),
            // treat it as a failure by checking if status is Ok and successful
            let is_success = status.map(|s| s.success()).unwrap_or(false);
            
            if is_success {
                compiled_count += 1;
                println!("cargo:warning=✅ Compiled {} PTX ({})", arch, name);
                if primary_ptx.is_empty() {
                    primary_ptx = ptx_output;
                }
            } else {
                println!("cargo:warning=⚠️  Skipped {} PTX ({}): nvcc not available for this arch", arch, name);
            }
        }
        
        if compiled_count == 0 {
            panic!("CUDA kernel compilation failed for all architectures. Make sure nvcc is available.");
        }
        
        // Set the primary PTX path (prefer highest architecture, fall back to lowest)
        // The runtime will try sm_80 first, then fall back to sm_75
        println!("cargo:warning=✅ Compiled {} CUDA PTX variants", compiled_count);
        println!("cargo:rustc-env=SSM_KERNELS_PTX={}", primary_ptx);
    }
}


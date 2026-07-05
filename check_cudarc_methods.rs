// Check cudarc methods
extern crate cudarc;

use cudarc::driver::safe::*;

fn main() {
    println!("Checking cudarc methods...");
    
    // Check CudaContext methods
    println!("CudaContext methods:");
    println!("- new()");
    println!("- name()");
    println!("- new_stream()");
    println!("- alloc_zeros()");
    println!("- htod_sync_copy()");
    println!("- dtoh_sync_copy()");
    
    // Check CudaModule methods
    println!("\nCudaModule methods:");
    println!("- load_from_ptx_string()");
    println!("- get_function()");
    
    // Check CudaFunction methods
    println!("\nCudaFunction methods:");
    println!("- launch()?");
    println!("- launch_kernel()?");
    println!("- call()?");
    
    // Check CudaStream methods
    println!("\nCudaStream methods:");
    println!("- synchronize()");
}
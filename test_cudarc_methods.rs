// Simple test to check cudarc API
use cudarc::driver::safe::*;

fn main() {
    println!("Testing cudarc v0.19.8 API...");
    
    // Create context
    let ctx = CudaContext::new(0).unwrap();
    
    // Check allocation methods
    println!("Testing allocation:");
    let slice1 = ctx.alloc_zeros::<f32>(100);
    println!("alloc_zeros: {}", slice1.is_ok());
    
    let slice2 = ctx.alloc::<f32>(100);
    println!("alloc: {}", slice2.is_ok());
    
    let slice3 = ctx.zeroed::<f32>(100);
    println!("zeroed: {}", slice3.is_ok());
    
    // Check copy methods
    println!("Testing copy methods:");
    let data = vec![1.0f32, 2.0, 3.0];
    let slice4 = ctx.htod_sync_copy(&data);
    println!("htod_sync_copy: {}", slice4.is_ok());
    
    let slice5 = ctx.htod_copy(&data);
    println!("htod_copy: {}", slice5.is_ok());
    
    let slice6 = ctx.copy_to(&data);
    println!("copy_to: {}", slice6.is_ok());
    
    let slice7 = ctx.copy_to_device(&data);
    println!("copy_to_device: {}", slice7.is_ok());
    
    // Check module loading
    println!("Testing module loading:");
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    
    let module1 = CudaModule::from_ptx(&ctx, ptx);
    println!("from_ptx: {}", module1.is_ok());
    
    let module2 = CudaModule::load_ptx(&ctx, ptx);
    println!("load_ptx: {}", module2.is_ok());
    
    let module3 = CudaModule::load_from_ptx_string(&ctx, ptx);
    println!("load_from_ptx_string: {}", module3.is_ok());
    
    // Check CudaSlice methods
    println!("Testing CudaSlice methods:");
    if let Ok(slice) = slice4 {
        let vec_result = slice.to_vec();
        println!("to_vec: {}", vec_result.is_ok());
        
        let htod_result = slice.htod_copy(&data);
        println!("htod_copy (from slice): {}", htod_result.is_ok());
    }
}
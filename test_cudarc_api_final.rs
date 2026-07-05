// Test cudarc API to find correct method names
extern crate cudarc;

use cudarc::driver::safe::*;

fn main() {
    println!("Testing cudarc v0.19.8 API");
    
    // Check CudaContext methods
    println!("\nCudaContext methods:");
    let ctx = CudaContext::new(0).unwrap();
    
    // Check allocation methods
    println!("Checking allocation methods...");
    let _zeros = ctx.alloc_zeros::<f32>(100);
    let _alloc = ctx.alloc::<f32>(100);
    let _zeroed = ctx.zeroed::<f32>(100);
    
    // Check copy methods
    println!("Checking copy methods...");
    let data = vec![1.0f32, 2.0, 3.0];
    let _slice1 = ctx.htod_sync_copy(&data);
    let _slice2 = ctx.htod_copy(&data);
    let _slice3 = ctx.copy_to(&data);
    let _slice4 = ctx.copy_to_device(&data);
    
    // Check CudaModule methods
    println!("\nCudaModule methods:");
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    let _module1 = CudaModule::load_from_ptx_string(&ctx, ptx);
    let _module2 = CudaModule::from_ptx(&ctx, ptx);
    let _module3 = CudaModule::load_ptx(&ctx, ptx);
    let _module4 = CudaModule::new(&ctx, ptx);
    
    // Check CudaFunction methods
    println!("\nCudaFunction methods:");
    if let Ok(module) = CudaModule::load_from_ptx_string(&ctx, ptx) {
        if let Ok(func) = module.get_function("main") {
            println!("Function obtained");
            // Check launch methods
            let stream = ctx.new_stream().unwrap();
            let _ = func.launch(&stream, (1, 1, 1), (1, 1, 1), &[]);
            let _ = func.launch_kernel(&stream, (1, 1, 1), (1, 1, 1), &[]);
            let _ = func.call(&stream, (1, 1, 1), (1, 1, 1), &[]);
        }
    }
    
    println!("Done testing");
}
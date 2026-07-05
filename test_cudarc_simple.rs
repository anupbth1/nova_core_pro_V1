// Simple test to check cudarc API
use cudarc::driver::safe::*;

fn main() {
    println!("Testing cudarc API...");
    
    // Create context
    let ctx = CudaContext::new(0).unwrap();
    
    // Check CudaContext methods
    println!("CudaContext methods:");
    println!("- new() works");
    
    // Check allocation
    println!("Testing allocation...");
    let zeros = ctx.alloc_zeros::<f32>(100);
    println!("alloc_zeros: {:?}", zeros.is_ok());
    
    let alloc = ctx.alloc::<f32>(100);
    println!("alloc: {:?}", alloc.is_ok());
    
    let zeroed = ctx.zeroed::<f32>(100);
    println!("zeroed: {:?}", zeroed.is_ok());
    
    // Check copy methods
    println!("Testing copy methods...");
    let data = vec![1.0f32, 2.0, 3.0];
    let slice1 = ctx.htod_sync_copy(&data);
    println!("htod_sync_copy: {:?}", slice1.is_ok());
    
    let slice2 = ctx.htod_copy(&data);
    println!("htod_copy: {:?}", slice2.is_ok());
    
    let slice3 = ctx.copy_to(&data);
    println!("copy_to: {:?}", slice3.is_ok());
    
    let slice4 = ctx.copy_to_device(&data);
    println!("copy_to_device: {:?}", slice4.is_ok());
    
    // Check CudaModule methods
    println!("Testing CudaModule methods...");
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    let module1 = CudaModule::load_from_ptx_string(&ctx, ptx);
    println!("load_from_ptx_string: {:?}", module1.is_ok());
    
    let module2 = CudaModule::from_ptx(&ctx, ptx);
    println!("from_ptx: {:?}", module2.is_ok());
    
    let module3 = CudaModule::load_ptx(&ctx, ptx);
    println!("load_ptx: {:?}", module3.is_ok());
    
    let module4 = CudaModule::new(&ctx, ptx);
    println!("new: {:?}", module4.is_ok());
    
    // Check CudaFunction methods
    println!("Testing CudaFunction methods...");
    if let Ok(module) = CudaModule::load_ptx(&ctx, ptx) {
        if let Ok(func) = module.get_function("main") {
            println!("Got function");
            let stream = ctx.new_stream().unwrap();
            
            // Check launch methods
            let result1 = func.launch(&stream, (1, 1, 1), (1, 1, 1), &[]);
            println!("launch: {:?}", result1.is_ok());
            
            let result2 = func.launch_kernel(&stream, (1, 1, 1), (1, 1, 1), &[]);
            println!("launch_kernel: {:?}", result2.is_ok());
            
            let result3 = func.call(&stream, (1, 1, 1), (1, 1, 1), &[]);
            println!("call: {:?}", result3.is_ok());
        }
    }
}
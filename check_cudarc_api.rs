// Check cudarc API by trying different method names
use cudarc::driver::safe::*;

fn main() {
    println!("Checking cudarc API...");
    
    // Create context
    let ctx = CudaContext::new(0).unwrap();
    
    // Check CudaContext methods
    println!("Checking CudaContext methods...");
    
    // Try different allocation methods
    let _ = ctx.alloc::<f32>(100);
    let _ = ctx.alloc_zeros::<f32>(100);
    let _ = ctx.zeroed::<f32>(100);
    
    // Try different copy methods
    let data = vec![1.0f32, 2.0, 3.0];
    let _ = ctx.htod_sync_copy(&data);
    let _ = ctx.htod_copy(&data);
    let _ = ctx.copy_to(&data);
    let _ = ctx.copy_to_device(&data);
    
    // Check CudaModule methods
    println!("Checking CudaModule methods...");
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    
    // Try different PTX loading methods
    let _ = CudaModule::from_ptx(&ctx, ptx);
    let _ = CudaModule::load_ptx(&ctx, ptx);
    let _ = CudaModule::load_from_ptx_string(&ctx, ptx);
    let _ = CudaModule::from_ptx_string(&ctx, ptx);
    let _ = CudaModule::new(&ctx, ptx);
    
    // Check CudaFunction methods
    println!("Checking CudaFunction methods...");
    
    // Check CudaStream methods
    println!("Checking CudaStream methods...");
    let stream = ctx.new_stream().unwrap();
    let _ = stream.synchronize();
}
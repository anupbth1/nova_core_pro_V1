// Simple test to check cudarc API
fn main() {
    // Just try to compile with cudarc to see what methods exist
    // We'll look at the compiler errors
    let ctx = cudarc::driver::safe::CudaContext::new(0).unwrap();
    
    // Try to create a module from PTX
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    
    // Try different method names
    let _ = cudarc::driver::safe::CudaModule::from_ptx(&ctx, ptx);
    let _ = cudarc::driver::safe::CudaModule::load_ptx(&ctx, ptx);
    let _ = cudarc::driver::safe::CudaModule::load_from_ptx(&ctx, ptx);
    let _ = cudarc::driver::safe::CudaModule::load_from_ptx_string(&ctx, ptx);
    let _ = cudarc::driver::safe::CudaModule::new(&ctx, ptx);
    let _ = cudarc::driver::safe::CudaModule::from_ptx_string(&ctx, ptx);
    
    // Try to get a function
    let module = cudarc::driver::safe::CudaModule::from_ptx(&ctx, ptx).unwrap();
    let func = module.get_function("main").unwrap();
    
    // Try to launch the function
    let stream = ctx.new_stream().unwrap();
    let _ = func.launch(&stream, (1, 1, 1), (1, 1, 1), &[]);
    let _ = func.launch_kernel(&stream, (1, 1, 1), (1, 1, 1), &[]);
    let _ = func.call(&stream, (1, 1, 1), (1, 1, 1), &[]);
    let _ = func.launch_kernel_async(&stream, (1, 1, 1), (1, 1, 1), &[]);
    
    // Check allocation methods
    let _ = ctx.alloc::<f32>(100);
    let _ = ctx.zeroed::<f32>(100);
    let _ = ctx.alloc_zeros::<f32>(100);
    
    // Check copy methods
    let data = vec![1.0f32, 2.0, 3.0];
    let _ = ctx.htod_copy(&data);
    let _ = ctx.htod_sync_copy(&data);
    let _ = ctx.copy_to(&data);
    let _ = ctx.copy_to_device(&data);
    
    let slice = ctx.alloc::<f32>(3).unwrap();
    let _ = ctx.dtoh_copy(&slice);
    let _ = ctx.dtoh_sync_copy(&slice);
    let _ = slice.download();
}
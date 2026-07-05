// Simple check of cudarc methods
fn main() {
    // Just check what methods exist by trying to compile
    // We'll look at the compiler errors
    let _ = cudarc::driver::safe::CudaContext::new(0);
    let _ = cudarc::driver::safe::CudaModule::from_ptx;
    let _ = cudarc::driver::safe::CudaModule::load_ptx;
    let _ = cudarc::driver::safe::CudaModule::load_from_ptx_string;
    let _ = cudarc::driver::safe::CudaModule::new;
    
    // Check CudaFunction methods
    let _ = cudarc::driver::safe::CudaFunction::launch;
    let _ = cudarc::driver::safe::CudaFunction::launch_kernel;
    let _ = cudarc::driver::safe::CudaFunction::call;
    
    // Check CudaContext methods
    let ctx = cudarc::driver::safe::CudaContext::new(0).unwrap();
    let _ = ctx.alloc_zeros::<f32>;
    let _ = ctx.zeroed::<f32>;
    let _ = ctx.alloc::<f32>;
    let _ = ctx.htod_sync_copy::<f32>;
    let _ = ctx.htod_copy::<f32>;
    let _ = ctx.dtoh_sync_copy::<f32>;
    let _ = ctx.dtoh_copy::<f32>;
    let _ = ctx.copy_to::<f32>;
    let _ = ctx.copy_to_device::<f32>;
}
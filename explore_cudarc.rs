// Explore cudarc v0.19.8 API
use cudarc::driver::safe::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Exploring cudarc v0.19.8 API...");
    
    // Create context
    let ctx = CudaContext::new(0)?;
    println!("Context created");
    
    // Check CudaContext methods
    println!("CudaContext methods:");
    println!("  - new() works");
    println!("  - name() works: {:?}", ctx.name());
    
    // Check allocation
    println!("Testing allocation...");
    let slice: CudaSlice<f32> = ctx.alloc(100)?;
    println!("  ctx.alloc() works");
    
    // Check zeroed allocation
    let zeros: CudaSlice<f32> = ctx.zeroed(100)?;
    println!("  ctx.zeroed() works");
    
    // Check copy methods
    let host_data = vec![1.0f32, 2.0, 3.0];
    let device_slice: CudaSlice<f32> = ctx.htod_copy(&host_data)?;
    println!("  ctx.htod_copy() works");
    
    let host_copy: Vec<f32> = ctx.dtoh_copy(&device_slice)?;
    println!("  ctx.dtoh_copy() works");
    
    // Check stream creation
    let stream: CudaStream = ctx.new_stream()?;
    println!("  ctx.new_stream() works");
    
    // Check stream sync
    stream.synchronize()?;
    println!("  stream.synchronize() works");
    
    // Check CudaModule methods
    println!("Checking CudaModule methods...");
    
    // Try to find the correct PTX loading method
    let ptx = r#".version 7.5
.target sm_75
.entry main() { ret; }"#;
    
    // Try different method names
    println!("Trying CudaModule::load_ptx...");
    match CudaModule::load_ptx(&ctx, ptx) {
        Ok(_) => println!("  CudaModule::load_ptx works"),
        Err(e) => println!("  CudaModule::load_ptx error: {:?}", e),
    }
    
    println!("Trying CudaModule::from_ptx...");
    match CudaModule::from_ptx(&ctx, ptx) {
        Ok(_) => println!("  CudaModule::from_ptx works"),
        Err(e) => println!("  CudaModule::from_ptx error: {:?}", e),
    }
    
    println!("Trying CudaModule::new...");
    match CudaModule::new(&ctx, ptx) {
        Ok(_) => println!("  CudaModule::new works"),
        Err(e) => println!("  CudaModule::new error: {:?}", e),
    }
    
    // Check what methods are available on CudaModule
    println!("CudaModule type: {:?}", std::any::type_name::<CudaModule>());
    
    Ok(())
}
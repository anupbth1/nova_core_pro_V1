// Quick test to check cudarc API
use cudarc::driver::safe::*;

fn main() {
    // Try to compile with cudarc to see what methods exist
    let ctx = CudaContext::new(0).unwrap();
    
    // Check available methods
    let stream = ctx.new_stream().unwrap();
    println!("Stream type: {:?}", stream);
    
    // Check if we can get stream as Arc
    let stream_ref = &stream;
    
    // Check PTX loading
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    let module = CudaModule::load_from_ptx_string(&ctx, ptx).unwrap();
    
    // Check function
    let func = module.get_function("main").unwrap();
    println!("Function type: {:?}", func);
    
    // Check if launch exists
    // func.launch(&stream, (1, 1, 1), (1, 1, 1), &[]);
    
    // Check allocation
    let zeros = ctx.alloc_zeros::<f32>(100).unwrap();
    println!("Allocated zeros");
    
    // Check copy methods
    let data = vec![1.0f32, 2.0, 3.0];
    let slice = ctx.htod_sync_copy(&data).unwrap();
    println!("htod copy");
    
    let cpu_data = ctx.dtoh_sync_copy(&slice).unwrap();
    println!("dtoh copy");
}
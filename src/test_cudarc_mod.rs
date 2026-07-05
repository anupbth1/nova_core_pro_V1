#[cfg(feature = "cuda")]
mod test {
    use cudarc::driver::safe::*;
    
    pub fn test_api() {
        println!("Testing cudarc API");
        
        // Check what methods are available
        let ctx = CudaContext::new(0).unwrap();
        
        // Check stream creation
        let stream = ctx.new_stream().unwrap();
        println!("Stream created");
        
        // Check PTX loading
        let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
        let module = CudaModule::load_from_ptx_string(&ctx, ptx).unwrap();
        println!("Module loaded");
        
        // Check function
        let func = module.get_function("main").unwrap();
        println!("Function obtained");
        
        // Check allocation
        let zeros = ctx.alloc_zeros::<f32>(100).unwrap();
        println!("Zeros allocated: {}", zeros.len());
        
        // Check copy
        let data = vec![1.0f32, 2.0, 3.0];
        let slice = ctx.htod_sync_copy(&data).unwrap();
        println!("Data copied to GPU: {}", slice.len());
        
        let cpu_data = ctx.dtoh_sync_copy(&slice).unwrap();
        println!("Data copied back: {:?}", cpu_data);
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    test::test_api();
    
    #[cfg(not(feature = "cuda"))]
    println!("CUDA feature not enabled");
}
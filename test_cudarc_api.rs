use cudarc::driver::safe::*;

fn test_cudarc_api() {
    // Test creating context
    match CudaContext::new(0) {
        Ok(ctx) => {
            println!("Context created successfully");
            
            // Test creating stream
            match ctx.new_stream() {
                Ok(stream) => {
                    println!("Stream created: {}", stream);
                    
                    // Test PTX loading
                    let ptx_src = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
                    match CudaModule::load_from_ptx_string(&ctx, ptx_src) {
                        Ok(module) => {
                            println!("Module loaded");
                            // Test getting function
                            match module.get_function("main") {
                                Ok(func) => {
                                    println!("Function obtained");
                                    // Check launch method signature
                                    println!("Function type: {:?}", func);
                                },
                                Err(e) => println!("Failed to get function: {:?}", e),
                            }
                        },
                        Err(e) => println!("Failed to load module: {:?}", e),
                    }
                    
                    // Test allocation methods
                    let size = 100;
                    match ctx.alloc_zeros::<f32>(size) {
                        Ok(slice) => println!("Allocated zeros: {}", slice.len()),
                        Err(e) => println!("Failed to alloc zeros: {:?}", e),
                    }
                    
                    // Test htod copy
                    let data = vec![1.0f32, 2.0, 3.0];
                    match ctx.htod_sync_copy(&data) {
                        Ok(slice) => println!("htod sync copy: {}", slice.len()),
                        Err(e) => println!("Failed htod copy: {:?}", e),
                    }
                    
                    // Test dtoh copy
                    let test_data = vec![4.0f32, 5.0, 6.0];
                    match ctx.htod_sync_copy(&test_data) {
                        Ok(slice) => {
                            match ctx.dtoh_sync_copy(&slice) {
                                Ok(cpu_data) => println!("dtoh sync copy successful: {}", cpu_data.len()),
                                Err(e) => println!("Failed dtoh copy: {:?}", e),
                            }
                        },
                        Err(e) => println!("Failed initial htod: {:?}", e),
                    }
                },
                Err(e) => println!("Failed to create stream: {:?}", e),
            }
        },
        Err(e) => println!("Failed to create context: {:?}", e),
    }
}

fn main() {
    test_cudarc_api();
}
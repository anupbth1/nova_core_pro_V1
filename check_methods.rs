// Check what methods are available on CudaFunction
use cudarc::driver::safe::*;

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let ptx = ".version 7.5\n.target sm_75\n.entry main() { ret; }";
    let module = CudaModule::load_from_ptx_string(&ctx, ptx).unwrap();
    let func = module.get_function("main").unwrap();
    
    // Check what methods are available on func
    println!("Function type: {}", std::any::type_name::<CudaFunction>());
    
    // Try to see available methods
    // func.launch(...);
}
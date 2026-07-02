//! Compare Nova with other LLMs

use crate::loom::NovaLoom;
use std::collections::HashMap;

pub fn compare_with_llama(_nova: &mut NovaLoom, _samples: usize) -> HashMap<String, f32> {
    let mut results = HashMap::new();
    results.insert("nova_vs_llama".to_string(), 0.5);
    results
}

pub fn run_comparison(_nova: &mut NovaLoom) {
    println!("📊 Comparison with Llama 3 (placeholder - requires llama.cpp)");
    println!("  Coming soon: Download llama.cpp and run side-by-side");
}
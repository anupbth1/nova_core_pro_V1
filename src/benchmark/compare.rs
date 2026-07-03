//! Compare Nova with other LLMs
//! PRIORITY 8: Self-contained comparison using Nova's own benchmarks as reference

use crate::loom::NovaLoom;
use crate::benchmark::tasks;
use std::collections::HashMap;

/// Run a self-contained comparison using Nova's benchmark suite.
/// Returns a map of task names to scores (0.0 - 1.0).
pub fn run_self_comparison(nova: &mut NovaLoom, samples: usize) -> HashMap<String, f32> {
    let mut results = HashMap::new();
    
    // Run all task categories and collect scores
    let task_groups: Vec<(&str, fn() -> Vec<tasks::BenchmarkTask>)> = vec![
        ("language", tasks::get_language_tasks),
        ("reasoning", tasks::get_reasoning_tasks),
        ("code", tasks::get_code_tasks),
        ("long_context", tasks::get_long_context_tasks),
        ("memory", tasks::get_memory_tasks),
    ];
    
    for (group_name, task_fn) in task_groups {
        let tasks_list = task_fn();
        let mut group_scores = Vec::new();
        
        for task in &tasks_list {
            let score = task.run(nova, samples);
            results.insert(format!("{}/{}", group_name, task.name), score);
            group_scores.push(score);
        }
        
        let avg = group_scores.iter().sum::<f32>() / group_scores.len() as f32;
        results.insert(format!("{}_average", group_name), avg);
    }
    
    // Overall average
    let all_scores: Vec<f32> = results.values().copied().collect();
    let overall = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
    results.insert("overall".to_string(), overall);
    
    results
}

/// Print a comparison report showing Nova's performance across all benchmarks.
pub fn print_comparison_report(results: &HashMap<String, f32>) {
    println!("\n📊 NOVA BENCHMARK COMPARISON REPORT");
    println!("{}", "═".repeat(60));
    
    let categories = ["language", "reasoning", "code", "long_context", "memory"];
    
    for category in &categories {
        println!("\n  📁 {}:", category.to_uppercase());
        let mut cat_scores: Vec<(&String, &f32)> = results.iter()
            .filter(|(k, _)| k.starts_with(&format!("{}/", category)))
            .collect();
        cat_scores.sort_by(|a, b| a.0.cmp(b.0));
        
        for (name, score) in &cat_scores {
            let bar = "█".repeat((*score * 20.0) as usize);
            let empty = "░".repeat(20 - (*score * 20.0) as usize);
            println!("    {:30} [{}{}] {:5.1}%", 
                name.split('/').nth(1).unwrap_or(name), 
                bar, empty, *score * 100.0);
        }
        
        if let Some(avg) = results.get(&format!("{}_average", category)) {
            println!("    {:30} {:>27.1}%", "─ AVERAGE ─", avg * 100.0);
        }
    }
    
    if let Some(overall) = results.get("overall") {
        println!("\n  {}", "─".repeat(40));
        println!("  🏆 OVERALL SCORE: {:5.1}%", overall * 100.0);
        let bar = "█".repeat((overall * 30.0) as usize);
        let empty = "░".repeat(30 - (overall * 30.0) as usize);
        println!("  [{}{}]", bar, empty);
    }
    
    println!("\n{}", "═".repeat(60));
}

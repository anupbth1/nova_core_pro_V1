//! Complete Nova Benchmark Suite

mod tasks;
mod metrics;
mod data;
mod compare;
mod improve;

use crate::loom::NovaLoom;
use std::collections::HashMap;
use std::time::Instant;

pub struct NovaBenchmark {
    model: NovaLoom,
    results: HashMap<String, f32>,
    detailed_results: HashMap<String, Vec<f32>>,
}

impl NovaBenchmark {
    pub fn new(model: NovaLoom) -> Self {
        Self {
            model,
            results: HashMap::new(),
            detailed_results: HashMap::new(),
        }
    }
    
    pub fn run_full_suite(&mut self) -> &HashMap<String, f32> {
        println!("\n{}", "═".repeat(60));
        println!("🏆 NOVA COMPLETE BENCHMARK SUITE");
        println!("{}", "═".repeat(60));
        
        self.run_language_understanding();
        self.run_reasoning_suite();
        self.run_code_suite();
        self.run_long_context();
        self.run_efficiency_suite();
        self.run_memory_suite();
        
        println!("{}", "═".repeat(60));
        println!("📊 FINAL RESULTS:");
        for (name, score) in &self.results {
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let avg = self.results.values().sum::<f32>() / self.results.len() as f32;
        println!("{}", "─".repeat(40));
        println!("🏆 OVERALL SCORE: {:.1}%", avg * 100.0);
        println!("{}", "═".repeat(60));
        
        &self.results
    }
    
    fn run_language_understanding(&mut self) {
        println!("\n📚 Language Understanding Suite");
        let tasks_list = tasks::get_language_tasks();
        
        for task in tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(name.clone(), score);
            self.detailed_results.insert(name.clone(), vec![score]);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
    }
    
    fn run_reasoning_suite(&mut self) {
        println!("\n🧠 Reasoning Suite");
        let tasks_list = tasks::get_reasoning_tasks();
        
        for task in tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(name.clone(), score);
            self.detailed_results.insert(name.clone(), vec![score]);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
    }
    
    fn run_code_suite(&mut self) {
        println!("\n💻 Code Suite");
        let tasks_list = tasks::get_code_tasks();
        
        for task in tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(name.clone(), score);
            self.detailed_results.insert(name.clone(), vec![score]);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
    }
    
    fn run_long_context(&mut self) {
        println!("\n📖 Long Context Suite");
        let tasks_list = tasks::get_long_context_tasks();
        
        for task in tasks_list {
            let score = task.run(&mut self.model, 5);
            let name = task.name.clone();
            self.results.insert(name.clone(), score);
            self.detailed_results.insert(name.clone(), vec![score]);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
    }
    
    fn run_efficiency_suite(&mut self) {
        println!("\n⚡ Efficiency Suite");
        
        let start = Instant::now();
        for _ in 0..100 {
            self.model.process("short test");
        }
        let speed = 100.0 / start.elapsed().as_secs_f32();
        let speed_score = (speed / 10000.0).min(1.0);
        
        let memory = self.model.memory_usage();
        let memory_val = memory as f32;
        let memory_score = if memory_val > 0.0 { (100.0 / memory_val).min(1.0) } else { 1.0 };
        
        self.results.insert("speed".to_string(), speed_score);
        self.results.insert("memory_efficiency".to_string(), memory_score);
        
        println!("  Speed: {:.0} tok/s ({:.1}%)", speed, speed_score * 100.0);
        println!("  Memory: {}MB ({:.1}%)", memory, memory_score * 100.0);
    }
    
    fn run_memory_suite(&mut self) {
        println!("\n💾 Memory Suite");
        let tasks_list = tasks::get_memory_tasks();
        
        for task in tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(name.clone(), score);
            self.detailed_results.insert(name.clone(), vec![score]);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
    }
    
    pub fn generate_training_data(&mut self) -> Vec<(String, String)> {
        let mut training_data = Vec::new();
        
        for (task_name, scores) in &self.detailed_results {
            let avg_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
            if avg_score < 0.6 {
                println!("🔴 Weak on: {} ({:.1}%)", task_name, avg_score * 100.0);
                let data = data::generate_for_task(task_name, 20);
                training_data.extend(data);
            }
        }
        
        training_data
    }
    
    pub fn auto_improve(&mut self) {
        println!("\n🔄 Auto-Improving Nova...");
        let training_data = self.generate_training_data();
        if !training_data.is_empty() {
            improve::fine_tune(&mut self.model, &training_data, 3);
        }
        improve::optimize_hyperparameters(&mut self.model, &self.results);
        println!("✅ Auto-improvement complete!");
    }
}

pub fn run_full_benchmark(_model: &mut NovaLoom, _samples: usize) -> HashMap<String, f32> {
    // Create a fresh model for benchmarking
    let mut bench = NovaBenchmark::new(NovaLoom::new(64, 5));

    bench.run_full_suite();
    bench.results
}

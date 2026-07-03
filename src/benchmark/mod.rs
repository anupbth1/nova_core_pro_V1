//! Complete Nova Benchmark Suite
//! PRIORITY 8: Integrated with proper evaluators, comparison reports, and auto-improvement

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
        
        // Print comparison report using the new compare module
        compare::print_comparison_report(&self.results);
        
        &self.results
    }
    
    fn run_language_understanding(&mut self) {
        println!("\n📚 Language Understanding Suite");
        let tasks_list = tasks::get_language_tasks();
        
        for task in &tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(format!("language/{}", name), score);
            self.detailed_results.entry(name.clone()).or_insert_with(Vec::new).push(score);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let scores: Vec<f32> = tasks_list.iter().map(|t| {
            *self.results.get(&format!("language/{}", t.name)).unwrap_or(&0.0)
        }).collect();
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;
        self.results.insert("language_average".to_string(), avg);
    }
    
    fn run_reasoning_suite(&mut self) {
        println!("\n🧠 Reasoning Suite");
        let tasks_list = tasks::get_reasoning_tasks();
        
        for task in &tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(format!("reasoning/{}", name), score);
            self.detailed_results.entry(name.clone()).or_insert_with(Vec::new).push(score);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let scores: Vec<f32> = tasks_list.iter().map(|t| {
            *self.results.get(&format!("reasoning/{}", t.name)).unwrap_or(&0.0)
        }).collect();
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;
        self.results.insert("reasoning_average".to_string(), avg);
    }
    
    fn run_code_suite(&mut self) {
        println!("\n💻 Code Suite");
        let tasks_list = tasks::get_code_tasks();
        
        for task in &tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(format!("code/{}", name), score);
            self.detailed_results.entry(name.clone()).or_insert_with(Vec::new).push(score);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let scores: Vec<f32> = tasks_list.iter().map(|t| {
            *self.results.get(&format!("code/{}", t.name)).unwrap_or(&0.0)
        }).collect();
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;
        self.results.insert("code_average".to_string(), avg);
    }
    
    fn run_long_context(&mut self) {
        println!("\n📖 Long Context Suite");
        let tasks_list = tasks::get_long_context_tasks();
        
        for task in &tasks_list {
            let score = task.run(&mut self.model, 5);
            let name = task.name.clone();
            self.results.insert(format!("long_context/{}", name), score);
            self.detailed_results.entry(name.clone()).or_insert_with(Vec::new).push(score);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let scores: Vec<f32> = tasks_list.iter().map(|t| {
            *self.results.get(&format!("long_context/{}", t.name)).unwrap_or(&0.0)
        }).collect();
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;
        self.results.insert("long_context_average".to_string(), avg);
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
        
        self.results.insert("efficiency/speed".to_string(), speed_score);
        self.results.insert("efficiency/memory".to_string(), memory_score);
        
        println!("  Speed: {:.0} tok/s ({:.1}%)", speed, speed_score * 100.0);
        println!("  Memory: {}MB ({:.1}%)", memory, memory_score * 100.0);
    }
    
    fn run_memory_suite(&mut self) {
        println!("\n💾 Memory Suite");
        let tasks_list = tasks::get_memory_tasks();
        
        for task in &tasks_list {
            let score = task.run(&mut self.model, 10);
            let name = task.name.clone();
            self.results.insert(format!("memory/{}", name), score);
            self.detailed_results.entry(name.clone()).or_insert_with(Vec::new).push(score);
            println!("  {}: {:.1}%", name, score * 100.0);
        }
        
        let scores: Vec<f32> = tasks_list.iter().map(|t| {
            *self.results.get(&format!("memory/{}", t.name)).unwrap_or(&0.0)
        }).collect();
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;
        self.results.insert("memory_average".to_string(), avg);
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
        println!("{}", "═".repeat(60));
        
        // Phase 1: Generate targeted training data from weak areas
        let training_data = self.generate_training_data();
        let initial_learned = self.model.learned_count();
        let initial_ngrams = self.model.ngram_count();
        let initial_vocab = self.model.vocab_size();
        
        println!("  Initial state: {} learned, {} ngrams, {} vocab", 
                 initial_learned, initial_ngrams, initial_vocab);
        
        // Phase 2: Run self-improvement cycles with convergence tracking
        if !training_data.is_empty() {
            println!("  Training on {} examples across {} epochs", 
                     training_data.len(), 5);
            
            // Use the model's built-in self-improvement method
            let improvements = self.model.run_self_improvement(&training_data, 5);
            println!("  Made {} improvements from training data", improvements);
        }
        
        // Phase 3: Run diagnostic and fix any issues found
        let issues = self.model.self_diagnostic();
        if !issues.is_empty() {
            println!("  Found {} issues during diagnostic:", issues.len());
            for issue in &issues {
                println!("    ⚠️  {}", issue);
            }
        } else {
            println!("  ✅ Model diagnostic passed - no issues found");
        }
        
        // Phase 4: Optimize hyperparameters based on benchmark results
        improve::optimize_hyperparameters(&mut self.model, &self.results);
        
        // Phase 5: Report improvements
        let final_learned = self.model.learned_count();
        let final_ngrams = self.model.ngram_count();
        let final_vocab = self.model.vocab_size();
        
        println!("\n  📈 Improvement Summary:");
        println!("     Learned: {} → {} (+{})", initial_learned, final_learned, final_learned - initial_learned);
        println!("     N-grams: {} → {} (+{})", initial_ngrams, final_ngrams, final_ngrams - initial_ngrams);
        println!("     Vocab:   {} → {} (+{})", initial_vocab, final_vocab, final_vocab - initial_vocab);
        
        // Phase 6: Run a quick post-improvement benchmark to verify progress
        println!("\n  📊 Post-improvement benchmark:");
        let mut post_scores = Vec::new();
        for (task_name, _) in &self.results {
            if !task_name.contains("average") && !task_name.contains("efficiency") {
                // Re-run a few tasks to check improvement
                let tasks_list = tasks::get_language_tasks();
                for task in &tasks_list {
                    if task.name == *task_name {
                        let score = task.run(&mut self.model, 5);
                        post_scores.push((task_name.clone(), score));
                        break;
                    }
                }
            }
        }
        
        // Show top improvements
        if !post_scores.is_empty() {
            post_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("     Best scores after improvement:");
            for (name, score) in post_scores.iter().take(3) {
                println!("       {}: {:.1}%", name, score * 100.0);
            }
        }
        
        println!("{}", "═".repeat(60));
        println!("✅ Auto-improvement complete!");
    }
    
    /// Run multiple auto-improvement cycles until convergence.
    /// Returns the number of cycles performed.
    pub fn auto_improve_until_convergence(&mut self, max_cycles: usize) -> usize {
        let mut cycles = 0;
        let mut prev_learned = self.model.learned_count();
        
        for cycle in 0..max_cycles {
            println!("\n🔄 Auto-Improvement Cycle {}/{}", cycle + 1, max_cycles);
            
            // Run benchmark to get current scores
            self.run_full_suite();
            
            // Run auto-improvement
            self.auto_improve();
            
            cycles += 1;
            
            // Check for convergence: if no new learned patterns, we've converged
            let current_learned = self.model.learned_count();
            if current_learned == prev_learned {
                println!("  ✅ Converged after {} cycles (no new patterns learned)", cycles);
                break;
            }
            prev_learned = current_learned;
        }
        
        cycles
    }
}

pub fn run_full_benchmark(model: &mut NovaLoom, samples: usize) -> HashMap<String, f32> {
    let mut bench = NovaBenchmark::new(NovaLoom::new(64, 5));
    
    // Run the full suite
    bench.run_full_suite();
    
    // Run self-comparison report
    let comparison = compare::run_self_comparison(model, samples);
    compare::print_comparison_report(&comparison);
    
    bench.results
}

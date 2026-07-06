//! Nova Profiler - Detailed timing measurements for all components
//!
//! Provides high-resolution timing for:
//! - dataset loading
//! - preprocessing
//! - pulse creation
//! - embeddings
//! - field propagation
//! - SSM layers
//! - forward pass
//! - loss computation
//! - optimizer step
//! - vocabulary search
//! - checkpoint writing
//! - chat inference

use std::collections::HashMap;
use std::time::Instant;

/// A simple high-resolution profiler
pub struct Profiler {
    /// Currently running timers
    pub running: HashMap<String, Instant>,
    /// Accumulated timings (key -> total seconds)
    pub timings: HashMap<String, f64>,
    /// Call counts
    pub counts: HashMap<String, usize>,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            running: HashMap::new(),
            timings: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Start timing a named section
    pub fn start(&mut self, name: &str) {
        self.running.insert(name.to_string(), Instant::now());
    }

    /// Stop timing a named section and accumulate
    pub fn stop(&mut self, name: &str) {
        if let Some(start) = self.running.remove(name) {
            let elapsed = start.elapsed().as_secs_f64();
            *self.timings.entry(name.to_string()).or_insert(0.0) += elapsed;
            *self.counts.entry(name.to_string()).or_insert(0) += 1;
        }
    }

    /// Print timing summary
    pub fn print_summary(&self) {
        if self.timings.is_empty() {
            println!("  (no timing data)");
            return;
        }

        let total: f64 = self.timings.values().sum();
        
        // Sort by time descending
        let mut sorted: Vec<(&String, &f64)> = self.timings.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n  📊 Timing Profile (total: {:.4}s):", total);
        for (name, time) in &sorted {
            let pct = if total > 0.0 { (*time / total) * 100.0 } else { 0.0 };
            let count = self.counts.get(*name).copied().unwrap_or(1);
            let avg = if count > 0 { *time / count as f64 } else { 0.0 };
            let bar_len = (pct / 3.0) as usize;
            let bar = "█".repeat(bar_len.min(30));
            println!("    {:25}: {:7.4}s ({:5.1}%) x{} avg={:.4}s {}", 
                name, time, pct, count, avg, bar);
        }
    }

    /// Reset all timings
    pub fn reset(&mut self) {
        self.running.clear();
        self.timings.clear();
        self.counts.clear();
    }

    /// Get total time
    pub fn total(&self) -> f64 {
        self.timings.values().sum()
    }
}
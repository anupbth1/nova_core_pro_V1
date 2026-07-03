//! Auto-improvement based on benchmark results
//! PRIORITY 9: Enhanced fine-tuning with version compatibility and convergence tracking

use crate::loom::NovaLoom;
use std::collections::HashMap;

/// Fine-tune the model on weak areas identified by benchmarks.
/// Uses targeted training data to improve specific capabilities.
/// PRIORITY 9: Now delegates to NovaLoom::run_self_improvement() for consistency.
pub fn fine_tune(model: &mut NovaLoom, training_data: &[(String, String)], epochs: usize) {
    println!("  Fine-tuning on {} examples for {} epochs", training_data.len(), epochs);
    let improvements = model.run_self_improvement(training_data, epochs);
    println!("  Fine-tuning complete: {} improvements made", improvements);
}

/// Check model version compatibility.
/// Returns a compatibility report string.
pub fn check_version_compatibility(model: &NovaLoom, required_dim: usize, required_cores: usize) -> String {
    let compatible = model.is_compatible_with(required_dim, required_cores);
    let version = model.model_version();
    
    format!(
        "Version Compatibility Check:\n\
         Model: {}\n\
         Required: dim={}, cores={}\n\
         Compatible: {}\n\
         {}",
        version,
        required_dim,
        required_cores,
        if compatible { "✅ YES" } else { "❌ NO" },
        if compatible {
            "The model can be used with the specified configuration."
        } else {
            "WARNING: The model may not work correctly with the specified configuration.\n\
             Consider reinitializing with the correct parameters."
        }
    )
}

/// Export model patterns to a portable format.
/// Returns a HashMap of input -> output learned associations.
pub fn export_model_patterns(model: &NovaLoom) -> HashMap<String, String> {
    model.export_learned_patterns()
}

/// Import model patterns from a previously exported snapshot.
pub fn import_model_patterns(model: &mut NovaLoom, patterns: HashMap<String, String>) {
    let count = patterns.len();
    model.import_learned_patterns(patterns);
    println!("  Imported {} learned patterns", count);
}

/// Run a full model diagnostic and return a health report.
pub fn run_diagnostic(model: &NovaLoom) -> String {
    let issues = model.self_diagnostic();
    let version = model.model_version();
    
    let mut report = format!(
        "=== Model Diagnostic Report ===\n\
         Version: {}\n\
         Health: {}\n\n",
        version,
        if issues.is_empty() { "✅ HEALTHY" } else { "⚠️  ISSUES FOUND" }
    );
    
    if !issues.is_empty() {
        report.push_str("Issues:\n");
        for (i, issue) in issues.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, issue));
        }
    } else {
        report.push_str("All systems operational.\n");
    }
    
    report.push_str(&format!(
        "\nModel Stats:\n\
         - Cores: {}\n\
         - Dimension: {}\n\
         - Max iterations: {}\n\
         - Convergence threshold: {:.3}\n\
         - Content convergence threshold: {:.3}\n\
         - Field diffusion: {:.3}\n\
         - Learned responses: {}\n\
         - N-gram patterns: {}\n\
         - Vocabulary size: {}\n\
         - Memory usage: {}MB\n",
        model.cores.len(),
        model.dim,
        model.max_iterations,
        model.convergence_threshold,
        model.content_convergence_threshold,
        model.get_field_diffusion(),
        model.learned_count(),
        model.ngram_count(),
        model.vocab_size(),
        model.memory_usage(),
    ));
    
    report
}

/// Optimize hyperparameters based on benchmark results.
/// Adjusts model parameters to improve performance on weak areas.
pub fn optimize_hyperparameters(model: &mut NovaLoom, results: &HashMap<String, f32>) {
    let avg_score = results.values().sum::<f32>() / results.len() as f32;
    
    if avg_score < 0.3 {
        println!("  Low score ({:.1}%) - significant adjustments needed", avg_score * 100.0);
        // Increase adaptive depth range for more processing
        model.set_adaptive_depth(3, 8);
        // Increase field diffusion for better information spread
        model.set_field_diffusion(0.15);
        // Strengthen core gate responses
        model.set_core_gate_strength(0.9);
        // Increase convergence threshold for more thorough processing
        model.set_convergence_threshold(0.85);
    } else if avg_score < 0.6 {
        println!("  Medium score ({:.1}%) - fine-tuning parameters", avg_score * 100.0);
        // Moderate adjustments
        model.set_adaptive_depth(2, 6);
        model.set_field_diffusion(0.1);
        model.set_core_gate_strength(0.8);
        model.set_convergence_threshold(0.75);
    } else if avg_score < 0.8 {
        println!("  Good score ({:.1}%) - minor refinements", avg_score * 100.0);
        // Small adjustments
        model.set_adaptive_depth(2, 5);
        model.set_field_diffusion(0.08);
        model.set_core_gate_strength(0.7);
        model.set_convergence_threshold(0.7);
    } else {
        println!("  Excellent score ({:.1}%) - no adjustments needed", avg_score * 100.0);
    }
    
    // Task-specific adjustments based on weak areas
    for (task, score) in results {
        if *score < 0.4 {
            let task_type = task.split('/').next().unwrap_or("");
            match task_type {
                "language" => {
                    println!("    → Boosting language processing");
                    model.set_field_diffusion(model.get_field_diffusion() * 1.2);
                },
                "reasoning" => {
                    println!("    → Boosting reasoning depth");
                    model.set_adaptive_depth(3, model.get_max_depth() + 2);
                },
                "code" => {
                    println!("    → Boosting code analysis");
                    model.set_core_gate_strength(model.get_core_gate_strength() * 1.1);
                },
                "memory" => {
                    println!("    → Boosting memory retention");
                    model.set_convergence_threshold(model.get_convergence_threshold() * 0.95);
                },
                _ => {}
            }
        }
    }
}

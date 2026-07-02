//! Nova Core - Main CLI Entry Point
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod pulse;
mod field;
mod core;
mod ssm;
mod loom;
mod benchmark;
mod trainer;
mod dataset;
mod model;


use clap::{Parser, Subcommand};

use colored::*;
use std::time::Instant;
use loom::NovaLoom;

use crate::benchmark::NovaBenchmark;
use crate::trainer::{NovaTrainer, init_global_thread_pool};
use crate::dataset::{NovaDataset, DatasetSource, HFDatasetRef, FilterCondition, ColumnMapping, DatasetFormat};
use crate::model::NovaModelManager;


#[derive(Parser)]
#[command(name = "nova")]
#[command(about = "🚀 Nova AI - Post-Transformer LLM", long_about = None)]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Global field dimension size (Terminal se dynamic set karne ke liye)
    #[arg(long, default_value_t = 64, global = true)]
    dim: usize,

    /// Global number of active cores
    #[arg(long, default_value_t = 5, global = true)]
    cores: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Process text input
    Run {
        #[arg(short, long)]
        input: String,
        /// Model name to load (optional, uses current model if not specified)
        #[arg(short = 'm', long)]
        model: Option<String>,
    },

    /// Run benchmark suite
    Bench {
        #[arg(short, long, default_value = "10")]
        samples: usize,
    },
    /// Interactive chat mode
    Chat,
    /// Show architecture information
    Info,
    /// Run speed test
    Speed {
        #[arg(short, long, default_value = "1000")]
        pulses: usize,
    },
    /// Run full benchmark suite
    FullBench,

    /// Auto-improve based on benchmarks
    Improve,

    /// Generate training data
    GenData,

    /// Train Nova Core on data
    Train {
        /// Number of training examples to generate
        #[arg(short = 'x', long, default_value = "100")]
        examples: usize,
        /// Number of training epochs
        #[arg(short = 'e', long, default_value = "10")]
        epochs: usize,
    },


    /// Interactive chat with trained model
    SmartChat {
        /// Model name to load at startup (optional)
        #[arg(short = 'm', long)]
        model: Option<String>,
    },


    // ===== NEW: Dataset Management Commands =====

    /// Manage datasets (load, filter, save)
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    // ===== NEW: Model Management Commands =====

    /// Manage models (list, save, load, delete, upload, download)
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    // ===== NEW: Train using Hugging Face datasets =====

    /// Train using datasets from Hugging Face (single-pass pro training)
    HfTrain {
        /// Hugging Face dataset name (e.g., "imdb", "wikitext", "tiny_shakespeare")
        #[arg(short, long)]
        dataset: String,

        /// Dataset subset/config (e.g., "wikitext-2-raw-v1")
        #[arg(long, default_value = "")]
        subset: String,


        /// Dataset split (train, test, validation)
        #[arg(short, long, default_value = "train")]
        split: String,

        /// Input column name (auto-detected if not specified)
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,

        /// Target column name (auto-detected if not specified)
        #[arg(short = 't', long, default_value = "")]
        target_col: String,

        /// Extra input columns to concatenate (comma-separated)
        #[arg(long, default_value = "")]
        extra_cols: String,

        /// Prompt template: e.g. "User: {user}\nAssistant: {assistant}"
        #[arg(long, default_value = "")]
        template: String,

        /// Max rows to download
        #[arg(short = 'm', long, default_value = "500")]
        max_rows: usize,

        /// Enable pro mode: adaptive iterations, pattern caching, smart learning
        #[arg(long, default_value = "false")]
        pro: bool,

        /// ULTRA-FAST mode: skip core iterations, direct pattern learning (100x faster)
        #[arg(long, default_value = "false")]
        ultra: bool,

        /// Model name to save after training
        #[arg(short = 'n', long, default_value = "hf-trained-model")]
        model_name: String,
    },

    // ===== NEW: Multi-Dataset Training =====

    /// Train ONE model on MULTIPLE Hugging Face datasets sequentially
    MultiHfTrain {

        /// Hugging Face dataset names (comma-separated, e.g. "imdb,wikitext,tiny_shakespeare")
        #[arg(short, long)]
        datasets: String,

        /// Dataset split for all datasets (train, test, validation)
        #[arg(short, long, default_value = "train")]
        split: String,

        /// Input column name (auto-detected if not specified, same for all datasets)
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,

        /// Target column name (auto-detected if not specified, same for all datasets)
        #[arg(short = 't', long, default_value = "")]
        target_col: String,

        /// Prompt template: e.g. "User: {user}\nAssistant: {assistant}"
        #[arg(long, default_value = "")]
        template: String,

        /// Max rows per dataset
        #[arg(short = 'm', long, default_value = "300")]
        max_rows: usize,

        /// Enable pro mode: adaptive iterations, pattern caching, smart learning
        #[arg(long, default_value = "false")]
        pro: bool,

        /// ULTRA-FAST mode: skip core iterations, direct pattern learning (100x faster)
        #[arg(long, default_value = "false")]
        ultra: bool,

        /// Model name to save after training
        #[arg(short = 'n', long, default_value = "multi-hf-trained-model")]
        model_name: String,
    },

}

// ===== NEW: Dataset Subcommands =====

#[derive(Subcommand)]
enum DatasetAction {
    /// Load a local dataset file (CSV, JSON, JSONL, TXT)
    Load {
        /// Path to dataset file
        #[arg(short, long)]
        file: String,

        /// Input column name (auto-detected if empty)
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,

        /// Target column name (auto-detected if empty)
        #[arg(short = 't', long, default_value = "")]
        target_col: String,

        /// Dataset format (auto-detect by default)
        #[arg(short = 'f', long)]
        format: Option<String>,

        /// Max rows to load (0 = all)
        #[arg(short = 'm', long, default_value = "0")]
        max_rows: usize,

        /// Input prefix
        #[arg(long, default_value = "")]
        prefix: String,

        /// Input suffix
        #[arg(long, default_value = "")]
        suffix: String,

        /// Extra input columns to concatenate (comma-separated)
        #[arg(long, default_value = "")]
        extra_cols: String,

        /// Prompt template: e.g. "User: {user}\nAssistant: {assistant}"
        #[arg(long, default_value = "")]
        template: String,
    },

    /// Load a dataset from Hugging Face
    Hf {
        /// Hugging Face dataset name
        #[arg(short, long)]
        name: String,

        /// Dataset subset/config
        #[arg(short = 's', long, default_value = "")]
        subset: String,

        /// Dataset split
        #[arg(short = 'p', long, default_value = "train")]
        split: String,

        /// Input column name (auto-detected if empty)
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,

        /// Target column name (auto-detected if empty)
        #[arg(short = 't', long, default_value = "")]
        target_col: String,

        /// Max rows to download
        #[arg(short = 'm', long, default_value = "1000")]
        max_rows: usize,

        /// Extra input columns to concatenate (comma-separated)
        #[arg(long, default_value = "")]
        extra_cols: String,

        /// Prompt template: e.g. "User: {user}\nAssistant: {assistant}"
        #[arg(long, default_value = "")]
        template: String,
    },

    /// Show dataset statistics
    Stats,

    /// Save dataset to JSONL file
    Save {
        /// Output file path
        #[arg(short, long)]
        output: String,
    },

    /// Clear all loaded datasets
    Clear,
}

// ===== NEW: Model Subcommands =====

#[derive(Subcommand)]
enum ModelAction {
    /// List all available models
    List,

    /// Save current model
    Save {
        /// Model name
        #[arg(short, long)]
        name: String,
    },

    /// Load a model
    Load {
        /// Model name to load
        #[arg(short, long)]
        name: String,
    },

    /// Delete a model
    Delete {
        /// Model name to delete
        #[arg(short, long)]
        name: String,
    },

    /// Upload model to Hugging Face Hub
    Upload {
        /// Model name to upload
        #[arg(short, long)]
        name: String,

        /// Hugging Face repo (e.g., "username/repo-name")
        #[arg(short, long)]
        repo: String,

        /// Hugging Face token (optional, uses HF_TOKEN env var)
        #[arg(short = 'k', long)]
        token: Option<String>,
    },

    /// Download model from Hugging Face Hub
    Download {
        /// Hugging Face repo (e.g., "username/repo-name")
        #[arg(short, long)]
        repo: String,

        /// Model name to save as
        #[arg(short, long)]
        name: String,

        /// Hugging Face token (optional, uses HF_TOKEN env var)
        #[arg(short = 'k', long)]
        token: Option<String>,
    },
}


fn print_header(nova: &NovaLoom) {
    println!("\n{}", "═".repeat(60));
    println!("{}", "🚀 NOVA CORE - Zero Transformer AI".bright_green().bold());
    println!("{}", "═".repeat(60));
    println!("{} {}", "📌 CURRENT MODEL:".bright_yellow(), nova.model_info().cyan());
    println!("{}", "═".repeat(60));
    println!("   • No Attention (O(n) field dynamics)");
    println!("   • No Tokens (continuous pulses)");
    println!("   • No Fixed Layers (adaptive depth cores)");
    println!("{}", "═".repeat(60));
}

/// Trim generated text to a clean sentence boundary.
/// Stops at the first sentence-ending punctuation (. ! ?) or at `max_words`,
/// whichever comes first. This prevents runaway n-gram generations.
fn trim_to_sentence(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result_words: Vec<&str> = Vec::new();

    for (i, word) in words.iter().enumerate() {
        result_words.push(word);

        // Stop at sentence-ending punctuation
        if word.ends_with('.') || word.ends_with('!') || word.ends_with('?')
            || word.ends_with("...") {
            break;
        }

        // Hard cap at max_words
        if i + 1 >= max_words {
            break;
        }
    }

    result_words.join(" ")
}

fn main() {
    // Initialize global thread pool with auto-detected optimal thread count
    init_global_thread_pool();
    
    let cli = Cli::parse();
    let mut nova = NovaLoom::new(cli.dim, cli.cores);
    
    match cli.command {
        Commands::Run { input, model } => {
            // Load model if specified
            if let Some(model_name) = &model {
                let model_mgr = NovaModelManager::new();
                match model_mgr.load_model(model_name) {
                    Ok((loaded_loom, vocabulary)) => {
                        let _ = std::mem::replace(&mut nova, loaded_loom);
                        nova.name = model_name.clone();
                        nova.vocabulary = vocabulary;
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to load model '{}': {}", model_name, e);
                        return;
                    }
                }
            }
            
            print_header(&nova);
            
            println!("\n📝 Input: {}", input.cyan());
            let start = Instant::now();
            
            // First try process() which checks learned_responses by hash
            let output = nova.process(&input);
            
            let duration = start.elapsed();
            println!("✨ Output: {}", output.green());
            println!("⏱️  Time: {:?}", duration);
            println!("\n📊 Stats: {}", nova.stats());
        }
        
        Commands::Bench { samples } => {
            println!("\n📊 Running NovaCore Benchmark ({} samples)...", samples);
            let start = Instant::now();
            let results = benchmark::run_full_benchmark(&mut nova, samples);
            let duration = start.elapsed();
            
            println!("\n{}", "═".repeat(60));
            println!("{}", "🏆 BENCHMARK RESULTS".yellow().bold());
            println!("{}", "═".repeat(60));
            
            for (name, score) in &results {
                let bar_len = (score * 30.0) as usize;
                let bar = "█".repeat(bar_len);
                let spaces = " ".repeat(30 - bar_len);
                println!("  {:<20} {}{} {:.1}%", 
                    name, bar, spaces, score * 100.0);
            }
            
            let avg = results.values().sum::<f32>() / results.len() as f32;
            println!("{}", "─".repeat(60));
            println!("  {:<20} {:.1}%", "AVERAGE", avg * 100.0);
            println!("{}", "═".repeat(60));
            println!("⏱️  Time: {:?}", duration);
        }
        
        Commands::Chat => {
            println!("\n💬 Interactive Chat Mode");
            println!("{}", "─".repeat(40));
            println!("Type 'exit' or 'quit' to stop");
            println!("Type 'stats' to see performance");
            println!("{}\n", "─".repeat(40));
            
            loop {
                print!("{} ", ">>>".bright_blue());
                use std::io::Write;
                std::io::stdout().flush().unwrap();
                
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();
                
                if input == "exit" || input == "quit" {
                    println!("👋 Goodbye!");
                    break;
                }
                
                if input == "stats" {
                    println!("📊 {}", nova.stats());
                    continue;
                }
                
                if input.is_empty() {
                    continue;
                }
                
                let start = Instant::now();
                let output = nova.process(input);
                let duration = start.elapsed();
                
                println!("{} {}", "nova:".bright_green(), output);
                println!("{} {:?}\n", "⏱️".dimmed(), duration);
            }
        }

        Commands::FullBench => {
            let mut bench = NovaBenchmark::new(NovaLoom::new(cli.dim, cli.cores));
            bench.run_full_suite();
        }
        Commands::Improve => {
            let mut bench = NovaBenchmark::new(NovaLoom::new(cli.dim, cli.cores));
            bench.run_full_suite();
            bench.auto_improve();
            println!("✅ Nova improved! Run benchmark again to see gains.");
        }
        Commands::GenData => {
            let mut bench = NovaBenchmark::new(NovaLoom::new(cli.dim, cli.cores));
            bench.run_full_suite();
            let data = bench.generate_training_data();
            println!("Generated {} training examples", data.len());
            // Save to file
        }

        
        Commands::Info => {
            println!("\n🔧 {} Architecture Info", "NOVA CORE".bright_green());
            println!("{}", "─".repeat(40));
            println!("  Field dimension:     128");
            println!("  Cores:               5 (syntax, semantic, memory, reasoning, pattern)");
            println!("  Adaptive depth:      1-15 iterations");
            println!("  Complexity:          O(n) per iteration");
            println!("  Memory usage:        ~50-100MB");
            println!("  File format:         .nova (binary, no pickle)");
            println!("{}", "─".repeat(40));
            println!("\n  Compared to Transformer:");
            println!("    Attention O(n²)  →  Field O(n)");
            println!("    Tokens (discrete) → Pulses (continuous)");
            println!("    Fixed depth      → Adaptive depth");
            println!("    Layers (linear)  → Cores (graph)");
        }
        
        Commands::Speed { pulses } => {
            println!("\n⚡ Speed Test: Processing {} pulses...", pulses);
            let long_text = "nova ".repeat(pulses);
            let start = Instant::now();
            let _ = nova.process(&long_text);
            let duration = start.elapsed();
            
            let rate = pulses as f64 / duration.as_secs_f64();
            println!("  ✅ Processed {} pulses in {:?}", pulses, duration);
            println!("  📈 Rate: {:.0} pulses/second", rate);
            
            if rate > 1000.0 {
                println!("  🚀 Excellent performance!");
            } else if rate > 100.0 {
                println!("  👍 Good performance");
            } else {
                println!("  💡 Tip: Run 'cargo build --release' for faster speed");
            }
        }

        Commands::Train { examples, epochs } => {
            print_header(&nova);
            println!("\n🎓 Training Nova Core LLM");
            println!("{}", "═".repeat(60));
            
            // Generate training data
            println!("📦 Generating {} training examples...", examples);
            let training_data = crate::trainer::NovaTrainer::generate_training_data(examples);
            
            // Create trainer and train (updates current model)
            let mut trainer = NovaTrainer::new();
            trainer.train(&mut nova, &training_data, epochs);
            
            // Update model name
            nova.name = "trained-model".to_string();
            
            // Auto-save the trained model
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, "trained-model") {
                Ok(path) => println!("✅ Trained model saved to: {}", path),
                Err(e) => eprintln!("⚠️  Failed to save model: {}", e),
            }
            
            // Run benchmark after training
            println!("\n📊 Running post-training benchmark...");
            let mut bench = NovaBenchmark::new(NovaLoom::new(cli.dim, cli.cores));
            bench.run_full_suite();

            
            println!("\n💡 Try 'nova run --input \"what color is the sky\"' to test the trained model!");
        }

        Commands::SmartChat { model } => {
            // Load model at startup if specified
            if let Some(model_name) = &model {
                let model_mgr = NovaModelManager::new();
                match model_mgr.load_model(model_name) {
                    Ok((loaded_loom, vocabulary)) => {
                        let _ = std::mem::replace(&mut nova, loaded_loom);
                        nova.name = model_name.clone();
                        nova.vocabulary = vocabulary.clone();
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to load model '{}': {}", model_name, e);
                        return;
                    }
                }
            }
            
            print_header(&nova);
            println!("\n🧠 Nova Smart Chat (Trained Mode)");
            println!("{}", "─".repeat(40));
            println!("Type 'exit' or 'quit' to stop");
            println!("Type 'train <N>' to train on N examples");
            println!("Type 'stats' to see performance");
            println!("Type 'load <name>' to load a trained model");
            println!("{}\n", "─".repeat(40));
            
            let mut trainer = NovaTrainer::new();
            
            loop {
                print!("{} ", "🧠".bright_cyan());
                use std::io::Write;
                std::io::stdout().flush().unwrap();
                
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();
                
                if input == "exit" || input == "quit" {
                    println!("👋 Goodbye!");
                    break;
                }
                
                if input == "stats" {
                    println!("📊 {}", nova.stats());
                    if !trainer.history.is_empty() {
                        let last = trainer.history.last().unwrap();
                        println!("📈 Last training: epoch={}, loss={:.4}, acc={:.1}%", 
                            last.epoch, last.loss, last.accuracy * 100.0);
                    }
                    println!("📚 Vocabulary: {} words", trainer.vocab_forward.len());
                    continue;
                }
                
                if input.starts_with("train ") {
                    if let Ok(n) = input[6..].trim().parse::<usize>() {
                        let data = NovaTrainer::generate_training_data(n);
                        trainer.train(&mut nova, &data, 5);
                        nova.name = "trained-model".to_string();
                        println!("✅ Training complete!");
                        print_header(&nova);
                    } else {
                        println!("❌ Usage: train <number_of_examples>");
                    }
                    continue;
                }
                
                if input.starts_with("load ") {
                    let name = input[5..].trim();
                    let model_mgr = NovaModelManager::new();
                    match model_mgr.load_model(name) {
                        Ok((loaded_loom, vocabulary)) => {
                            let _ = std::mem::replace(&mut nova, loaded_loom);
                            nova.name = name.to_string();
                            nova.vocabulary = vocabulary.clone();
                            trainer.vocab_forward = vocabulary;
                            trainer.vocab_initialized = true;
                            println!("✅ Model '{}' loaded! Vocab: {} words", name, trainer.vocab_forward.len());
                            print_header(&nova);
                        }
                        Err(e) => eprintln!("❌ Failed to load model: {}", e),
                    }
                    continue;
                }
                
                if input.is_empty() {
                    continue;
                }
                
                let start = Instant::now();
                
                // Use the model's process method which handles learned_responses, n-gram patterns, and fallback
                let raw_output = nova.process(input);

                // Post-process: trim to a clean sentence boundary (max 20 words).
                // Prevents runaway generations from flooding the terminal.
                let output = trim_to_sentence(&raw_output, 20);
                
                let duration = start.elapsed();
                
                println!("{} {}", "nova:".bright_green(), output);
                println!("{} {:?}\n", "⏱️".dimmed(), duration);


            }
        }


        // ===== NEW: Dataset Commands =====

        Commands::Dataset { action } => {
            let mut dataset = NovaDataset::new();

            match action {
                DatasetAction::Load { file, input_col, target_col, format: _format, max_rows, prefix, suffix, extra_cols, template } => {
                    let source = dataset.add_file(&file);
                    
                    // If template is provided, use it
                    if !template.is_empty() {
                        source.column_mapping.prompt_template = Some(template.clone());
                    }
                    
                    // Set columns if provided (otherwise auto-detect will kick in)
                    if !input_col.is_empty() {
                        source.column_mapping.input_column = input_col;
                    }
                    if !target_col.is_empty() {
                        source.column_mapping.target_column = target_col;
                    }
                    
                    source.column_mapping.input_prefix = prefix;
                    source.column_mapping.input_suffix = suffix;
                    
                    // Parse extra columns (comma-separated)
                    if !extra_cols.is_empty() {
                        let cols: Vec<String> = extra_cols.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        if !cols.is_empty() {
                            source.column_mapping.extra_input_columns = cols;
                        }
                    }
                    
                    if max_rows > 0 { source.max_rows = max_rows; }

                    println!("\n📂 Loading dataset from: {}", file);
                    dataset.load_all();
                    dataset.print_stats();
                }

                DatasetAction::Hf { name, subset, split, input_col, target_col, max_rows, extra_cols, template } => {
                    let mut hf = HFDatasetRef::new(&name)
                        .with_split(&split)
                        .with_max_rows(max_rows);
                    
                    // If template is provided, use it
                    if !template.is_empty() {
                        hf.column_mapping.prompt_template = Some(template.clone());
                    }
                    
                    // Set columns if provided (otherwise auto-detect will kick in)
                    if !input_col.is_empty() {
                        hf.column_mapping.input_column = input_col;
                    }
                    if !target_col.is_empty() {
                        hf.column_mapping.target_column = target_col;
                    }
                    
                    // Parse extra columns (comma-separated)
                    if !extra_cols.is_empty() {
                        let cols: Vec<String> = extra_cols.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        if !cols.is_empty() {
                            hf.column_mapping.extra_input_columns = cols;
                        }
                    }
                    
                    if !subset.is_empty() {
                        hf = hf.with_subset(&subset);
                    }
                    dataset.add_hf_dataset(hf);
                    dataset.load_all();
                    dataset.print_stats();
                }

                DatasetAction::Stats => {
                    dataset.print_stats();
                }

                DatasetAction::Save { output } => {
                    match dataset.save_to_jsonl(&output) {
                        Ok(()) => println!("✅ Dataset saved to {}", output),
                        Err(e) => eprintln!("❌ Failed to save: {}", e),
                    }
                }

                DatasetAction::Clear => {
                    println!("🗑️ Cleared all loaded datasets");
                }
            }
        }

        // ===== NEW: Model Commands =====

        Commands::Model { action } => {
            let model_mgr = NovaModelManager::new();

            match action {
                ModelAction::List => {
                    model_mgr.list_models();
                }

                ModelAction::Save { name } => {
                    println!("📌 Saving current model: {}", nova.model_info().cyan());
                    match model_mgr.save_model(&nova, &name) {
                        Ok(path) => println!("✅ Model saved to: {}", path),
                        Err(e) => eprintln!("❌ Failed to save model: {}", e),
                    }
                }

                ModelAction::Load { name } => {
                    match model_mgr.load_model(&name) {
                        Ok((loaded_loom, vocabulary)) => {
                            let _ = std::mem::replace(&mut nova, loaded_loom);
                            nova.name = name.clone();
                            nova.vocabulary = vocabulary.clone();
                            println!("✅ Model '{}' loaded successfully!", name);
                            if !vocabulary.is_empty() {
                                println!("   📚 Vocabulary: {} words", vocabulary.len());
                            }
                            print_header(&nova);
                        }
                        Err(e) => eprintln!("❌ Failed to load model: {}", e),
                    }
                }


                ModelAction::Delete { name } => {
                    match model_mgr.delete_model(&name) {
                        Ok(()) => println!("✅ Model '{}' deleted", name),
                        Err(e) => eprintln!("❌ Failed to delete model: {}", e),
                    }
                }

                ModelAction::Upload { name, repo, token } => {
                    let hf_token = token.unwrap_or_else(|| {
                        std::env::var("HF_TOKEN").unwrap_or_default()
                    });
                    if hf_token.is_empty() {
                        eprintln!("❌ No Hugging Face token provided. Use --token or set HF_TOKEN env var.");
                        return;
                    }
                    match model_mgr.upload_to_hf(&name, &repo, &hf_token) {
                        Ok(()) => println!("✅ Model '{}' uploaded to {}", name, repo),
                        Err(e) => eprintln!("❌ Upload failed: {}", e),
                    }
                }

                ModelAction::Download { repo, name, token } => {
                    let hf_token = token.unwrap_or_else(|| {
                        std::env::var("HF_TOKEN").unwrap_or_default()
                    });
                    match model_mgr.download_from_hf(&repo, &name, &hf_token) {
                        Ok(path) => println!("✅ Model downloaded to: {}", path),
                        Err(e) => eprintln!("❌ Download failed: {}", e),
                    }
                }
            }
        }

        // ===== NEW: HF Train Command =====

        Commands::HfTrain { dataset: ds_name, subset, split, input_col, target_col, extra_cols, template, max_rows, pro, ultra, model_name } => {

            print_header(&nova);
            println!("{}", "═".repeat(60));
            if pro {
                println!("{}", "🔥 PRO MODE: Single-Pass Adaptive Training".bright_green().bold());
            } else {
                println!("{}", "🎓 Training Nova Core with Hugging Face Dataset".bright_green());
            }
            println!("{}", "═".repeat(60));

            // Step 1: Download dataset from Hugging Face
            println!("📦 Step 1: Downloading dataset '{}'...", ds_name);
            let mut dataset = NovaDataset::new();
            
            // Build column mapping with new features
            let mut hf = HFDatasetRef::new(&ds_name)
                .with_split(&split)
                .with_max_rows(max_rows);
            
            // If template is provided, use it (overrides input_col/target_col)
            if !template.is_empty() {
                hf.column_mapping.prompt_template = Some(template.clone());
                println!("    📝 Using prompt template");
            }
            
            // If input_col/target_col are provided, use them
            // Otherwise auto-detect will kick in
            if !input_col.is_empty() {
                hf.column_mapping.input_column = input_col.clone();
            }
            if !target_col.is_empty() {
                hf.column_mapping.target_column = target_col.clone();
            }
            
            // Parse extra columns (comma-separated)
            if !extra_cols.is_empty() {
                let cols: Vec<String> = extra_cols.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !cols.is_empty() {
                    hf.column_mapping.extra_input_columns = cols;
                    println!("    📚 Extra input columns: {}", extra_cols);
                }
            }
            
            if !subset.is_empty() {
                hf = hf.with_subset(&subset);
            }
            dataset.add_hf_dataset(hf);
            dataset.load_all();

            if dataset.examples.is_empty() {
                eprintln!("❌ No training examples loaded. Aborting.");
                return;
            }

            // Step 2: Split into train/validation
            println!("\n📊 Step 2: Splitting dataset...");
            let (train_data, val_data) = dataset.train_val_split(0.1);
            println!("   Train: {} examples", train_data.len());
            println!("   Validation: {} examples", val_data.len());

            // Step 3: Train the model (single pass - no epochs!)
            println!("\n🎯 Step 3: Training model (single pass)...");
            let mut trainer = NovaTrainer::new();
            
            if ultra {
                trainer.train_one_pass_ultra(&mut nova, &train_data);
            } else {
                trainer.train_one_pass(&mut nova, &train_data);
            }
            nova.name = model_name.clone();

            // Step 4: Evaluate on validation set
            println!("\n📈 Step 4: Evaluating on validation set...");
            if !val_data.is_empty() {
                let val_loss = trainer.compute_loss(
                    &nova.text_to_pulses(&val_data[0].input),
                    &val_data[0].target
                );
                println!("   Validation loss: {:.4}", val_loss);
            }

            // Step 5: Save the model
            println!("\n💾 Step 5: Saving model...");
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, &model_name) {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("❌ Failed to save model: {}", e),
            }

            println!("\n{}", "═".repeat(60));
            println!("✅ Training complete! Model '{}' is ready.", model_name);
            println!("💡 Try: nova model list");
            println!("💡 Try: nova model load --name {}", model_name);
        }

        // ===== NEW: Multi-HF Train Command =====

        Commands::MultiHfTrain { datasets, split, input_col, target_col, template, max_rows, pro, ultra, model_name } => {
            print_header(&nova);
            println!("{}", "═".repeat(60));
            if pro {
                println!("{}", "🔥 PRO MODE: Multi-Dataset Adaptive Training".bright_green().bold());
            } else {
                println!("{}", "🎓 Training ONE model on MULTIPLE datasets".bright_green());
            }
            println!("{}", "═".repeat(60));

            // Parse dataset names (comma-separated)
            let dataset_names: Vec<&str> = datasets.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if dataset_names.is_empty() {
                eprintln!("❌ No datasets specified. Use --datasets with comma-separated names.");
                return;
            }

            println!("📋 Datasets to train on: {}", dataset_names.join(", "));
            println!("📊 Max rows per dataset: {}", max_rows);
            println!("💾 Final model name: {}", model_name);
            println!("{}", "═".repeat(60));

            // Create ONE trainer for ALL datasets (shared vocabulary)
            let mut trainer = NovaTrainer::new();
            let mut total_examples = 0;

            for (idx, ds_name) in dataset_names.iter().enumerate() {
                println!("\n{}", "═".repeat(60));
                println!("📦 Dataset {}/{}: '{}'", idx + 1, dataset_names.len(), ds_name);
                println!("{}", "═".repeat(60));

                // Step 1: Download dataset from Hugging Face
                println!("📥 Downloading dataset '{}'...", ds_name);
                let mut dataset = NovaDataset::new();
                
                let mut hf = HFDatasetRef::new(ds_name)
                    .with_split(&split)
                    .with_max_rows(max_rows);
                
                // If template is provided, use it
                if !template.is_empty() {
                    hf.column_mapping.prompt_template = Some(template.clone());
                    println!("    📝 Using prompt template");
                }
                
                // If input_col/target_col are provided, use them
                if !input_col.is_empty() {
                    hf.column_mapping.input_column = input_col.clone();
                }
                if !target_col.is_empty() {
                    hf.column_mapping.target_column = target_col.clone();
                }
                
                dataset.add_hf_dataset(hf);
                dataset.load_all();

                if dataset.examples.is_empty() {
                    eprintln!("⚠️  No examples loaded from '{}'. Skipping.", ds_name);
                    continue;
                }

                println!("   ✅ Loaded {} examples from '{}'", dataset.examples.len(), ds_name);

                // Step 2: Train the model on this dataset (single pass)
                println!("🎯 Training on '{}'...", ds_name);
                if ultra {
                    trainer.train_one_pass_ultra(&mut nova, &dataset.examples);
                } else {
                    trainer.train_one_pass(&mut nova, &dataset.examples);
                }
                
                total_examples += dataset.examples.len();
                println!("   ✅ Training on '{}' complete!", ds_name);
            }

            if total_examples == 0 {
                eprintln!("❌ No training data loaded from any dataset. Aborting.");
                return;
            }

            // Step 3: Update model name
            nova.name = model_name.clone();

            // Step 4: Save the final model
            println!("\n{}", "═".repeat(60));
            println!("💾 Saving final model after training on {} total examples...", total_examples);
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, &model_name) {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("❌ Failed to save model: {}", e),
            }

            println!("\n{}", "═".repeat(60));
            println!("✅ Multi-dataset training complete!");
            println!("   📚 Total examples trained: {}", total_examples);
            println!("   💾 Model '{}' is ready.", model_name);
            println!("{}", "═".repeat(60));
            println!("💡 Try: nova model list");
            println!("💡 Try: nova model load --name {}", model_name);
            println!("💡 Try: nova run --input \"hello\" --model {}", model_name);
        }
    }
}

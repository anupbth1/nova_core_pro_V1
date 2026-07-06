//! Nova Core - Main CLI Entry Point
//! Post-Transformer LLM with O(n) field dynamics instead of O(n²) attention.
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod pulse;
mod field;
mod core;
mod ssm;
mod embedding;
mod loom;
mod trainer;
mod optimizer;
mod dataset;
mod model;

use clap::{Parser, Subcommand};
use colored::*;
use std::time::Instant;
use loom::NovaLoom;
use embedding::{VOCAB_SIZE, EMBED_DIM};
use trainer::{TrainingConfig, TrainingExample, NovaTrainer};
use dataset::{NovaDataset, HFDatasetRef, DatasetFormat};
use model::NovaModelManager;

#[derive(Parser)]
#[command(name = "nova")]
#[command(about = "🚀 Nova AI - Post-Transformer LLM with O(n) field dynamics", long_about = None)]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Model dimension (global)
    #[arg(long, default_value_t = 256, global = true)]
    dim: usize,

    /// Number of cores (global)
    #[arg(long, default_value_t = 5, global = true)]
    cores: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Process text input through neural network
    Run {
        #[arg(short, long)]
        input: String,
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
        #[arg(short = 'x', long, default_value = "100")]
        examples: usize,
        #[arg(short = 'e', long, default_value = "10")]
        epochs: usize,
    },
    /// Interactive chat with trained model
    SmartChat {
        /// Model name to load at startup
        #[arg(short, long)]
        model: Option<String>,
    },
    /// Dataset management commands
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },
    /// Model management commands
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Train using Hugging Face dataset (single-pass)
    HfTrain {
        /// Hugging Face dataset name
        #[arg(short, long)]
        dataset: String,
        /// Dataset subset/config
        #[arg(long, default_value = "")]
        subset: String,
        /// Dataset split
        #[arg(short, long, default_value = "train")]
        split: String,
        /// Input column name
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,
        /// Target column name
        #[arg(short = 't', long, default_value = "")]
        target_col: String,
        /// Max rows to download
        #[arg(short = 'm', long, default_value = "500")]
        max_rows: usize,
        /// Model dimension
        #[arg(long, default_value = "256")]
        dim: usize,
        /// Model name to save
        #[arg(short = 'n', long, default_value = "hf-trained-model")]
        model_name: String,
    },
    /// Train ONE model on MULTIPLE Hugging Face datasets sequentially
    MultiHfTrain {
        /// Hugging Face dataset names (comma-separated)
        #[arg(short, long)]
        datasets: String,
        /// Dataset split for all datasets
        #[arg(short, long, default_value = "train")]
        split: String,
        /// Input column name (auto-detected if empty)
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,
        /// Target column name (auto-detected if empty)
        #[arg(short = 't', long, default_value = "")]
        target_col: String,
        /// Max rows per dataset (0 = ALL rows)
        #[arg(short = 'm', long, default_value = "1000")]
        max_rows: usize,
        /// Model dimension
        #[arg(long, default_value = "256")]
        dim: usize,
        /// Number of cores
        #[arg(long, default_value = "5")]
        cores: usize,
        /// Model name to save after training
        #[arg(short = 'n', long, default_value = "multi-hf-trained-model")]
        model_name: String,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// Load a local dataset file
    Load {
        #[arg(short, long)]
        file: String,
        #[arg(short = 'i', long, default_value = "")]
        input_col: String,
        #[arg(short = 't', long, default_value = "")]
        target_col: String,
        #[arg(short = 'f', long)]
        format: Option<String>,
        #[arg(short = 'm', long, default_value = "0")]
        max_rows: usize,
    },
    /// Load a dataset from Hugging Face
    Hf {
        #[arg(short, long)]
        name: String,
        #[arg(long, default_value = "")]
        subset: String,
        #[arg(short, long, default_value = "train")]
        split: String,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// List available models
    List,
    /// Load a model
    Load {
        #[arg(short, long)]
        name: String,
    },
    /// Delete a model
    Delete {
        #[arg(short, long)]
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let mut nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
    nova.use_neural = true;

    match &cli.command {
        Commands::Run { input, model } => {
            // Load model if specified
            if let Some(model_name) = model {
                let model_mgr = NovaModelManager::new();
                match model_mgr.load_model(model_name) {
                    Ok((loaded_loom, _)) => {
                        nova = loaded_loom;
                        println!("✅ Loaded model: {}", model_name);
                    }
                    Err(e) => eprintln!("⚠️ Failed to load model: {}", e),
                }
            }

            println!("{}", "═".repeat(60));
            println!("🚀 Nova Core - Processing");
            println!("{}", "═".repeat(60));
            println!("Input: {}", input);

            let start = Instant::now();
            let output = nova.generate_text(input, 50);
            let elapsed = start.elapsed();

            println!("Output: {}", output);
            println!("Time: {:?}", elapsed);
        }

        Commands::Bench { samples } => {
            println!("📊 Benchmark: {} samples", samples);
            let start = Instant::now();
            for _ in 0..*samples {
                let _ = nova.generate_text("test input", 10);
            }
            let elapsed = start.elapsed();
            let rate = *samples as f64 / elapsed.as_secs_f64();
            println!("  ✅ {} samples in {:?}", samples, elapsed);
            println!("  📈 Rate: {:.2} samples/sec", rate);
        }

        Commands::Chat => {
            println!("{}", "═".repeat(60));
            println!("💬 Nova Chat - Type 'exit' to quit");
            println!("{}", "═".repeat(60));

            loop {
                print!("You: ");
                use std::io::{self, Write};
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "exit" || input == "quit" { break; }
                if input.is_empty() { continue; }

                let start = Instant::now();
                let output = nova.generate_text(input, 100);
                let elapsed = start.elapsed();

                println!("Nova: {}", output);
                println!("      ({:?})", elapsed);
                println!();
            }
        }

        Commands::Info => {
            println!("{}", "═".repeat(60));
            println!("🔧 NOVA CORE Architecture Info");
            println!("{}", "─".repeat(40));
            let total_params = nova.num_params();
            println!("  Architecture:    Post-Transformer LLM");
            println!("  Complexity:      O(n) - Linear");
            println!("  Embedding:       {} x {} = {} params", VOCAB_SIZE, EMBED_DIM, VOCAB_SIZE * EMBED_DIM);
            println!("  Cores:           5 (syntax, semantic, memory, reasoning, pattern)");
            println!("  Each Core:       LayerNorm → SSM (Mamba) → GLU → Residual");
            println!("  Field:           Global state aggregator (replaces attention)");
            println!("  Total Params:    {}", total_params);
            println!("{}", "─".repeat(40));
            println!("  Compared to Transformer:");
            println!("    Attention O(n²)  →  Field O(n)");
            println!("    Tokens (discrete) → Pulses (continuous)");
            println!("    Fixed depth       → Adaptive depth");
            println!("    Layers (linear)   → Cores (graph)");
            println!("{}", "═".repeat(60));
        }

        Commands::Speed { pulses } => {
            println!("⚡ Speed Test: Processing {} pulses...", pulses);
            let start = Instant::now();
            for _ in 0..*pulses {
                let _ = nova.generate_text("nova", 5);
            }
            let duration = start.elapsed();
            let rate = *pulses as f64 / duration.as_secs_f64();
            println!("  ✅ Processed {} calls in {:?}", pulses, duration);
            println!("  📈 Rate: {:.0} calls/second", rate);
            if rate > 100.0 {
                println!("  🚀 Excellent performance!");
            } else {
                println!("  💡 Run 'cargo build --release' for faster speed");
            }
        }

        Commands::FullBench => {
            println!("📊 Running full benchmark suite...");
            let start = Instant::now();
            for _ in 0..20 {
                let _ = nova.generate_text("benchmark test", 10);
            }
            let elapsed = start.elapsed();
            println!("  ✅ Complete in {:?}", elapsed);
        }

        Commands::Improve => {
            println!("🔧 Auto-improve is not available in this version.");
            println!("💡 Train with: nova train --examples 100 --epochs 5");
        }

        Commands::GenData => {
            println!("📝 Generate training data (placeholder)");
            let data = vec![
                "the cat sat on the mat".to_string(),
                "the dog ran in the park".to_string(),
                "birds fly in the sky".to_string(),
                "fish swim in the water".to_string(),
            ];
            println!("  Generated {} examples", data.len());
        }

        Commands::Train { examples, epochs } => {
            println!("{}", "═".repeat(60));
            println!("🎯 Training Nova Core");
            println!("{}", "═".repeat(60));

            let training_data = vec![
                "the cat sat on the mat".to_string(),
                "the dog ran in the park".to_string(),
                "birds fly in the sky".to_string(),
                "fish swim in the water".to_string(),
                "the sun is bright today".to_string(),
                "the moon shines at night".to_string(),
            ];

            let config = TrainingConfig {
                batch_size: 2,
                seq_length: 32,
                learning_rate: 3e-4,
                max_epochs: *epochs,
                warmup_steps: 10,
                total_steps: training_data.len().max(100),
                grad_clip: 1.0,
                eval_every: 10,
                save_every: 100,
            };

            nova.init_trainer(config);
            println!("🏋️  Training for {} epochs on {} examples...", epochs, examples);

            let start = Instant::now();
            for epoch in 0..*epochs {
                let loss = nova.train(&training_data);
                println!("  Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            }
            let elapsed = start.elapsed();
            println!("✅ Training complete! Time: {:?}", elapsed);

            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, "trained-model") {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("⚠️ Failed to save model: {}", e),
            }
        }

        Commands::SmartChat { model } => {
            if let Some(model_name) = model {
                let model_mgr = NovaModelManager::new();
                match model_mgr.load_model(model_name) {
                    Ok((loaded_loom, _)) => {
                        nova = loaded_loom;
                        println!("✅ Loaded model: {}", model_name);
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to load model '{}': {}", model_name, e);
                        return;
                    }
                }
            }

            println!("\n🌺 Nova Smart Chat");
            println!("{}", "─".repeat(40));
            println!("Type 'exit' or 'quit' to stop");
            println!("Type 'train <N>' to train on N examples");
            println!("Type 'load <name>' to load a trained model");
            println!("Type 'save <name>' to save the current model");
            println!("Type 'stats' to see performance");
            println!("{}\n", "─".repeat(40));

            let mut response_timeout_secs: u64 = 30;

            loop {
                print!("🌺 ");
                use std::io::Write;
                std::io::stdout().flush().unwrap();

                let input = {
                    let (tx, rx) = std::sync::mpsc::channel();
                    std::thread::spawn(move || {
                        let mut buf = String::new();
                        if std::io::stdin().read_line(&mut buf).is_ok() {
                            let _ = tx.send(buf);
                        }
                    });
                    match rx.recv_timeout(std::time::Duration::from_secs(response_timeout_secs)) {
                        Ok(line) => line.trim().to_string(),
                        Err(_) => {
                            println!("\n⏰ No input for {}s. Type something or 'exit'.", response_timeout_secs);
                            continue;
                        }
                    }
                };

                if input == "exit" || input == "quit" {
                    println!("👋 Goodbye!");
                    break;
                }

                if input == "stats" {
                    println!("📊 Nova Stats");
                    println!("  Parameters: {}", nova.num_params());
                    println!("  Cores: {}", nova.cores.len());
                    println!("  Timeout: {}s", response_timeout_secs);
                    continue;
                }

                if input.starts_with("timeout ") {
                    if let Ok(secs) = input[8..].trim().parse::<u64>() {
                        response_timeout_secs = secs.max(5).min(300);
                        println!("✅ Timeout set to {}s", response_timeout_secs);
                    }
                    continue;
                }

                if input.starts_with("train ") {
                    if let Ok(n) = input[6..].trim().parse::<usize>() {
                        let config = TrainingConfig {
                            batch_size: 2, seq_length: 32, learning_rate: 3e-4,
                            max_epochs: 5, warmup_steps: 10,
                            total_steps: n.max(100), grad_clip: 1.0,
                            eval_every: 10, save_every: 100,
                        };
                        nova.init_trainer(config);
                        let data: Vec<String> = (0..n).map(|i| format!("training example {}", i)).collect();
                        for epoch in 0..5 {
                            let loss = nova.train(&data);
                            println!("  Epoch {}/5: loss = {:.6}", epoch + 1, loss);
                        }
                        println!("✅ Training complete!");
                    } else {
                        println!("❌ Usage: train <number>");
                    }
                    continue;
                }

                if input.starts_with("load ") {
                    let name = input[5..].trim();
                    let model_mgr = NovaModelManager::new();
                    match model_mgr.load_model(name) {
                        Ok((loaded_loom, _)) => {
                            nova = loaded_loom;
                            println!("✅ Model '{}' loaded!", name);
                        }
                        Err(e) => eprintln!("❌ Failed to load model: {}", e),
                    }
                    continue;
                }

                if input.starts_with("save ") {
                    let name = input[5..].trim();
                    let model_mgr = NovaModelManager::new();
                    match model_mgr.save_model(&nova, name) {
                        Ok(path) => println!("✅ Model saved to: {}", path),
                        Err(e) => eprintln!("❌ Failed to save model: {}", e),
                    }
                    continue;
                }

                if input.is_empty() { continue; }

                let start = Instant::now();
                let output = nova.generate_text(&input, 100);
                let duration = start.elapsed();

                // Trim to clean output
                let words: Vec<&str> = output.split_whitespace().collect();
                let trimmed = if words.len() > 30 {
                    words[..30].join(" ") + "..."
                } else {
                    words.join(" ")
                };

                println!("{}", trimmed.bright_green());
                if trimmed.len() > 100 {
                    println!("  ({:?}, {} chars)", duration, trimmed.len());
                } else {
                    println!("  ({:?})", duration);
                }
            }
        }

        Commands::Dataset { action } => {
            match action {
                DatasetAction::Load { file, input_col, target_col, format, max_rows } => {
                    println!("📦 Loading dataset from: {}", file);
                    let mut dataset = NovaDataset::new();
                    let mut src = dataset.add_file(file);
                    if !input_col.is_empty() { src.column_mapping.input_column = input_col.clone(); }
                    if !target_col.is_empty() { src.column_mapping.target_column = target_col.clone(); }
                    if *max_rows > 0 { src.max_rows = *max_rows; }
                    dataset.load_all();
                    dataset.print_stats();
                }
                DatasetAction::Hf { name, subset, split } => {
                    println!("🤗 Loading dataset '{}'...", name);
                    let mut dataset = NovaDataset::new();
                    let mut hf = HFDatasetRef::new(name).with_split(split);
                    if !subset.is_empty() { hf.subset = subset.clone(); }
                    dataset.add_hf_dataset(hf);
                    dataset.load_all();
                    dataset.print_stats();
                }
            }
        }

        Commands::Model { action } => {
            let model_mgr = NovaModelManager::new();
            match action {
                ModelAction::List => {
                    model_mgr.list_models();
                }
                ModelAction::Load { name } => {
                    match model_mgr.load_model(name) {
                        Ok((loaded_loom, _)) => {
                            nova = loaded_loom;
                            println!("✅ Model '{}' loaded!", name);
                        }
                        Err(e) => eprintln!("❌ Failed to load model: {}", e),
                    }
                }
                ModelAction::Delete { name } => {
                    match model_mgr.delete_model(name) {
                        Ok(()) => println!("✅ Model '{}' deleted", name),
                        Err(e) => eprintln!("❌ Failed to delete model: {}", e),
                    }
                }
            }
        }

        Commands::HfTrain { dataset: ds_name, subset, split, input_col, target_col, max_rows, dim, model_name } => {
            println!("{}", "═".repeat(60));
            println!("🎯 Hugging Face Training");
            println!("{}", "═".repeat(60));

            let dim = *dim;
            let mut nova = NovaLoom::new(dim, VOCAB_SIZE);

            println!("📥 Step 1: Downloading dataset '{}'...", ds_name);
            let mut dataset = NovaDataset::new();
            let mut hf = HFDatasetRef::new(ds_name).with_split(split).with_max_rows(*max_rows);
            if !subset.is_empty() { hf.subset = subset.clone(); }
            if !input_col.is_empty() { hf.column_mapping.input_column = input_col.clone(); }
            if !target_col.is_empty() { hf.column_mapping.target_column = target_col.clone(); }
            dataset.add_hf_dataset(hf);
            let examples = dataset.load_all().to_vec();

            if examples.is_empty() {
                eprintln!("❌ No examples loaded from '{}'", ds_name);
                return;
            }
            println!("   ✅ Loaded {} examples", examples.len());

            println!("\n🎯 Step 2: Training model...");
            let config = TrainingConfig {
                batch_size: 2, seq_length: 64, learning_rate: 3e-4,
                max_epochs: 1, warmup_steps: 50,
                total_steps: examples.len().max(100), grad_clip: 1.0,
                eval_every: 100, save_every: 1000,
            };
            nova.init_trainer(config);

            let texts: Vec<String> = examples.iter().map(|ex| {
                if ex.target.is_empty() || ex.target == ex.input {
                    ex.input.clone()
                } else {
                    format!("{} {}", ex.input, ex.target)
                }
            }).collect();

            let loss = nova.train(&texts);
            println!("   ✅ Loss: {:.6}", loss);

            println!("\n💾 Step 3: Saving model...");
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, model_name) {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("❌ Failed to save model: {}", e),
            }

            println!("\n✅ Training complete!");
            println!("💡 Try: nova smart-chat --model {}", model_name);
        }

        Commands::MultiHfTrain { datasets, split, input_col, target_col, max_rows, dim, cores, model_name } => {
            println!("{}", "═".repeat(60));
            println!("🚀 MULTI-DATASET TRAINING");
            println!("{}", "═".repeat(60));

            let dataset_names: Vec<&str> = datasets.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            if dataset_names.is_empty() {
                eprintln!("❌ No datasets specified.");
                return;
            }

            println!("📋 Datasets: {}", dataset_names.join(", "));
            println!("📊 Max rows/dataset: {}", max_rows);
            println!("🔧 Dim: {}, Cores: {}", dim, cores);
            println!("💾 Model: {}", model_name);
            println!("{}", "═".repeat(60));

            let mut nova = NovaLoom::new(*dim, VOCAB_SIZE);
            let config = TrainingConfig {
                batch_size: 2, seq_length: 64, learning_rate: 3e-4,
                max_epochs: 1, warmup_steps: 50,
                total_steps: 10000, grad_clip: 1.0,
                eval_every: 100, save_every: 1000,
            };
            nova.init_trainer(config);

            let mut total_examples = 0;

            for (idx, ds_name) in dataset_names.iter().enumerate() {
                println!("\n📦 Dataset {}/{}: '{}'", idx + 1, dataset_names.len(), ds_name);
                println!("📥 Downloading...");

                let mut dataset = NovaDataset::new();
                let hf = HFDatasetRef::new(ds_name).with_split(split).with_max_rows(*max_rows);
                dataset.add_hf_dataset(hf);
                let examples = dataset.load_all().to_vec();

                if examples.is_empty() {
                    eprintln!("   ⚠️  No examples. Skipping.");
                    continue;
                }
                println!("   ✅ Loaded {} examples", examples.len());

                let texts: Vec<String> = examples.iter().map(|ex| {
                    if ex.target.is_empty() || ex.target == ex.input {
                        ex.input.clone()
                    } else {
                        format!("{} {}", ex.input, ex.target)
                    }
                }).collect();

                let loss = nova.train(&texts);
                println!("   ✅ Loss: {:.6}", loss);
                total_examples += examples.len();
            }

            if total_examples == 0 {
                eprintln!("❌ No data loaded.");
                return;
            }

            println!("\n💾 Saving model...");
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, model_name) {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("❌ Failed to save model: {}", e),
            }

            println!("\n✅ Multi-dataset training complete!");
            println!("   📚 Total examples: {}", total_examples);
            println!("   💾 Model: '{}'", model_name);
        }
    }
}
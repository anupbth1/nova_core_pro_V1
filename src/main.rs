//! Nova Core - Main CLI Entry Point
//! Post-Transformer LLM with O(n) field dynamics instead of O(n²) attention.
//! Architecture: Embedding → SSM+GLU Cores → Field Aggregation → Prediction
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
use trainer::{TrainingConfig, TrainingExample};
use dataset::{NovaDataset, HFDatasetRef, DatasetFormat};
use model::NovaModelManager;

#[derive(Parser)]
#[command(name = "nova")]
#[command(about = "🚀 Nova AI - Post-Transformer LLM with O(n) field dynamics", long_about = None)]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process text input through neural network
    Run {
        #[arg(short, long)]
        input: String,
        #[arg(short = 't', long, default_value = "50")]
        max_tokens: usize,
    },
    /// Interactive chat mode
    Chat {
        #[arg(short = 't', long, default_value = "100")]
        max_tokens: usize,
    },
    /// Show architecture information
    Info,
    /// Train on text data
    Train {
        #[arg(short = 'x', long, default_value = "100")]
        examples: usize,
        #[arg(short = 'e', long, default_value = "3")]
        epochs: usize,
        #[arg(short, long, default_value = "data.txt")]
        data_file: String,
    },
    /// Evaluate model on test text
    Eval {
        #[arg(short, long)]
        input: String,
    },
    /// Show parameter count
    Params,
    /// Train ONE model on MULTIPLE Hugging Face datasets sequentially
    MultiHfTrain {
        /// Hugging Face dataset names (comma-separated, e.g. "imdb,wikitext,tiny_shakespeare")
        #[arg(short, long)]
        datasets: String,
        /// Dataset split for all datasets (train, test, validation)
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
        /// Model dimension (64/128/256/512)
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

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run { input, max_tokens } => {
            println!("{}", "═".repeat(60));
            println!("🚀 Nova Core v0.1.0 - Neural Processing");
            println!("{}", "═".repeat(60));
            println!("Input: {}", input);
            println!("{}", "─".repeat(60));

            let start = Instant::now();
            let mut nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            nova.use_neural = true;

            let output = nova.generate_text(input, *max_tokens);

            let elapsed = start.elapsed();
            println!("Output: {}", output);
            println!("{}", "─".repeat(60));
            println!("Time: {:?}", elapsed);
            println!("{}", "═".repeat(60));
        }

        Commands::Chat { max_tokens } => {
            println!("{}", "═".repeat(60));
            println!("💬 Nova Chat Mode - Type 'exit' to quit");
            println!("{}", "═".repeat(60));

            let mut nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            nova.use_neural = true;

            loop {
                print!("You: ");
                use std::io::{self, Write};
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "exit" || input == "quit" {
                    break;
                }

                let start = Instant::now();
                let output = nova.generate_text(input, *max_tokens);
                let elapsed = start.elapsed();

                println!("Nova: {}", output);
                println!("      ({:?})", elapsed);
                println!();
            }
        }

        Commands::Info => {
            println!("{}", "═".repeat(60));
            println!("🚀 Nova Core Architecture v0.1.0");
            println!("{}", "═".repeat(60));

            let nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            let total_params = nova.num_params();

            println!("Architecture: Post-Transformer LLM");
            println!("Complexity:   O(n) - Linear in sequence length");
            println!("No Attention: Uses field dynamics instead of O(n²) attention");
            println!();
            println!("Components:");
            println!("  • Embedding: {} x {} = {} params",
                VOCAB_SIZE, EMBED_DIM, VOCAB_SIZE * EMBED_DIM);
            println!("  • Cores: 5 specialized (Syntax, Semantic, Memory, Reasoning, Pattern)");
            println!("  • Each Core: LayerNorm → SSM (Mamba scan) → GLU → Residual");
            println!("  • Field: Global state aggregator (replaces attention)");
            println!("  • Total Parameters: {}", total_params);
            println!();
            println!("Inference:");
            println!("  • Autoregressive generation with top-k sampling");
            println!("  • Cross-entropy loss for next-token prediction");
            println!("  • Cosine similarity for output logits");
            println!("{}", "═".repeat(60));
        }

        Commands::Train { examples, epochs, data_file } => {
            println!("{}", "═".repeat(60));
            println!("🎯 Training Nova Core");
            println!("{}", "═".repeat(60));

            let dataset = match std::fs::read_to_string(data_file) {
                Ok(content) => {
                    let lines: Vec<String> = content.lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect();
                    println!("📚 Loaded {} examples from {}", lines.len(), data_file);
                    lines
                }
                Err(e) => {
                    println!("📝 Could not read {}: {}", data_file, e);
                    println!("📝 Generating synthetic training data...");
                    let synthetic = vec![
                        "the cat sat on the mat".to_string(),
                        "the dog ran in the park".to_string(),
                        "birds fly in the sky".to_string(),
                        "fish swim in the water".to_string(),
                        "the sun is bright today".to_string(),
                        "the moon shines at night".to_string(),
                        "apples grow on trees".to_string(),
                        "books are full of knowledge".to_string(),
                        "music brings people together".to_string(),
                        "learning never stops".to_string(),
                        "hello world this is a test".to_string(),
                        "the quick brown fox jumps".to_string(),
                        "artificial intelligence is amazing".to_string(),
                        "rust is a systems programming language".to_string(),
                        "neural networks process information".to_string(),
                        "deep learning transforms data".to_string(),
                        "machines learn from examples".to_string(),
                        "data is the new oil".to_string(),
                        "programming requires practice".to_string(),
                        "algorithms solve problems efficiently".to_string(),
                    ];
                    println!("📚 Generated {} synthetic examples", synthetic.len());
                    synthetic
                }
            };

            if dataset.is_empty() {
                eprintln!("❌ No training data available");
                return;
            }

            let config = TrainingConfig {
                batch_size: 4,
                seq_length: 32,
                learning_rate: 3e-4,
                max_epochs: *epochs,
                warmup_steps: 10,
                total_steps: dataset.len().max(100),
                grad_clip: 1.0,
                eval_every: 10,
                save_every: 100,
            };

            let mut nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            nova.init_trainer(config);

            println!("\n🏋️  Starting training for {} epochs...", epochs);
            let start = Instant::now();

            for epoch in 0..*epochs {
                let loss = nova.train(&dataset);
                println!("  Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            }

            let elapsed = start.elapsed();
            println!("\n✅ Training complete! Time: {:?}", elapsed);

            println!("\n{}", "─".repeat(60));
            println!("📝 Sample generations:");
            for prompt in &["the", "hello", "birds", "music"] {
                let output = nova.generate_text(prompt, 20);
                println!("  '{}' → '{}'", prompt, output);
            }
            println!("{}", "═".repeat(60));
        }

        Commands::Eval { input } => {
            let mut nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            nova.init_trainer(TrainingConfig::default());

            let (loss, predictions) = {
                let trainer = nova.trainer.as_mut().unwrap();
                trainer.forward(input)
            };

            let decoded = nova.embedding.detokenize(&predictions);
            println!("Input: {}", input);
            println!("Prediction: {}", decoded);
            println!("Loss: {:.6}", loss);
        }

        Commands::Params => {
            let nova = NovaLoom::new(EMBED_DIM, VOCAB_SIZE);
            let params = nova.num_params();
            println!("Total parameters: {}", params);
            println!("Vocabulary: {}", VOCAB_SIZE);
            println!("Embedding dimension: {}", EMBED_DIM);
            println!("Embedding parameters: {}", VOCAB_SIZE * EMBED_DIM);
            println!("Core parameters: {}", params - VOCAB_SIZE * EMBED_DIM);
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
                eprintln!("❌ No datasets specified. Use --datasets with comma-separated names.");
                return;
            }

            let dim = *dim;
            let vocab_size = VOCAB_SIZE;
            let num_cores = *cores.min(&9);

            println!("📋 Datasets: {}", dataset_names.join(", "));
            println!("📊 Max rows/dataset: {}", max_rows);
            println!("🔧 Dim: {}, Cores: {}", dim, num_cores);
            println!("💾 Model: {}", model_name);
            println!("{}", "═".repeat(60));

            // Create model with specified dimensions
            let mut nova = NovaLoom::new(dim, vocab_size);
            let config = TrainingConfig {
                batch_size: 2,
                seq_length: 64,
                learning_rate: 3e-4,
                max_epochs: 1,
                warmup_steps: 50,
                total_steps: 10000,
                grad_clip: 1.0,
                eval_every: 100,
                save_every: 1000,
            };
            nova.init_trainer(config);

            let mut total_examples = 0;

            for (idx, ds_name) in dataset_names.iter().enumerate() {
                println!("\n{}", "═".repeat(60));
                println!("📦 Dataset {}/{}: '{}'", idx + 1, dataset_names.len(), ds_name);
                println!("{}", "═".repeat(60));

                // Download dataset using Python
                println!("📥 Downloading '{}'...", ds_name);
                let examples = match download_hf_dataset(ds_name, split, *max_rows) {
                    Ok(examples) => {
                        println!("   ✅ Loaded {} examples", examples.len());
                        examples
                    }
                    Err(e) => {
                        eprintln!("   ⚠️  Failed: {}. Skipping.", e);
                        continue;
                    }
                };

                // Train on this dataset
                println!("🎯 Training on '{}'...", ds_name);
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
                eprintln!("❌ No training data loaded from any dataset.");
                return;
            }

            // Save model
            println!("\n{}", "═".repeat(60));
            println!("💾 Saving model '{}' after {} examples...", model_name, total_examples);
            let model_mgr = NovaModelManager::new();
            match model_mgr.save_model(&nova, model_name) {
                Ok(path) => println!("✅ Model saved to: {}", path),
                Err(e) => eprintln!("❌ Failed to save model: {}", e),
            }

            println!("\n{}", "═".repeat(60));
            println!("✅ Multi-dataset training complete!");
            println!("   📚 Total examples: {}", total_examples);
            println!("   💾 Model: '{}'", model_name);
            println!("{}", "═".repeat(60));
        }
    }
}

/// Download dataset from Hugging Face using src/dataset.rs NovaDataset infrastructure
fn download_hf_dataset(name: &str, split: &str, max_rows: usize) -> Result<Vec<TrainingExample>, String> {
    let mut dataset = NovaDataset::new();
    let hf = HFDatasetRef::new(name)
        .with_split(split)
        .with_max_rows(max_rows);
    dataset.add_hf_dataset(hf);
    let examples = dataset.load_all().to_vec();
    if examples.is_empty() {
        Err(format!("No examples loaded from '{}'", name))
    } else {
        Ok(examples)
    }
}
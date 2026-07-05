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

use clap::{Parser, Subcommand};
use colored::*;
use std::time::Instant;
use loom::NovaLoom;
use embedding::{VOCAB_SIZE, EMBED_DIM};
use trainer::TrainingConfig;

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

            // Try to load dataset from file
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
                    // Generate synthetic training data
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

            // Test generation
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
    }
}
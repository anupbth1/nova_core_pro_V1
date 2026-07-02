//! Nova Model Manager - Save, load, and manage Nova Core models
//!
//! Features:
//! - Save/load models in .nova format (binary)
//! - List available models
//! - Upload/download models from Hugging Face Hub
//! - Model configuration management
//! - Multiple model versions

use crate::core::NovaCore;
use crate::field::NovaField;
use crate::loom::NovaLoom;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub description: String,
    pub dim: usize,
    pub num_cores: usize,
    pub core_names: Vec<String>,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub created_at: String,
    pub trained_on: String,
    pub accuracy: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "nova-core".to_string(),
            version: "0.1.0".to_string(),
            description: "Nova Core - Post-Transformer LLM".to_string(),
            dim: 64,
            num_cores: 5,
            core_names: vec!["syntax".into(), "semantic".into(), "memory".into(), "reasoning".into(), "pattern".into()],
            max_iterations: 8,
            convergence_threshold: 0.08,
            created_at: chrono_now(),
            trained_on: "none".to_string(),
            accuracy: 0.0,
        }
    }
}

/// Complete model snapshot for saving/loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub config: ModelConfig,
    pub cores: Vec<CoreSnapshot>,
    pub field_state: Vec<f32>,
    pub field_momentum: Vec<f32>,
    pub field_update_count: usize,
    pub vocabulary: HashMap<String, Vec<f32>>,
    /// Reverse vocabulary: hash of embedding -> word (optional for backward compat)
    #[serde(default)]
    pub vocab_reverse: HashMap<u64, String>,
    /// N-gram patterns: sequence hash -> next word predictions (optional for backward compat)
    #[serde(default)]
    pub ngram_patterns: HashMap<u64, Vec<(String, f32)>>,
    /// All unique words seen during training (optional for backward compat)
    #[serde(default)]
    pub all_words: Vec<String>,
    /// Learned responses: input hash -> output text
    pub learned_responses: HashMap<u64, String>,
    /// Original input texts for learned responses (hash -> original input text)
    pub learned_inputs: HashMap<u64, String>,
}



/// Core state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSnapshot {
    pub id: usize,
    pub name: String,
    pub memory: Vec<f32>,
    pub internal_state: Vec<f32>,
    pub gate: f32,
    /// SSM parameters for this core (flat memory layout)
    pub ssm_delta: Vec<f32>,
    pub ssm_delta_bias: Vec<f32>,
    pub ssm_a_log: Vec<f32>,
    pub ssm_b: Vec<f32>,
    pub ssm_c: Vec<f32>,
    pub ssm_d: Vec<f32>,
    pub ssm_h: Vec<f32>,
    pub ssm_time_mix_x: Vec<f32>,
    pub ssm_time_mix_w: Vec<f32>,
    pub ssm_time_mix_key: Vec<f32>,
    pub ssm_time_mix_value: Vec<f32>,
    pub ssm_time_mix_receptance: Vec<f32>,
    pub ssm_prev_x: Vec<f32>,
    pub use_ssm: bool,
    pub use_time_mixing: bool,
}


/// Nova Model Manager
pub struct NovaModelManager {
    /// Directory where models are stored
    pub models_dir: PathBuf,
    /// Currently loaded model configs
    pub available_models: Vec<ModelConfig>,
}

impl NovaModelManager {
    pub fn new() -> Self {
        let models_dir = PathBuf::from("models");
        let mut mgr = Self {
            models_dir,
            available_models: Vec::new(),
        };
        mgr.scan_models();
        mgr
    }
    
    pub fn with_dir(dir: &str) -> Self {
        let models_dir = PathBuf::from(dir);
        let mut mgr = Self {
            models_dir,
            available_models: Vec::new(),
        };
        mgr.scan_models();
        mgr
    }
    
    /// Scan models directory for .nova files
    pub fn scan_models(&mut self) -> &[ModelConfig] {
        self.available_models.clear();
        
        if !self.models_dir.exists() {
            return &self.available_models;
        }
        
        if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "nova").unwrap_or(false) {
                    if let Some(config) = self.read_config(&path) {
                        self.available_models.push(config);
                    }
                }
            }
        }
        
        &self.available_models
    }
    
    /// Read config from a .nova file
    fn read_config(&self, path: &Path) -> Option<ModelConfig> {
        let content = std::fs::read_to_string(path).ok()?;
        // Try parsing as ModelSnapshot first (JSON format with config nested)
        if let Ok(snapshot) = serde_json::from_str::<ModelSnapshot>(&content) {
            return Some(snapshot.config);
        }
        // Fallback: try parsing directly as ModelConfig (legacy format with --- separator)
        if let Some(config_part) = content.split("\n---\n").next() {
            if let Ok(config) = serde_json::from_str::<ModelConfig>(config_part) {
                return Some(config);
            }
        }
        None
    }
    
    /// Save model to .nova file
    pub fn save_model(&self, model: &NovaLoom, name: &str) -> Result<String, String> {
        // Create models directory
        std::fs::create_dir_all(&self.models_dir).map_err(|e| format!("Failed to create models dir: {}", e))?;
        
        let config = ModelConfig {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            description: format!("Nova Core model '{}'", name),
            dim: model.dim,
            num_cores: model.cores.len(),
            core_names: model.cores.iter().map(|c| c.name.clone()).collect(),
            max_iterations: model.max_iterations,
            convergence_threshold: model.convergence_threshold,
            created_at: chrono_now(),
            trained_on: "custom".to_string(),
            accuracy: 0.0,
        };
        
        let snapshot = ModelSnapshot {
            config,
            cores: model.cores.iter().map(|c| {
                let ssm = &c.ssm;
                CoreSnapshot {
                    id: c.id,
                    name: c.name.clone(),
                    memory: c.memory.clone(),
                    internal_state: c.internal_state.clone(),
                    gate: c.gate,
                    ssm_delta: ssm.delta.clone(),
                    ssm_delta_bias: ssm.delta_bias.clone(),
                    ssm_a_log: ssm.a_log.clone(),
                    ssm_b: ssm.b.clone(),
                    ssm_c: ssm.c.clone(),
                    ssm_d: ssm.d.clone(),
                    ssm_h: ssm.h.clone(),
                    ssm_time_mix_x: ssm.time_mix_x.clone(),
                    ssm_time_mix_w: ssm.time_mix_w.clone(),
                    ssm_time_mix_key: ssm.time_mix_key.clone(),
                    ssm_time_mix_value: ssm.time_mix_value.clone(),
                    ssm_time_mix_receptance: ssm.time_mix_receptance.clone(),
                    ssm_prev_x: ssm.prev_x.clone(),
                    use_ssm: c.use_ssm,
                    use_time_mixing: c.use_time_mixing,
                }
            }).collect(),

            field_state: model.field.state().to_vec(),
            field_momentum: vec![], // Field momentum is internal
            field_update_count: 0,
            vocabulary: model.vocabulary.clone(),
            vocab_reverse: model.vocab_reverse.clone(),
            ngram_patterns: model.ngram_patterns.clone(),
            all_words: model.all_words.clone(),
            learned_responses: model.learned_responses.clone(),
            learned_inputs: model.learned_inputs.clone(),
        };


        
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        
        let filepath = self.models_dir.join(format!("{}.nova", name));
        std::fs::write(&filepath, &json)
            .map_err(|e| format!("Failed to write model: {}", e))?;
        
        println!("  💾 Model saved to: {}", filepath.display());
        Ok(filepath.to_string_lossy().to_string())
    }
    
    /// Load model from .nova file
    pub fn load_model(&self, name: &str) -> Result<(NovaLoom, HashMap<String, Vec<f32>>), String> {
        // Strip .nova extension if user provided it
        let clean_name = name.strip_suffix(".nova").unwrap_or(name);
        let filepath = self.models_dir.join(format!("{}.nova", clean_name));
        if !filepath.exists() {
            return Err(format!("Model '{}' not found at {}", name, filepath.display()));
        }
        
        let content = std::fs::read_to_string(&filepath)
            .map_err(|e| format!("Failed to read model: {}", e))?;
        
        // Try to parse as JSON (Rust format)
        let snapshot_result: Result<ModelSnapshot, _> = serde_json::from_str(&content);
        
        let snapshot = match snapshot_result {
            Ok(s) => s,
            Err(json_err) => {
                // Check if it's a binary format (starts with NOVA magic bytes)
                if content.as_bytes().starts_with(b"NOVA") {
                    return Err(format!(
                        "Binary .nova format detected. This file was created by an older Python converter.\n\
                         Use 'convert_to_nova_json.py' to convert HF models to the correct JSON format.\n\
                         Or train a new model with: nova train --examples 100 --epochs 10\n\
                         JSON parse error: {}", json_err
                    ));
                }
                return Err(format!("Failed to parse model '{}': {}\n\
                    The file exists but is not valid JSON. Try using convert_to_nova_json.py to convert HF models.", 
                    name, json_err));
            }
        };
        
        // Reconstruct NovaLoom
        let mut loom = NovaLoom::new(snapshot.config.dim, snapshot.config.num_cores);

        
        // Restore cores (including SSM state)
        for (i, core_snap) in snapshot.cores.iter().enumerate() {
            if i < loom.cores.len() {
                loom.cores[i].memory = core_snap.memory.clone();
                loom.cores[i].internal_state = core_snap.internal_state.clone();
                loom.cores[i].gate = core_snap.gate;
                loom.cores[i].use_ssm = core_snap.use_ssm;
                loom.cores[i].use_time_mixing = core_snap.use_time_mixing;
                
                // Restore SSM parameters if they exist in the snapshot
                if !core_snap.ssm_delta.is_empty() {
                    loom.cores[i].ssm.load_from_projection(
                        core_snap.ssm_delta.clone(),
                        core_snap.ssm_delta_bias.clone(),
                        core_snap.ssm_a_log.clone(),
                        core_snap.ssm_b.clone(),
                        core_snap.ssm_c.clone(),
                        core_snap.ssm_d.clone(),
                    );
                    // Restore hidden state (flat memory layout)
                    if core_snap.ssm_h.len() == loom.cores[i].ssm.h.len() {
                        loom.cores[i].ssm.h.copy_from_slice(&core_snap.ssm_h);
                    }
                    // Restore time-mix parameters
                    if core_snap.ssm_time_mix_x.len() == loom.cores[i].ssm.d_inner {
                        loom.cores[i].ssm.time_mix_x.copy_from_slice(&core_snap.ssm_time_mix_x);
                        loom.cores[i].ssm.time_mix_w.copy_from_slice(&core_snap.ssm_time_mix_w);
                        loom.cores[i].ssm.time_mix_key.copy_from_slice(&core_snap.ssm_time_mix_key);
                        loom.cores[i].ssm.time_mix_value.copy_from_slice(&core_snap.ssm_time_mix_value);
                        loom.cores[i].ssm.time_mix_receptance.copy_from_slice(&core_snap.ssm_time_mix_receptance);
                        loom.cores[i].ssm.prev_x.copy_from_slice(&core_snap.ssm_prev_x);
                    }
                }
            }
        }

        
        // Restore field state using the new setter methods
        if !snapshot.field_state.is_empty() {
            loom.field.set_state(&snapshot.field_state);
        }
        if !snapshot.field_momentum.is_empty() {
            loom.field.set_momentum(&snapshot.field_momentum);
        }
        loom.field.set_update_count(snapshot.field_update_count);
        
        // Restore vocabulary and reverse vocabulary
        loom.vocabulary = snapshot.vocabulary.clone();
        loom.vocab_reverse = snapshot.vocab_reverse.clone();
        
        // Build reverse vocabulary from vocabulary if not saved (backward compat)
        if loom.vocab_reverse.is_empty() && !loom.vocabulary.is_empty() {
            for (word, vec) in &loom.vocabulary {
                let hash: u64 = vec.iter().fold(0u64, |acc, &x| {
                    acc.wrapping_mul(31).wrapping_add((x * 1000.0) as u64)
                });
                loom.vocab_reverse.insert(hash, word.clone());
            }
        }
        
        // Restore n-gram patterns for text generation
        loom.ngram_patterns = snapshot.ngram_patterns.clone();
        loom.all_words = snapshot.all_words.clone();
        
        // Build all_words from vocabulary if not saved (backward compat)
        if loom.all_words.is_empty() && !loom.vocabulary.is_empty() {
            loom.all_words = loom.vocabulary.keys().cloned().collect();
        }
        
        // Restore learned responses
        loom.learned_responses = snapshot.learned_responses.clone();
        // Restore learned inputs for word-overlap matching
        loom.learned_inputs = snapshot.learned_inputs.clone();
        
        println!("  📂 Model '{}' loaded (dim={}, cores={}, vocab={}, learned={}, ngrams={})", 
            name, snapshot.config.dim, snapshot.cores.len(), snapshot.vocabulary.len(),
            snapshot.learned_responses.len(), snapshot.ngram_patterns.len());

        
        Ok((loom, snapshot.vocabulary))

    }

    
    /// List all available models
    pub fn list_models(&self) {
        println!("\n{}", "═".repeat(60));
        println!("📦 AVAILABLE MODELS");
        println!("{}", "═".repeat(60));
        
        if self.available_models.is_empty() {
            println!("  (No models found. Train one with 'nova train' or 'nova hf-train')");
            return;
        }
        
        for config in &self.available_models {
            println!("  📁 {} v{}", config.name, config.version);
            println!("     📝 {}", config.description);
            println!("     📐 Dim: {}, Cores: {}", config.dim, config.num_cores);
            println!("     🎯 Accuracy: {:.1}%", config.accuracy * 100.0);
            println!("     📅 {}", config.created_at);
            println!();
        }
    }
    
    /// Delete a model
    pub fn delete_model(&self, name: &str) -> Result<(), String> {
        let filepath = self.models_dir.join(format!("{}.nova", name));
        if filepath.exists() {
            std::fs::remove_file(&filepath)
                .map_err(|e| format!("Failed to delete model: {}", e))?;
            println!("  🗑️ Deleted model '{}'", name);
            Ok(())
        } else {
            Err(format!("Model '{}' not found", name))
        }
    }
    
    /// Upload model to Hugging Face Hub
    pub fn upload_to_hf(&self, name: &str, hf_repo: &str, token: &str) -> Result<(), String> {
        let filepath = self.models_dir.join(format!("{}.nova", name));
        if !filepath.exists() {
            return Err(format!("Model '{}' not found", name));
        }
        
        // Create Python upload script
        let script = format!(r#"
import json, sys, os, time
try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("pip install huggingface_hub")
    sys.exit(1)

try:
    api = HfApi(token="{token}")
    
    # Ensure repo exists
    api.create_repo(repo_id="{repo}", exist_ok=True)
    time.sleep(2)
    
    # Upload model file
    api.upload_file(
        path_or_fileobj=r"{local_path}",
        path_in_repo="{model_name}.nova",
        repo_id="{repo}",
    )
    
    # Upload a README
    readme = f"""---
title: {{model_name}}
tags:
- nova-core
- post-transformer
- rust
---
# {{model_name}}

Nova Core model - A post-transformer LLM without attention, tokens, or layers.

## Architecture
- Field Dynamics (O(n)) instead of Attention (O(n²))
- Continuous Pulses instead of discrete tokens
- Adaptive Depth Cores instead of fixed layers

## Training
Trained on {{examples}} examples for {{epochs}} epochs.

## Usage
Download and use with Nova Core CLI.
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id="{repo}",
    )
    
    print(f"✅ Model uploaded to https://huggingface.co/{{repo}}")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"#,
            token = token,
            repo = hf_repo,
            local_path = filepath.to_string_lossy().to_string().replace("\\", "\\\\"),
            model_name = name,
        );
        
        let script_path = std::env::temp_dir().join("nova_hf_upload.py");
        std::fs::write(&script_path, &script)
            .map_err(|e| format!("Failed to write script: {}", e))?;
        
        println!("  📤 Uploading '{}' to Hugging Face Hub...", name);
        
        let output = std::process::Command::new("python")
            .arg(&script_path)
            .output()
            .map_err(|e| format!("Failed to run Python: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if output.status.success() {
            println!("  {}", stdout.trim());
            Ok(())
        } else {
            Err(format!("Upload failed: {}", stderr.trim()))
        }
    }
    
    /// Download model from Hugging Face Hub
    pub fn download_from_hf(&self, hf_repo: &str, model_name: &str, token: &str) -> Result<String, String> {
        std::fs::create_dir_all(&self.models_dir)
            .map_err(|e| format!("Failed to create models dir: {}", e))?;
        
        let script = format!(r#"
import json, sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed")
    sys.exit(1)

try:
    local_path = hf_hub_download(
        repo_id="{repo}",
        filename="{model_name}.nova",
        token="{token}" if "{token}" else None,
    )
    print(local_path)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"#,
            repo = hf_repo,
            model_name = model_name,
            token = token,
        );
        
        let script_path = std::env::temp_dir().join("nova_hf_download_model.py");
        std::fs::write(&script_path, &script)
            .map_err(|e| format!("Failed to write script: {}", e))?;
        
        println!("  📥 Downloading '{}' from Hugging Face Hub...", model_name);
        
        let output = std::process::Command::new("python")
            .arg(&script_path)
            .output()
            .map_err(|e| format!("Failed to run Python: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if output.status.success() {
            let downloaded_path = stdout.trim();
            // Copy to models directory
            let dest = self.models_dir.join(format!("{}.nova", model_name));
            std::fs::copy(downloaded_path, &dest)
                .map_err(|e| format!("Failed to copy model: {}", e))?;
            println!("  ✅ Model downloaded to: {}", dest.display());
            Ok(dest.to_string_lossy().to_string())
        } else {
            Err(format!("Download failed: {}", stderr.trim()))
        }
    }
}

impl Default for NovaModelManager {
    fn default() -> Self { Self::new() }
}

/// Get current time as string
fn chrono_now() -> String {
    // Simple timestamp without chrono dependency
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    
    // Simple date calculation
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    
    // Year calculation from days since epoch
    let year = 1970 + (days as f64 / 365.25) as u64;
    let day_of_year = (days % 365) as u64;
    let month = 1 + day_of_year / 31;
    let day = 1 + day_of_year % 31;
    
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", year, month.min(12), day.min(28), hours, minutes, seconds)
}

//! Nova Model Manager - Save, load, and manage Nova Core models
//!
//! Uses binary format: [4 config_len][config_json][4 emb_len][raw_f32_embeddings]
//! Binary format is ~10x faster and 5x smaller than JSON for the 8M float embedding table.

use crate::embedding::NovaEmbedding;
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
    pub created_at: String,
    pub trained_on: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "nova-core".to_string(),
            version: "0.1.0".to_string(),
            description: "Nova Core - Post-Transformer LLM".to_string(),
            dim: 256,
            num_cores: 5,
            core_names: vec!["syntax".into(), "semantic".into(), "memory".into(), "reasoning".into(), "pattern".into()],
            created_at: now_string(),
            trained_on: "none".to_string(),
        }
    }
}

pub struct NovaModelManager {
    pub models_dir: PathBuf,
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

    fn read_config(&self, path: &Path) -> Option<ModelConfig> {
        if let Ok(data) = std::fs::read(path) {
            if data.len() < 4 { return None; }
            let config_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
            if 4 + config_len > data.len() { return None; }
            if let Ok(config_json) = std::str::from_utf8(&data[4..4 + config_len]) {
                if let Ok(config) = serde_json::from_str::<ModelConfig>(config_json) {
                    return Some(config);
                }
            }
        }
        None
    }

    /// Fast binary save: [4 config_len][config_json][4 emb_len][raw_f32_embeddings]
    pub fn save_model(&self, model: &NovaLoom, name: &str) -> Result<String, String> {
        std::fs::create_dir_all(&self.models_dir)
            .map_err(|e| format!("Failed to create models dir: {}", e))?;

        let config = ModelConfig {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            description: format!("Nova Core model '{}'", name),
            dim: model.cores.first().map(|c| c.ssm_stack.dim).unwrap_or(256),
            num_cores: model.cores.len(),
            core_names: model.cores.iter().map(|c| c.name.clone()).collect(),
            created_at: now_string(),
            trained_on: "custom".to_string(),
        };

        let config_json = serde_json::to_string(&config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        let config_bytes = config_json.as_bytes();

        // Convert f32 embedding table to raw bytes
        let emb = &model.embedding.token_embeddings;
        let emb_byte_len = emb.len() * 4; // 4 bytes per f32
        let emb_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(emb.as_ptr() as *const u8, emb_byte_len)
        };

        // Format: [4 config_len][config_json][4 emb_len][emb_bytes]
        let mut output = Vec::with_capacity(4 + config_bytes.len() + 4 + emb_bytes.len());
        output.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(config_bytes);
        output.extend_from_slice(&(emb_byte_len as u32).to_le_bytes());
        output.extend_from_slice(emb_bytes);

        let filepath = self.models_dir.join(format!("{}.nova", name));
        std::fs::write(&filepath, &output)
            .map_err(|e| format!("Failed to write model: {}", e))?;

        let file_size_mb = output.len() / (1024 * 1024);
        println!("  💾 Model saved to: {} ({} MB)", filepath.display(), file_size_mb);
        Ok(filepath.to_string_lossy().to_string())
    }

    /// Fast binary load: parse [4 config_len][config_json][4 emb_len][raw_f32]
    pub fn load_model(&self, name: &str) -> Result<(NovaLoom, HashMap<String, Vec<f32>>), String> {
        let clean_name = name.strip_suffix(".nova").unwrap_or(name);
        let filepath = self.models_dir.join(format!("{}.nova", clean_name));

        let data = std::fs::read(&filepath)
            .map_err(|e| format!("Failed to read model file: {}", e))?;

        if data.len() < 8 {
            return Err("File too small".to_string());
        }

        let config_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let config_start = 4;
        let emb_len_start = config_start + config_len;

        if emb_len_start + 4 > data.len() {
            return Err("Corrupted model file (config extends past file)".to_string());
        }

        let emb_byte_len = u32::from_le_bytes([
            data[emb_len_start], data[emb_len_start+1], 
            data[emb_len_start+2], data[emb_len_start+3]
        ]) as usize;
        let emb_start = emb_len_start + 4;

        if emb_start + emb_byte_len > data.len() {
            return Err("Corrupted model file (embeddings extend past file)".to_string());
        }

        // Parse config
        let config_json = std::str::from_utf8(&data[config_start..emb_len_start])
            .map_err(|_| "Invalid config UTF-8".to_string())?;
        let config: ModelConfig = serde_json::from_str(config_json)
            .map_err(|e| format!("Config parse error: {}", e))?;

        println!("  📂 Loading model '{}'...", clean_name);
        println!("     Dim: {}, Cores: {}", config.dim, config.num_cores);

        let mut loom = NovaLoom::new(config.dim, 32768);

        // Convert raw bytes back to f32 slice
        let emb_count = emb_byte_len / 4;
        if emb_count == loom.embedding.token_embeddings.len() {
            let emb_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    data[emb_start..].as_ptr() as *const f32,
                    emb_count
                )
            };
            loom.embedding.token_embeddings.copy_from_slice(emb_slice);
        }

        println!("  ✅ Model '{}' loaded successfully!", clean_name);
        Ok((loom, HashMap::new()))
    }

    pub fn delete_model(&self, name: &str) -> Result<(), String> {
        let filepath = self.models_dir.join(format!("{}.nova", name));
        if filepath.exists() {
            std::fs::remove_file(&filepath)
                .map_err(|e| format!("Failed to delete model: {}", e))?;
            Ok(())
        } else {
            Err(format!("Model '{}' not found", name))
        }
    }

    pub fn list_models(&self) {
        println!("\n{}", "═".repeat(60));
        println!("📦 AVAILABLE MODELS");
        println!("{}", "═".repeat(60));
        if self.available_models.is_empty() {
            println!("  (No models found. Train one with 'nova train' or 'nova multi-hf-train')");
            return;
        }
        for config in &self.available_models {
            println!("  📁 {} v{}", config.name, config.version);
            println!("     📐 Dim: {}, Cores: {}", config.dim, config.num_cores);
            println!("     📅 {}", config.created_at);
            println!();
        }
    }
}

fn now_string() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let year = 1970 + (days as f64 / 365.25) as u64;
    let day_of_year = (days % 365) as u64;
    let month = 1 + day_of_year / 31;
    let day = 1 + day_of_year % 31;
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", year, month.min(12), day.min(28), hours, minutes, seconds)
}

impl Default for NovaModelManager {
    fn default() -> Self { Self::new() }
}
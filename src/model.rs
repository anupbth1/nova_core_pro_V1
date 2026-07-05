//! Nova Model Manager - Save, load, and manage Nova Core models
//!
//! Features:
//! - Save/load models in .nova format (binary JSON)
//! - List available models
//! - Upload/download models from Hugging Face Hub

use crate::core::NovaCore;
use crate::field::NovaField;
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

/// SSM layer snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmLayerSnapshot {
    pub a_log: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub h: Vec<f32>,
    pub delta: Vec<f32>,
    pub delta_bias: Vec<f32>,
    pub d: Vec<f32>,
    pub ssm_norm_weight: Vec<f32>,
    pub ssm_norm_bias: Vec<f32>,
    pub glu_gate_weight: Vec<f32>,
    pub glu_gate_bias: Vec<f32>,
    pub glu_up_weight: Vec<f32>,
    pub glu_up_bias: Vec<f32>,
    pub glu_down_weight: Vec<f32>,
    pub glu_down_bias: Vec<f32>,
    pub glu_norm_weight: Vec<f32>,
    pub glu_norm_bias: Vec<f32>,
}

/// Core snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSnapshot {
    pub id: usize,
    pub name: String,
    pub internal_state: Vec<f32>,
    pub gate: f32,
    pub output_weight: Vec<f32>,
    pub output_bias: Vec<f32>,
    pub output_norm_weight: Vec<f32>,
    pub output_norm_bias: Vec<f32>,
    pub ssm_layers: Vec<SsmLayerSnapshot>,
}

/// Complete model snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub config: ModelConfig,
    pub cores: Vec<CoreSnapshot>,
    pub field_content: Vec<f32>,
    pub field_momentum: Vec<f32>,
    pub field_ssm: Vec<SsmLayerSnapshot>,
    pub token_embeddings: Vec<f32>,
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

    pub fn with_dir(dir: &str) -> Self {
        let models_dir = PathBuf::from(dir);
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
        let content = std::fs::read_to_string(path).ok()?;
        if let Ok(snapshot) = serde_json::from_str::<ModelSnapshot>(&content) {
            return Some(snapshot.config);
        }
        None
    }

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

        let cores: Vec<CoreSnapshot> = model.cores.iter().map(|c| {
            let ssm_layers = c.ssm_stack.layers.iter().map(|layer| {
                SsmLayerSnapshot {
                    a_log: layer.a_log.clone(),
                    b: layer.b.clone(),
                    c: layer.c.clone(),
                    h: layer.h.clone(),
                    delta: layer.delta.clone(),
                    delta_bias: layer.delta_bias.clone(),
                    d: layer.d.clone(),
                    ssm_norm_weight: layer.ssm_norm_weight.clone(),
                    ssm_norm_bias: layer.ssm_norm_bias.clone(),
                    glu_gate_weight: layer.glu.as_ref().map(|g| g.gate_weight.clone()).unwrap_or_default(),
                    glu_gate_bias: layer.glu.as_ref().map(|g| g.gate_bias.clone()).unwrap_or_default(),
                    glu_up_weight: layer.glu.as_ref().map(|g| g.up_weight.clone()).unwrap_or_default(),
                    glu_up_bias: layer.glu.as_ref().map(|g| g.up_bias.clone()).unwrap_or_default(),
                    glu_down_weight: layer.glu.as_ref().map(|g| g.down_weight.clone()).unwrap_or_default(),
                    glu_down_bias: layer.glu.as_ref().map(|g| g.down_bias.clone()).unwrap_or_default(),
                    glu_norm_weight: layer.glu.as_ref().map(|g| g.norm_weight.clone()).unwrap_or_default(),
                    glu_norm_bias: layer.glu.as_ref().map(|g| g.norm_bias.clone()).unwrap_or_default(),
                }
            }).collect();

            CoreSnapshot {
                id: c.id,
                name: c.name.clone(),
                internal_state: c.internal_state.clone(),
                gate: c.gate,
                output_weight: c.output_weight.clone(),
                output_bias: c.output_bias.clone(),
                output_norm_weight: c.output_norm_weight.clone(),
                output_norm_bias: c.output_norm_bias.clone(),
                ssm_layers,
            }
        }).collect();

        let field_ssm_layer = SsmLayerSnapshot {
            a_log: model.field.ssm.a_log.clone(),
            b: model.field.ssm.b.clone(),
            c: model.field.ssm.c.clone(),
            h: model.field.ssm.h.clone(),
            delta: model.field.ssm.delta.clone(),
            delta_bias: model.field.ssm.delta_bias.clone(),
            d: model.field.ssm.d.clone(),
            ssm_norm_weight: model.field.ssm.ssm_norm_weight.clone(),
            ssm_norm_bias: model.field.ssm.ssm_norm_bias.clone(),
            glu_gate_weight: model.field.ssm.glu.as_ref().map(|g| g.gate_weight.clone()).unwrap_or_default(),
            glu_gate_bias: model.field.ssm.glu.as_ref().map(|g| g.gate_bias.clone()).unwrap_or_default(),
            glu_up_weight: model.field.ssm.glu.as_ref().map(|g| g.up_weight.clone()).unwrap_or_default(),
            glu_up_bias: model.field.ssm.glu.as_ref().map(|g| g.up_bias.clone()).unwrap_or_default(),
            glu_down_weight: model.field.ssm.glu.as_ref().map(|g| g.down_weight.clone()).unwrap_or_default(),
            glu_down_bias: model.field.ssm.glu.as_ref().map(|g| g.down_bias.clone()).unwrap_or_default(),
            glu_norm_weight: model.field.ssm.glu.as_ref().map(|g| g.norm_weight.clone()).unwrap_or_default(),
            glu_norm_bias: model.field.ssm.glu.as_ref().map(|g| g.norm_bias.clone()).unwrap_or_default(),
        };

        let snapshot = ModelSnapshot {
            config,
            cores,
            field_content: model.field.content.clone(),
            field_momentum: model.field.momentum.clone(),
            field_ssm: vec![field_ssm_layer],
            token_embeddings: model.embedding.token_embeddings.clone(),
        };

        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Failed to serialize: {}", e))?;

        let filepath = self.models_dir.join(format!("{}.nova", name));
        std::fs::write(&filepath, &json)
            .map_err(|e| format!("Failed to write model: {}", e))?;

        println!("  💾 Model saved to: {}", filepath.display());
        Ok(filepath.to_string_lossy().to_string())
    }

    pub fn load_model(&self, name: &str) -> Result<(NovaLoom, HashMap<String, Vec<f32>>), String> {
        let clean_name = name.strip_suffix(".nova").unwrap_or(name);
        let filepath = self.models_dir.join(format!("{}.nova", clean_name));

        let json = std::fs::read_to_string(&filepath)
            .map_err(|e| format!("Failed to read model file: {}", e))?;

        let snapshot: ModelSnapshot = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse model: {}", e))?;

        println!("  📂 Loading model '{}'...", clean_name);
        println!("     Dim: {}, Cores: {}", snapshot.config.dim, snapshot.config.num_cores);

        // Build new NovaLoom with saved parameters
        let dim = snapshot.config.dim;
        let mut loom = NovaLoom::new(dim, 32768);

        // Restore token embeddings
        if snapshot.token_embeddings.len() == loom.embedding.token_embeddings.len() {
            loom.embedding.token_embeddings = snapshot.token_embeddings;
        }

        // Restore core parameters
        for (i, core_snap) in snapshot.cores.iter().enumerate() {
            if i < loom.cores.len() {
                let core = &mut loom.cores[i];
                core.internal_state = core_snap.internal_state.clone();
                core.gate = core_snap.gate;
                core.output_weight = core_snap.output_weight.clone();
                core.output_bias = core_snap.output_bias.clone();
                core.output_norm_weight = core_snap.output_norm_weight.clone();
                core.output_norm_bias = core_snap.output_norm_bias.clone();

                // Restore SSM layers
                for (j, layer_snap) in core_snap.ssm_layers.iter().enumerate() {
                    if j < core.ssm_stack.layers.len() {
                        let layer = &mut core.ssm_stack.layers[j];
                        layer.a_log = layer_snap.a_log.clone();
                        layer.b = layer_snap.b.clone();
                        layer.c = layer_snap.c.clone();
                        layer.h = layer_snap.h.clone();
                        layer.delta = layer_snap.delta.clone();
                        layer.delta_bias = layer_snap.delta_bias.clone();
                        layer.d = layer_snap.d.clone();
                        layer.ssm_norm_weight = layer_snap.ssm_norm_weight.clone();
                        layer.ssm_norm_bias = layer_snap.ssm_norm_bias.clone();

                        if let Some(ref mut glu) = layer.glu {
                            glu.gate_weight = layer_snap.glu_gate_weight.clone();
                            glu.gate_bias = layer_snap.glu_gate_bias.clone();
                            glu.up_weight = layer_snap.glu_up_weight.clone();
                            glu.up_bias = layer_snap.glu_up_bias.clone();
                            glu.down_weight = layer_snap.glu_down_weight.clone();
                            glu.down_bias = layer_snap.glu_down_bias.clone();
                            glu.norm_weight = layer_snap.glu_norm_weight.clone();
                            glu.norm_bias = layer_snap.glu_norm_bias.clone();
                        }
                    }
                }
            }
        }

        // Restore field
        loom.field.content = snapshot.field_content;
        loom.field.momentum = snapshot.field_momentum;
        if let Some(layer_snap) = snapshot.field_ssm.first() {
            loom.field.ssm.a_log = layer_snap.a_log.clone();
            loom.field.ssm.b = layer_snap.b.clone();
            loom.field.ssm.c = layer_snap.c.clone();
            loom.field.ssm.h = layer_snap.h.clone();
            loom.field.ssm.delta = layer_snap.delta.clone();
            loom.field.ssm.delta_bias = layer_snap.delta_bias.clone();
            loom.field.ssm.d = layer_snap.d.clone();
            loom.field.ssm.ssm_norm_weight = layer_snap.ssm_norm_weight.clone();
            loom.field.ssm.ssm_norm_bias = layer_snap.ssm_norm_bias.clone();
            if let Some(ref mut glu) = loom.field.ssm.glu {
                glu.gate_weight = layer_snap.glu_gate_weight.clone();
                glu.gate_bias = layer_snap.glu_gate_bias.clone();
                glu.up_weight = layer_snap.glu_up_weight.clone();
                glu.up_bias = layer_snap.glu_up_bias.clone();
                glu.down_weight = layer_snap.glu_down_weight.clone();
                glu.down_bias = layer_snap.glu_down_bias.clone();
                glu.norm_weight = layer_snap.glu_norm_weight.clone();
                glu.norm_bias = layer_snap.glu_norm_bias.clone();
            }
        }

        println!("  ✅ Model '{}' loaded successfully!", clean_name);
        Ok((loom, HashMap::new()))
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
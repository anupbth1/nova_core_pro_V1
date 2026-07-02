//! Nova Dataset Manager - Download, filter, and preprocess datasets from Hugging Face
//!
//! Supports multiple dataset formats:
//! - CSV (.csv) - Comma separated values
//! - JSON (.json) - JSON array of objects
//! - JSONL (.jsonl) - JSON lines format
//! - Parquet (.parquet) - Columnar storage
//! - Text (.txt) - Plain text files
//!
//! NEW FEATURES:
//! - Auto-detect columns: Common patterns like user/assistant, question/answer,
//!   instruction/output, text/label are automatically detected.
//! - Multi-column concatenation: Multiple columns can be merged into one input.
//! - Prompt template: Custom format strings like "User: {user}\nAssistant: {assistant}"

use crate::trainer::TrainingExample;
use rand::Rng;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Dataset format types
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetFormat {
    CSV,
    JSON,
    JSONL,
    Parquet,
    Text,
    Unknown,
}

impl DatasetFormat {
    pub fn from_extension(path: &str) -> Self {
        let path_lower = path.to_lowercase();
        if path_lower.ends_with(".csv") { Self::CSV }
        else if path_lower.ends_with(".jsonl") || path_lower.ends_with(".ndjson") { Self::JSONL }
        else if path_lower.ends_with(".json") { Self::JSON }
        else if path_lower.ends_with(".parquet") { Self::Parquet }
        else if path_lower.ends_with(".txt") { Self::Text }
        else { Self::Unknown }
    }
}

/// Common column name patterns for auto-detection
const INPUT_PATTERNS: &[&str] = &[
    "user", "question", "instruction", "input", "prompt", "text", "sentence",
    "context", "source", "query", "problem", "premise", "hypothesis", "content",
];
const TARGET_PATTERNS: &[&str] = &[
    "assistant", "answer", "output", "response", "target", "label", "completion",
    "reply", "result", "solution", "conclusion", "summary", "translation",
    "label", "category", "class", "sentiment", "score",
];

/// Column mapping configuration
#[derive(Debug, Clone)]
pub struct ColumnMapping {
    /// Column name for input text
    pub input_column: String,
    /// Column name for target text
    pub target_column: String,
    /// Additional columns to concatenate into input (multi-column support)
    pub extra_input_columns: Vec<String>,
    /// Optional prefix to add to input
    pub input_prefix: String,
    /// Optional suffix to add to input
    pub input_suffix: String,
    /// Prompt template: e.g. "User: {user}\nAssistant: {assistant}"
    /// If set, overrides input_column/target_column with template-based extraction
    pub prompt_template: Option<String>,
    /// Whether columns were auto-detected (for display)
    pub auto_detected: bool,
}

impl Default for ColumnMapping {
    fn default() -> Self {
        Self {
            input_column: "text".to_string(),
            target_column: "label".to_string(),
            extra_input_columns: Vec::new(),
            input_prefix: String::new(),
            input_suffix: String::new(),
            prompt_template: None,
            auto_detected: false,
        }
    }
}

impl ColumnMapping {
    /// Auto-detect input and target columns from a list of available column names.
    /// Returns true if both input and target were found.
    pub fn auto_detect(&mut self, available_columns: &[String]) -> bool {
        let col_set: HashSet<&str> = available_columns.iter().map(|s| s.as_str()).collect();
        
        // Helper: find first matching column from a list of patterns
        let find_match = |patterns: &[&str]| -> Option<String> {
            for pattern in patterns {
                if col_set.contains(pattern) {
                    return Some(pattern.to_string());
                }
                // Also check case-insensitive
                for col in &col_set {
                    if col.eq_ignore_ascii_case(pattern) {
                        return Some(col.to_string());
                    }
                }
            }
            None
        };
        
        let input_found = find_match(INPUT_PATTERNS);
        let target_found = find_match(TARGET_PATTERNS);
        
        if let Some(ref input) = input_found {
            self.input_column = input.clone();
        }
        if let Some(ref target) = target_found {
            self.target_column = target.clone();
        }
        
        // If input and target are the same (e.g., both "text"), try harder
        if self.input_column == self.target_column {
            // Try to find a different target column
            for pattern in TARGET_PATTERNS {
                if col_set.contains(pattern) && *pattern != self.input_column {
                    self.target_column = pattern.to_string();
                    break;
                }
            }
        }
        
        let found = self.input_column != "text" || target_found.is_some();
        self.auto_detected = found;
        found
    }
    
    /// Apply prompt template to extract input and target from a JSON object.
    /// Template format: "User: {user}\nAssistant: {assistant}"
    /// {column_name} gets replaced with the value from the JSON object.
    pub fn apply_template(&self, obj: &serde_json::Map<String, serde_json::Value>) -> Option<(String, String)> {
        let template = self.prompt_template.as_ref()?;
        
        // Find all {column_name} placeholders
        let mut result = template.clone();
        let mut input_cols_found = Vec::new();
        let mut target_cols_found = Vec::new();
        
        for (key, value) in obj {
            let placeholder = format!("{{{}}}", key);
            if result.contains(&placeholder) {
                let val_str = value.as_str().unwrap_or("").to_string();
                result = result.replace(&placeholder, &val_str);
                
                // Track which columns were used
                if INPUT_PATTERNS.contains(&key.as_str()) || key == &self.input_column {
                    input_cols_found.push(key.clone());
                }
                if TARGET_PATTERNS.contains(&key.as_str()) || key == &self.target_column {
                    target_cols_found.push(key.clone());
                }
            }
        }
        
        // If template has {assistant} or {output} etc., extract target from there
        // Otherwise use the last column value as target
        let target = if !target_cols_found.is_empty() {
            obj.get(&target_cols_found[0])
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            // Try to find target from remaining columns not used in template
            for pattern in TARGET_PATTERNS {
                let pattern_str = pattern.to_string();
                if let Some(val) = obj.get(&pattern_str).and_then(|v| v.as_str()) {
                    if !result.contains(val) {
                        target_cols_found.push(pattern.to_string());
                        break;
                    }
                }
            }
            String::new()
        };
        
        if result.contains('{') {
            // Some placeholders weren't filled — skip this row
            None
        } else {
            Some((result, target))
        }
    }
}

/// Filter conditions for dataset rows
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// Keep rows where column equals value
    Equals { column: String, value: String },
    /// Keep rows where column contains value
    Contains { column: String, value: String },
    /// Keep rows where column length is > N
    MinLength { column: String, min: usize },
    /// Keep rows where column length is < N
    MaxLength { column: String, max: usize },
    /// Keep rows where column matches regex
    Regex { column: String, pattern: String },
    /// Remove rows with empty values in these columns
    NonEmpty { columns: Vec<String> },
}

/// A single dataset source
#[derive(Debug, Clone)]
pub struct DatasetSource {
    /// Path to dataset file
    pub path: PathBuf,
    /// Format of the dataset
    pub format: DatasetFormat,
    /// Column mapping for this dataset
    pub column_mapping: ColumnMapping,
    /// Filters to apply
    pub filters: Vec<FilterCondition>,
    /// Name of this dataset
    pub name: String,
    /// Max rows to load (0 = all)
    pub max_rows: usize,
}

impl DatasetSource {
    pub fn new(path: &str) -> Self {
        let path_buf = PathBuf::from(path);
        let name = path_buf.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "dataset".to_string());
        
        Self {
            format: DatasetFormat::from_extension(path),
            path: path_buf,
            column_mapping: ColumnMapping::default(),
            filters: Vec::new(),
            name,
            max_rows: 0,
        }
    }
    
    pub fn with_mapping(mut self, input_col: &str, target_col: &str) -> Self {
        self.column_mapping.input_column = input_col.to_string();
        self.column_mapping.target_column = target_col.to_string();
        self
    }
    
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.column_mapping.input_prefix = prefix.to_string();
        self
    }
    
    pub fn with_suffix(mut self, suffix: &str) -> Self {
        self.column_mapping.input_suffix = suffix.to_string();
        self
    }
    
    pub fn with_filter(mut self, filter: FilterCondition) -> Self {
        self.filters.push(filter);
        self
    }
    
    pub fn with_max_rows(mut self, max: usize) -> Self {
        self.max_rows = max;
        self
    }
}

/// Hugging Face dataset reference
#[derive(Debug, Clone)]
pub struct HFDatasetRef {
    /// HF dataset name (e.g., "imdb", "wikitext", "tiny_shakespeare")
    pub name: String,
    /// Subset/config name (e.g., "wikitext-2-raw-v1")
    pub subset: String,
    /// Split (e.g., "train", "test", "validation")
    pub split: String,
    /// Column mapping
    pub column_mapping: ColumnMapping,
    /// Max rows to download
    pub max_rows: usize,
}

impl HFDatasetRef {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            subset: String::new(),
            split: "train".to_string(),
            column_mapping: ColumnMapping::default(),
            max_rows: 1000,
        }
    }
    
    pub fn with_subset(mut self, subset: &str) -> Self {
        self.subset = subset.to_string();
        self
    }
    
    pub fn with_split(mut self, split: &str) -> Self {
        self.split = split.to_string();
        self
    }
    
    pub fn with_mapping(mut self, input_col: &str, target_col: &str) -> Self {
        self.column_mapping.input_column = input_col.to_string();
        self.column_mapping.target_column = target_col.to_string();
        self
    }
    
    pub fn with_max_rows(mut self, max: usize) -> Self {
        self.max_rows = max;
        self
    }
}

/// The Nova Dataset Manager
pub struct NovaDataset {
    /// Loaded training examples
    pub examples: Vec<TrainingExample>,
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
    /// Registered dataset sources
    pub sources: Vec<DatasetSource>,
    /// Hugging Face dataset references
    pub hf_datasets: Vec<HFDatasetRef>,
}

impl NovaDataset {
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            metadata: HashMap::new(),
            sources: Vec::new(),
            hf_datasets: Vec::new(),
        }
    }
    
    /// Add a local file as dataset source
    pub fn add_file(&mut self, path: &str) -> &mut DatasetSource {
        let source = DatasetSource::new(path);
        self.sources.push(source);
        self.sources.last_mut().unwrap()
    }
    
    /// Add a Hugging Face dataset reference
    pub fn add_hf_dataset(&mut self, hf: HFDatasetRef) {
        self.hf_datasets.push(hf);
    }
    
    /// Load and parse a CSV file
    fn load_csv(&self, source: &DatasetSource) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let content = match std::fs::read_to_string(&source.path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ❌ Error reading CSV {}: {}", source.path.display(), e);
                return examples;
            }
        };
        
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() < 2 { return examples; }
        
        // Parse header
        let header: Vec<&str> = lines[0].split(',').map(|s| s.trim().trim_matches('"')).collect();
        let input_idx = header.iter().position(|&h| h == source.column_mapping.input_column);
        let target_idx = header.iter().position(|&h| h == source.column_mapping.target_column);
        
        if input_idx.is_none() || target_idx.is_none() {
            eprintln!("  ❌ Columns '{}/{}' not found in CSV header: {:?}", 
                source.column_mapping.input_column, source.column_mapping.target_column, header);
            return examples;
        }
        let input_idx = input_idx.unwrap();
        let target_idx = target_idx.unwrap();
        
        for (i, line) in lines.iter().enumerate().skip(1) {
            if source.max_rows > 0 && examples.len() >= source.max_rows { break; }
            if line.trim().is_empty() { continue; }
            
            let fields: Vec<&str> = line.split(',').map(|s| s.trim().trim_matches('"')).collect();
            if fields.len() <= input_idx.max(target_idx) { continue; }
            
            let input = format!("{}{}{}", 
                source.column_mapping.input_prefix,
                fields[input_idx],
                source.column_mapping.input_suffix
            );
            let target = fields[target_idx].to_string();
            
            if input.is_empty() || target.is_empty() { continue; }
            
            // Apply filters
            if self.apply_filters(&source.filters, &header, &fields) {
                examples.push(TrainingExample { input, target });
            }
        }
        
        examples
    }
    
    /// Extract input from a JSON object using column mapping.
    /// Supports:
    /// 1. Prompt template: "User: {user}\nAssistant: {assistant}"
    /// 2. Multi-column concatenation: extra_input_columns merged with main input
    /// 3. Standard single column
    fn extract_from_json_obj(&self, obj: &serde_json::Map<String, serde_json::Value>, source: &DatasetSource) -> Option<TrainingExample> {
        let cm = &source.column_mapping;
        
        // 1. If prompt template is set, use it
        if let Some(template) = &cm.prompt_template {
            if let Some((input, target)) = cm.apply_template(obj) {
                if !input.is_empty() && !target.is_empty() {
                    return Some(TrainingExample { input, target });
                }
            }
            return None;
        }
        
        // 2. Get main input column
        let main_input = obj.get(&cm.input_column)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let main_input = match main_input {
            Some(s) => s,
            None => return None,
        };
        
        // 3. Get extra input columns and concatenate
        let mut input_parts = vec![main_input];
        for extra_col in &cm.extra_input_columns {
            if let Some(val) = obj.get(extra_col).and_then(|v| v.as_str()) {
                if !val.is_empty() {
                    input_parts.push(val.to_string());
                }
            }
        }
        
        let input = format!("{}{}{}",
            cm.input_prefix,
            input_parts.join(" "),
            cm.input_suffix
        );
        
        // 4. Get target
        let target = obj.get(&cm.target_column)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();
        
        if input.is_empty() || target.is_empty() {
            None
        } else {
            Some(TrainingExample { input, target })
        }
    }
    
    /// Auto-detect columns from the first JSON object in a file.
    /// Returns the detected column names.
    fn auto_detect_columns_from_json(&self, content: &str, source: &DatasetSource) -> Vec<String> {
        // Try to parse first object to get available columns
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(content) {
            let first_obj = if let Some(arr) = json_val.as_array() {
                arr.first().and_then(|v| v.as_object())
            } else {
                json_val.as_object()
            };
            
            if let Some(obj) = first_obj {
                let columns: Vec<String> = obj.keys().cloned().collect();
                return columns;
            }
        }
        Vec::new()
    }
    
    /// Auto-detect columns from the first line of a JSONL file.
    fn auto_detect_columns_from_jsonl(&self, content: &str) -> Vec<String> {
        if let Some(first_line) = content.lines().next() {
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(first_line.trim()) {
                if let Some(obj) = json_val.as_object() {
                    return obj.keys().cloned().collect();
                }
            }
        }
        Vec::new()
    }
    
    /// Load and parse a JSON file
    fn load_json(&self, source: &DatasetSource) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let content = match std::fs::read_to_string(&source.path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ❌ Error reading JSON {}: {}", source.path.display(), e);
                return examples;
            }
        };
        
        // Auto-detect columns if not explicitly set
        let mut source_mut = source.clone();
        if !source.column_mapping.auto_detected {
            let columns = self.auto_detect_columns_from_json(&content, source);
            if !columns.is_empty() {
                let cols_str: Vec<String> = columns.clone();
                source_mut.column_mapping.auto_detect(&cols_str);
                if source_mut.column_mapping.auto_detected {
                    println!("    🔍 Auto-detected: input='{}', target='{}'",
                        source_mut.column_mapping.input_column,
                        source_mut.column_mapping.target_column);
                }
            }
        }
        
        // Try to parse as JSON array
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(arr) = json_val.as_array() {
                for item in arr {
                    if source.max_rows > 0 && examples.len() >= source.max_rows { break; }
                    if let Some(obj) = item.as_object() {
                        if let Some(ex) = self.extract_from_json_obj(obj, &source_mut) {
                            examples.push(ex);
                        }
                    }
                }
            }
        }
        
        examples
    }
    
    /// Load and parse a JSONL file
    fn load_jsonl(&self, source: &DatasetSource) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let content = match std::fs::read_to_string(&source.path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ❌ Error reading JSONL {}: {}", source.path.display(), e);
                return examples;
            }
        };
        
        // Auto-detect columns if not explicitly set
        let mut source_mut = source.clone();
        if !source.column_mapping.auto_detected {
            let columns = self.auto_detect_columns_from_jsonl(&content);
            if !columns.is_empty() {
                let cols_str: Vec<String> = columns;
                source_mut.column_mapping.auto_detect(&cols_str);
                if source_mut.column_mapping.auto_detected {
                    println!("    🔍 Auto-detected: input='{}', target='{}'",
                        source_mut.column_mapping.input_column,
                        source_mut.column_mapping.target_column);
                }
            }
        }
        
        for line in content.lines() {
            if source.max_rows > 0 && examples.len() >= source.max_rows { break; }
            let line = line.trim();
            if line.is_empty() { continue; }
            
            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(obj) = json_val.as_object() {
                    if let Some(ex) = self.extract_from_json_obj(obj, &source_mut) {
                        examples.push(ex);
                    }
                }
            }
        }
        
        examples
    }
    
    /// Load and parse a plain text file
    fn load_text(&self, source: &DatasetSource) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let content = match std::fs::read_to_string(&source.path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ❌ Error reading text {}: {}", source.path.display(), e);
                return examples;
            }
        };
        
        // Split by double newline for paragraphs
        let paragraphs: Vec<&str> = content.split("\n\n")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        // Create next-sentence prediction pairs
        for i in 0..paragraphs.len().saturating_sub(1) {
            if source.max_rows > 0 && examples.len() >= source.max_rows { break; }
            let input = format!("{}{}", source.column_mapping.input_prefix, paragraphs[i]);
            let target = paragraphs[i + 1].to_string();
            if !input.is_empty() && !target.is_empty() {
                examples.push(TrainingExample { input, target });
            }
        }
        
        examples
    }
    
    /// Apply filter conditions to a row
    fn apply_filters(&self, filters: &[FilterCondition], header: &[&str], fields: &[&str]) -> bool {
        for filter in filters {
            match filter {
                FilterCondition::Equals { column, value } => {
                    if let Some(idx) = header.iter().position(|&h| h == column) {
                        if fields.get(idx).map(|f| f.trim()) != Some(value.as_str()) {
                            return false;
                        }
                    }
                }
                FilterCondition::Contains { column, value } => {
                    if let Some(idx) = header.iter().position(|&h| h == column) {
                        if !fields.get(idx).map(|f| f.contains(value.as_str())).unwrap_or(false) {
                            return false;
                        }
                    }
                }
                FilterCondition::MinLength { column, min } => {
                    if let Some(idx) = header.iter().position(|&h| h == column) {
                        if fields.get(idx).map(|f| f.len() < *min).unwrap_or(true) {
                            return false;
                        }
                    }
                }
                FilterCondition::MaxLength { column, max } => {
                    if let Some(idx) = header.iter().position(|&h| h == column) {
                        if fields.get(idx).map(|f| f.len() > *max).unwrap_or(false) {
                            return false;
                        }
                    }
                }
                FilterCondition::Regex { column, pattern } => {
                    if let Some(idx) = header.iter().position(|&h| h == column) {
                        if let Some(field) = fields.get(idx) {
                            if let Ok(re) = regex_lite::Regex::new(pattern) {
                                if !re.is_match(field) {
                                    return false;
                                }
                            }
                        }
                    }
                }
                FilterCondition::NonEmpty { columns } => {
                    for col in columns {
                        if let Some(idx) = header.iter().position(|&h| h == col) {
                            if fields.get(idx).map(|f| f.trim().is_empty()).unwrap_or(true) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
    
    /// Load all registered datasets
    pub fn load_all(&mut self) -> &[TrainingExample] {
        println!("\n{}", "═".repeat(60));
        println!("📦 LOADING DATASETS");
        println!("{}", "═".repeat(60));
        
        let mut total = 0;
        
        // Load local files
        for source in &self.sources {
            let examples = match source.format {
                DatasetFormat::CSV => self.load_csv(source),
                DatasetFormat::JSON => self.load_json(source),
                DatasetFormat::JSONL => self.load_jsonl(source),
                DatasetFormat::Text => self.load_text(source),
                _ => {
                    eprintln!("  ⚠️ Unsupported format for: {}", source.path.display());
                    Vec::new()
                }
            };
            
            println!("  📄 {}: {} examples", source.name, examples.len());
            total += examples.len();
            self.examples.extend(examples);
        }
        
        // Load Hugging Face datasets (via Python bridge)
        for hf in &self.hf_datasets {
            println!("  🤗 Downloading '{}' (split: {})...", hf.name, hf.split);
            match self.download_hf_dataset(hf) {
                Ok(examples) => {
                    println!("    ✅ Loaded {} examples", examples.len());
                    total += examples.len();
                    self.examples.extend(examples);
                }
                Err(e) => {
                    eprintln!("    ❌ Failed: {}", e);
                }
            }
        }
        
        println!("{}", "─".repeat(60));
        println!("📊 Total: {} training examples loaded", total);
        println!("{}", "═".repeat(60));
        
        &self.examples
    }
    
    /// Download dataset from Hugging Face using Python bridge.
    /// Uses direct HTTP download from Hugging Face Hub API.
    /// Completely bypasses the `datasets` library which has issues with v5+.
    fn download_hf_dataset(&self, hf: &HFDatasetRef) -> Result<Vec<TrainingExample>, String> {
        // Build the Python script using string concatenation to avoid Rust format!() escaping issues
        let name = &hf.name;
        let subset = &hf.subset;
        let split = &hf.split;
        let max_rows = hf.max_rows;
        let input_col = if hf.column_mapping.input_column.is_empty() { "text".to_string() } else { hf.column_mapping.input_column.clone() };
        let target_col = if hf.column_mapping.target_column.is_empty() { "text".to_string() } else { hf.column_mapping.target_column.clone() };
        
        let script = format!(
r#"import json, sys, os, urllib.request, urllib.error, io
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

name = "{name}"
subset = "{subset}"
split = "{split}"
max_rows = {max_rows}
input_col = "{input_col}"
target_col = "{target_col}"

examples = []

def try_fetch(url):
    try:
        req = urllib.request.Request(url, headers={{"User-Agent": "NovaAI/1.0"}})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except Exception:
        return None

def parse_jsonl(data):
    text = data.decode("utf-8")
    items = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items

# Strategy 1: Get dataset info from HF API
api_url = "https://huggingface.co/api/datasets/" + name
info_data = try_fetch(api_url)
if info_data:
    try:
        info = json.loads(info_data)
        siblings = info.get("siblings", [])
        
        # Find parquet and jsonl files
        data_files = []
        for s in siblings:
            rf = s.get("rfilename", "")
            if rf.endswith(".parquet") or rf.endswith(".jsonl"):
                data_files.append(rf)
        
        # Filter by split if possible
        split_files = [f for f in data_files if split in f]
        if not split_files:
            split_files = data_files
        
        for rf in split_files[:5]:
            file_url = "https://huggingface.co/datasets/" + name + "/resolve/main/" + rf
            file_data = try_fetch(file_url)
            if file_data:
                if rf.endswith(".parquet"):
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(io.BytesIO(file_data))
                        col_names = table.column_names
                        # Auto-detect columns if needed
                        detected_input = input_col
                        detected_target = target_col
                        if detected_input not in col_names:
                            for c in col_names:
                                if c in ["text", "user", "question", "input", "prompt", "sentence", "content"]:
                                    detected_input = c
                                    break
                            else:
                                detected_input = col_names[0]
                        if detected_target not in col_names:
                            for c in col_names:
                                if c in ["label", "assistant", "answer", "output", "response", "target", "completion"]:
                                    detected_target = c
                                    break
                            else:
                                detected_target = col_names[-1]
                        
                        for i in range(table.num_rows):
                            if max_rows > 0 and len(examples) >= max_rows:
                                break
                            row = {{col_names[j]: table.column(j)[i].as_py() for j in range(len(col_names))}}
                            input_val = str(row.get(detected_input, ""))
                            target_val = str(row.get(detected_target, ""))
                            if input_val and target_val:
                                examples.append({{"input": input_val, "target": target_val}})
                    except ImportError:
                        pass
                elif rf.endswith(".jsonl"):
                    items = parse_jsonl(file_data)
                    for item in items:
                        if max_rows > 0 and len(examples) >= max_rows:
                            break
                        if isinstance(item, dict):
                            input_val = str(item.get(input_col, item.get("text", "")))
                            target_val = str(item.get(target_col, item.get("text", "")))
                            if input_val and target_val:
                                examples.append({{"input": input_val, "target": target_val}})
                
                if max_rows > 0 and len(examples) >= max_rows:
                    break
    except Exception as e:
        pass

# Strategy 2: Try raw data URLs
if not examples:
    urls = [
        "https://huggingface.co/datasets/" + name + "/raw/main/" + split + "/data.jsonl",
        "https://huggingface.co/datasets/" + name + "/raw/main/data.jsonl",
        "https://huggingface.co/datasets/" + name + "/resolve/main/data.jsonl",
    ]
    for url in urls:
        data = try_fetch(url)
        if data:
            text = data.decode("utf-8")
            if text.strip().startswith("[") or text.strip().startswith("{{"):
                try:
                    if text.strip().startswith("["):
                        items = json.loads(text)
                    else:
                        items = parse_jsonl(data)
                    for item in items:
                        if max_rows > 0 and len(examples) >= max_rows:
                            break
                        if isinstance(item, dict):
                            input_val = str(item.get(input_col, item.get("text", "")))
                            target_val = str(item.get(target_col, item.get("text", "")))
                            if input_val and target_val:
                                examples.append({{"input": input_val, "target": target_val}})
                except Exception:
                    pass
            else:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                for i in range(len(lines) - 1):
                    if max_rows > 0 and len(examples) >= max_rows:
                        break
                    examples.append({{"input": lines[i], "target": lines[i + 1]}})
            if examples:
                break

# Strategy 3: Try huggingface_hub
if not examples:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(repo_id=name, repo_type="dataset")
        data_files = [f for f in files if f.endswith(".jsonl") or f.endswith(".txt")]
        for rf in data_files[:3]:
            try:
                file_path = hf_hub_download(repo_id=name, filename=rf, repo_type="dataset")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if rf.endswith(".jsonl"):
                    items = parse_jsonl(content.encode())
                    for item in items:
                        if max_rows > 0 and len(examples) >= max_rows:
                            break
                        if isinstance(item, dict):
                            input_val = str(item.get(input_col, item.get("text", "")))
                            target_val = str(item.get(target_col, item.get("text", "")))
                            if input_val and target_val:
                                examples.append({{"input": input_val, "target": target_val}})
                elif rf.endswith(".txt"):
                    lines = [l.strip() for l in content.split("\n") if l.strip()]
                    for i in range(len(lines) - 1):
                        if max_rows > 0 and len(examples) >= max_rows:
                            break
                        examples.append({{"input": lines[i], "target": lines[i + 1]}})
            except Exception:
                pass
            if examples:
                break
    except ImportError:
        pass

print(json.dumps(examples))
"#,
            name = hf.name,
            subset = hf.subset,
            split = hf.split,
            max_rows = hf.max_rows,
            input_col = input_col,
            target_col = target_col,
        );
        
        let script_path = std::env::temp_dir().join("nova_hf_direct.py");
        std::fs::write(&script_path, &script).map_err(|e| format!("Failed to write download script: {}", e))?;
        
        let output = std::process::Command::new("python")
            .arg(&script_path)
            .output()
            .map_err(|e| format!("Failed to run Python: {}. Is Python installed?", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if !output.status.success() {
            return Err(format!("Download failed. stdout: {}, stderr: {}", stdout.trim(), stderr.trim()));
        }
        
        // Parse JSON output
        if let Ok(examples) = serde_json::from_str::<Vec<TrainingExample>>(&stdout) {
            if examples.is_empty() {
                Err(format!("No examples could be downloaded for dataset '{}'. The dataset may not have Parquet/JSONL files available on the Hub.", hf.name))
            } else {
                Ok(examples)
            }
        } else {
            Err(format!("Failed to parse Python output: {}", stdout.trim()))
        }
    }
    
    /// Save dataset to JSONL file
    pub fn save_to_jsonl(&self, path: &str) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = std::io::BufWriter::new(file);
        
        for ex in &self.examples {
            let line = serde_json::json!({
                "input": ex.input,
                "target": ex.target
            });
            writeln!(&mut writer, "{}", line).map_err(|e| format!("Failed to write: {}", e))?;
        }
        
        println!("  💾 Saved {} examples to {}", self.examples.len(), path);
        Ok(())
    }
    
    /// Split dataset into train/validation
    pub fn train_val_split(&self, val_ratio: f32) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.examples.len()).collect();
        
        // Shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        let val_count = (self.examples.len() as f32 * val_ratio) as usize;
        let train_count = self.examples.len() - val_count;
        
        let train: Vec<TrainingExample> = indices[..train_count].iter()
            .map(|&i| self.examples[i].clone())
            .collect();
        let val: Vec<TrainingExample> = indices[train_count..].iter()
            .map(|&i| self.examples[i].clone())
            .collect();
        
        (train, val)
    }
    
    /// Print dataset statistics
    pub fn print_stats(&self) {
        println!("\n📊 Dataset Statistics:");
        println!("  Total examples: {}", self.examples.len());
        
        if self.examples.is_empty() { return; }
        
        let avg_input_len: f32 = self.examples.iter().map(|e| e.input.len() as f32).sum::<f32>() / self.examples.len() as f32;
        let avg_target_len: f32 = self.examples.iter().map(|e| e.target.len() as f32).sum::<f32>() / self.examples.len() as f32;
        
        println!("  Avg input length: {:.1} chars", avg_input_len);
        println!("  Avg target length: {:.1} chars", avg_target_len);
        
        // Count unique words
        let mut words = std::collections::HashSet::new();
        for ex in &self.examples {
            for w in ex.input.split_whitespace() {
                words.insert(w.to_lowercase());
            }
            for w in ex.target.split_whitespace() {
                words.insert(w.to_lowercase());
            }
        }
        println!("  Unique words: {}", words.len());
        
        // Show sample
        if let Some(first) = self.examples.first() {
            println!("\n  📝 Sample:");
            println!("    Input:  {}", &first.input[..first.input.len().min(100)]);
            println!("    Target: {}", &first.target[..first.target.len().min(100)]);
        }
    }
}

impl Default for NovaDataset {
    fn default() -> Self { Self::new() }
}

// Helper for writeln macro
use std::io::Write;

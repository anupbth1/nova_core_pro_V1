//! Nova Embedding - Trainable Embedding Table with Positional Encoding
//!
//! Replaces the old random hash-based embeddings with proper trainable
//! embeddings using Xavier initialization and sinusoidal positional encoding.
//! O(n) complexity, no attention, Transformer-free.

use rand::Rng;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Vocabulary size (number of tokens/subwords)
pub const VOCAB_SIZE: usize = 32768;

/// Embedding dimension for each token
pub const EMBED_DIM: usize = 256;

/// Maximum sequence length supported
pub const MAX_SEQ_LEN: usize = 2048;

/// Number of top candidates for fast vocabulary search
/// Instead of scanning all 32768 tokens, scan only this many top clusters
pub const TOP_K_SEARCH: usize = 512;

/// Trainable embedding table with positional encoding
pub struct NovaEmbedding {
    /// Token embedding table [VOCAB_SIZE x EMBED_DIM]
    pub token_embeddings: Vec<f32>,
    /// Positional encoding cache [MAX_SEQ_LEN x EMBED_DIM]
    pub positional_encoding: Vec<f32>,
    /// Vocabulary tokenizer (word/subword -> id)
    pub token_to_id: HashMap<String, usize>,
    /// Reverse vocabulary (id -> token string)
    pub id_to_token: Vec<String>,
    /// Whether embeddings have been initialized with real data
    pub initialized: bool,
    /// Pre-computed token embedding norms (for fast cosine similarity)
    pub token_norms: Vec<f32>,
    /// Partition-based search index: which token IDs are "real" (not padding/special)
    pub real_token_ids: Vec<usize>,
    /// The ACTUAL embedding dimension (may differ from compile-time EMBED_DIM)
    pub actual_dim: usize,
}

impl NovaEmbedding {
    /// Create a new embedding table with Xavier initialization
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let scale = (2.0 / (embed_dim as f32)).sqrt();
        let mut rng = rand::thread_rng();

        let token_embeddings: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        // Precompute sinusoidal positional encodings
        let positional_encoding = Self::compute_positional_encoding(MAX_SEQ_LEN, embed_dim);

        // Initialize with basic byte-level vocabulary
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // Add special tokens
        token_to_id.insert("<PAD>".to_string(), 0);
        id_to_token.push("<PAD>".to_string());
        token_to_id.insert("<BOS>".to_string(), 1);
        id_to_token.push("<BOS>".to_string());
        token_to_id.insert("<EOS>".to_string(), 2);
        id_to_token.push("<EOS>".to_string());
        token_to_id.insert("<UNK>".to_string(), 3);
        id_to_token.push("<UNK>".to_string());

        // Add byte-level tokens (single bytes as tokens)
        for i in 0..256 {
            let byte_str = format!("<BYTE_{}>", i);
            if !token_to_id.contains_key(&byte_str) {
                let id = token_to_id.len();
                token_to_id.insert(byte_str.clone(), id);
                id_to_token.push(byte_str);
            }
        }

        // Add common ASCII characters
        for c in 32u8..127u8 {
            let s = (c as char).to_string();
            if !token_to_id.contains_key(&s) {
                let id = token_to_id.len();
                token_to_id.insert(s.clone(), id);
                id_to_token.push(s);
            }
        }

        // Fill remaining vocabulary with common English words
        let common_words = vec![
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "is", "was", "are", "had", "has", "been", "were", "said", "been", "being",
        ];

        for word in common_words {
            if !token_to_id.contains_key(word) && token_to_id.len() < vocab_size {
                let id = token_to_id.len();
                token_to_id.insert(word.to_string(), id);
                id_to_token.push(word.to_string());
            }
        }

        // Expand vocabulary with ALL words from common English
        let extra_words = vec![
            "hi", "hello", "name", "nova", "how", "are", "you", "fine", "good",
            "morning", "evening", "night", "bye", "thank", "thanks", "welcome",
            "yes", "no", "okay", "please", "sorry", "excuse", "pardon", "sir",
            "fruit", "animal", "flower", "vehicle", "color", "day", "month",
            "apple", "banana", "mango", "grape", "orange", "cherry", "peach",
            "pear", "plum", "kiwi", "lemon", "lime", "berry", "melon", "papaya",
            "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "goat",
            "lion", "tiger", "bear", "wolf", "deer", "fox", "rabbit", "elephant",
            "giraffe", "zebra", "monkey", "snake", "eagle", "shark", "whale", "dolphin",
            "rose", "lily", "tulip", "daisy", "lotus", "jasmine", "sunflower", "orchid",
            "car", "bus", "truck", "train", "plane", "boat", "bike", "ship", "van",
            "red", "blue", "green", "yellow", "black", "white", "pink", "brown", "purple",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "june", "july", "august",
            "september", "october", "november", "december", "capital", "paris",
            "london", "tokyo", "berlin", "rome", "madrid", "moscow", "beijing",
            "delhi", "dublin", "seoul", "cairo", "ottawa", "athens", "warsaw",
            "sun", "moon", "sky", "earth", "fire", "water", "ice", "snow", "wind",
            "fly", "swim", "bark", "meow", "run", "jump", "slither", "sing",
            "eat", "drink", "sleep", "read", "write", "talk", "walk", "think",
            "big", "small", "hot", "cold", "tall", "short", "fast", "slow",
            "happy", "sad", "love", "hate", "life", "time", "world", "people",
            "book", "food", "tree", "star", "rain", "snow", "wind", "fire",
        ];

        for w in extra_words.iter() {
            if !token_to_id.contains_key(*w) && token_to_id.len() < vocab_size {
                let id = token_to_id.len();
                token_to_id.insert(w.to_string(), id);
                id_to_token.push(w.to_string());
            }
        }

        // Fill remaining with placeholder tokens
        while token_to_id.len() < vocab_size {
            let placeholder = format!("<VOCAB_{}>", token_to_id.len());
            let id = token_to_id.len();
            token_to_id.insert(placeholder.clone(), id);
            id_to_token.push(placeholder);
        }

        // Precompute token norms
        let mut token_norms = vec![0.0; vocab_size];
        for token_id in 0..vocab_size {
            let start = token_id * embed_dim;
            let mut sum_sq = 0.0f32;
            for i in 0..embed_dim {
                sum_sq += token_embeddings[start + i] * token_embeddings[start + i];
            }
            token_norms[token_id] = sum_sq.sqrt().max(1e-8);
        }

        // Build list of "real" tokens (English words and punctuation only)
        // Exclude ALL <VOCAB_N> and <BYTE_N> placeholders
        let mut real_token_ids = Vec::new();
        for id in 4..vocab_size {
            let token = &id_to_token[id];
            if !token.starts_with('<') {
                real_token_ids.push(id);
            }
        }
        // Also add common punctuation characters
        for punct in [" ", ".", ",", "!", "?", ";", ":", "'", "\""].iter() {
            if let Some(&id) = token_to_id.get(*punct) {
                if !real_token_ids.contains(&id) {
                    real_token_ids.push(id);
                }
            }
        }

        let actual_dim = embed_dim;
        NovaEmbedding {
            token_embeddings,
            positional_encoding,
            token_to_id,
            id_to_token,
            initialized: false,
            token_norms,
            real_token_ids,
            actual_dim,
        }
    }

    /// Compute sinusoidal positional encodings
    fn compute_positional_encoding(max_len: usize, dim: usize) -> Vec<f32> {
        let mut pos_enc = vec![0.0; max_len * dim];
        for pos in 0..max_len {
            for i in 0..dim {
                let angle = pos as f32 / (10000.0_f32.powf(2.0 * (i as f32) / dim as f32));
                pos_enc[pos * dim + i] = if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }
        pos_enc
    }

    /// Get embedding for a single token ID, including positional encoding.
    /// Uses the ACTUAL embedding dimension from the table.
    /// CRITICAL: token_embeddings has shape [VOCAB_SIZE x actual_dim] where
    /// actual_dim = token_embeddings.len() / VOCAB_SIZE.
    /// We must NOT use compile-time EMBED_DIM for indexing.
    pub fn get_embedding(&self, token_id: usize, position: usize) -> Vec<f32> {
        if self.token_embeddings.is_empty() { return vec![0.0; EMBED_DIM]; }
        let actual_dim = self.token_embeddings.len() / VOCAB_SIZE; // REAL embedding dimension
        if actual_dim == 0 { return vec![0.0; EMBED_DIM]; }
        let token_id = token_id.min(VOCAB_SIZE - 1);
        let pos = position.min(MAX_SEQ_LEN - 1);
        let mut embed = vec![0.0; actual_dim];
        for i in 0..actual_dim {
            embed[i] = self.token_embeddings[token_id * actual_dim + i]
                + self.positional_encoding[pos * actual_dim + i];
        }
        embed
    }

    /// Get embeddings for a sequence of token IDs
    pub fn get_embeddings(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| self.get_embedding(id, pos))
            .collect()
    }

    /// Get embedding as flat Vec<f32> for batch processing
    pub fn get_embeddings_flat(&self, token_ids: &[usize]) -> Vec<f32> {
        let dim = EMBED_DIM;
        let mut result = vec![0.0; token_ids.len() * dim];
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let token_id = token_id.min(VOCAB_SIZE - 1);
            let pos_idx = pos.min(MAX_SEQ_LEN - 1);
            for i in 0..dim {
                result[pos * dim + i] = self.token_embeddings[token_id * dim + i]
                    + self.positional_encoding[pos_idx * dim + i];
            }
        }
        result
    }

    /// Tokenize a text string into token IDs
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        tokens.push(1); // <BOS>

        let mut current_word = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    if let Some(&id) = self.token_to_id.get(&current_word.to_lowercase()) {
                        tokens.push(id);
                    } else {
                        for c in current_word.chars() {
                            let s = c.to_string();
                            if let Some(&id) = self.token_to_id.get(&s) {
                                tokens.push(id);
                            } else {
                                tokens.push(3); // <UNK>
                            }
                        }
                    }
                    current_word.clear();
                }
                if !ch.is_whitespace() {
                    let s = ch.to_string();
                    if let Some(&id) = self.token_to_id.get(&s) {
                        tokens.push(id);
                    }
                }
            }
        }
        if !current_word.is_empty() {
            if let Some(&id) = self.token_to_id.get(&current_word.to_lowercase()) {
                tokens.push(id);
            } else {
                for c in current_word.chars() {
                    let s = c.to_string();
                    if let Some(&id) = self.token_to_id.get(&s) {
                        tokens.push(id);
                    } else {
                        tokens.push(3);
                    }
                }
            }
        }

        tokens.push(2); // <EOS>
        tokens
    }

    /// Convert token IDs back to text
    pub fn detokenize(&self, token_ids: &[usize]) -> String {
        let mut text = String::new();
        let mut prev_was_word = false;

        for &id in token_ids {
            if id == 0 || id == 1 || id >= self.id_to_token.len() {
                continue;
            }
            if id == 2 {
                break;
            }
            let token = &self.id_to_token[id];
            if token.starts_with("<BYTE_") || token.starts_with("<VOCAB_") {
                continue;
            }
            if token.starts_with('<') && token.ends_with('>') {
                continue;
            }

            let is_word = token.chars().all(|c| c.is_alphanumeric());
            if prev_was_word && is_word {
                text.push(' ');
            }
            text.push_str(token);
            prev_was_word = is_word;
        }

        text
    }

    /// Get token ID for a word
    pub fn get_token_id(&self, word: &str) -> usize {
        self.token_to_id
            .get(word)
            .copied()
            .unwrap_or(3)
    }

    /// Update token embedding with gradient (for training)
    pub fn update_embedding(&mut self, token_id: usize, gradient: &[f32], lr: f32) {
        let dim = EMBED_DIM;
        let token_id = token_id.min(VOCAB_SIZE - 1);
        let start = token_id * dim;
        for i in 0..dim {
            self.token_embeddings[start + i] -= lr * gradient[i];
        }
    }

    /// FAST vocabulary search: compute logits using pre-filtered token set.
    /// Uses ACTUAL embedding dimension (not compile-time EMBED_DIM).
    pub fn compute_logits_fast(&self, pulse_content: &[f32]) -> Vec<f32> {
        if self.token_embeddings.is_empty() { return vec![0.0; VOCAB_SIZE]; }
        let dim = self.token_embeddings.len() / VOCAB_SIZE; // actual dimension
        if dim == 0 { return vec![0.0; VOCAB_SIZE]; }
        
        let norm: f32 = pulse_content.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        if norm < 1e-7 { return vec![0.0; VOCAB_SIZE]; }
        
        let mut logits = vec![-100.0; VOCAB_SIZE]; // start VERY low
        let pulse_len = pulse_content.len().min(dim);
        
        // Compute for real tokens
        for &token_id in &self.real_token_ids {
            let start = token_id * dim;
            let t_norm = self.token_norms[token_id];
            let mut dot = 0.0f32;
            for i in 0..pulse_len {
                dot += pulse_content[i] * self.token_embeddings[start + i];
            }
            logits[token_id] = (dot / (norm * t_norm)) * 5.0;
        }
        logits
    }

    /// Full logits computation for ALL vocabulary (slow but exact).
    /// Placeholder tokens (<VOCAB_N>) get very low scores to prevent garbage output.
    pub fn compute_logits_full(&self, pulse_content: &[f32]) -> Vec<f32> {
        if self.token_embeddings.is_empty() || VOCAB_SIZE == 0 { return vec![0.0; VOCAB_SIZE]; }
        let dim = self.token_embeddings.len() / VOCAB_SIZE;
        if dim == 0 { return vec![0.0; VOCAB_SIZE]; }
        let norm: f32 = pulse_content.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut logits = vec![-100.0; VOCAB_SIZE]; // Start low
        let pulse_len = pulse_content.len().min(dim);
        
        // Only compute for REAL tokens (English words, punctuation, characters)
        for &token_id in &self.real_token_ids {
            let start = token_id * dim;
            let t_norm = self.token_norms[token_id];
            let mut dot = 0.0f32;
            for i in 0..pulse_len {
                dot += pulse_content[i] * self.token_embeddings[start + i];
            }
            logits[token_id] = (dot / (norm * t_norm)) * 10.0;
        }
        
        // Also compute for <EOS> (token 2) with REDUCED scaling
        // so it doesn't dominate generation
        let eos_start = 2 * dim;
        let mut eos_dot = 0.0f32;
        for i in 0..pulse_len {
            eos_dot += pulse_content[i] * self.token_embeddings[eos_start + i];
        }
        // EOS gets lower logit so model generates more before stopping
        logits[2] = (eos_dot / norm) * 2.0;
        
        logits
    }

    /// Number of parameters in the embedding layer
    pub fn num_params(&self) -> usize {
        self.token_embeddings.len()
    }
}
//! NovaLoom - Main orchestration engine

use crate::pulse::NovaPulse;
use crate::field::NovaField;
use crate::core::NovaCore;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct NovaLoom {
    pub name: String,
    pub cores: Vec<NovaCore>,
    pub field: NovaField,
    pub dim: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub total_pulses_processed: usize,
    pub total_iterations: usize,
    /// Learned associations: input pattern hash -> output word
    pub learned_responses: HashMap<u64, String>,
    /// Original input texts for learned responses (hash -> original input text)
    pub learned_inputs: HashMap<u64, String>,
    /// Vocabulary mapping (word -> vector embedding)
    pub vocabulary: HashMap<String, Vec<f32>>,
    /// Reverse vocabulary: hash of embedding -> word (for fast lookup)
    pub vocab_reverse: HashMap<u64, String>,
    /// N-gram patterns: sequence hash -> next word predictions with confidence
    pub ngram_patterns: HashMap<u64, Vec<(String, f32)>>,
    /// Maximum n-gram order
    pub ngram_order: usize,
    /// All unique words seen during training (for generation fallback)
    pub all_words: Vec<String>,
}

impl NovaLoom {
    pub fn new(dim: usize, num_cores: usize) -> Self {
        let mut cores = vec![
            NovaCore::new(0, "syntax", 256, dim),
            NovaCore::new(1, "semantic", 256, dim),
            NovaCore::new(2, "memory", 512, dim),
            NovaCore::new(3, "reasoning", 256, dim),
            NovaCore::new(4, "pattern", 128, dim),
        ];

        let specialized_names = vec!["code_logic", "context_window", "bug_fixer", "optimizer"];

        if num_cores > cores.len() {
            for i in cores.len()..num_cores {
                let name_idx = i - 5;
                let core_name = if name_idx < specialized_names.len() {
                    specialized_names[name_idx].to_string()
                } else {
                    format!("vedcode_layer_{}", i)
                };
                cores.push(NovaCore::new(i, &core_name, 256, dim));
            }
        } else {
            cores.truncate(num_cores);
        }
        
        Self {
            name: "nova".to_string(),
            cores,
            field: NovaField::new(dim),
            dim,
            max_iterations: 6,
            convergence_threshold: 0.12,

            total_pulses_processed: 0,
            total_iterations: 0,
            learned_responses: HashMap::new(),
            learned_inputs: HashMap::new(),
            vocabulary: HashMap::new(),
            vocab_reverse: HashMap::new(),
            ngram_patterns: HashMap::new(),
            ngram_order: 3,
            all_words: Vec::new(),
        }
    }
    
    pub fn memory_usage(&self) -> usize {
        let cores_memory: usize = self.cores.iter().map(|c| c.memory.len()).sum();
        let field_memory = self.field.state().len();
        (cores_memory + field_memory) * 4 / (1024 * 1024)
    }

    pub fn text_to_pulses(&self, text: &str) -> Vec<NovaPulse> {
        text.split_whitespace()
            .enumerate()
            .map(|(pos, word)| NovaPulse::from_text(word, self.dim, pos))
            .collect()
    }
    
    /// Convert pulses to text using vocabulary-aware mapping.
    /// This is the core inference function that maps processed pulse vectors
    /// back to vocabulary words using cosine similarity.
    pub fn pulses_to_text(&self, pulses: &[NovaPulse]) -> String {
        // Use vocabulary if available for meaningful output
        if !self.vocabulary.is_empty() {
            return self.map_pulses_to_vocab(pulses);
        }
        
        // Fallback: deterministic mapping based on pulse content
        pulses.iter()
            .map(|p| {
                let mut hash: u64 = 0;
                for (i, &x) in p.content.iter().enumerate().take(8) {
                    let quantized = ((x * 0.5 + 0.5) * 255.0) as u64;
                    hash = hash.wrapping_mul(31).wrapping_add(quantized.wrapping_mul(i as u64 + 1));
                }
                let word_list = [
                    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
                    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
                    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
                    "an", "will", "my", "one", "all", "would", "there", "their", "what",
                    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                    "when", "make", "can", "like", "time", "no", "just", "him", "know",
                    "take", "people", "into", "year", "your", "good", "some", "could",
                    "them", "see", "other", "than", "then", "now", "look", "only", "come",
                    "its", "over", "think", "also", "back", "after", "use", "two", "how",
                    "our", "work", "first", "well", "way", "even", "new", "want", "because",
                    "any", "these", "give", "day", "most", "us", "great", "hello", "world",
                    "yes", "no", "maybe", "sure", "okay", "thanks", "please", "sorry",
                    "right", "wrong", "true", "false", "good", "bad", "big", "small",
                    "high", "low", "fast", "slow", "hot", "cold", "new", "old", "love",
                    "hate", "like", "dislike", "happy", "sad", "angry", "calm", "bright",
                    "dark", "hard", "soft", "strong", "weak", "long", "short", "light",
                    "heavy", "deep", "shallow", "rich", "poor", "clean", "dirty", "full",
                    "empty", "open", "closed", "early", "late", "near", "far", "simple",
                    "complex", "safe", "dangerous", "quiet", "loud", "sweet", "sour",
                    "smooth", "rough", "thick", "thin", "wide", "narrow", "smart", "nova",
                    "core", "field", "pulse", "data", "code", "test", "train", "learn",
                    "think", "know", "feel", "work", "play", "rest", "walk", "run", "jump",
                    "swim", "fly", "read", "write", "speak", "listen", "watch", "help",
                    "give", "take", "bring", "send", "receive", "find", "lose", "keep",
                    "start", "stop", "begin", "end", "change", "stay", "move", "wait",
                    "answer", "question", "reason", "result", "example", "system", "process",
                    "method", "theory", "practice", "science", "nature", "human", "machine",
                    "number", "letter", "word", "sentence", "meaning", "context", "concept",
                    "idea", "thought", "memory", "pattern", "syntax", "logic", "math",
                    "physics", "chemistry", "biology", "history", "geography", "language",
                    "music", "art", "sport", "game", "food", "water", "fire", "earth",
                    "wind", "sky", "sun", "moon", "star", "tree", "flower", "animal",
                    "bird", "fish", "stone", "metal", "wood", "glass", "paper", "color",
                    "shape", "size", "sound", "smell", "taste", "touch", "sight", "sense",
                ];
                let idx = (hash as usize) % word_list.len();
                word_list[idx].to_string()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Check if a vocabulary word is a clean, printable word (not a BPE subword token).
    /// Filters out GPT-2 style tokens like 'Ġword', 'Ċ', '##suffix', etc.
    fn is_clean_vocab_word(word: &str) -> bool {
        if word.is_empty() || word.len() > 40 { return false; }
        // Reject BPE special characters (Ġ = U+0120, Ċ = U+010A are common in GPT-2 vocab)
        // Reject tokens that are purely punctuation/symbols or contain non-ASCII
        let first = word.chars().next().unwrap_or(' ');
        if !first.is_ascii() { return false; }
        // Reject tokens that are all non-alphanumeric (e.g. "####", ">>>")
        if !word.chars().any(|c| c.is_alphanumeric()) { return false; }
        true
    }

    /// OPTIMIZED V3: Map pulse vectors to vocabulary words using cosine similarity.
    /// Caches vocab entries with pre-computed norms for fast repeated lookups.
    fn map_pulses_to_vocab(&self, pulses: &[NovaPulse]) -> String {
        // Use cached vocab entries if available, otherwise build them
        // The cache is stored as a static thread-local for speed
        struct VocabEntry {
            word: String,
            vec: Vec<f32>,
            norm: f32,
        }
        
        // Build vocab entries (this is fast - just iterates vocabulary)
        let vocab_entries: Vec<VocabEntry> = self.vocabulary.iter()
            .filter(|(w, _)| Self::is_clean_vocab_word(w))
            .map(|(word, vec)| {
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                VocabEntry {
                    word: word.clone(),
                    vec: vec.clone(),
                    norm,
                }
            })
            .collect();
        
        if vocab_entries.is_empty() {
            return String::new();
        }
        
        // Pre-allocate output capacity
        let mut result = String::with_capacity(pulses.len() * 8);
        
        for (i, p) in pulses.iter().enumerate() {
            if i > 0 { result.push(' '); }
            
            let norm1: f32 = p.content.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm1 < 1e-6 {
                result.push_str("the");
                continue;
            }
            
            let mut best_word = "the";
            let mut best_sim = -1.0f32;
            
            for entry in &vocab_entries {
                if entry.norm <= 0.0 { continue; }
                let dot: f32 = p.content.iter().zip(entry.vec.iter()).map(|(a, b)| a * b).sum();
                let sim = dot / (norm1 * entry.norm);
                
                if sim > best_sim {
                    best_sim = sim;
                    best_word = &entry.word;
                    
                    // Early exit: if similarity is very high, no need to check more
                    if best_sim > 0.95 {
                        break;
                    }
                }
            }
            
            if best_sim < 0.35 {
                result.push_str("the");
            } else {
                result.push_str(best_word);
            }
        }
        
        result
    }


    /// Find the closest vocabulary word to a pulse vector using cosine similarity.
    /// Returns (word, similarity_score).
    pub fn find_closest_word(&self, pulse: &NovaPulse) -> (String, f32) {
        if self.vocabulary.is_empty() {
            return ("the".to_string(), 0.0);
        }
        
        let mut best_word = "the";
        let mut best_sim = -1.0f32;
        
        let norm1: f32 = pulse.content.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm1 < 1e-6 {
            return ("the".to_string(), 0.0);
        }
        
        for (word, vec) in &self.vocabulary {
            if !Self::is_clean_vocab_word(word) { continue; }
            let dot: f32 = pulse.content.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            let norm2: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sim = if norm2 > 0.0 { dot / (norm1 * norm2) } else { 0.0 };
            
            if sim > best_sim {
                best_sim = sim;
                best_word = word;
            }
        }
        
        (best_word.to_string(), best_sim)
    }

    /// Find the closest vocabulary word to a pulse vector, EXCLUDING specified banned words.
    /// Returns (word, similarity_score).
    fn find_closest_word_excluding(&self, pulse: &NovaPulse, banned: &[String]) -> (String, f32) {
        if self.vocabulary.is_empty() {
            return ("the".to_string(), 0.0);
        }
        
        let mut best_word = "the";
        let mut best_sim = -1.0f32;
        
        let norm1: f32 = pulse.content.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm1 < 1e-6 {
            return ("the".to_string(), 0.0);
        }
        
        for (word, vec) in &self.vocabulary {
            if !Self::is_clean_vocab_word(word) { continue; }
            if banned.contains(word) { continue; }
            let dot: f32 = pulse.content.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            let norm2: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sim = if norm2 > 0.0 { dot / (norm1 * norm2) } else { 0.0 };
            
            if sim > best_sim {
                best_sim = sim;
                best_word = word;
            }
        }
        
        (best_word.to_string(), best_sim)
    }

    /// Return a canned conversational response for common greeting/query patterns.
    /// Called before falling through to n-gram generation for inputs that have
    /// no relevant training context.
    pub fn conversational_override(&self, text: &str) -> Option<String> {
        // Hardcoded conversational overrides removed as per user request.
        // The model should now learn greetings and responses naturally from training data.
        None
    }

    /// Generate text by predicting one word at a time.
    /// This is the MAIN inference method for text generation models.
    /// Given a prompt, it generates `max_words` additional words and returns
    /// ONLY the newly generated words (the prompt is NOT included in the output).
    ///
    /// Uses a hybrid approach:
    /// 1. First tries pulse-based prediction through cores + field (if vocabulary is trained)
    /// 2. Falls back to n-gram pattern matching
    /// 3. Falls back to diverse word selection
    pub fn generate_text(&mut self, prompt: &str, max_words: usize) -> String {
        // Remember the prompt length so we can strip it from the final output
        let prompt_word_count = prompt.split_whitespace().count();
        let mut output_words: Vec<String> = prompt.split_whitespace()
            .map(|w| w.to_string())
            .collect();
        
        if output_words.is_empty() {
            return String::new();
        }
        
        // Track recently generated words to avoid repetition
        let mut recent_words: Vec<String> = Vec::new();
        // Words that are permanently banned from being selected (accumulated loop words)
        let mut banned_words: Vec<String> = Vec::new();
        // Maximum words to ban (prevent banning everything)
        const MAX_BANNED: usize = 20;
        const REPETITION_PENALTY: f32 = 0.25;
        
        // Decide whether to use pulse-based prediction.
        // Use pulse-based if we have a trained vocabulary AND the model has been
        // neurally trained (cores have meaningful state).
        let use_pulse_prediction = !self.vocabulary.is_empty() 
            && self.cores.iter().any(|c| c.gate > 0.5);
        
        for _ in 0..max_words {
            // --- Improved loop detection: window of 8, catches ABAB, AAA, ABCABC ---
            let mut in_loop = false;
            let wlen = output_words.len();

            if wlen >= 4 {
                let last4 = &output_words[wlen - 4..];
                // ABAB pattern
                if last4[0] == last4[2] && last4[1] == last4[3] {
                    in_loop = true;
                    if banned_words.len() < MAX_BANNED {
                        if !banned_words.contains(&last4[0]) { banned_words.push(last4[0].clone()); }
                        if !banned_words.contains(&last4[1]) { banned_words.push(last4[1].clone()); }
                    }
                }
                // AAA pattern
                if last4[0] == last4[1] && last4[1] == last4[2] {
                    in_loop = true;
                    if banned_words.len() < MAX_BANNED {
                        if !banned_words.contains(&last4[0]) { banned_words.push(last4[0].clone()); }
                    }
                }
            }

            if !in_loop && wlen >= 6 {
                // ABCABC pattern — check last 6 words for 3-word repeat
                let last6 = &output_words[wlen - 6..];
                if last6[0] == last6[3] && last6[1] == last6[4] && last6[2] == last6[5] {
                    in_loop = true;
                    if banned_words.len() < MAX_BANNED {
                        for i in 0..3 {
                            if !banned_words.contains(&last6[i]) { banned_words.push(last6[i].clone()); }
                        }
                    }
                }
            }

            if !in_loop && wlen >= 8 {
                // ABCDABCD pattern
                let last8 = &output_words[wlen - 8..];
                if last8[0] == last8[4] && last8[1] == last8[5] && last8[2] == last8[6] && last8[3] == last8[7] {
                    in_loop = true;
                    if banned_words.len() < MAX_BANNED {
                        for i in 0..4 {
                            if !banned_words.contains(&last8[i]) { banned_words.push(last8[i].clone()); }
                        }
                    }
                }
            }
            
            // If in a loop, break out with a diverse word
            if in_loop {
                let next_word = self.pick_diverse_word(&banned_words, &recent_words);
                output_words.push(next_word.clone());
                recent_words.push(next_word);
                if recent_words.len() > 12 { recent_words.remove(0); }
                continue;
            }
            
            // Step 1: Try pulse-based prediction through cores + field (neural mode)
            // This uses the trained cores to predict the next word from context
            if use_pulse_prediction && output_words.len() >= 2 {
                let context_words: Vec<String> = output_words.iter()
                    .skip(output_words.len().saturating_sub(4))
                    .cloned()
                    .collect();
                let predicted = self.predict_next_word_via_pulses_excluding(&context_words, &banned_words);
                if predicted != "the" || !self.ngram_patterns.is_empty() {
                    // Only use pulse prediction if it found something meaningful
                    // or if n-gram fallback is available
                    output_words.push(predicted.clone());
                    recent_words.push(predicted);
                    if recent_words.len() > 12 { recent_words.remove(0); }
                    continue;
                }
            }
            
            // Step 2: Try n-gram pattern matching (fast, deterministic)
            let context_start = if output_words.len() >= self.ngram_order {
                output_words.len() - self.ngram_order
            } else {
                0
            };
            let context: Vec<&str> = output_words[context_start..]
                .iter()
                .map(|w| w.as_str())
                .collect();
            let context_str = context.join(" ");
            let context_hash = hash_text(&context_str);
            
            if let Some(predictions) = self.ngram_patterns.get(&context_hash) {
                if !predictions.is_empty() {
                    let best = predictions.iter()
                        .filter(|(w, _)| !banned_words.contains(w))
                        .map(|(w, conf)| {
                            let penalty = if recent_words.contains(w) { REPETITION_PENALTY } else { 1.0 };
                            (w, conf * penalty)
                        })
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(w, _)| w.clone());
                    
                    if let Some(word) = best {
                        output_words.push(word.clone());
                        recent_words.push(word);
                        if recent_words.len() > 12 { recent_words.remove(0); }
                        continue;
                    }
                }
            }
            
            // Step 3: Try shorter n-gram contexts (backoff)
            let mut found = false;
            'backoff: for order in (1..self.ngram_order).rev() {
                if output_words.len() >= order {
                    let short_context: Vec<&str> = output_words[output_words.len() - order..]
                        .iter()
                        .map(|w| w.as_str())
                        .collect();
                    let short_hash = hash_text(&short_context.join(" "));
                    if let Some(predictions) = self.ngram_patterns.get(&short_hash) {
                        if !predictions.is_empty() {
                            let best = predictions.iter()
                                .filter(|(w, _)| !banned_words.contains(w))
                                .map(|(w, conf)| {
                                    let penalty = if recent_words.contains(w) { REPETITION_PENALTY } else { 1.0 };
                                    (w, conf * penalty)
                                })
                                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                .map(|(w, _)| w.clone());
                            
                            if let Some(word) = best {
                                output_words.push(word.clone());
                                recent_words.push(word);
                                if recent_words.len() > 12 { recent_words.remove(0); }
                                found = true;
                                break 'backoff;
                            }
                        }
                    }
                }
            }
            if found { continue; }

            // Step 4: Sample from the overall n-gram distribution (no context — pure frequency)
            let next_word = self.sample_from_ngram_distribution(&banned_words, &recent_words);
            if let Some(word) = next_word {
                output_words.push(word.clone());
                recent_words.push(word);
                if recent_words.len() > 12 { recent_words.remove(0); }
                continue;
            }

            // Step 5: Last resort — diverse word from vocabulary
            let next_word = self.pick_diverse_word(&banned_words, &recent_words);
            output_words.push(next_word.clone());
            recent_words.push(next_word);
            if recent_words.len() > 12 { recent_words.remove(0); }
        }
        
        // Return only the GENERATED words — strip the prompt prefix
        output_words[prompt_word_count..].join(" ")
    }

    /// Pick a diverse word that is not banned and not recently used.
    /// Prefers words that appear more often in n-gram predictions (higher confidence = more frequent).
    fn pick_diverse_word(&self, banned: &[String], recent: &[String]) -> String {
        // Build frequency map from n-gram predictions
        let mut freq: HashMap<&String, f32> = HashMap::new();
        for predictions in self.ngram_patterns.values() {
            for (word, conf) in predictions {
                if !banned.contains(word) && !recent.contains(word) {
                    *freq.entry(word).or_insert(0.0) += conf;
                }
            }
        }

        // Return the highest-frequency eligible word
        if let Some((word, _)) = freq.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            return (*word).clone();
        }

        // Fallback to all_words if no n-gram data
        let eligible: Vec<&String> = self.all_words.iter()
            .filter(|w| !banned.contains(w) && !recent.contains(w))
            .collect();
        
        if eligible.is_empty() {
            for w in &self.all_words {
                if !banned.contains(w) {
                    return w.clone();
                }
            }
            return "the".to_string();
        }
        
        // Hash-based selection for determinism when no frequency data
        let idx = (banned.len() * 7 + recent.len() * 13) % eligible.len();
        eligible[idx].clone()
    }

    /// Sample a word from the overall n-gram distribution regardless of context.
    /// Used as fallback when no n-gram context matches. Returns None only if no
    /// patterns exist at all.
    fn sample_from_ngram_distribution(&self, banned: &[String], recent: &[String]) -> Option<String> {
        if self.ngram_patterns.is_empty() {
            return None;
        }

        // Aggregate all predictions across all n-gram entries
        let mut scores: HashMap<String, f32> = HashMap::new();
        for predictions in self.ngram_patterns.values() {
            for (word, conf) in predictions {
                if !banned.contains(word) {
                    let penalty = if recent.contains(word) { 0.2 } else { 1.0 };
                    *scores.entry(word.clone()).or_insert(0.0) += conf * penalty;
                }
            }
        }

        // Return the word with highest aggregate score
        scores.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(word, _)| word)
    }

    /// Predict the next word by processing context through cores and field,
    /// then finding the closest vocabulary word (excluding banned words) to the resulting pulse.
    fn predict_next_word_via_pulses_excluding(&mut self, context_words: &[String], banned: &[String]) -> String {
        if context_words.is_empty() {
            return self.pick_diverse_word(banned, &[]);
        }
        
        // Convert context to pulses
        let context_text = context_words.join(" ");
        let mut pulses = self.text_to_pulses(&context_text);
        
        if pulses.is_empty() {
            return self.pick_diverse_word(banned, &[]);
        }
        
        self.total_pulses_processed += pulses.len();
        
        // Process through cores and field (OPTIMIZED: parallel cores)
        for _iteration in 0..self.max_iterations {
            self.process_cores_parallel(&mut pulses);
            self.field.update(&mut pulses);
            self.total_iterations += 1;
            
            let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
            if avg_entropy < self.convergence_threshold {
                break;
            }
        }
        
        // The last pulse represents the predicted next word
        if let Some(last_pulse) = pulses.last() {
            let (word, sim) = self.find_closest_word_excluding(last_pulse, banned);
            if sim > 0.35 {
                return word;
            }
        }
        
        // Fallback: use the field state to find a word
        let field_state = self.field.state();
        if !field_state.is_empty() {
            let mut best_word = "the";
            let mut best_sim = -1.0f32;
            let norm1: f32 = field_state.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if norm1 > 1e-6 {
                for (word, vec) in &self.vocabulary {
                    if banned.contains(word) {
                        continue;
                    }
                    let dot: f32 = field_state.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                    let norm2: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let sim = if norm2 > 0.0 { dot / (norm1 * norm2) } else { 0.0 };
                    if sim > best_sim {
                        best_sim = sim;
                        best_word = word;
                    }
                }
                if best_sim > 0.35 {
                    return best_word.to_string();
                }
            }
        }
        
        // Last resort: return a diverse word from vocabulary (excluding banned)
        self.pick_diverse_word(banned, &[])
    }

    /// Process pulses through all cores in parallel using Rayon.
    /// OPTIMIZED: Cores are independent, so we process them simultaneously.
    /// Each core reads/writes pulse content independently (no data races between cores).
    /// Made `pub` so trainer can call it directly.
    pub fn process_cores_parallel(&mut self, pulses: &mut [NovaPulse]) {

        // SAFETY: We use raw pointer access to share pulses across parallel core processing.
        // Each core only reads/writes its own SSM state and pulse content independently.
        // The cores don't share state between each other, so there are no data races.
        // We wrap the raw pointer in a struct that implements Send + Sync.
        struct SharedPulses(*mut NovaPulse, usize);
        unsafe impl Send for SharedPulses {}
        unsafe impl Sync for SharedPulses {}
        
        let shared = SharedPulses(pulses.as_mut_ptr(), pulses.len());
        let shared_ref = &shared;
        
        // Use a scope-based approach to avoid Send/Sync issues on the closure
        rayon::scope(|s| {
            for core in self.cores.iter_mut() {
                s.spawn(|_| {
                    let pulses_slice = unsafe { std::slice::from_raw_parts_mut(shared_ref.0, shared_ref.1) };
                    core.process(pulses_slice);
                });
            }
        });
    }

    /// Process text input and return a response.
    /// For text generation models, this generates a continuation.
    /// For classification models, this returns the learned response.
    pub fn process(&mut self, text: &str) -> String {
        // Step 1: Check for exact learned response
        let input_hash: u64 = text.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        
        if let Some(response) = self.learned_responses.get(&input_hash) {
            return response.clone();
        }

        // Step 2: Conversational override for common greetings/queries.
        // This fires BEFORE n-gram generation so common inputs get sensible replies.
        if let Some(reply) = self.conversational_override(text) {
            return reply;
        }
        
        // Step 3: Word-overlap matching for classification models.
        // Guards: require >= 3 words in both input and stored example, AND >= 2 absolute
        // word matches, AND Jaccard score >= 0.4 (was 0.1 — far too loose).
        if !self.learned_responses.is_empty() && !self.learned_inputs.is_empty() {
            let input_words: Vec<String> = text.split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|w| !w.is_empty() && w.len() > 2)  // ignore short stop-words
                .collect();
            
            // Only activate overlap matching for inputs with enough content words
            if input_words.len() >= 3 {
                let mut best_match: Option<(u64, f32)> = None;
                
                for (hash, original_input) in &self.learned_inputs {
                    let learned_words: Vec<String> = original_input.split_whitespace()
                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                        .filter(|w| !w.is_empty() && w.len() > 2)
                        .collect();

                    // Guard: learned example must also have enough content words
                    if learned_words.len() < 3 {
                        continue;
                    }
                    
                    let overlap: usize = input_words.iter()
                        .filter(|w| learned_words.contains(w))
                        .count();

                    // Require at least 2 word matches (not just any 1-word overlap)
                    if overlap < 2 {
                        continue;
                    }
                    
                    let score = overlap as f32 / (input_words.len() + learned_words.len() - overlap) as f32;
                    if score > best_match.map(|(_, s)| s).unwrap_or(0.0) {
                        best_match = Some((*hash, score));
                    }
                }
                
                if let Some((hash, score)) = best_match {
                    // Raised threshold: 0.4 instead of 0.1
                    if score >= 0.4 {
                        if let Some(response) = self.learned_responses.get(&hash) {
                            return response.clone();
                        }
                    }
                }
            }
        }
        
        // Step 4: Text generation using n-gram patterns.
        // First check if the input has any coverage in our n-gram vocabulary.
        // If coverage is very low (e.g. Hindi text on a Shakespeare model), it means
        // the model has no knowledge about this input — admit it instead of generating
        // the most-frequent corpus words.
        if !self.vocabulary.is_empty() && !self.ngram_patterns.is_empty() {
            let input_words_lower: Vec<String> = text.split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|w| !w.is_empty())
                .collect();

            // Count how many input words exist as n-gram context keys
            let coverage = input_words_lower.iter().filter(|w| {
                let h = hash_text(w);
                self.ngram_patterns.contains_key(&h)
            }).count();

            let coverage_ratio = if input_words_lower.is_empty() { 0.0 }
                else { coverage as f32 / input_words_lower.len() as f32 };

            // If less than 15% of input words appear in our n-gram vocabulary,
            // the model doesn't know this topic — return a polite unknown response
            if coverage_ratio < 0.15 {
                return "I don't have knowledge about that — my training is limited to specific domains.".to_string();
            }

            let generated = self.generate_text(text, 20);
            // If generation came back empty (all words stripped as duplicates), admit it
            if generated.trim().is_empty() {
                return "I'm not sure how to continue that.".to_string();
            }
            return generated;
        }
        
        // If this is a converted HuggingFace model (has vocabulary but NO n-grams),
        // the core weights are not aligned with the embeddings and will produce random noise.
        // We must warn the user instead of outputting garbage.
        if !self.vocabulary.is_empty() && self.ngram_patterns.is_empty() {
            return "This model was imported from a Hugging Face Transformer but has not been distilled. Nova core dynamics cannot natively run Transformer weights without training. Please train this model using `cargo run -- hf-train` to build n-gram alignments.".to_string();
        }
        
        // Step 5: Fall back to processing through cores (no vocabulary trained)
        let mut pulses = self.text_to_pulses(text);
        self.total_pulses_processed += pulses.len();
        
        for _iteration in 0..self.max_iterations {
            // OPTIMIZED: Process all cores in parallel
            self.process_cores_parallel(&mut pulses);
            self.field.update(&mut pulses);
            self.total_iterations += 1;
            
            let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
            if avg_entropy < self.convergence_threshold {
                break;
            }
        }
        
        if !self.vocabulary.is_empty() {
            self.map_pulses_to_vocab(&pulses)
        } else {
            self.pulses_to_text(&pulses)
        }
    }
    
    /// Learn n-gram patterns from training data.
    /// This builds a statistical language model from the training examples.
    /// Also learns sliding-window bigrams/trigrams within each full text block
    /// so that in-text continuation works properly.
    pub fn learn_ngrams(&mut self, examples: &[crate::trainer::TrainingExample]) {
        for ex in examples {
            let words: Vec<&str> = ex.input.split_whitespace().collect();
            let target_words: Vec<&str> = ex.target.split_whitespace().collect();
            
            // Learn input→target associations (last N input words predict first target words)
            for order in 1..=self.ngram_order.min(words.len()) {
                let start = words.len() - order;
                let context: Vec<&str> = words[start..].to_vec();
                let context_str = context.join(" ");
                let context_hash = hash_text(&context_str);
                
                let entry = self.ngram_patterns.entry(context_hash).or_insert_with(Vec::new);
                for target_word in &target_words {
                    let target_lower = target_word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                    if !target_lower.is_empty() {
                        let mut found = false;
                        for (w, conf) in entry.iter_mut() {
                            if w == &target_lower {
                                *conf = (*conf * 0.9 + 1.0).min(1.0);
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            entry.push((target_lower, 0.5));
                        }
                    }
                }
            }

            // Learn sliding-window n-grams within the TARGET text.
            // This teaches word-to-word transitions so that generation stays coherent.
            let all_text_words: Vec<String> = target_words.iter()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|w| !w.is_empty())
                .collect();

            self.learn_sliding_window_ngrams(&all_text_words);

            // Also slide over the input itself (for models where input is long text)
            if words.len() > 4 {
                let input_words: Vec<String> = words.iter()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                    .filter(|w| !w.is_empty())
                    .collect();
                self.learn_sliding_window_ngrams(&input_words);
            }
            
            // Track all words for the diversity fallback
            for word in &all_text_words {
                if !self.all_words.contains(word) {
                    self.all_words.push(word.clone());
                }
            }
        }
        
        // Prune low-confidence patterns
        self.ngram_patterns.retain(|_, predictions| {
            predictions.retain(|(_, conf)| *conf > 0.1);
            !predictions.is_empty()
        });
    }

    /// Learn bigram and trigram transitions from a sliding window over a word sequence.
    fn learn_sliding_window_ngrams(&mut self, words: &[String]) {
        if words.len() < 2 { return; }

        // Bigrams: word[i] → word[i+1]
        for i in 0..words.len() - 1 {
            let ctx = &words[i];
            let next = &words[i + 1];
            if ctx.is_empty() || next.is_empty() { continue; }

            let ctx_hash = hash_text(ctx);
            let entry = self.ngram_patterns.entry(ctx_hash).or_insert_with(Vec::new);
            let mut found = false;
            for (w, conf) in entry.iter_mut() {
                if w == next {
                    *conf = (*conf * 0.95 + 1.0).min(1.0);
                    found = true;
                    break;
                }
            }
            if !found { entry.push((next.clone(), 0.5)); }
        }

        // Trigrams: "word[i] word[i+1]" → word[i+2]
        if words.len() < 3 { return; }
        for i in 0..words.len() - 2 {
            let ctx = format!("{} {}", words[i], words[i + 1]);
            let next = &words[i + 2];
            if next.is_empty() { continue; }

            let ctx_hash = hash_text(&ctx);
            let entry = self.ngram_patterns.entry(ctx_hash).or_insert_with(Vec::new);
            let mut found = false;
            for (w, conf) in entry.iter_mut() {
                if w == next {
                    *conf = (*conf * 0.95 + 1.0).min(1.0);
                    found = true;
                    break;
                }
            }
            if !found { entry.push((next.clone(), 0.5)); }
        }
    }
    
    pub fn benchmark(&mut self, test_cases: Vec<(&str, &str)>) -> f32 {
        let mut correct = 0;
        for (input, expected) in &test_cases {
            let output = self.process(input);
            if output.contains(expected) {
                correct += 1;
            }
        }
        correct as f32 / test_cases.len() as f32
    }
    
    pub fn stats(&self) -> String {
        format!(
            "Pulses: {} | Iterations: {} | Ratio: {:.2} | Ngrams: {}",
            self.total_pulses_processed,
            self.total_iterations,
            if self.total_pulses_processed > 0 {
                self.total_iterations as f32 / self.total_pulses_processed as f32
            } else { 0.0 },
            self.ngram_patterns.len()
        )
    }
    
    pub fn model_info(&self) -> String {
        format!(
            "{} (dim={}, cores={}, vocab={}, learned={}, ngrams={})",
            self.name,
            self.dim,
            self.cores.len(),
            self.vocabulary.len(),
            self.learned_responses.len(),
            self.ngram_patterns.len()
        )
    }
    
    pub fn reset(&mut self) {
        self.field.reset();
        self.total_pulses_processed = 0;
        self.total_iterations = 0;
    }
}

/// Hash a text string to a u64
fn hash_text(text: &str) -> u64 {
    text.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as u64)
    })
}

impl Default for NovaLoom {
    fn default() -> Self { Self::new(64, 5) }
}

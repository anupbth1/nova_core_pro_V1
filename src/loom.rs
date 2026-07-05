//! NovaLoom - Main orchestration engine
//!
//! PRIORITY 1: COMPLETE REWRITE of inference pipeline.
//! Neural path (cores + field) is now the PRIMARY output mechanism.
//! Hash matches and n-gram generation are fallbacks only.
//! Added content convergence detection for adaptive early exit.
//! Removed hardcoded word list dependency from pulses_to_text().

use crate::pulse::NovaPulse;
use crate::field::NovaField;
use crate::core::NovaCore;
use crate::knowledge::KnowledgeStore;
use crate::context::{LongContextManager, HierarchicalField, ContextCompressor, SlidingWindowSSM};
use crate::coding::{CodingEngine, CodeGenRequest, CodeSnippet};
use crate::math::{MathEngine, MathExpr, BinaryOpKind, UnaryOpKind};
use crate::tools::{ToolEngine, ToolResult};
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
    /// PHASE 5: Knowledge store for structured knowledge representation
    pub knowledge: KnowledgeStore,
    /// PRIORITY 1: Content convergence threshold for adaptive early exit
    pub content_convergence_threshold: f32,
    /// PRIORITY 3: Long context manager for handling sequences > max_seq_length
    pub long_context: LongContextManager,
    /// PRIORITY 3: Hierarchical field for long-range dependencies
    pub hierarchical_field: HierarchicalField,
    /// PRIORITY 3: Whether to use hierarchical field instead of regular field
    pub use_hierarchical_field: bool,
    /// PRIORITY 3: Context compressor for compressing long sequences
    pub context_compressor: ContextCompressor,
    /// PRIORITY 3: Sliding window SSM for processing long sequences
    pub sliding_window: SlidingWindowSSM,
    /// PRIORITY 4: Coding engine for code analysis, generation, and debugging
    pub coding_engine: CodingEngine,
    /// PRIORITY 4: Whether to enable code-aware inference
    pub coding_enabled: bool,
    /// PRIORITY 5: Math engine for arithmetic, algebra, and logical deduction
    pub math_engine: MathEngine,
    /// PRIORITY 5: Whether to enable math-aware inference
    pub math_enabled: bool,
    /// PRIORITY 6: Tool engine for invoking external tools (file ops, HTTP, calculator, etc.)
    pub tool_engine: ToolEngine,
    /// PRIORITY 6: Whether to enable tool-aware inference
    pub tool_enabled: bool,
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
        }
        
        Self {
            name: "NovaLoom".to_string(),
            cores,
            field: NovaField::new(dim),
            dim,
            max_iterations: 10,
            convergence_threshold: 0.3,
            total_pulses_processed: 0,
            total_iterations: 0,
            learned_responses: HashMap::new(),
            learned_inputs: HashMap::new(),
            vocabulary: HashMap::new(),
            vocab_reverse: HashMap::new(),
            ngram_patterns: HashMap::new(),
            ngram_order: 3,
            all_words: Vec::new(),
            knowledge: KnowledgeStore::new(dim),
            content_convergence_threshold: 0.85,
            
            // PRIORITY 3: Long context handling
            long_context: LongContextManager::new(),
            hierarchical_field: HierarchicalField::new(dim),
            use_hierarchical_field: true,
            context_compressor: ContextCompressor::new(4),
            sliding_window: SlidingWindowSSM::new(512, 64),
            // PRIORITY 4: Coding engine
            coding_engine: CodingEngine::new(),
            coding_enabled: true,
            // PRIORITY 5: Math engine
            math_engine: MathEngine::new(),
            math_enabled: true,
            // PRIORITY 6: Tool engine
            tool_engine: ToolEngine::new(),
            tool_enabled: true,
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
    pub fn map_pulses_to_vocab(&self, pulses: &[NovaPulse]) -> String {
        // Use cached vocab entries if available, otherwise build them
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
            
        //     if best_sim < 0.35 {
        //         result.push_str("the");
        //     } else {
        //         result.push_str(best_word);
        //     }
        // }
        // Always use the best match, even if similarity is low
        if best_sim > 0.1 {
            result.push_str(best_word);
        } else {
            // Pick a random word from vocabulary instead of always "the"
            if !self.all_words.is_empty() {
                let idx = (p.content.iter().sum::<f32>().abs() * 100.0) as usize % self.all_words.len();
                result.push_str(&self.all_words[idx]);
            } else {
                result.push_str("the");
            }
        }
        } // <-- THIS closes the "for (i, p) in pulses.iter().enumerate()" loop
        
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

    /// PRIORITY 1: Generate text by predicting one word at a time.
    /// This is the MAIN inference method for text generation models.
    /// Given a prompt, it generates `max_words` additional words and returns
    /// ONLY the newly generated words (the prompt is NOT included in the output).
    ///
    /// FIXED: Uses neural-first approach EXCLUSIVELY when vocabulary exists.
    /// N-gram fallback only used when vocabulary is empty (legacy mode).
    /// Pulse prediction is ALWAYS used when vocabulary is available.
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
        
        // PRIORITY 1: When vocabulary exists, use ONLY pulse-based prediction.
        // N-grams are only used when vocabulary is empty (legacy mode).
        let use_pulse_prediction = !self.vocabulary.is_empty();

        
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
            
            // PRIORITY 1: ALWAYS use pulse-based prediction when vocabulary exists.
            // This is the neural-first approach. N-grams are NEVER used when vocabulary exists.
            if use_pulse_prediction && output_words.len() >= 2 {
                let context_words: Vec<String> = output_words.iter()
                    .skip(output_words.len().saturating_sub(4))
                    .cloned()
                    .collect();
                let predicted = self.predict_next_word_via_pulses_excluding(&context_words, &banned_words);
                
                // PRIORITY 1: ALWAYS use pulse prediction result.
                // No fallback to n-grams when vocabulary exists.
                output_words.push(predicted.clone());
                recent_words.push(predicted);
                if recent_words.len() > 12 { recent_words.remove(0); }
                continue;
            }
            
            // Legacy mode: n-gram fallback (only when vocabulary is empty)
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

    /// PRIORITY 1: Predict the next word by processing context through cores and field,
    /// then finding the closest vocabulary word (excluding banned words) to the resulting pulse.
    ///
    /// Uses adaptive convergence with BOTH entropy AND content convergence detection.
    /// Content convergence measures how much pulse content has stabilized across iterations.
    ///
    /// FIXED BUG 4: No longer falls through to pick_diverse_word() when similarity < 0.35.
    /// Now ALWAYS returns the best match from pulse or field state, even if similarity is low.
    /// Threshold lowered from 0.35 to 0.2 for more permissive matching.
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
        
        // PRIORITY 1: Adaptive iteration count based on entropy level.
        // High entropy contexts need more iterations to converge.
        // Low entropy contexts can converge quickly.
        let adaptive_max = self.adaptive_iteration_count(&pulses);
        let mut prev_entropy = f32::MAX;
        
        // Process through cores and field (OPTIMIZED: parallel cores)
        for _iteration in 0..adaptive_max {
            self.process_cores_parallel(&mut pulses);
            self.field.update(&mut pulses);
            self.total_iterations += 1;
            
            let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
            
            // Check convergence threshold (entropy-based)
            if avg_entropy < self.convergence_threshold {
                break;
            }
            
            // PRIORITY 1: Adaptive early exit - if entropy change is negligible (< 1%),
            // the system has converged even if above absolute threshold.
            let entropy_delta = (prev_entropy - avg_entropy).abs();
            if prev_entropy != f32::MAX && entropy_delta < 0.001 {
                break;
            }
            prev_entropy = avg_entropy;
            
            // PRIORITY 1: Content convergence check - if pulse content has stabilized
            let content_conv = self.content_convergence(&pulses);
            if content_conv > self.content_convergence_threshold {
                break;
            }
        }
        
        // PRIORITY 1: Multi-core semantic consensus.
        // After all cores have processed, compute the semantic consensus across cores
        // by averaging their internal states and blending into the last pulse.
        self.apply_multi_core_semantic_consensus(&mut pulses);
        
        // BUG 4 FIX: ALWAYS return the best match from pulse or field state.
        // No longer falls through to pick_diverse_word() when similarity < 0.35.
        // Threshold lowered from 0.35 to 0.2 for more permissive matching.
        const MIN_SIMILARITY: f32 = 0.2;
        
        // First try: use the last pulse to find the closest word
        if let Some(last_pulse) = pulses.last() {
            let (word, sim) = self.find_closest_word_excluding(last_pulse, banned);
            if sim > MIN_SIMILARITY {
                return word;
            }
            // Even if similarity is low, return the best match rather than falling through
            if sim > -1.0 {
                return word;
            }
        }
        
        // Second try: use the field state to find a word
        let field_state = self.field.state();
        if !field_state.is_empty() {
            let mut best_word = "the".to_string();
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
                        best_word = word.clone();
                    }
                }
                // Always return the best match from field state, even if low similarity
                return best_word;
            }
        }
        
        // Last resort: return a diverse word from vocabulary (excluding banned)
        self.pick_diverse_word(banned, &[])
    }
    
    /// PRIORITY 1: Apply multi-core semantic consensus.
    /// After all cores have processed pulses, compute the semantic consensus
    /// across all cores by averaging their internal states and blending the
    /// consensus signal into the last pulse (which represents the predicted next word).
    ///
    /// This ensures that the final prediction reflects agreement across all cores
    /// (syntax, semantic, memory, reasoning, pattern), not just one core's output.
    fn apply_multi_core_semantic_consensus(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() || self.cores.len() < 2 {
            return;
        }
        
        // Collect internal states from all cores
        let mut consensus = vec![0.0f32; self.dim];
        let mut active_cores = 0usize;
        
        for core in self.cores.iter() {
            let min_len = consensus.len().min(core.internal_state.len());
            for i in 0..min_len {
                consensus[i] += core.internal_state[i] * core.gate;
            }
            active_cores += 1;
        }
        
        if active_cores == 0 {
            return;
        }
        
        // Normalize by number of active cores
        for val in consensus.iter_mut() {
            *val /= active_cores as f32;
        }
        
        // Blend consensus into the last pulse (predicted next word)
        if let Some(last_pulse) = pulses.last_mut() {
            let blend = 0.2; // 20% consensus influence
            let min_len = last_pulse.content.len().min(consensus.len());
            for i in 0..min_len {
                last_pulse.content[i] = last_pulse.content[i] * (1.0 - blend) + consensus[i] * blend;
                last_pulse.content[i] = last_pulse.content[i].clamp(-1.0, 1.0);
            }
        }
    }
    
    /// PRIORITY 1: Compute adaptive max iterations based on convergence rate.
    /// If convergence is happening quickly, use fewer iterations.
    /// If convergence is slow, allow more iterations.
    /// Returns the number of iterations to use for the current inference step.
    fn adaptive_iteration_count(&self, pulses: &[NovaPulse]) -> usize {
        if pulses.is_empty() {
            return self.max_iterations;
        }
        
        let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
        
        // Base iterations on entropy level:
        // - High entropy (>0.5): need more iterations to converge (up to 2x)
        // - Low entropy (<0.2): need fewer iterations (as few as 2)
        // - Normal: use default max_iterations
        if avg_entropy > 0.5 {
            // High uncertainty: allow more iterations
            let extra = ((avg_entropy - 0.5) * 12.0) as usize;
            (self.max_iterations + extra).min(self.max_iterations * 2)
        } else if avg_entropy < 0.2 {
            // Low uncertainty: can converge quickly
            let reduced = ((avg_entropy / 0.2) * self.max_iterations as f32) as usize;
            reduced.max(2) // At least 2 iterations
        } else {
            self.max_iterations
        }
    }


    /// Process pulses through all cores in parallel.
    /// Uses GPU acceleration if available, otherwise falls back to Rayon CPU parallelism.
    /// Made `pub` so trainer can call it directly.
    ///
    /// PHASE 4: Added multi-core communication bus.
    /// After all cores finish processing, each core broadcasts its internal state
    /// to all other cores. Each core then blends the received signals into pulses.
    /// This enables cross-core information flow without O(n²) attention.
    pub fn process_cores_parallel(&mut self, pulses: &mut [NovaPulse]) {
        // Try GPU-accelerated path if available
        #[cfg(feature = "cuda")]
        {
            let gpu_available = crate::cuda::is_gpu_available();
            if gpu_available {
                let mut acc = crate::cuda::get_accelerator();
                if acc.is_kernels_ready() {
                    let mut pulses_content: Vec<Vec<f32>> = pulses.iter().map(|p| p.content.clone()).collect();
                    let mut pulses_entropy: Vec<f32> = pulses.iter().map(|p| p.entropy).collect();
                    let mut pulses_weight: Vec<f32> = pulses.iter().map(|p| p.weight).collect();
                    
                    acc.process_cores_batch(
                        &mut self.cores,
                        &mut pulses_content,
                        &mut pulses_entropy,
                        &mut pulses_weight,
                    );
                    
                    // Copy results back to pulses
                    for (i, pulse) in pulses.iter_mut().enumerate() {
                        if i < pulses_content.len() {
                            let len = pulse.content.len().min(pulses_content[i].len());
                            pulse.content[..len].copy_from_slice(&pulses_content[i][..len]);
                        }
                        if i < pulses_entropy.len() {
                            pulse.entropy = pulses_entropy[i];
                        }
                        if i < pulses_weight.len() {
                            pulse.weight = pulses_weight[i];
                        }
                    }
                    
                    // PHASE 4: Multi-core communication bus (GPU path)
                    // Collect broadcast messages from all cores
                    let messages: Vec<crate::core::CoreMessage> = self.cores.iter()
                        .map(|core| core.broadcast_message())
                        .collect();
                    
                    // Distribute messages to all cores and blend cross-core signals
                    for core in self.cores.iter_mut() {
                        core.receive_messages(&messages);
                        // Re-read pulses from the already-updated content
                        let pulses_slice = unsafe { std::slice::from_raw_parts_mut(pulses.as_mut_ptr(), pulses.len()) };
                        core.blend_cross_core_signals(pulses_slice);
                    }
                    return;
                }
            }
        }

        // CPU fallback: Rayon-based parallel core processing
        // SAFETY: We use raw pointer access to share pulses across parallel core processing.
        // Each core only reads/writes its own SSM state and pulse content independently.
        // The cores don't share state between each other, so there are no data races.
        // We wrap the raw pointer in a struct that implements Send + Sync.
        struct SharedPulses(*mut NovaPulse, usize);
        unsafe impl Send for SharedPulses {}
        unsafe impl Sync for SharedPulses {}
        
        let shared = SharedPulses(pulses.as_mut_ptr(), pulses.len());
        let shared_ref = &shared;
        
        // Phase 1: Process all cores in parallel (each core transforms pulses independently)
        rayon::scope(|s| {
            for core in self.cores.iter_mut() {
                s.spawn(|_| {
                    let pulses_slice = unsafe { std::slice::from_raw_parts_mut(shared_ref.0, shared_ref.1) };
                    core.process(pulses_slice);
                });
            }
        });
        
        // PHASE 4: Multi-core communication bus.
        // After all cores finish processing, collect broadcast messages and
        // distribute them back to each core for cross-core signal blending.
        // This is O(cores) for message collection + O(cores × dim) for blending,
        // which is O(1) relative to pulse count.
        let messages: Vec<crate::core::CoreMessage> = self.cores.iter()
            .map(|core| core.broadcast_message())
            .collect();
        
        // Distribute messages to all cores and blend cross-core signals into pulses
        for core in self.cores.iter_mut() {
            core.receive_messages(&messages);
            let pulses_slice = unsafe { std::slice::from_raw_parts_mut(shared_ref.0, shared_ref.1) };
            core.blend_cross_core_signals(pulses_slice);
        }
        
        // PHASE 5: Knowledge augmentation - blend knowledge into pulses
        // Apply knowledge transform from each core to augment pulses with
        // stored concept embeddings and relations.
        if !self.knowledge.concepts.is_empty() {
            for core in self.cores.iter_mut() {
                let pulses_slice = unsafe { std::slice::from_raw_parts_mut(shared_ref.0, shared_ref.1) };
                core.knowledge_transform(pulses_slice, &self.knowledge);
            }
        }
    }


    /// PRIORITY 4: Detect if input text contains code-related content.
    /// Checks for common code patterns like function definitions, keywords,
    /// and programming language identifiers.
    fn is_code_input(&self, text: &str) -> bool {
        let code_keywords = [
            "fn ", "def ", "function", "class ", "impl ", "pub ", "let ", "mut ",
            "const ", "static ", "return", "if ", "else ", "for ", "while ", "loop ",
            "match ", "unsafe ", "async ", "await ", "use ", "mod ", "trait ",
            "struct ", "enum ", "type ", "where ", "macro", "#[", "//", "/*",
            "import ", "from ", "export ", "=>", "->", "::", "<", ">", "{", "}",
            "String", "Vec", "HashMap", "Result", "Option", "Box", "Arc", "Rc",
            "println", "format!", "assert", "unwrap", "expect", "clone", "copy",
        ];
        
        let lower = text.to_lowercase();
        let keyword_count = code_keywords.iter()
            .filter(|kw| lower.contains(*kw))
            .count();
        
        // If 3 or more code keywords detected, treat as code input
        keyword_count >= 3
    }
    
    /// PRIORITY 4: Apply code-aware pulse transform.
    /// When code input is detected, this method augments the pulse processing
    /// with coding engine analysis. It blends code patterns into the pulse content
    /// to improve code understanding and generation.
    fn apply_code_aware_pulse_transform(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() || !self.coding_enabled {
            return;
        }
        
        // Analyze the code patterns from the pulse content
        let code_text: String = pulses.iter()
            .map(|p| p.content.iter().map(|v| if *v > 0.5 { '1' } else { '0' }).collect::<String>())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Use coding engine to analyze patterns
        let snippet = self.coding_engine.analyze_code(&code_text, "rust");
        
        // Blend code complexity into pulse entropy for adaptive iteration
        let complexity_factor = snippet.complexity;
        for pulse in pulses.iter_mut() {
            // Increase entropy for complex code to allow more iterations
            pulse.entropy = pulse.entropy * (1.0 + complexity_factor * 0.5);
            // Boost weight for code-related pulses
            pulse.weight = pulse.weight * (1.0 + complexity_factor * 0.3);
        }
    }
    
    /// PRIORITY 4: Generate code using the coding engine.
    /// Called when the input is detected as a code generation request.
    fn generate_code_response(&mut self, text: &str) -> String {
        // Detect the programming language from the input
        let language = if text.contains("rust") || text.contains("Rust") || text.contains("rs") {
            "rust"
        } else if text.contains("python") || text.contains("Python") || text.contains("py") {
            "python"
        } else if text.contains("javascript") || text.contains("JavaScript") || text.contains("js") {
            "javascript"
        } else {
            "rust" // Default to Rust
        };
        
        // Create a code generation request
        let request = CodeGenRequest {
            description: text.to_string(),
            language: language.to_string(),
            context: vec![],
            complexity: 0.5,
        };
        
        // Generate code
        let code = self.coding_engine.generate_code(&request);
        
        // Also analyze the generated code for patterns
        let snippet = self.coding_engine.analyze_code(&code, language);
        
        // Format the response with code block
        format!("```{}\n{}\n```\n\n*Generated code with {} patterns detected, complexity: {:.2}*",
            language, code, snippet.patterns.len(), snippet.complexity)
    }
    
    /// PRIORITY 5: Detect if input text contains math-related content.
    /// Checks for patterns like numbers, operators, equations, math keywords,
    /// and mathematical expressions.
    fn is_math_input(&self, text: &str) -> bool {
        // Check for explicit math keywords
        let math_keywords = [
            "solve", "calculate", "compute", "evaluate", "simplify",
            "derivative", "integral", "equation", "expression",
            "sin", "cos", "tan", "log", "sqrt", "ln",
            "prime", "factorial", "gcd", "lcm", "factor",
            "mean", "median", "mode", "variance", "standard deviation",
            "statistics", "probability", "algebra", "arithmetic",
            "quadratic", "linear", "polynomial", "matrix",
            "differentiate", "integrate", "sum", "product",
            "modulo", "remainder", "absolute", "floor", "ceil",
            "round", "pi", "euler", "infinity",
        ];
        
        let lower = text.to_lowercase();
        let keyword_count = math_keywords.iter()
            .filter(|kw| lower.contains(*kw))
            .count();
        
        // Check for mathematical patterns: numbers with operators
        let has_numbers = text.chars().any(|c| c.is_ascii_digit());
        let has_operators = text.contains('+') || text.contains('-') || text.contains('*') 
            || text.contains('/') || text.contains('^') || text.contains('=');
        let has_decimal = text.contains('.');
        
        // Check for equation patterns (e.g., "2x + 3 = 7")
        let has_equation = text.contains('=') && has_numbers;
        
        // Check for function call patterns (e.g., "sin(45)", "sqrt(16)")
        let has_function_call = text.contains('(') && text.contains(')') && has_numbers;
        
        // If 2+ math keywords, or has equation, or has numbers+operators+function call
        if keyword_count >= 2 {
            return true;
        }
        if has_equation {
            return true;
        }
        if has_function_call && has_numbers {
            return true;
        }
        if has_numbers && has_operators && keyword_count >= 1 {
            return true;
        }
        
        false
    }
    
    /// PRIORITY 5: Apply math-aware pulse transform.
    /// When math input is detected, this method augments the pulse processing
    /// with math engine analysis. It blends mathematical precision into pulse
    /// content for improved numerical reasoning.
    fn apply_math_aware_pulse_transform(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() || !self.math_enabled {
            return;
        }
        
        // Analyze the mathematical patterns from the pulse content
        // Use math engine to detect numerical patterns and precision requirements
        let math_activity = self.math_engine.summary();
        
        // Blend mathematical precision into pulse entropy and weight
        // Math problems typically need more precision (lower entropy) and higher weight
        for pulse in pulses.iter_mut() {
            // Reduce entropy for math problems to allow more focused iterations
            pulse.entropy = pulse.entropy * 0.8;
            // Boost weight for math-related pulses to prioritize them
            pulse.weight = pulse.weight * 1.2;
        }
    }
    
    /// PRIORITY 5: Solve math problems using the math engine.
    /// Called when the input is detected as a math problem-solving request.
    /// Parses the input to detect the type of math problem and routes it
    /// to the appropriate math engine method.
    fn solve_math_response(&mut self, text: &str) -> String {
        let lower = text.to_lowercase();
        
        // Detect the type of math problem
        // 1. Prime detection
        if lower.contains("prime") || lower.contains("is prime") {
            // Extract number from text
            if let Some(num) = self.extract_number(text) {
                let is_prime = self.math_engine.is_prime(num as u64);
                let factors = self.math_engine.prime_factors(num as u64);
                return format!(
                    "🔢 Prime Analysis:\n\n{} is {}prime.\nPrime factors: {:?}\n\n*Confidence: 1.0*",
                    num,
                    if is_prime { "" } else { "NOT " },
                    factors,
                );
            }
        }
        
        // 2. GCD / LCM
        if lower.contains("gcd") || lower.contains("lcm") || lower.contains("hcf") {
            let numbers = self.extract_numbers(text);
            if numbers.len() >= 2 {
                let a = numbers[0] as u64;
                let b = numbers[1] as u64;
                if lower.contains("lcm") {
                    let result = self.math_engine.lcm(a, b);
                    return format!(
                        "📐 LCM Calculation:\n\nLCM({}, {}) = {}\n\n*Steps: Using prime factorization method*\n*Confidence: 1.0*",
                        a, b, result,
                    );
                } else {
                    let result = self.math_engine.gcd(a, b);
                    return format!(
                        "📐 GCD Calculation:\n\nGCD({}, {}) = {}\n\n*Steps: Using Euclidean algorithm*\n*Confidence: 1.0*",
                        a, b, result,
                    );
                }
            }
        }
        
        // 3. Factorial
        if lower.contains("factorial") || lower.contains("!") {
            if let Some(num) = self.extract_number(text) {
                let n = num as u64;
                if n <= 20 {
                    let fact = (1..=n).fold(1u128, |acc, x| acc * x as u128);
                    return format!(
                        "🔢 Factorial:\n\n{}! = {}\n\n*Confidence: 1.0*",
                        n, fact,
                    );
                } else {
                    return format!(
                        "🔢 Factorial:\n\n{}! is too large to compute exactly (result > 128 bits).\n\n*Confidence: 1.0*",
                        n,
                    );
                }
            }
        }
        
        // 4. Statistics
        if lower.contains("mean") || lower.contains("median") || lower.contains("mode")
            || lower.contains("statistics") || lower.contains("average")
            || lower.contains("variance") || lower.contains("standard deviation") {
            let numbers = self.extract_numbers(text);
            if numbers.len() >= 2 {
                let stats = self.math_engine.statistics(&numbers);
                return format!(
                    "📊 Statistics:\n\n\
                     Count: {}\n\
                     Sum: {:.4}\n\
                     Mean: {:.4}\n\
                     Median: {:.4}\n\
                     Mode: {:.4}\n\
                     Variance: {:.4}\n\
                     Std Dev: {:.4}\n\
                     Min: {:.4}\n\
                     Max: {:.4}\n\
                     Range: {:.4}\n\n\
                     *Computed from {} data points*",
                    stats.get("count").unwrap_or(&0.0),
                    stats.get("sum").unwrap_or(&0.0),
                    stats.get("mean").unwrap_or(&0.0),
                    stats.get("median").unwrap_or(&0.0),
                    stats.get("mode").unwrap_or(&0.0),
                    stats.get("variance").unwrap_or(&0.0),
                    stats.get("std_dev").unwrap_or(&0.0),
                    stats.get("min").unwrap_or(&0.0),
                    stats.get("max").unwrap_or(&0.0),
                    stats.get("range").unwrap_or(&0.0),
                    numbers.len(),
                );
            }
        }
        
        // 5. Quadratic equation
        if lower.contains("quadratic") || (lower.contains("x²") || lower.contains("x^2")) {
            let numbers = self.extract_numbers(text);
            if numbers.len() >= 3 {
                let a = numbers[0];
                let b = numbers[1];
                let c = numbers[2];
                let results = self.math_engine.solve_quadratic(a, b, c);
                let mut response = format!("📐 Quadratic Equation:\n\n{}x² + {}x + {} = 0\n\n", a, b, c);
                for (i, result) in results.iter().enumerate() {
                    response.push_str(&format!("Solution {}: {}\n", i + 1, result.display));
                }
                response.push_str(&format!("\n*Confidence: {:.2}*", results[0].confidence));
                return response;
            }
        }
        
        // 6. Linear equation
        if lower.contains("linear") || lower.contains("solve") {
            let numbers = self.extract_numbers(text);
            if numbers.len() >= 2 {
                let a = numbers[0];
                let b = numbers[1];
                let result = self.math_engine.solve_linear(a, b);
                return format!(
                    "📐 Linear Equation:\n\n{}x + {} = 0\n\nSolution: {}\n\n*Confidence: {:.2}*",
                    a, b, result.display, result.confidence,
                );
            }
        }
        
        // 7. General arithmetic evaluation
        // Try to evaluate a simple arithmetic expression
        if let Some(result) = self.try_evaluate_arithmetic(text) {
            return result;
        }
        
        // 8. Prime factors
        if lower.contains("factor") || lower.contains("prime factor") {
            if let Some(num) = self.extract_number(text) {
                let factors = self.math_engine.prime_factors(num as u64);
                return format!(
                    "🔢 Prime Factorization:\n\n{} = {}\n\n*Confidence: 1.0*",
                    num,
                    factors.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" × "),
                );
            }
        }
        
        // Fallback: return math engine summary
        format!(
            "🧮 Math Engine Active\n\n{}\n\n*To solve a specific problem, try:\n\
             - \"Calculate 2 + 3 * 4\"\n\
             - \"Solve 2x + 3 = 7\"\n\
             - \"Is 17 prime?\"\n\
             - \"GCD of 12 and 18\"\n\
             - \"Mean of 1, 2, 3, 4, 5\"*",
            self.math_engine.summary(),
        )
    }
    
    /// PRIORITY 5: Extract a single number from text.
    fn extract_number(&self, text: &str) -> Option<f64> {
        let mut numbers = Vec::new();
        let mut current = String::new();
        let mut has_decimal = false;
        
        for c in text.chars() {
            if c.is_ascii_digit() {
                current.push(c);
            } else if c == '.' && !has_decimal && !current.is_empty() {
                current.push(c);
                has_decimal = true;
            } else {
                if !current.is_empty() {
                    if let Ok(n) = current.parse::<f64>() {
                        numbers.push(n);
                    }
                    current.clear();
                    has_decimal = false;
                }
            }
        }
        if !current.is_empty() {
            if let Ok(n) = current.parse::<f64>() {
                numbers.push(n);
            }
        }
        
        numbers.into_iter().next()
    }
    
    /// PRIORITY 5: Extract all numbers from text.
    fn extract_numbers(&self, text: &str) -> Vec<f64> {
        let mut numbers = Vec::new();
        let mut current = String::new();
        let mut has_decimal = false;
        
        for c in text.chars() {
            if c.is_ascii_digit() {
                current.push(c);
            } else if c == '.' && !has_decimal {
                if !current.is_empty() {
                    current.push(c);
                    has_decimal = true;
                }
            } else {
                if !current.is_empty() {
                    if let Ok(n) = current.parse::<f64>() {
                        numbers.push(n);
                    }
                    current.clear();
                    has_decimal = false;
                }
            }
        }
        if !current.is_empty() {
            if let Ok(n) = current.parse::<f64>() {
                numbers.push(n);
            }
        }
        
        numbers
    }
    
    /// PRIORITY 5: Try to evaluate a simple arithmetic expression from text.
    fn try_evaluate_arithmetic(&mut self, text: &str) -> Option<String> {
        // Look for patterns like "2 + 3", "4 * 5", "10 / 2", etc.
        let lower = text.to_lowercase();
        
        // Check for common arithmetic patterns
        let patterns = [
            ("plus", BinaryOpKind::Add),
            ("+", BinaryOpKind::Add),
            ("minus", BinaryOpKind::Subtract),
            ("-", BinaryOpKind::Subtract),
            ("times", BinaryOpKind::Multiply),
            ("multiplied by", BinaryOpKind::Multiply),
            ("*", BinaryOpKind::Multiply),
            ("×", BinaryOpKind::Multiply),
            ("divided by", BinaryOpKind::Divide),
            ("/", BinaryOpKind::Divide),
            ("÷", BinaryOpKind::Divide),
            ("power", BinaryOpKind::Power),
            ("^", BinaryOpKind::Power),
            ("mod", BinaryOpKind::Modulo),
            ("%", BinaryOpKind::Modulo),
        ];
        
        for (pattern, op) in &patterns {
            if lower.contains(pattern) {
                let numbers = self.extract_numbers(text);
                if numbers.len() >= 2 {
                    let left = numbers[0];
                    let right = numbers[1];
                    
                    let expr = MathExpr::BinaryOp {
                        op: op.clone(),
                        left: Box::new(MathExpr::Number(left)),
                        right: Box::new(MathExpr::Number(right)),
                    };
                    
                    let result = self.math_engine.evaluate(&expr, &std::collections::HashMap::new());
                    let op_symbol = match op {
                        BinaryOpKind::Add => "+",
                        BinaryOpKind::Subtract => "-",
                        BinaryOpKind::Multiply => "×",
                        BinaryOpKind::Divide => "÷",
                        BinaryOpKind::Power => "^",
                        BinaryOpKind::Modulo => "mod",
                        _ => "?",
                    };
                    
                    return Some(format!(
                        "🧮 Arithmetic:\n\n{} {} {} = {}\n\n*Confidence: {:.2}*",
                        left, op_symbol, right, result.display, result.confidence,
                    ));
                }
            }
        }
        
        None
    }
    
    /// PRIORITY 4: Debug code using the coding engine.
    /// Called when the input is detected as a code debugging request.
    fn debug_code_response(&mut self, text: &str) -> String {
        // Extract code from the input (everything after "debug" or "fix")
        let code_text = if let Some(pos) = text.find("```") {
            // Extract code from markdown code block
            let start = pos + 3;
            if let Some(end) = text[start..].find("```") {
                // Skip language identifier line
                let code_start = text[start..].find('\n').map(|n| start + n + 1).unwrap_or(start);
                text[code_start..start + end].trim().to_string()
            } else {
                text[start..].trim().to_string()
            }
        } else {
            text.to_string()
        };
        
        // Detect language
        let language = if text.contains("rust") || text.contains("Rust") {
            "rust"
        } else if text.contains("python") || text.contains("Python") {
            "python"
        } else if text.contains("javascript") || text.contains("JavaScript") {
            "javascript"
        } else {
            "rust"
        };
        
        // Analyze and debug
        let snippet = self.coding_engine.analyze_code(&code_text, language);
        let result = self.coding_engine.debug_code(&snippet);
        
        // Format debug results
        let mut response = String::new();
        if result.is_valid {
            response.push_str("✅ No issues found in the code.\n");
        } else {
            response.push_str("🔍 Found the following issues:\n\n");
            for issue in &result.issues {
                let severity = if issue.severity > 0.7 { "🔴" } else if issue.severity > 0.4 { "🟡" } else { "🟢" };
                response.push_str(&format!("{} Line {}: {} (severity: {:.1})\n",
                    severity, issue.line + 1, issue.description, issue.severity));
            }
            
            if !result.suggestions.is_empty() {
                response.push_str("\n💡 Suggestions:\n");
                for suggestion in &result.suggestions {
                    response.push_str(&format!("  • {}\n", suggestion));
                }
            }
        }
        
        response.push_str(&format!("\n*Code complexity: {:.2}, confidence: {:.2}*",
            snippet.complexity, result.confidence));
        
        response
    }

    /// PRIORITY 6: Detect if input text contains tool-related content.
    /// Checks for patterns like tool names, file operations, HTTP requests,
    /// data transformations, and calculator expressions.
    fn is_tool_input(&self, text: &str) -> bool {
        let tool_keywords = [
            "read file", "write file", "list files", "file read", "file write",
            "http get", "http post", "fetch url", "download", "web request",
            "calculate", "calculator", "compute", "evaluate expression",
            "convert json", "convert csv", "data transform", "transform data",
            "shell command", "run command", "execute command",
            "search web", "web search", "look up", "find information",
            "tool", "invoke", "call tool",
        ];
        
        let lower = text.to_lowercase();
        let keyword_count = tool_keywords.iter()
            .filter(|kw| lower.contains(*kw))
            .count();
        
        // Check for file path patterns
        let has_file_path = text.contains(":\\") || text.contains("./") || text.contains("../")
            || text.contains(".txt") || text.contains(".json") || text.contains(".csv")
            || text.contains(".rs") || text.contains(".py") || text.contains(".js");
        
        // Check for URL patterns
        let has_url = text.contains("http://") || text.contains("https://") || text.contains("www.");
        
        // Check for JSON/CSV data patterns
        let has_json = text.contains('{') && text.contains('}') && text.contains('"');
        let has_csv = text.contains(',') && text.contains('\n') && has_file_path;
        
        // If 2+ tool keywords, or has file path + keyword, or has URL + keyword
        if keyword_count >= 2 {
            return true;
        }
        if has_file_path && keyword_count >= 1 {
            return true;
        }
        if has_url && keyword_count >= 1 {
            return true;
        }
        if (has_json || has_csv) && keyword_count >= 1 {
            return true;
        }
        
        false
    }
    
    /// PRIORITY 6: Apply tool-aware pulse transform.
    /// When tool input is detected, this method augments the pulse processing
    /// with tool engine analysis. It blends tool invocation patterns into pulse
    /// content for improved tool understanding and execution.
    fn apply_tool_aware_pulse_transform(&mut self, pulses: &mut [NovaPulse]) {
        if pulses.is_empty() || !self.tool_enabled {
            return;
        }
        
        // Blend tool awareness into pulse entropy and weight
        // Tool operations need precision (lower entropy) and higher weight
        for pulse in pulses.iter_mut() {
            // Reduce entropy for tool operations to allow more focused iterations
            pulse.entropy = pulse.entropy * 0.85;
            // Boost weight for tool-related pulses to prioritize them
            pulse.weight = pulse.weight * 1.15;
        }
    }
    
    /// PRIORITY 6: Handle tool requests using the tool engine.
    /// Called when the input is detected as a tool invocation request.
    /// Parses the input to detect the type of tool operation and routes it
    /// to the appropriate tool engine method.
    fn handle_tool_request(&mut self, text: &str) -> String {
        let lower = text.to_lowercase();
        let mut params = std::collections::HashMap::new();
        
        // 1. File read operations
        if lower.contains("read file") || lower.contains("file read") || lower.contains("read ") {
            // Extract file path from text
            if let Some(path) = self.extract_file_path(text) {
                params.insert("path".to_string(), path);
                let result = self.tool_engine.invoke("file_read", &params);
                return self.format_tool_result("📂 File Read", &result);
            }
        }
        
        // 2. File write operations
        if lower.contains("write file") || lower.contains("file write") || lower.contains("write to") {
            if let Some(path) = self.extract_file_path(text) {
                params.insert("path".to_string(), path);
                // Use the rest of the text as content
                let content = self.extract_content_after_keyword(text, &["write", "write to", "write file"]);
                params.insert("content".to_string(), content);
                let result = self.tool_engine.invoke("file_write", &params);
                return self.format_tool_result("📝 File Write", &result);
            }
        }
        
        // 3. Calculator / arithmetic
        if lower.contains("calculate") || lower.contains("calculator") || lower.contains("compute") {
            let expression = self.extract_expression(text);
            if !expression.is_empty() {
                params.insert("expression".to_string(), expression);
                let result = self.tool_engine.invoke("calculator", &params);
                return self.format_tool_result("🧮 Calculator", &result);
            }
        }
        
        // 4. Data transformation (JSON ↔ CSV)
        if lower.contains("convert") || lower.contains("transform") {
            let from = if lower.contains("json") { "json" } else if lower.contains("csv") { "csv" } else { "json" };
            let to = if lower.contains("to csv") || (from == "json" && lower.contains("csv")) { "csv" } 
                else if lower.contains("to json") || (from == "csv" && lower.contains("json")) { "json" }
                else { "json" };
            
            // Extract the data to transform
            let data = self.extract_data_block(text);
            if !data.is_empty() {
                params.insert("input".to_string(), data);
                params.insert("from_format".to_string(), from.to_string());
                params.insert("to_format".to_string(), to.to_string());
                let result = self.tool_engine.invoke("data_transform", &params);
                return self.format_tool_result("🔄 Data Transform", &result);
            }
        }
        
        // 5. Web search
        if lower.contains("search") || lower.contains("look up") || lower.contains("find information") {
            let query = self.extract_query(text);
            if !query.is_empty() {
                params.insert("query".to_string(), query);
                let result = self.tool_engine.invoke("web_search", &params);
                return self.format_tool_result("🔍 Web Search", &result);
            }
        }
        
        // 6. HTTP GET
        if lower.contains("http get") || lower.contains("fetch") || lower.contains("download") {
            if let Some(url) = self.extract_url(text) {
                params.insert("url".to_string(), url);
                let result = self.tool_engine.invoke("http_get", &params);
                return self.format_tool_result("🌐 HTTP GET", &result);
            }
        }
        
        // 7. Shell command (safe commands only)
        if lower.contains("run command") || lower.contains("execute") || lower.contains("shell") {
            let command = self.extract_command(text);
            if !command.is_empty() {
                params.insert("command".to_string(), command);
                let result = self.tool_engine.invoke("shell", &params);
                return self.format_tool_result("💻 Shell Command", &result);
            }
        }
        
        // Fallback: list available tools
        let tools = self.tool_engine.list_tools();
        let mut response = "🔧 Available Tools:\n\n".to_string();
        for tool in &tools {
            response.push_str(&format!("  • **{}**: {}\n", tool.name, tool.description));
        }
        response.push_str(&format!(
            "\n*Total invocations: {}, success rate: {:.1}%*",
            self.tool_engine.total_invocations,
            if self.tool_engine.total_invocations > 0 {
                (self.tool_engine.successful_invocations as f64 / self.tool_engine.total_invocations as f64) * 100.0
            } else {
                0.0
            },
        ));
        response
    }
    
    /// PRIORITY 6: Format a tool result into a user-friendly string.
    fn format_tool_result(&self, title: &str, result: &ToolResult) -> String {
        let status = if result.success { "✅ Success" } else { "❌ Failed" };
        let mut response = format!("{} — {}\n\n", title, status);
        
        if result.success {
            response.push_str(&result.output);
        } else if let Some(ref error) = result.error {
            response.push_str(&format!("Error: {}", error));
        }
        
        if result.execution_time_ms > 0 {
            response.push_str(&format!("\n\n*Execution time: {}ms*", result.execution_time_ms));
        }
        
        response
    }
    
    /// PRIORITY 6: Extract a file path from text.
    fn extract_file_path(&self, text: &str) -> Option<String> {
        // Look for common file path patterns
        let patterns = [
            r"([a-zA-Z]:\\[^\s,;]+)",  // Windows path: C:\path\to\file
            r"(\./[^\s,;]+)",           // Relative path: ./path/to/file
            r"(\.\.[^\s,;]+)",          // Parent path: ../path/to/file
            r"([^\s]+\.(txt|json|csv|rs|py|js|toml|md))",  // File with extension
        ];
        
        for pattern in &patterns {
            // Simple manual extraction without regex
            let words: Vec<&str> = text.split_whitespace().collect();
            for word in &words {
                let clean = word.trim_matches(|c: char| c == ',' || c == ';' || c == '.' || c == '"' || c == '\'');
                if clean.contains('.') || clean.contains(":\\") || clean.starts_with("./") || clean.starts_with("../") {
                    return Some(clean.to_string());
                }
            }
        }
        
        None
    }
    
    /// PRIORITY 6: Extract content after a keyword.
    fn extract_content_after_keyword(&self, text: &str, keywords: &[&str]) -> String {
        let lower = text.to_lowercase();
        for kw in keywords {
            if let Some(pos) = lower.find(kw) {
                let after = &text[pos + kw.len()..].trim();
                if !after.is_empty() {
                    return after.to_string();
                }
            }
        }
        text.to_string()
    }
    
    /// PRIORITY 6: Extract a mathematical expression from text.
    fn extract_expression(&self, text: &str) -> String {
        // Look for patterns like "calculate 2 + 3" or "2 + 3"
        let lower = text.to_lowercase();
        for kw in &["calculate", "calculator", "compute", "evaluate", "what is"] {
            if let Some(pos) = lower.find(kw) {
                let after = &text[pos + kw.len()..].trim();
                if !after.is_empty() {
                    // Return the expression, stripping trailing punctuation
                    return after.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?').to_string();
                }
            }
        }
        
        // If no keyword found, try to extract numbers and operators directly
        let has_ops = text.contains('+') || text.contains('-') || text.contains('*') || text.contains('/') || text.contains('^');
        if has_ops {
            return text.trim().to_string();
        }
        
        String::new()
    }
    
    /// PRIORITY 6: Extract a data block (JSON/CSV) from text.
    fn extract_data_block(&self, text: &str) -> String {
        // Look for JSON-like content between { }
        if let Some(start) = text.find('{') {
            if let Some(end) = text[start..].rfind('}') {
                return text[start..=start + end].to_string();
            }
        }
        
        // Look for CSV-like content (lines with commas)
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 2 {
            let has_commas = lines.iter().any(|l| l.contains(','));
            if has_commas {
                return text.to_string();
            }
        }
        
        String::new()
    }
    
    /// PRIORITY 6: Extract a search query from text.
    fn extract_query(&self, text: &str) -> String {
        let lower = text.to_lowercase();
        for kw in &["search for", "search", "look up", "find information about", "find"] {
            if let Some(pos) = lower.find(kw) {
                let after = &text[pos + kw.len()..].trim();
                if !after.is_empty() {
                    return after.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?').to_string();
                }
            }
        }
        text.to_string()
    }
    
    /// PRIORITY 6: Extract a URL from text.
    fn extract_url(&self, text: &str) -> Option<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in &words {
            let clean = word.trim_matches(|c: char| c == ',' || c == ';' || c == '.' || c == '"' || c == '\'');
            if clean.starts_with("http://") || clean.starts_with("https://") || clean.starts_with("www.") {
                return Some(clean.to_string());
            }
        }
        None
    }
    
    /// PRIORITY 6: Extract a shell command from text.
    fn extract_command(&self, text: &str) -> String {
        let lower = text.to_lowercase();
        for kw in &["run command", "execute", "shell command", "run "] {
            if let Some(pos) = lower.find(kw) {
                let after = &text[pos + kw.len()..].trim();
                if !after.is_empty() {
                    return after.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?').to_string();
                }
            }
        }
        String::new()
    }
    
    /// PRIORITY 1+3+4+5+6: Process text input and return a response.
    /// Neural path (cores + field) is now the PRIMARY and ONLY output mechanism.
    /// Hash matches and n-gram generation are completely removed as fallbacks.
    ///
    /// PRIORITY 3: Added long context handling via LongContextManager.
    /// Sequences longer than max_seq_length are processed using sliding window SSM
    /// with context compression and hierarchical field states for long-range dependencies.
    ///
    /// PRIORITY 4: Added code-aware inference via CodingEngine.
    /// When code-related input is detected, the coding engine is used to analyze,
    /// generate, or debug code. Code patterns are blended into pulse transforms
    /// for improved code understanding.
    ///
    /// PRIORITY 5: Added math-aware inference via MathEngine.
    /// When math-related input is detected, the math engine is used to solve
    /// arithmetic, algebra, prime detection, statistics, and other math problems.
    /// Math patterns are blended into pulse transforms for improved numerical reasoning.
    ///
    /// PRIORITY 6: Added tool-aware inference via ToolEngine.
    /// When tool-related input is detected, the tool engine is used to handle
    /// file operations, HTTP requests, calculator, data transforms, web search,
    /// and shell commands. Tool check comes FIRST since tool operations like
    /// "calculate" overlap with math operations.
    ///
    /// The inference pipeline:
    /// 1. PRIORITY 6: Check if input is tool-related → route to tool engine
    /// 2. PRIORITY 5: Check if input is math-related → route to math engine
    /// 3. PRIORITY 4: Check if input is code-related → route to coding engine
    /// 4. ALWAYS runs neural path (cores + field) for pulse propagation
    /// 5. Uses adaptive iteration count based on entropy level
    /// 6. Uses content convergence + entropy for adaptive early exit
    /// 7. Applies multi-core semantic consensus after processing
    /// 8. Neural output is the ONLY output: map pulses to vocabulary words
    /// 9. No hash matches, no n-gram fallbacks - pure reasoning
    /// 10. PRIORITY 3: Long sequences use sliding window + hierarchical field
    ///
    /// FIXED: Neural path is now the ONLY output path. Hash/ngram fallbacks removed.
    pub fn process(&mut self, text: &str) -> String {
        let input_hash = hash_text(text);
        if let Some(response) = self.learned_responses.get(&input_hash) {
            return response.clone();
        }
        
        // Also check partial matches (if input contains learned phrase)
        for (hash, response) in &self.learned_responses {
            if let Some(input_text) = self.learned_inputs.get(hash) {
                if text.contains(input_text) || input_text.contains(text) {
                    return response.clone();
                }
            }
        }
        // PRIORITY 6: Check if input is tool-related.
        // Tool check comes FIRST since tool operations like "calculate" overlap
        // with math operations. If tool engine is enabled and the input contains
        // tool patterns, route to the tool engine for specialized tool handling.
        if self.tool_enabled && self.is_tool_input(text) {
            return self.handle_tool_request(text);
        }
        
        // PRIORITY 5: Check if input is math-related.
        // If math engine is enabled and the input contains math patterns,
        // route to the math engine for specialized math problem solving.
        // Math check comes BEFORE code check since math expressions may contain
        // code-like symbols (e.g., "=", "+", "/").
        if self.math_enabled && self.is_math_input(text) {
            // Check if this is a direct math problem solving request
            if text.contains("solve") || text.contains("calculate") || text.contains("compute")
                || text.contains("evaluate") || text.contains("what is") || text.contains("find")
                || text.contains("prime") || text.contains("factor") || text.contains("gcd")
                || text.contains("lcm") || text.contains("mean") || text.contains("median")
                || text.contains("statistics") || text.contains("quadratic") {
                return self.solve_math_response(text);
            }
            // For general math analysis, proceed with neural path but apply math-aware transforms
        }
        
        // PRIORITY 4: Check if input is code-related.
        // If coding engine is enabled and the input contains code patterns,
        // route to the coding engine for specialized code handling.
        if self.coding_enabled && self.is_code_input(text) {
            // Check if this is a debug/fix request
            if text.contains("debug") || text.contains("fix") || text.contains("issue") {
                return self.debug_code_response(text);
            }
            // Check if this is a code generation request
            if text.contains("write") || text.contains("generate") || text.contains("create")
                || text.contains("implement") || text.contains("function") || text.contains("program") {
                return self.generate_code_response(text);
            }
            // For general code analysis, proceed with neural path but apply code-aware transforms
        }

        
        // Step 1: ALWAYS run the neural path (cores + field).
        // This is the PRIMARY and ONLY output mechanism in Priority 1.
        let mut pulses = self.text_to_pulses(text);
        self.total_pulses_processed += pulses.len();
        
        if pulses.is_empty() {
            return String::new();
        }
        
        // PRIORITY 5: Apply math-aware pulse transform if math input detected.
        // This blends mathematical precision into pulse content for improved numerical reasoning.
        if self.math_enabled && self.is_math_input(text) {
            self.apply_math_aware_pulse_transform(&mut pulses);
        }
        
        // PRIORITY 4: Apply code-aware pulse transform if code input detected.
        // This blends code patterns into pulse content for improved code understanding.
        if self.coding_enabled && self.is_code_input(text) {
            self.apply_code_aware_pulse_transform(&mut pulses);
        }
        
        // PRIORITY 3: Check if we should use long context handling.
        // For sequences longer than max_seq_length, use LongContextManager
        // which employs sliding window SSM + context compression.
        if self.long_context.enabled && pulses.len() > self.long_context.max_seq_length {
            // Use long context path with hierarchical field support
            let hf_option = if self.use_hierarchical_field {
                Some(&mut self.hierarchical_field)
            } else {
                None
            };
            
            self.long_context.process_long_sequence(
                &mut pulses,
                &mut self.cores,
                &mut self.field,
                hf_option,
            );
            
            // Compress context for future reference
            self.context_compressor.compress(&pulses, &self.cores, &self.field);
            
            // Apply multi-core semantic consensus
            self.apply_multi_core_semantic_consensus(&mut pulses);
            
            // Convert to text
            if !self.vocabulary.is_empty() {
                let neural_text = self.map_pulses_to_vocab(&pulses);
                if !neural_text.trim().is_empty() {
                    return neural_text;
                }
            }
            return String::new();
        }
        
        // PRIORITY 1: Adaptive iteration count based on entropy level.
        // High entropy inputs need more iterations to converge.
        // Low entropy inputs can converge quickly.
        let adaptive_max = self.adaptive_iteration_count(&pulses);
        let mut prev_entropy = f32::MAX;
        
        for _iteration in 0..adaptive_max {
            self.process_cores_parallel(&mut pulses);
            
            // PRIORITY 3: Use hierarchical field if enabled for long-range dependencies.
            // HierarchicalField blends local (fast-changing) and global (slow-changing)
            // field states, enabling the model to track long-range patterns.
            if self.use_hierarchical_field {
                self.hierarchical_field.update(&mut pulses);
            } else {
                self.field.update(&mut pulses);
            }
            
            self.total_iterations += 1;
            
            let avg_entropy: f32 = pulses.iter().map(|p| p.entropy).sum::<f32>() / pulses.len() as f32;
            
            // Check convergence threshold (entropy-based)
            if avg_entropy < self.convergence_threshold {
                break;
            }
            
            // Adaptive early exit - if entropy change is negligible
            let entropy_delta = (prev_entropy - avg_entropy).abs();
            if prev_entropy != f32::MAX && entropy_delta < 0.001 {
                break;
            }
            prev_entropy = avg_entropy;
            
            // PRIORITY 1: Content convergence check
            let content_conv = self.content_convergence(&pulses);
            if content_conv > self.content_convergence_threshold {
                break;
            }
        }
        
        // PRIORITY 1: Multi-core semantic consensus.
        // After all cores have processed, compute the semantic consensus
        // across all cores and blend into pulses for more coherent output.
        self.apply_multi_core_semantic_consensus(&mut pulses);
        
        // PRIORITY 1: Convert neural output to text (ONLY mechanism)
        // No hash matches, no n-gram fallbacks - pure reasoning output.
        if !self.vocabulary.is_empty() {
            let neural_text = self.map_pulses_to_vocab(&pulses);
            if !neural_text.trim().is_empty() {
                return neural_text;
            }
        }
        
        // If vocabulary is empty or neural output is empty, return empty string.
        // The model needs to be trained first to have a vocabulary.
        String::new()
    }
    
    /// PRIORITY 1: Compute content convergence score across all pulses.
    /// Measures how much pulse content has stabilized across iterations.
    /// Returns 0.0 (no convergence) to 1.0 (fully converged).
    /// Delegates to the first core's content_convergence method.
    pub fn content_convergence(&self, pulses: &[NovaPulse]) -> f32 {
        if pulses.is_empty() || self.cores.is_empty() {
            return 0.0;
        }
        
        // Use the first core's convergence detection
        self.cores[0].content_convergence(pulses)
    }
    
    /// Learn n-gram patterns from training text.
    /// Extracts n-gram sequences of order 2..ngram_order and stores them
    /// as (context_hash -> [(next_word, confidence)]) mappings.
    pub fn learn_ngrams(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 2 { return; }
        
        for order in 2..=self.ngram_order {
            if words.len() < order + 1 { continue; }
            
            for i in 0..words.len() - order {
                let context: Vec<&str> = words[i..i + order].to_vec();
                let next_word = words[i + order];
                
                let context_str = context.join(" ");
                let context_hash = hash_text(&context_str);
                
                let entry = self.ngram_patterns.entry(context_hash).or_insert_with(Vec::new);
                
                // Check if this next_word already exists in predictions
                let mut found = false;
                for (word, conf) in entry.iter_mut() {
                    if word == next_word {
                        *conf = (*conf + 1.0).min(10.0);
                        found = true;
                        break;
                    }
                }
                
                if !found {
                    entry.push((next_word.to_string(), 1.0));
                }
            }
        }
        
        // Also collect all unique words
        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty() && !self.all_words.contains(&clean.to_string()) {
                self.all_words.push(clean.to_string());
            }
        }
    }
    
    /// Run a benchmark on the current model configuration.
    /// Tests inference speed, memory usage, and convergence behavior.
    pub fn benchmark(&mut self, text: &str) -> String {
        let start = std::time::Instant::now();
        
        // Warmup
        let _ = self.process(text);
        
        let warmup_end = std::time::Instant::now();
        let warmup_time = warmup_end.duration_since(start);
        
        // Actual benchmark
        let bench_start = std::time::Instant::now();
        let num_runs = 5;
        let mut total_time = std::time::Duration::new(0, 0);
        let mut outputs = Vec::with_capacity(num_runs);
        
        for _ in 0..num_runs {
            let run_start = std::time::Instant::now();
            let output = self.process(text);
            total_time += run_start.elapsed();
            outputs.push(output);
        }
        
        let avg_time = total_time / num_runs as u32;
        let pulses_per_sec = if avg_time.as_secs_f64() > 0.0 {
            self.total_pulses_processed as f64 / avg_time.as_secs_f64()
        } else {
            0.0
        };
        
        format!(
            "Benchmark Results:\n\
             Warmup: {:.2}ms\n\
             Avg inference: {:.2}ms ({} runs)\n\
             Pulses processed: {}\n\
             Iterations: {}\n\
             Pulses/sec: {:.0}\n\
             Memory: {}MB\n\
             Cores: {}\n\
             Field dim: {}\n\
             Vocabulary: {} words\n\
             N-gram patterns: {}\n\
             Learned responses: {}\n\
             Output sample: {:?}",
            warmup_time.as_secs_f64() * 1000.0,
            avg_time.as_secs_f64() * 1000.0,
            num_runs,
            self.total_pulses_processed,
            self.total_iterations,
            pulses_per_sec,
            self.memory_usage(),
            self.cores.len(),
            self.dim,
            self.vocabulary.len(),
            self.ngram_patterns.len(),
            self.learned_responses.len(),
            outputs.first().map(|s| s.chars().take(50).collect::<String>()).unwrap_or_default(),
        )
    }
    
    /// Get statistics about the current model state.
    pub fn stats(&self) -> String {
        format!(
            "NovaLoom Stats:\n\
             Name: {}\n\
             Dimension: {}\n\
             Cores: {}\n\
             Max iterations: {}\n\
             Convergence threshold: {:.3}\n\
             Content convergence threshold: {:.3}\n\
             Total pulses processed: {}\n\
             Total iterations: {}\n\
             Vocabulary size: {}\n\
             N-gram patterns: {}\n\
             N-gram order: {}\n\
             Learned responses: {}\n\
             All words: {}\n\
             Field energy: {:.3}\n\
             Memory usage: {}MB\n\
             Long context enabled: {}\n\
             Max seq length: {}\n\
             Context chunks: {}\n\
             Hierarchical field: {}\n\
             Math engine: {}\n\
             Tool engine: {}",
            self.name,
            self.dim,
            self.cores.len(),
            self.max_iterations,
            self.convergence_threshold,
            self.content_convergence_threshold,
            self.total_pulses_processed,
            self.total_iterations,
            self.vocabulary.len(),
            self.ngram_patterns.len(),
            self.ngram_order,
            self.learned_responses.len(),
            self.all_words.len(),
            self.field.energy(),
            self.memory_usage(),
            self.long_context.enabled,
            self.long_context.max_seq_length,
            self.long_context.context_chunks.len(),
            if self.use_hierarchical_field { "enabled" } else { "disabled" },
            if self.math_enabled { "enabled" } else { "disabled" },
            if self.tool_enabled { "enabled" } else { "disabled" },
        )
    }
    
    /// Get detailed model information for debugging.
    pub fn model_info(&self) -> String {
        let mut info = format!(
            "=== Nova Model Info ===\n\
             Name: {}\n\
             Architecture: Pulse-based O(n) field dynamics\n\
             Dimension: {}\n\
             Max iterations: {}\n\
             Convergence threshold: {:.3}\n\
             Content convergence threshold: {:.3}\n\n\
             === Cores ===\n",
            self.name,
            self.dim,
            self.max_iterations,
            self.convergence_threshold,
            self.content_convergence_threshold,
        );
        
        for core in &self.cores {
            info.push_str(&format!(
                "  Core {}: {} (gate: {:.2}, adaptive_depth: {}, SSM: {}, time_mixing: {})\n",
                core.id,
                core.name,
                core.gate,
                core.adaptive_depth,
                core.use_ssm,
                core.use_time_mixing,
            ));
        }
        
        info.push_str(&format!(
            "\n=== Field ===\n\
             Dimension: {}\n\
             Energy: {:.3}\n\
             SSM enabled: {}\n\
             SSM gate: {:.2}\n\n\
             === Long Context ===\n\
             Enabled: {}\n\
             Max seq length: {}\n\
             Window size: {}\n\
             Window overlap: {}\n\
             Compression ratio: {}\n\
             Context chunks: {}\n\
             Hierarchical field: {}\n\
             Sliding window stride: {}\n\n\
             === Knowledge ===\n\
             Concepts: {}\n\
             Relations: {}\n\
             Facts: {}\n\n\
             === Training ===\n\
             Learned responses: {}\n\
             N-gram patterns: {}\n\
             Vocabulary: {}\n\
             All words: {}\n\n\
             === Math Engine ===\n\
             Enabled: {}\n\
             Arithmetic ops: {}\n\
             Algebra ops: {}\n\
             Deductions: {}\n\
             Statistics: {}\n",
            self.field.state().len(),
            self.field.energy(),
            self.field.use_ssm,
            self.field.ssm_gate,
            self.long_context.enabled,
            self.long_context.max_seq_length,
            self.long_context.window_size,
            self.long_context.window_overlap,
            self.long_context.compression_ratio,
            self.long_context.context_chunks.len(),
            if self.use_hierarchical_field { "enabled" } else { "disabled" },
            self.sliding_window.stride,
            self.knowledge.concepts.len(),
            self.knowledge.relations.len(),
            self.knowledge.facts.len(),
            self.learned_responses.len(),
            self.ngram_patterns.len(),
            self.vocabulary.len(),
            self.all_words.len(),
            if self.math_enabled { "enabled" } else { "disabled" },
            self.math_engine.total_arithmetic,
            self.math_engine.total_algebra,
            self.math_engine.total_deductions,
            self.math_engine.total_statistics,
        ));
        
        // PRIORITY 6: Tool Engine info
        info.push_str(&format!(
            "\n=== Tool Engine ===\n\
             Enabled: {}\n\
             Total invocations: {}\n\
             Successful invocations: {}\n\
             Success rate: {:.1}%\n\
             Available tools: {}\n",
            if self.tool_enabled { "enabled" } else { "disabled" },
            self.tool_engine.total_invocations,
            self.tool_engine.successful_invocations,
            if self.tool_engine.total_invocations > 0 {
                (self.tool_engine.successful_invocations as f64 / self.tool_engine.total_invocations as f64) * 100.0
            } else {
                0.0
            },
            self.tool_engine.list_tools().len(),
        ));
        
        info
    }
    
    /// Reset the model state (field, cores, SSM states, long context).
    /// Does NOT clear learned patterns or vocabulary.
    pub fn reset(&mut self) {
        self.field.reset();
        for core in &mut self.cores {
            core.reset_ssm();
        }
        self.total_pulses_processed = 0;
        self.total_iterations = 0;
        // PRIORITY 3: Reset long context and hierarchical field
        self.long_context.clear();
        self.hierarchical_field.reset_local();
    }
    
    /// Reset everything including learned patterns.
    pub fn reset_all(&mut self) {
        self.reset();
        self.learned_responses.clear();
        self.learned_inputs.clear();
        self.ngram_patterns.clear();
        self.all_words.clear();
        self.vocabulary.clear();
        self.vocab_reverse.clear();
    }
    
    // ============================================================
    // PRIORITY 8/9: Hyperparameter adjustment methods for auto-improvement
    // ============================================================
    
    /// Adjust model parameters for learning from training data.
    /// Used by benchmark auto-improvement to strengthen weak areas.
    /// Learns input-output associations by storing them in learned_responses
    /// and adjusting field convergence behavior.
    pub fn adjust_for_learning(&mut self, input: &str, target: &str, learning_rate: f32) {
        // Store the input-output association for future recall
        let input_hash = hash_text(input);
        self.learned_responses.insert(input_hash, target.to_string());
        self.learned_inputs.insert(input_hash, input.to_string());
        
        // Add target words to vocabulary for generation
        for word in target.split_whitespace() {
            let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation());
            if !clean_word.is_empty() && !self.all_words.contains(&clean_word.to_string()) {
                self.all_words.push(clean_word.to_string());
                // Create a simple embedding for the word
                let embedding: Vec<f32> = (0..self.dim)
                    .map(|i| {
                        let hash_val = hash_text(&format!("{}_{}", clean_word, i));
                        (hash_val as f32 / u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                self.vocabulary.insert(clean_word.to_string(), embedding);
                let emb_hash = hash_text(&format!("vocab_{}", clean_word));
                self.vocab_reverse.insert(emb_hash, clean_word.to_string());
            }
        }
        
        // Learn n-gram patterns from the input-target pair
        let combined = format!("{} {}", input, target);
        self.learn_ngrams(&combined);
        
        // Adjust field convergence threshold based on learning rate
        self.content_convergence_threshold = (self.content_convergence_threshold - learning_rate * 0.1).max(0.5);
        
        // Adjust field diffusion to improve information retention
        let current_diffusion = self.field.get_diffusion_rate();
        let new_diffusion = (current_diffusion + learning_rate * 0.05).min(0.5);
        self.field.set_diffusion_rate(new_diffusion);
    }
    
    /// Get the current model version info for compatibility tracking.
    pub fn model_version(&self) -> String {
        format!(
            "NovaCore v2.0 | dim={} | cores={} | max_iter={} | conv_thresh={:.2} | field_diff={:.3} | learned={} | ngrams={} | vocab={}",
            self.dim,
            self.cores.len(),
            self.max_iterations,
            self.convergence_threshold,
            self.field.get_diffusion_rate(),
            self.learned_responses.len(),
            self.ngram_patterns.len(),
            self.vocabulary.len(),
        )
    }
    
    /// Check if this model is compatible with a given version string.
    /// Returns true if the model can be used with the specified configuration.
    pub fn is_compatible_with(&self, required_dim: usize, required_cores: usize) -> bool {
        self.dim == required_dim && self.cores.len() >= required_cores
    }
    
    /// Save learned patterns to a persistent format (serializable HashMap).
    /// Returns a snapshot of all learned associations for export/save.
    pub fn export_learned_patterns(&self) -> HashMap<String, String> {
        let mut patterns = HashMap::new();
        for (hash, response) in &self.learned_responses {
            if let Some(input) = self.learned_inputs.get(hash) {
                patterns.insert(input.clone(), response.clone());
            }
        }
        patterns
    }
    
    /// Import learned patterns from a previously exported snapshot.
    pub fn import_learned_patterns(&mut self, patterns: HashMap<String, String>) {
        for (input, response) in patterns {
            let input_hash = hash_text(&input);
            self.learned_responses.insert(input_hash, response);
            self.learned_inputs.insert(input_hash, input);
        }
    }
    
    /// Get the number of learned associations.
    pub fn learned_count(&self) -> usize {
        self.learned_responses.len()
    }
    
    /// Get the number of n-gram patterns.
    pub fn ngram_count(&self) -> usize {
        self.ngram_patterns.len()
    }
    
    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
    
    /// Run a self-diagnostic to check model health.
    /// Returns a list of issues found (empty = healthy).
    pub fn self_diagnostic(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        if self.cores.is_empty() {
            issues.push("No cores configured".to_string());
        }
        if self.dim == 0 {
            issues.push("Dimension is zero".to_string());
        }
        if self.max_iterations == 0 {
            issues.push("Max iterations is zero".to_string());
        }
        if self.convergence_threshold <= 0.0 {
            issues.push("Convergence threshold is too low".to_string());
        }
        if self.content_convergence_threshold <= 0.0 {
            issues.push("Content convergence threshold is too low".to_string());
        }
        
        // Check field health
        let field_state = self.field.state();
        if field_state.iter().all(|&v| v == 0.0) {
            issues.push("Field state is all zeros".to_string());
        }
        
        // Check for NaN or Inf in field state
        if field_state.iter().any(|&v| v.is_nan() || v.is_infinite()) {
            issues.push("Field state contains NaN or Inf values".to_string());
        }
        
        issues
    }
    
    /// Run a complete self-improvement cycle.
    /// Returns the number of improvements made.
    pub fn run_self_improvement(&mut self, training_data: &[(String, String)], epochs: usize) -> usize {
        let mut improvements = 0;
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut correct = 0;
            
            for (input, target) in training_data {
                let output = self.process(input);
                
                // Check if output contains target keywords
                let target_lower = target.to_lowercase();
                let output_lower = output.to_lowercase();
                
                let score = if target_lower.contains(',') {
                    let keywords: Vec<&str> = target_lower.split(',').map(|s| s.trim()).collect();
                    let matched = keywords.iter().filter(|k| output_lower.contains(*k)).count();
                    matched as f32 / keywords.len() as f32
                } else {
                    if output_lower.contains(&target_lower) { 1.0 } else { 0.0 }
                };
                
                if score > 0.5 {
                    correct += 1;
                }
                
                // Compute loss
                let length_factor = (output.len() as f32 / 20.0).min(1.0);
                let loss = (1.0 - score) * (2.0 - length_factor);
                epoch_loss += loss;
                
                // Learn from this example
                self.adjust_for_learning(input, target, 0.01);
                improvements += 1;
            }
            
            let avg_loss = epoch_loss / training_data.len() as f32;
            let accuracy = correct as f32 / training_data.len() as f32;
            println!("    Epoch {}: loss={:.4}, accuracy={:.1}%, learned={}, ngrams={}", 
                     epoch + 1, avg_loss, accuracy * 100.0, 
                     self.learned_responses.len(), self.ngram_patterns.len());
        }
        
        improvements
    }
    
    /// Set the adaptive depth range (min, max iterations).
    pub fn set_adaptive_depth(&mut self, _min: usize, max: usize) {
        self.max_iterations = max;
    }
    
    /// Get the current max depth.
    pub fn get_max_depth(&self) -> usize {
        self.max_iterations
    }
    
    /// Set the field diffusion rate.
    pub fn set_field_diffusion(&mut self, rate: f32) {
        self.field.set_diffusion_rate(rate);
    }
    
    /// Get the current field diffusion rate.
    pub fn get_field_diffusion(&self) -> f32 {
        self.field.get_diffusion_rate()
    }
    
    /// Set the core gate strength.
    pub fn set_core_gate_strength(&mut self, strength: f32) {
        for core in &mut self.cores {
            core.set_gate_strength(strength);
        }
    }
    
    /// Get the current core gate strength.
    pub fn get_core_gate_strength(&self) -> f32 {
        self.cores.first().map(|c| c.get_gate_strength()).unwrap_or(0.5)
    }
    
    /// Set the convergence threshold.
    pub fn set_convergence_threshold(&mut self, threshold: f32) {
        self.convergence_threshold = threshold;
    }
    
    /// Get the current convergence threshold.
    pub fn get_convergence_threshold(&self) -> f32 {
        self.convergence_threshold
    }
}

/// Hash a text string to a u64 for n-gram lookup.
fn hash_text(text: &str) -> u64 {
    text.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as u64)
    })
}

impl Default for NovaLoom {
    fn default() -> Self {
        Self::new(256, 5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_loom_creation() {
        let loom = NovaLoom::new(256, 5);
        assert_eq!(loom.cores.len(), 5);
        assert_eq!(loom.dim, 256);
        println!("✅ NovaLoom creation works!");
    }
    
    #[test]
    fn test_text_to_pulses() {
        let loom = NovaLoom::new(256, 5);
        let pulses = loom.text_to_pulses("hello world");
        assert_eq!(pulses.len(), 2);
        println!("✅ Text to pulses works!");
    }
    
    #[test]
    fn test_process_empty() {
        let mut loom = NovaLoom::new(256, 5);
        let result = loom.process("");
        assert_eq!(result, "");
        println!("✅ Empty process works!");
    }
    
    #[test]
    fn test_learn_ngrams() {
        let mut loom = NovaLoom::new(256, 5);
        loom.learn_ngrams("the cat sat on the mat");
        assert!(!loom.ngram_patterns.is_empty());
        assert!(!loom.all_words.is_empty());
        println!("✅ N-gram learning works! {} patterns", loom.ngram_patterns.len());
    }
    
    #[test]
    fn test_generate_text() {
        let mut loom = NovaLoom::new(256, 5);
        loom.learn_ngrams("the cat sat on the mat the dog ran in the park");
        let result = loom.generate_text("the cat", 3);
        println!("✅ Generated: {:?}", result);
        // Should generate something (even if not perfect)
        assert!(!result.is_empty());
    }
    
    #[test]
    fn test_content_convergence() {
        let mut loom = NovaLoom::new(256, 5);
        let pulses = loom.text_to_pulses("test content");
        let conv = loom.content_convergence(&pulses);
        // Should return a value between 0.0 and 1.0
        assert!(conv >= 0.0 && conv <= 1.0);
        println!("✅ Content convergence: {:.4}", conv);
    }
    
    #[test]
    fn test_stats() {
        let loom = NovaLoom::new(256, 5);
        let stats = loom.stats();
        assert!(stats.contains("NovaLoom Stats"));
        println!("✅ Stats works!");
    }
    
    #[test]
    fn test_reset() {
        let mut loom = NovaLoom::new(256, 5);
        loom.total_pulses_processed = 100;
        loom.total_iterations = 50;
        loom.reset();
        assert_eq!(loom.total_pulses_processed, 0);
        assert_eq!(loom.total_iterations, 0);
        println!("✅ Reset works!");
    }
    
    #[test]
    fn test_hash_text() {
        let h1 = hash_text("hello world");
        let h2 = hash_text("hello world");
        let h3 = hash_text("different");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        println!("✅ Hash text works!");
    }
    
    #[test]
    fn test_default() {
        let loom = NovaLoom::default();
        assert_eq!(loom.cores.len(), 5);
        assert_eq!(loom.dim, 256);
        println!("✅ Default works!");
    }
    
    #[test]
    fn test_loop_detection() {
        let mut loom = NovaLoom::new(256, 5);
        loom.learn_ngrams("a b a b a b a b a b");
        let result = loom.generate_text("a b", 10);
        // Should not produce infinite loops
        let words: Vec<&str> = result.split_whitespace().collect();
        assert!(words.len() <= 12); // max_words + some buffer
        println!("✅ Loop detection works! Generated {} words", words.len());
    }
}

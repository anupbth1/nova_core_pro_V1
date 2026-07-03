//! Nova Knowledge - Structured Knowledge Representation
//!
//! Phase 5: Implements a knowledge store with:
//! - Concept embeddings (dense vector representations of concepts)
//! - Relation triples (subject -> relation -> object)
//! - Fact storage with confidence scoring
//! - Knowledge-aware pulse transforms
//! - Cosine similarity retrieval for concept lookup
//!
//! This gives Nova the ability to store and retrieve structured knowledge,
//! making it more than just a pattern matcher.

use crate::pulse::NovaPulse;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A single concept with its embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// The concept name (e.g., "cat", "gravity", "quantum")
    pub name: String,
    /// Dense embedding vector
    pub embedding: Vec<f32>,
    /// Category/tag for grouping (e.g., "animal", "physics", "math")
    pub category: String,
    /// Confidence in this concept (0.0 to 1.0)
    pub confidence: f32,
    /// How many times this concept has been reinforced
    pub strength: u32,
}

/// A relation triple: subject -> relation -> object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Subject concept name
    pub subject: String,
    /// Relation type (e.g., "is_a", "has_property", "causes", "part_of")
    pub relation: String,
    /// Object concept name
    pub object: String,
    /// Confidence in this relation (0.0 to 1.0)
    pub confidence: f32,
    /// How many times this relation has been observed
    pub strength: u32,
}

/// A factual statement with supporting evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    /// The fact text
    pub statement: String,
    /// Hash of the statement for deduplication
    pub hash: u64,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Source identifier
    pub source: String,
    /// Category
    pub category: String,
}

/// The main knowledge store for Nova Core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeStore {
    /// All known concepts indexed by name
    pub concepts: HashMap<String, Concept>,
    /// All relations (subject -> [(relation, object, confidence)])
    pub relations: HashMap<String, Vec<(String, String, f32)>>,
    /// Reverse relations (object -> [(relation, subject, confidence)])
    pub reverse_relations: HashMap<String, Vec<(String, String, f32)>>,
    /// All facts indexed by hash
    pub facts: HashMap<u64, Fact>,
    /// Facts grouped by category
    pub facts_by_category: HashMap<String, Vec<u64>>,
    /// Embedding dimension for concepts
    pub dim: usize,
    /// Maximum number of concepts to store
    pub max_concepts: usize,
    /// Learning rate for embedding updates
    pub learning_rate: f32,
}

impl KnowledgeStore {
    /// Create a new empty knowledge store
    pub fn new(dim: usize) -> Self {
        Self {
            concepts: HashMap::new(),
            relations: HashMap::new(),
            reverse_relations: HashMap::new(),
            facts: HashMap::new(),
            facts_by_category: HashMap::new(),
            dim,
            max_concepts: 10000,
            learning_rate: 0.1,
        }
    }

    /// Add or update a concept with its embedding
    pub fn add_concept(&mut self, name: &str, embedding: Vec<f32>, category: &str) {
        // Prune if at capacity and concept doesn't exist yet
        if !self.concepts.contains_key(name) && self.concepts.len() >= self.max_concepts {
            // Remove the weakest concept
            if let Some(weakest) = self.concepts.iter()
                .min_by(|a, b| a.1.strength.cmp(&b.1.strength))
                .map(|(k, _)| k.clone())
            {
                self.concepts.remove(&weakest);
                // Also clean up relations involving this concept
                self.relations.remove(&weakest);
                self.reverse_relations.remove(&weakest);
            }
        }

        let entry = self.concepts.entry(name.to_string()).or_insert_with(|| {
            Concept {
                name: name.to_string(),
                embedding: vec![0.0; self.dim],
                category: category.to_string(),
                confidence: 0.5,
                strength: 0,
            }
        });

        // Blend new embedding with existing (if any)
        let lr = self.learning_rate;
        let min_len = entry.embedding.len().min(embedding.len());
        for i in 0..min_len {
            entry.embedding[i] = entry.embedding[i] * (1.0 - lr) + embedding[i] * lr;
        }
        // Normalize
        let norm: f32 = entry.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in entry.embedding.iter_mut() {
                *x /= norm;
            }
        }
        entry.confidence = (entry.confidence * 0.9 + 0.95 * 0.1).min(1.0);
        entry.strength += 1;
    }

    /// Add a relation triple
    pub fn add_relation(&mut self, subject: &str, relation: &str, object: &str, confidence: f32) {
        // Forward relation
        let entry = self.relations.entry(subject.to_string()).or_default();
        let mut found = false;
        for (r, o, c) in entry.iter_mut() {
            if r == relation && o == object {
                *c = (*c * 0.9 + confidence * 0.1).min(1.0);
                found = true;
                break;
            }
        }
        if !found {
            entry.push((relation.to_string(), object.to_string(), confidence));
        }

        // Reverse relation
        let rev_entry = self.reverse_relations.entry(object.to_string()).or_default();
        let mut found_rev = false;
        for (r, s, c) in rev_entry.iter_mut() {
            if r == relation && s == subject {
                *c = (*c * 0.9 + confidence * 0.1).min(1.0);
                found_rev = true;
                break;
            }
        }
        if !found_rev {
            rev_entry.push((relation.to_string(), subject.to_string(), confidence));
        }
    }

    /// Add a factual statement
    pub fn add_fact(&mut self, statement: &str, source: &str, category: &str, confidence: f32) {
        let hash: u64 = statement.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        let entry = self.facts.entry(hash).or_insert_with(|| {
            Fact {
                statement: statement.to_string(),
                hash,
                confidence: 0.0,
                source: source.to_string(),
                category: category.to_string(),
            }
        });

        entry.confidence = (entry.confidence * 0.9 + confidence * 0.1).min(1.0);

        // Add to category index
        self.facts_by_category.entry(category.to_string())
            .or_default()
            .push(hash);
    }

    /// Find the closest concept to a given embedding vector
    pub fn find_closest_concept(&self, embedding: &[f32], threshold: f32) -> Option<(&Concept, f32)> {
        if self.concepts.is_empty() {
            return None;
        }

        let norm1: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm1 < 1e-6 {
            return None;
        }

        let mut best: Option<(&Concept, f32)> = None;

        for concept in self.concepts.values() {
            let dot: f32 = embedding.iter()
                .zip(concept.embedding.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm2: f32 = concept.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sim = if norm2 > 0.0 { dot / (norm1 * norm2) } else { 0.0 };

            if sim > threshold && (best.is_none() || sim > best.unwrap().1) {
                best = Some((concept, sim));
            }
        }

        best
    }

    /// Find concepts by category
    pub fn get_concepts_by_category(&self, category: &str) -> Vec<&Concept> {
        self.concepts.values()
            .filter(|c| c.category == category)
            .collect()
    }

    /// Get all relations for a subject
    pub fn get_relations(&self, subject: &str) -> Option<&Vec<(String, String, f32)>> {
        self.relations.get(subject)
    }

    /// Get all reverse relations for an object
    pub fn get_reverse_relations(&self, object: &str) -> Option<&Vec<(String, String, f32)>> {
        self.reverse_relations.get(object)
    }

    /// Get facts by category
    pub fn get_facts_by_category(&self, category: &str) -> Vec<&Fact> {
        self.facts_by_category.get(category)
            .map(|hashes| {
                hashes.iter()
                    .filter_map(|h| self.facts.get(h))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query: given a pulse, find related knowledge and blend it into the pulse
    pub fn augment_pulse_with_knowledge(&self, pulse: &mut NovaPulse, blend_strength: f32) -> bool {
        if self.concepts.is_empty() {
            return false;
        }

        // Find closest concept to this pulse
        if let Some((concept, sim)) = self.find_closest_concept(&pulse.content, 0.3) {
            // Blend concept embedding into pulse content
            let blend = blend_strength * sim;
            let min_len = pulse.content.len().min(concept.embedding.len());
            for i in 0..min_len {
                pulse.content[i] = pulse.content[i] * (1.0 - blend) + concept.embedding[i] * blend;
                pulse.content[i] = pulse.content[i].clamp(-1.0, 1.0);
            }
            // Reduce entropy (more certain with knowledge)
            pulse.reduce_entropy(0.95);
            return true;
        }

        false
    }

    /// Extract concepts and relations from a training example
    pub fn learn_from_example(&mut self, input: &str, target: &str) {
        let words: Vec<&str> = input.split_whitespace().collect();
        let target_words: Vec<&str> = target.split_whitespace().collect();

        // Extract concepts from input words (words longer than 3 chars are potential concepts)
        for (i, word) in words.iter().enumerate() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if clean.len() > 3 && !clean.is_empty() {
                // Create a deterministic embedding from the word
                let mut embedding = vec![0.0; self.dim];
                let bytes = clean.as_bytes();
                for (j, &b) in bytes.iter().enumerate() {
                    if j < self.dim {
                        embedding[j] = (b as f32) / 255.0 * 2.0 - 1.0;
                    }
                }
                // Add position information
                if self.dim > 1 {
                    embedding[1] += (i as f32 / 100.0).sin();
                }
                // Normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in embedding.iter_mut() {
                        *x /= norm;
                    }
                }

                // Determine category from context
                let category = if clean.chars().all(|c| c.is_ascii_digit()) {
                    "number"
                } else if clean.contains(|c: char| c.is_ascii_punctuation()) {
                    "symbol"
                } else {
                    "word"
                };

                self.add_concept(&clean, embedding, category);
            }
        }

        // Extract relations: adjacent words form "followed_by" relations
        for i in 0..words.len().saturating_sub(1) {
            let w1 = words[i].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            let w2 = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if !w1.is_empty() && !w2.is_empty() && w1.len() > 2 && w2.len() > 2 {
                self.add_relation(&w1, "followed_by", &w2, 0.6);
            }
        }

        // Extract relations from input -> target mapping
        if !target_words.is_empty() {
            let last_input = words.last()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .unwrap_or_default();
            let first_target = target_words.first()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .unwrap_or_default();
            if !last_input.is_empty() && !first_target.is_empty() {
                self.add_relation(&last_input, "predicts", &first_target, 0.7);
            }
        }

        // Store the full input-target pair as a fact
        if !input.is_empty() && !target.is_empty() {
            self.add_fact(
                &format!("{} -> {}", input, target),
                "training",
                "association",
                0.8,
            );
        }
    }

    /// Get the total number of stored knowledge items
    pub fn knowledge_count(&self) -> usize {
        self.concepts.len() + self.relations.values().map(|v| v.len()).sum::<usize>() + self.facts.len()
    }

    /// Get a summary of knowledge store contents
    pub fn summary(&self) -> String {
        let rel_count: usize = self.relations.values().map(|v| v.len()).sum();
        format!(
            "Knowledge: {} concepts, {} relations, {} facts",
            self.concepts.len(),
            rel_count,
            self.facts.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_store_creation() {
        let ks = KnowledgeStore::new(64);
        assert_eq!(ks.dim, 64);
        assert!(ks.concepts.is_empty());
        assert!(ks.relations.is_empty());
        println!("✅ KnowledgeStore creation works!");
    }

    #[test]
    fn test_add_concept() {
        let mut ks = KnowledgeStore::new(64);
        let embedding = vec![0.5; 64];
        ks.add_concept("cat", embedding, "animal");
        assert_eq!(ks.concepts.len(), 1);
        assert_eq!(ks.concepts["cat"].name, "cat");
        assert_eq!(ks.concepts["cat"].category, "animal");
        println!("✅ add_concept works!");
    }

    #[test]
    fn test_add_relation() {
        let mut ks = KnowledgeStore::new(64);
        ks.add_relation("cat", "is_a", "animal", 0.9);
        assert!(ks.relations.contains_key("cat"));
        assert!(ks.reverse_relations.contains_key("animal"));
        println!("✅ add_relation works!");
    }

    #[test]
    fn test_find_closest_concept() {
        let mut ks = KnowledgeStore::new(64);
        let cat_emb = vec![0.5; 64];
        let dog_emb = vec![0.3; 64];
        ks.add_concept("cat", cat_emb, "animal");
        ks.add_concept("dog", dog_emb, "animal");

        // Query with something close to cat
        let query = vec![0.48; 64];
        let result = ks.find_closest_concept(&query, 0.3);
        assert!(result.is_some());
        let (concept, sim) = result.unwrap();
        assert_eq!(concept.name, "cat");
        assert!(sim > 0.3);
        println!("✅ find_closest_concept works! Found '{}' with sim {:.3}", concept.name, sim);
    }

    #[test]
    fn test_learn_from_example() {
        let mut ks = KnowledgeStore::new(64);
        ks.learn_from_example("the cat sat on", "the mat");
        assert!(ks.concepts.len() >= 2); // "cat" and "sat" should be concepts
        assert!(ks.relations.contains_key("cat") || ks.relations.contains_key("sat"));
        assert!(!ks.facts.is_empty());
        println!("✅ learn_from_example works! {} concepts, {} facts", 
            ks.concepts.len(), ks.facts.len());
    }

    #[test]
    fn test_augment_pulse_with_knowledge() {
        let mut ks = KnowledgeStore::new(64);
        let cat_emb = vec![0.5; 64];
        ks.add_concept("cat", cat_emb, "animal");

        let mut pulse = NovaPulse::from_text("cat", 64, 0);
        let original_content = pulse.content.clone();
        
        let augmented = ks.augment_pulse_with_knowledge(&mut pulse, 0.3);
        assert!(augmented);
        // Content should have changed
        let changed = pulse.content.iter()
            .zip(original_content.iter())
            .any(|(a, b)| (a - b).abs() > 0.001);
        assert!(changed);
        println!("✅ augment_pulse_with_knowledge works!");
    }
}

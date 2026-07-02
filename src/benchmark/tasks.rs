//! All benchmark tasks for LLM evaluation

use rand::Rng;
use crate::loom::NovaLoom;

pub struct BenchmarkTask {
    pub name: String,
    pub generator: fn() -> (String, String),
    pub evaluator: fn(&str, &str) -> f32,
}

impl BenchmarkTask {
    pub fn run(&self, model: &mut NovaLoom, num_samples: usize) -> f32 {
        let mut scores = Vec::new();
        for _ in 0..num_samples {
            let (question, expected) = (self.generator)();
            let answer = model.process(&question);
            let score = (self.evaluator)(&answer, &expected);
            scores.push(score);
        }
        scores.iter().sum::<f32>() / scores.len() as f32
    }
}

pub fn get_language_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            name: "sentiment_analysis".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let texts = vec![
                    ("This is great!", "positive"),
                    ("I hate this", "negative"),
                    ("Not bad", "neutral"),
                    ("Excellent work", "positive"),
                    ("Terrible", "negative"),
                ];
                let (text, sentiment) = texts[rng.gen_range(0..texts.len())];
                (format!("Sentiment: {}", text), sentiment.to_string())
            },
            evaluator: |answer, expected| {
                if answer.to_lowercase().contains(expected) { 1.0 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "named_entity".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let entities = vec![
                    ("John lives in Paris", "PERSON:John, LOC:Paris"),
                    ("Apple Inc. is in California", "ORG:Apple, LOC:California"),
                ];
                let (text, entities) = entities[rng.gen_range(0..entities.len())];
                (format!("Entities: {}", text), entities.to_string())
            },
            evaluator: |answer, expected| {
                if answer.to_lowercase().contains(&expected.to_lowercase()) { 0.5 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "paraphrase_detection".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let pairs = vec![
                    ("The cat sat on the mat", "The mat had a cat sitting on it", "yes"),
                    ("The dog ran fast", "The bird flew high", "no"),
                ];
                let (s1, s2, expected) = pairs[rng.gen_range(0..pairs.len())];
                (format!("Same meaning? '{}' vs '{}'", s1, s2), expected.to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 0.7 } else { 0.0 }
            },
        },
    ]
}

pub fn get_reasoning_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            name: "logical_deduction".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let problems = vec![
                    ("All humans are mortal. Socrates is human. Therefore?", "mortal"),
                    ("If it rains, ground gets wet. It rained. Therefore?", "wet"),
                ];
                let (premise, conclusion) = problems[rng.gen_range(0..problems.len())];
                (premise.to_string(), conclusion.to_string())
            },
            evaluator: |answer, expected| {
                if answer.to_lowercase().contains(expected) { 1.0 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "mathematical_reasoning".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let a: i32 = rng.gen_range(1..50);
                let b: i32 = rng.gen_range(1..50);
                (format!("{} + {} = ?", a, b), (a + b).to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 1.0 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "analogical_reasoning".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let analogies = vec![
                    ("doctor is to hospital as teacher is to", "school"),
                    ("bird is to fly as fish is to", "swim"),
                ];
                let (question, answer) = analogies[rng.gen_range(0..analogies.len())];
                (question.to_string(), answer.to_string())
            },
            evaluator: |answer, expected| {
                if answer.to_lowercase().contains(expected) { 1.0 } else { 0.0 }
            },
        },
    ]
}

pub fn get_code_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            name: "code_completion".to_string(),
            generator: || {
                let snippets = vec![
                    ("def add(a, b):\n    return ", "a + b"),
                    ("for i in range(10):\n    print(", "i)"),
                ];
                let (code, completion) = snippets[0];
                (code.to_string(), completion.to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 0.5 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "bug_detection".to_string(),
            generator: || {
                let bugs = vec![
                    ("x = 10\ny = 0\nz = x / y", "division by zero"),
                    ("arr = [1,2,3]\nprint(arr[3])", "index out of range"),
                ];
                let (code, bug) = bugs[0];
                (format!("Find bug: {}", code), bug.to_string())
            },
            evaluator: |answer, expected| {
                if answer.to_lowercase().contains(expected) { 0.7 } else { 0.0 }
            },
        },
    ]
}

pub fn get_long_context_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            name: "long_summary".to_string(),
            generator: || {
                let long_text = "The quick brown fox jumps over the lazy dog. ".
                    repeat(20);
                (long_text, "fox jumps over dog".to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 0.5 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "information_retrieval".to_string(),
            generator: || {
                let context = "Alice has a cat named Whiskers. Bob has a dog named Rex.";
                (format!("Question: What is Alice's cat's name? Context: {}", context), "Whiskers".to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 1.0 } else { 0.0 }
            },
        },
    ]
}

pub fn get_memory_tasks() -> Vec<BenchmarkTask> {
    vec![
        BenchmarkTask {
            name: "short_term_memory".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let numbers: Vec<i32> = (0..5).map(|_| rng.gen_range(1..100)).collect();
                let list = numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ");
                (format!("Remember: {}. What was the 3rd number?", list), numbers[2].to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 1.0 } else { 0.0 }
            },
        },
        BenchmarkTask {
            name: "working_memory".to_string(),
            generator: || {
                let instructions = "Add 5 and 3, then multiply by 2, then subtract 4";
                (instructions.to_string(), "12".to_string())
            },
            evaluator: |answer, expected| {
                if answer.contains(expected) { 0.8 } else { 0.0 }
            },
        },
    ]
}
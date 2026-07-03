//! All benchmark tasks for LLM evaluation
//! PRIORITY 8: Comprehensive benchmark suite with proper evaluators

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

/// Proper evaluator: checks if answer contains expected keyword (binary 0.0 or 1.0)
fn contains_evaluator(answer: &str, expected: &str) -> f32 {
    if answer.to_lowercase().contains(&expected.to_lowercase()) { 1.0 } else { 0.0 }
}

/// Partial evaluator: checks for multiple keywords, returns partial credit
fn partial_contains_evaluator(answer: &str, expected: &str) -> f32 {
    let expected_lower = expected.to_lowercase();
    let answer_lower = answer.to_lowercase();
    
    // Split expected by comma for multi-keyword matching
    let keywords: Vec<&str> = expected_lower.split(',').map(|s| s.trim()).collect();
    if keywords.len() <= 1 {
        return if answer_lower.contains(&expected_lower) { 1.0 } else { 0.0 };
    }
    
    let matched = keywords.iter().filter(|k| answer_lower.contains(*k)).count();
    matched as f32 / keywords.len() as f32
}

/// Exact match evaluator
fn exact_match_evaluator(answer: &str, expected: &str) -> f32 {
    if answer.trim().to_lowercase() == expected.trim().to_lowercase() { 1.0 } else { 0.0 }
}

/// Number match evaluator: extracts numbers and compares
fn number_match_evaluator(answer: &str, expected: &str) -> f32 {
    let extract_numbers = |s: &str| -> Vec<i32> {
        s.split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter_map(|w| w.parse::<i32>().ok())
            .collect()
    };
    let answer_nums = extract_numbers(answer);
    let expected_nums = extract_numbers(expected);
    
    if answer_nums.is_empty() || expected_nums.is_empty() {
        return contains_evaluator(answer, expected);
    }
    
    if answer_nums == expected_nums { 1.0 } else { 0.0 }
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
                    ("What a wonderful day", "positive"),
                    ("This is disappointing", "negative"),
                    ("It's okay I guess", "neutral"),
                ];
                let (text, sentiment) = texts[rng.gen_range(0..texts.len())];
                (format!("Sentiment: {}", text), sentiment.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "named_entity".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let entities = vec![
                    ("John lives in Paris", "John, Paris"),
                    ("Apple Inc. is in California", "Apple, California"),
                    ("Microsoft was founded by Bill Gates", "Microsoft, Bill Gates"),
                    ("The Eiffel Tower is in Paris", "Eiffel Tower, Paris"),
                ];
                let (text, entities) = entities[rng.gen_range(0..entities.len())];
                (format!("Entities: {}", text), entities.to_string())
            },
            evaluator: partial_contains_evaluator,
        },
        BenchmarkTask {
            name: "paraphrase_detection".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let pairs = vec![
                    ("The cat sat on the mat", "The mat had a cat sitting on it", "yes"),
                    ("The dog ran fast", "The bird flew high", "no"),
                    ("She sells seashells", "Seashells are sold by her", "yes"),
                    ("The sun is bright", "The moon is dark", "no"),
                ];
                let (s1, s2, expected) = pairs[rng.gen_range(0..pairs.len())];
                (format!("Same meaning? '{}' vs '{}'", s1, s2), expected.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "text_classification".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let texts = vec![
                    ("Breaking news: earthquake hits city", "news"),
                    ("How to bake a cake step by step", "tutorial"),
                    ("Product review: amazing quality", "review"),
                    ("Scientists discover new species", "science"),
                ];
                let (text, category) = texts[rng.gen_range(0..texts.len())];
                (format!("Classify: {}", text), category.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "question_answering".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let qa = vec![
                    ("What is the capital of France?", "Paris"),
                    ("Who wrote Romeo and Juliet?", "Shakespeare"),
                    ("What is the largest planet?", "Jupiter"),
                    ("What is the boiling point of water?", "100"),
                ];
                let (q, a) = qa[rng.gen_range(0..qa.len())];
                (q.to_string(), a.to_string())
            },
            evaluator: contains_evaluator,
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
                    ("All birds have wings. A penguin is a bird. Therefore?", "wings"),
                    ("If you study, you pass. You studied. Therefore?", "pass"),
                ];
                let (premise, conclusion) = problems[rng.gen_range(0..problems.len())];
                (premise.to_string(), conclusion.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "mathematical_reasoning".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let a: i32 = rng.gen_range(1..100);
                let b: i32 = rng.gen_range(1..100);
                (format!("{} + {} = ?", a, b), (a + b).to_string())
            },
            evaluator: number_match_evaluator,
        },
        BenchmarkTask {
            name: "analogical_reasoning".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let analogies = vec![
                    ("doctor is to hospital as teacher is to", "school"),
                    ("bird is to fly as fish is to", "swim"),
                    ("hand is to glove as foot is to", "sock"),
                    ("puppy is to dog as kitten is to", "cat"),
                ];
                let (question, answer) = analogies[rng.gen_range(0..analogies.len())];
                (question.to_string(), answer.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "causal_reasoning".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let causes = vec![
                    ("What happens when you drop a glass?", "break"),
                    ("What happens when you heat ice?", "melt"),
                    ("What happens when you plant a seed?", "grow"),
                    ("What happens when you push a ball?", "roll"),
                ];
                let (q, a) = causes[rng.gen_range(0..causes.len())];
                (q.to_string(), a.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "arithmetic_chain".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let a: i32 = rng.gen_range(1..20);
                let b: i32 = rng.gen_range(1..10);
                let c: i32 = rng.gen_range(1..10);
                let result = a + b * c;
                (format!("{} + {} * {} = ?", a, b, c), result.to_string())
            },
            evaluator: number_match_evaluator,
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
                    ("if x > 0:\n    print(", "positive"),
                    ("while count < 10:\n    count += ", "1"),
                ];
                let (code, completion) = snippets[0];
                (code.to_string(), completion.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "bug_detection".to_string(),
            generator: || {
                let bugs = vec![
                    ("x = 10\ny = 0\nz = x / y", "division by zero"),
                    ("arr = [1,2,3]\nprint(arr[3])", "index out of range"),
                    ("def foo():\n    return x\nprint(foo())", "undefined"),
                    ("x = 'hello'\nx = x + 5", "type error"),
                ];
                let (code, bug) = bugs[0];
                (format!("Find bug: {}", code), bug.to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "code_explanation".to_string(),
            generator: || {
                let snippets = vec![
                    ("def square(x): return x * x", "returns the square"),
                    ("for i in range(5): print(i)", "prints numbers"),
                    ("if x % 2 == 0: print('even')", "checks if even"),
                ];
                let (code, explanation) = snippets[0];
                (format!("Explain: {}", code), explanation.to_string())
            },
            evaluator: contains_evaluator,
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
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "information_retrieval".to_string(),
            generator: || {
                let context = "Alice has a cat named Whiskers. Bob has a dog named Rex.";
                (format!("Question: What is Alice's cat's name? Context: {}", context), "Whiskers".to_string())
            },
            evaluator: contains_evaluator,
        },
        BenchmarkTask {
            name: "long_context_reasoning".to_string(),
            generator: || {
                let context = "First, John went to the store. Then he met Mary. They decided to go to the park. At the park, they saw a dog. The dog was chasing a ball.";
                (format!("Question: Who did John meet? Context: {}", context), "Mary".to_string())
            },
            evaluator: contains_evaluator,
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
                let idx = rng.gen_range(0..5);
                (format!("Remember: {}. What was the {}th number?", list, idx + 1), numbers[idx].to_string())
            },
            evaluator: number_match_evaluator,
        },
        BenchmarkTask {
            name: "working_memory".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let a: i32 = rng.gen_range(1..10);
                let b: i32 = rng.gen_range(1..10);
                let c: i32 = rng.gen_range(1..5);
                let result = (a + b) * c;
                (format!("Add {} and {}, then multiply by {}. What is the result?", a, b, c), result.to_string())
            },
            evaluator: number_match_evaluator,
        },
        BenchmarkTask {
            name: "instruction_following".to_string(),
            generator: || {
                let mut rng = rand::thread_rng();
                let instructions = vec![
                    ("Write the word 'hello' three times", "hello hello hello"),
                    ("List the numbers 1, 2, 3 in order", "1, 2, 3"),
                    ("Write the opposite of 'hot'", "cold"),
                ];
                let (instr, expected) = instructions[rng.gen_range(0..instructions.len())];
                (instr.to_string(), expected.to_string())
            },
            evaluator: contains_evaluator,
        },
    ]
}

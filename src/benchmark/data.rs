//! Training data generation from benchmarks
//! PRIORITY 8: Comprehensive data generation for all task types

use rand::Rng;

/// Generate synthetic training data for a specific task type.
/// Used by auto-improvement to create targeted training examples.
pub fn generate_for_task(task_name: &str, count: usize) -> Vec<(String, String)> {
    let mut data = Vec::new();
    let mut rng = rand::thread_rng();
    
    match task_name {
        "sentiment_analysis" => {
            for _ in 0..count {
                let sentiment = if rng.gen_bool(0.5) { "positive" } else { "negative" };
                let words = match sentiment {
                    "positive" => vec!["great", "excellent", "amazing", "wonderful", "fantastic"],
                    "negative" => vec!["bad", "terrible", "awful", "horrible", "disappointing"],
                    _ => vec!["okay"],
                };
                let word = words[rng.gen_range(0..words.len())];
                let text = format!("This is {}!", word);
                data.push((format!("Sentiment: {}", text), sentiment.to_string()));
            }
        },
        "named_entity" => {
            for _ in 0..count {
                let entities = [
                    ("Alice works at Google", "Alice, Google"),
                    ("Bob lives in London", "Bob, London"),
                    ("Tesla was founded by Elon Musk", "Tesla, Elon Musk"),
                    ("The Statue of Liberty is in New York", "Statue of Liberty, New York"),
                ];
                let (text, ents) = entities[rng.gen_range(0..entities.len())];
                data.push((format!("Entities: {}", text), ents.to_string()));
            }
        },
        "paraphrase_detection" => {
            for _ in 0..count {
                let pairs = [
                    ("The cat sat on the mat", "The mat had a cat sitting on it", "yes"),
                    ("The dog ran fast", "The bird flew high", "no"),
                    ("She sells seashells", "Seashells are sold by her", "yes"),
                    ("The sun is bright", "The moon is dark", "no"),
                    ("He wrote a letter", "A letter was written by him", "yes"),
                ];
                let (s1, s2, expected) = pairs[rng.gen_range(0..pairs.len())];
                data.push((format!("Same meaning? '{}' vs '{}'", s1, s2), expected.to_string()));
            }
        },
        "text_classification" => {
            for _ in 0..count {
                let texts = [
                    ("Breaking news: earthquake hits city", "news"),
                    ("How to bake a cake step by step", "tutorial"),
                    ("Product review: amazing quality", "review"),
                    ("Scientists discover new species", "science"),
                    ("Stock market reaches new high", "finance"),
                ];
                let (text, category) = texts[rng.gen_range(0..texts.len())];
                data.push((format!("Classify: {}", text), category.to_string()));
            }
        },
        "question_answering" => {
            for _ in 0..count {
                let qa = [
                    ("What is the capital of France?", "Paris"),
                    ("Who wrote Romeo and Juliet?", "Shakespeare"),
                    ("What is the largest planet?", "Jupiter"),
                    ("What is the boiling point of water?", "100"),
                    ("What is the speed of light?", "299792458"),
                ];
                let (q, a) = qa[rng.gen_range(0..qa.len())];
                data.push((q.to_string(), a.to_string()));
            }
        },
        "mathematical_reasoning" => {
            for _ in 0..count {
                let a: i32 = rng.gen_range(1..100);
                let b: i32 = rng.gen_range(1..100);
                let result = a + b;
                data.push((format!("{} + {} = ?", a, b), result.to_string()));
            }
        },
        "logical_deduction" => {
            for _ in 0..count {
                let problems = [
                    ("All humans are mortal. Socrates is human. Therefore?", "mortal"),
                    ("If it rains, ground gets wet. It rained. Therefore?", "wet"),
                    ("All birds have wings. A penguin is a bird. Therefore?", "wings"),
                    ("If you study, you pass. You studied. Therefore?", "pass"),
                    ("All mammals have fur. A whale is a mammal. Therefore?", "fur"),
                ];
                let (premise, conclusion) = problems[rng.gen_range(0..problems.len())];
                data.push((premise.to_string(), conclusion.to_string()));
            }
        },
        "analogical_reasoning" => {
            for _ in 0..count {
                let analogies = [
                    ("doctor is to hospital as teacher is to", "school"),
                    ("bird is to fly as fish is to", "swim"),
                    ("hand is to glove as foot is to", "sock"),
                    ("puppy is to dog as kitten is to", "cat"),
                    ("rain is to umbrella as sun is to", "sunglasses"),
                ];
                let (question, answer) = analogies[rng.gen_range(0..analogies.len())];
                data.push((question.to_string(), answer.to_string()));
            }
        },
        "causal_reasoning" => {
            for _ in 0..count {
                let causes = [
                    ("What happens when you drop a glass?", "break"),
                    ("What happens when you heat ice?", "melt"),
                    ("What happens when you plant a seed?", "grow"),
                    ("What happens when you push a ball?", "roll"),
                    ("What happens when you mix red and blue?", "purple"),
                ];
                let (q, a) = causes[rng.gen_range(0..causes.len())];
                data.push((q.to_string(), a.to_string()));
            }
        },
        "arithmetic_chain" => {
            for _ in 0..count {
                let a: i32 = rng.gen_range(1..20);
                let b: i32 = rng.gen_range(1..10);
                let c: i32 = rng.gen_range(1..10);
                let result = a + b * c;
                data.push((format!("{} + {} * {} = ?", a, b, c), result.to_string()));
            }
        },
        "short_term_memory" => {
            for _ in 0..count {
                let numbers: Vec<i32> = (0..5).map(|_| rng.gen_range(1..100)).collect();
                let idx = rng.gen_range(0..5);
                let list = numbers.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ");
                data.push((
                    format!("Remember: {}. What was the {}th number?", list, idx + 1),
                    numbers[idx].to_string()
                ));
            }
        },
        "working_memory" => {
            for _ in 0..count {
                let a: i32 = rng.gen_range(1..10);
                let b: i32 = rng.gen_range(1..10);
                let c: i32 = rng.gen_range(1..5);
                let result = (a + b) * c;
                data.push((
                    format!("Add {} and {}, then multiply by {}. What is the result?", a, b, c),
                    result.to_string()
                ));
            }
        },
        "instruction_following" => {
            for _ in 0..count {
                let instructions = [
                    ("Write the word 'hello' three times", "hello hello hello"),
                    ("List the numbers 1, 2, 3 in order", "1, 2, 3"),
                    ("Write the opposite of 'hot'", "cold"),
                    ("Write the opposite of 'fast'", "slow"),
                ];
                let (instr, expected) = instructions[rng.gen_range(0..instructions.len())];
                data.push((instr.to_string(), expected.to_string()));
            }
        },
        "code_completion" => {
            for _ in 0..count {
                let snippets = [
                    ("def add(a, b):\n    return ", "a + b"),
                    ("for i in range(10):\n    print(", "i)"),
                    ("if x > 0:\n    print(", "positive"),
                    ("while count < 10:\n    count += ", "1"),
                    ("def square(x):\n    return x ", "* x"),
                ];
                let (code, completion) = snippets[rng.gen_range(0..snippets.len())];
                data.push((code.to_string(), completion.to_string()));
            }
        },
        "bug_detection" => {
            for _ in 0..count {
                let bugs = [
                    ("x = 10\ny = 0\nz = x / y", "division by zero"),
                    ("arr = [1,2,3]\nprint(arr[3])", "index out of range"),
                    ("def foo():\n    return x\nprint(foo())", "undefined"),
                    ("x = 'hello'\nx = x + 5", "type error"),
                    ("print('hello'", "syntax error"),
                ];
                let (code, bug) = bugs[rng.gen_range(0..bugs.len())];
                data.push((format!("Find bug: {}", code), bug.to_string()));
            }
        },
        "code_explanation" => {
            for _ in 0..count {
                let snippets = [
                    ("def square(x): return x * x", "returns the square"),
                    ("for i in range(5): print(i)", "prints numbers"),
                    ("if x % 2 == 0: print('even')", "checks if even"),
                    ("while True: print('loop')", "infinite loop"),
                ];
                let (code, explanation) = snippets[rng.gen_range(0..snippets.len())];
                data.push((format!("Explain: {}", code), explanation.to_string()));
            }
        },
        "long_summary" => {
            for _ in 0..count {
                let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
                data.push((text, "fox jumps over dog".to_string()));
            }
        },
        "information_retrieval" => {
            for _ in 0..count {
                let contexts = [
                    ("Alice has a cat named Whiskers. Bob has a dog named Rex.", "Whiskers", "Alice's cat"),
                    ("The capital of Japan is Tokyo. The capital of France is Paris.", "Tokyo", "capital of Japan"),
                    ("Einstein developed relativity. Newton developed gravity.", "relativity", "Einstein developed"),
                ];
                let (ctx, answer, _) = contexts[rng.gen_range(0..contexts.len())];
                data.push((format!("Question: What is {}? Context: {}", answer, ctx), answer.to_string()));
            }
        },
        "long_context_reasoning" => {
            for _ in 0..count {
                let contexts = [
                    ("First, John went to the store. Then he met Mary. They went to the park.", "Mary", "John meet"),
                    ("Alice baked a cake. Bob made coffee. Charlie set the table.", "coffee", "Bob made"),
                    ("The sun rose. The birds sang. The flowers bloomed.", "birds", "sang"),
                ];
                let (ctx, answer, _) = contexts[rng.gen_range(0..contexts.len())];
                data.push((format!("Question: Who did {}? Context: {}", answer, ctx), answer.to_string()));
            }
        },
        _ => {
            // Generic Q&A pairs for unknown task types
            for i in 0..count {
                let questions = [
                    ("What is the color of the sky?", "blue"),
                    ("How many legs does a dog have?", "4"),
                    ("What is 2 + 2?", "4"),
                    ("What color is grass?", "green"),
                ];
                let (q, a) = questions[i % questions.len()];
                data.push((q.to_string(), a.to_string()));
            }
        }
    }
    
    data
}

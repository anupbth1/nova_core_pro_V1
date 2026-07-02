//! Training data generation from benchmarks

use rand::Rng;

pub fn generate_for_task(task_name: &str, count: usize) -> Vec<(String, String)> {
    let mut data = Vec::new();
    let mut rng = rand::thread_rng();
    
    match task_name {
        "sentiment_analysis" => {
            for _ in 0..count {
                let sentiment = if rng.gen_bool(0.5) { "positive" } else { "negative" };
                let words = match sentiment {
                    "positive" => vec!["great", "excellent", "amazing", "wonderful"],
                    "negative" => vec!["bad", "terrible", "awful", "horrible"],
                    _ => vec!["okay"],
                };
                let word = words[rng.gen_range(0..words.len())];
                let text = format!("This is {}!", word);
                data.push((format!("Sentiment: {}", text), sentiment.to_string()));
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
        _ => {
            // Generic Q&A pairs
            for i in 0..count {
                data.push((
                    format!("What is question {}?", i),
                    format!("Answer {}", i)
                ));
            }
        }
    }
    
    data
}
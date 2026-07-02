//! Advanced metrics for LLM evaluation

use std::collections::HashMap;

pub struct Metrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub perplexity: f32,
}

pub fn calculate_metrics(predictions: &[String], targets: &[String]) -> Metrics {
    let mut correct = 0;
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        if pred == target {
            correct += 1;
            tp += 1;
        } else {
            fp += 1;
            fn_ += 1;
        }
    }
    
    let accuracy = correct as f32 / predictions.len() as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else { 0.0 };
    
    Metrics {
        accuracy,
        precision,
        recall,
        f1_score,
        perplexity: 1.0 / accuracy.max(0.01),
    }
}
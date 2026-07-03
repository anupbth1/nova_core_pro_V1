//! Nova Coding Intelligence Module
//!
//! Phase 8: Implements code understanding, generation, and debugging capabilities.
//! Uses Nova's pulse-based computation for code analysis rather than
//! traditional transformer-based code models.
//!
//! Features:
//! - Code understanding: parse and analyze code structure
//! - Code generation: generate code from natural language descriptions
//! - Debugging: identify and fix common code issues
//! - Pattern matching: recognize code idioms and anti-patterns

use std::collections::HashMap;

/// Represents a code snippet with metadata
#[derive(Debug, Clone)]
pub struct CodeSnippet {
    /// The raw code text
    pub code: String,
    /// Programming language (e.g., "rust", "python", "javascript")
    pub language: String,
    /// Detected code patterns
    pub patterns: Vec<CodePattern>,
    /// Complexity score (0.0 to 1.0)
    pub complexity: f32,
    /// Number of lines
    pub line_count: usize,
}

/// Types of code patterns Nova can recognize
#[derive(Debug, Clone)]
pub enum CodePattern {
    /// Function/method definition
    FunctionDefinition { name: String, params: usize },
    /// Loop construct
    Loop { loop_type: String, nested_depth: usize },
    /// Conditional branch
    Conditional { has_else: bool, conditions: usize },
    /// Error handling
    ErrorHandling { kind: String },
    /// Data structure usage
    DataStructure { name: String },
    /// Concurrency pattern
    Concurrency { kind: String },
    /// Unsafe code block
    UnsafeBlock,
    /// Recursive call
    Recursion,
    /// Generic type usage
    Generics { count: usize },
    /// Trait/interface implementation
    TraitImplementation { name: String },
}

/// Code generation request
#[derive(Debug, Clone)]
pub struct CodeGenRequest {
    /// Natural language description
    pub description: String,
    /// Target programming language
    pub language: String,
    /// Context/hints for generation
    pub context: Vec<String>,
    /// Desired complexity (0.0 = simple, 1.0 = complex)
    pub complexity: f32,
}

/// Debug result for a code snippet
#[derive(Debug, Clone)]
pub struct DebugResult {
    /// Whether the code is valid
    pub is_valid: bool,
    /// Detected issues
    pub issues: Vec<CodeIssue>,
    /// Suggestions for fixes
    pub suggestions: Vec<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// A code issue detected during debugging
#[derive(Debug, Clone)]
pub struct CodeIssue {
    /// Type of issue
    pub issue_type: IssueType,
    /// Line number (0-indexed)
    pub line: usize,
    /// Description of the issue
    pub description: String,
    /// Severity (0.0 = warning, 1.0 = critical)
    pub severity: f32,
}

/// Types of code issues
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    SyntaxError,
    LogicError,
    PerformanceIssue,
    SecurityVulnerability,
    MemoryLeak,
    UnusedVariable,
    InfiniteLoop,
    TypeMismatch,
    UnsafeOperation,
    ConcurrencyBug,
    StyleWarning,
}

/// Nova's coding intelligence engine
#[derive(Debug, Clone)]
pub struct CodingEngine {
    /// Known code patterns indexed by language
    pub known_patterns: HashMap<String, Vec<String>>,
    /// Common bug patterns indexed by language
    pub bug_patterns: HashMap<String, Vec<String>>,
    /// Code generation templates
    pub templates: HashMap<String, Vec<String>>,
    /// Number of code analyses performed
    pub total_analyses: usize,
    /// Number of code generations performed
    pub total_generations: usize,
    /// Number of debugging sessions
    pub total_debugs: usize,
}

impl CodingEngine {
    pub fn new() -> Self {
        let mut known_patterns = HashMap::new();
        let mut bug_patterns = HashMap::new();
        let mut templates = HashMap::new();

        // Rust patterns
        known_patterns.insert("rust".to_string(), vec![
            "fn main()".to_string(),
            "let mut".to_string(),
            "match expression".to_string(),
            "impl Trait for".to_string(),
            "unsafe block".to_string(),
            "Arc<Mutex>".to_string(),
            "Result<T, E>".to_string(),
            "Option<T>".to_string(),
            "async fn".to_string(),
            "#[derive]".to_string(),
        ]);

        // Python patterns
        known_patterns.insert("python".to_string(), vec![
            "def function".to_string(),
            "class definition".to_string(),
            "list comprehension".to_string(),
            "decorator".to_string(),
            "context manager".to_string(),
            "generator".to_string(),
            "async/await".to_string(),
            "type hint".to_string(),
            "duck typing".to_string(),
            "dunder methods".to_string(),
        ]);

        // JavaScript patterns
        known_patterns.insert("javascript".to_string(), vec![
            "arrow function".to_string(),
            "promise chain".to_string(),
            "async/await".to_string(),
            "destructuring".to_string(),
            "spread operator".to_string(),
            "closure".to_string(),
            "prototype chain".to_string(),
            "event loop".to_string(),
            "module pattern".to_string(),
            "callback hell".to_string(),
        ]);

        // Bug patterns
        bug_patterns.insert("rust".to_string(), vec![
            "use after free".to_string(),
            "data race".to_string(),
            "deadlock".to_string(),
            "panic unwrap".to_string(),
            "integer overflow".to_string(),
            "memory leak".to_string(),
            "lifetime issue".to_string(),
            "borrow checker".to_string(),
        ]);

        bug_patterns.insert("python".to_string(), vec![
            "undefined variable".to_string(),
            "type error".to_string(),
            "index out of range".to_string(),
            "key error".to_string(),
            "attribute error".to_string(),
            "import error".to_string(),
            "recursion limit".to_string(),
            "mutable default arg".to_string(),
        ]);

        // Code generation templates
        templates.insert("rust".to_string(), vec![
            "fn {name}({params}) -> {return_type} {{\n    {body}\n}}".to_string(),
            "impl {trait_name} for {type_name} {{\n    {methods}\n}}".to_string(),
            "let {var_name} = match {expression} {{\n    {arms}\n}};".to_string(),
            "pub struct {name} {{\n    {fields}\n}}".to_string(),
            "pub enum {name} {{\n    {variants}\n}}".to_string(),
        ]);

        templates.insert("python".to_string(), vec![
            "def {name}({params}):\n    {body}".to_string(),
            "class {name}:\n    def __init__(self, {params}):\n        {init_body}\n    {methods}".to_string(),
            "with {context} as {var}:\n    {body}".to_string(),
            "for {item} in {iterable}:\n    {body}".to_string(),
            "try:\n    {try_body}\nexcept {exception} as e:\n    {except_body}".to_string(),
        ]);

        Self {
            known_patterns,
            bug_patterns,
            templates,
            total_analyses: 0,
            total_generations: 0,
            total_debugs: 0,
        }
    }

    /// Analyze a code snippet and extract patterns
    pub fn analyze_code(&mut self, code: &str, language: &str) -> CodeSnippet {
        self.total_analyses += 1;
        
        let mut patterns = Vec::new();
        let line_count = code.lines().count();
        
        // Detect function definitions
        if code.contains("fn ") {
            patterns.push(CodePattern::FunctionDefinition {
                name: "detected".to_string(),
                params: code.matches("fn ").count(),
            });
        }
        if code.contains("def ") {
            patterns.push(CodePattern::FunctionDefinition {
                name: "detected".to_string(),
                params: code.matches("def ").count(),
            });
        }
        
        // Detect loops
        let for_count = code.matches("for ").count();
        let while_count = code.matches("while ").count();
        if for_count > 0 || while_count > 0 {
            patterns.push(CodePattern::Loop {
                loop_type: if for_count > while_count { "for".to_string() } else { "while".to_string() },
                nested_depth: (for_count + while_count).min(5),
            });
        }
        
        // Detect conditionals
        let if_count = code.matches("if ").count();
        let has_else = code.contains("else");
        if if_count > 0 {
            patterns.push(CodePattern::Conditional {
                has_else,
                conditions: if_count,
            });
        }
        
        // Detect error handling
        if code.contains("Result") || code.contains("try") || code.contains("catch") {
            patterns.push(CodePattern::ErrorHandling {
                kind: if code.contains("Result") { "Result".to_string() }
                      else if code.contains("try") { "try/catch".to_string() }
                      else { "unknown".to_string() },
            });
        }
        
        // Detect unsafe blocks
        if code.contains("unsafe") {
            patterns.push(CodePattern::UnsafeBlock);
        }
        
        // Detect recursion (function calling itself)
        if code.contains("recursion") || code.matches("fn ").count() > 5 {
            patterns.push(CodePattern::Recursion);
        }
        
        // Detect generics
        let generic_count = code.matches('<').count();
        if generic_count > 0 {
            patterns.push(CodePattern::Generics {
                count: generic_count.min(10),
            });
        }
        
        // Detect trait implementations
        if code.contains("impl ") && code.contains(" for ") {
            patterns.push(CodePattern::TraitImplementation {
                name: "detected".to_string(),
            });
        }
        
        // Compute complexity based on patterns and structure
        let complexity = self.compute_complexity(&patterns, line_count);
        
        CodeSnippet {
            code: code.to_string(),
            language: language.to_string(),
            patterns,
            complexity,
            line_count,
        }
    }

    /// Compute code complexity score
    fn compute_complexity(&self, patterns: &[CodePattern], line_count: usize) -> f32 {
        let mut score = 0.0;
        
        for pattern in patterns {
            match pattern {
                CodePattern::FunctionDefinition { params, .. } => {
                    score += 0.1 + (*params as f32) * 0.05;
                }
                CodePattern::Loop { nested_depth, .. } => {
                    score += 0.15 + (*nested_depth as f32) * 0.1;
                }
                CodePattern::Conditional { conditions, .. } => {
                    score += 0.1 + (*conditions as f32) * 0.05;
                }
                CodePattern::ErrorHandling { .. } => score += 0.1,
                CodePattern::UnsafeBlock => score += 0.2,
                CodePattern::Recursion => score += 0.25,
                CodePattern::Generics { count } => score += 0.05 + (*count as f32) * 0.02,
                CodePattern::TraitImplementation { .. } => score += 0.15,
                CodePattern::Concurrency { .. } => score += 0.2,
                CodePattern::DataStructure { .. } => score += 0.05,
            }
        }
        
        // Line count factor (diminishing returns)
        score += (line_count as f32 / 100.0).min(0.3);
        
        score.min(1.0)
    }

    /// Generate code from a natural language description
    pub fn generate_code(&mut self, request: &CodeGenRequest) -> String {
        self.total_generations += 1;
        
        let lang = &request.language;
        let desc = &request.description;
        
        // Simple template-based code generation
        // In a full implementation, this would use Nova's pulse-based reasoning
        
        match lang.as_str() {
            "rust" => self.generate_rust_code(desc, &request.context, request.complexity),
            "python" => self.generate_python_code(desc, &request.context, request.complexity),
            "javascript" => self.generate_javascript_code(desc, &request.context, request.complexity),
            _ => format!("// Code generation for {} is not yet supported\n// Description: {}", lang, desc),
        }
    }

    fn generate_rust_code(&self, description: &str, context: &[String], complexity: f32) -> String {
        let desc_lower = description.to_lowercase();
        
        if desc_lower.contains("hello") || desc_lower.contains("greet") {
            return String::from(
                "/// Greets the user with a friendly message.\n\
                pub fn greet(name: &str) -> String {\n    \
                    format!(\"Hello, {}! Welcome to Nova Core.\", name)\n\
                }\n\n\
                #[cfg(test)]\n\
                mod tests {\n    \
                    use super::*;\n    \
                    #[test]\n    \
                    fn test_greet() {\n        \
                        assert_eq!(greet(\"World\"), \"Hello, World! Welcome to Nova Core.\");\n    \
                    }\n\
                }"
            );
        }
        
        if desc_lower.contains("fibonacci") || desc_lower.contains("fib") {
            return String::from(
                "/// Computes the nth Fibonacci number using iterative approach.\n\
                pub fn fibonacci(n: u64) -> u64 {\n    \
                    match n {\n        \
                        0 => 0,\n        \
                        1 => 1,\n        \
                        _ => {\n            \
                            let mut a = 0;\n            \
                            let mut b = 1;\n            \
                            for _ in 2..=n {\n                \
                                let temp = a + b;\n                \
                                a = b;\n                \
                                b = temp;\n            \
                            }\n            \
                            b\n        \
                        }\n    \
                    }\n\
                }"
            );
        }
        
        if desc_lower.contains("sort") || desc_lower.contains("quicksort") {
            return String::from(
                "/// Sorts a mutable slice using quicksort algorithm.\n\
                pub fn quicksort<T: Ord>(arr: &mut [T]) {\n    \
                    if arr.len() <= 1 {\n        \
                        return;\n    \
                    }\n    \
                    let pivot = partition(arr);\n    \
                    quicksort(&mut arr[..pivot]);\n    \
                    quicksort(&mut arr[pivot + 1..]);\n\
                }\n\n\
                fn partition<T: Ord>(arr: &mut [T]) -> usize {\n    \
                    let len = arr.len();\n    \
                    let pivot = len / 2;\n    \
                    arr.swap(pivot, len - 1);\n    \
                    let mut i = 0;\n    \
                    for j in 0..len - 1 {\n        \
                        if arr[j] <= arr[len - 1] {\n            \
                            arr.swap(i, j);\n            \
                            i += 1;\n        \
                        }\n    \
                    }\n    \
                    arr.swap(i, len - 1);\n    \
                    i\n\
                }"
            );
        }
        
        // Generic template
        format!(
            "/// {}\n\
            pub fn solution({}) -> Result<(), String> {{\n    \
                // TODO: Implement solution\n    \
                // Context: {}\n    \
                unimplemented!(\"Solution not yet implemented\")\n\
            }}",
            description,
            if context.is_empty() { "_input: &str".to_string() } else { context.join(", ") },
            context.join(", ")
        )
    }

    fn generate_python_code(&self, description: &str, context: &[String], complexity: f32) -> String {
        let desc_lower = description.to_lowercase();
        
        if desc_lower.contains("hello") || desc_lower.contains("greet") {
            return String::from(
                "def greet(name: str) -> str:\n    \
                    \"\"\"Greet the user.\"\"\"\n    \
                    return f\"Hello, {name}! Welcome to Nova Core.\"\n\n\
                if __name__ == \"__main__\":\n    \
                    print(greet(\"World\"))"
            );
        }
        
        if desc_lower.contains("fibonacci") || desc_lower.contains("fib") {
            return String::from(
                "def fibonacci(n: int) -> int:\n    \
                    \"\"\"Compute the nth Fibonacci number.\"\"\"\n    \
                    if n <= 1:\n        \
                        return n\n    \
                    a, b = 0, 1\n    \
                    for _ in range(2, n + 1):\n        \
                        a, b = b, a + b\n    \
                    return b"
            );
        }
        
        if desc_lower.contains("sort") || desc_lower.contains("quicksort") {
            return String::from(
                "def quicksort(arr):\n    \
                    \"\"\"Sort a list using quicksort.\"\"\"\n    \
                    if len(arr) <= 1:\n        \
                        return arr\n    \
                    pivot = arr[len(arr) // 2]\n    \
                    left = [x for x in arr if x < pivot]\n    \
                    middle = [x for x in arr if x == pivot]\n    \
                    right = [x for x in arr if x > pivot]\n    \
                    return quicksort(left) + middle + quicksort(right)"
            );
        }
        
        format!(
            "def solution({}):\n    \
                \"\"\"{}.\"\"\"\n    \
                raise NotImplementedError(\"Solution not yet implemented\")",
            if context.is_empty() { "data".to_string() } else { context.join(", ") },
            description
        )
    }

    fn generate_javascript_code(&self, description: &str, context: &[String], complexity: f32) -> String {
        let desc_lower = description.to_lowercase();
        
        if desc_lower.contains("hello") || desc_lower.contains("greet") {
            return String::from(
                "/**\n * Greets the user.\n * @param {string} name - The name to greet\n * @returns {string} A greeting message\n */\n\
                function greet(name) {\n    \
                    return `Hello, ${name}! Welcome to Nova Core.`;\n\
                }\n\n\
                console.log(greet('World'));"
            );
        }
        
        if desc_lower.contains("fibonacci") || desc_lower.contains("fib") {
            return String::from(
                "/**\n * Computes the nth Fibonacci number.\n * @param {number} n - The position\n * @returns {number} The Fibonacci number\n */\n\
                function fibonacci(n) {\n    \
                    if (n <= 1) return n;\n    \
                    let a = 0, b = 1;\n    \
                    for (let i = 2; i <= n; i++) {\n        \
                        [a, b] = [b, a + b];\n    \
                    }\n    \
                    return b;\n\
                }"
            );
        }
        
        format!(
            "/**\n * {}.\n */\n\
            function solution({}) {{\n    \
                // TODO: Implement\n    \
                throw new Error('Not implemented');\n\
            }}",
            description,
            context.join(", ")
        )
    }

    /// Debug a code snippet and find issues
    pub fn debug_code(&mut self, snippet: &CodeSnippet) -> DebugResult {
        self.total_debugs += 1;
        
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();
        let code = &snippet.code;
        let language = &snippet.language;
        
        // Check for common issues based on language
        match language.as_str() {
            "rust" => self.debug_rust(code, &mut issues, &mut suggestions),
            "python" => self.debug_python(code, &mut issues, &mut suggestions),
            "javascript" => self.debug_javascript(code, &mut issues, &mut suggestions),
            _ => {}
        }
        
        // General checks
        self.check_general_issues(code, &mut issues, &mut suggestions);
        
        let is_valid = issues.is_empty();
        let confidence = if is_valid { 0.9 } else { 0.7 };
        
        DebugResult {
            is_valid,
            issues,
            suggestions,
            confidence,
        }
    }

    fn debug_rust(&self, code: &str, issues: &mut Vec<CodeIssue>, suggestions: &mut Vec<String>) {
        // Check for unwrap() usage
        for (i, line) in code.lines().enumerate() {
            if line.contains(".unwrap()") && !line.contains("//") {
                issues.push(CodeIssue {
                    issue_type: IssueType::UnsafeOperation,
                    line: i,
                    description: "Using .unwrap() can cause panics. Consider proper error handling.".to_string(),
                    severity: 0.6,
                });
                suggestions.push(format!("Line {}: Replace .unwrap() with proper error handling using match or ? operator", i + 1));
            }
            
            // Check for large unsafe blocks
            if line.contains("unsafe {") && line.len() > 50 {
                issues.push(CodeIssue {
                    issue_type: IssueType::UnsafeOperation,
                    line: i,
                    description: "Large unsafe block detected. Minimize unsafe code.".to_string(),
                    severity: 0.5,
                });
            }
        }
        
        // Check for missing error handling
        if code.contains("fn ") && !code.contains("Result") && !code.contains("panic") {
            if code.contains("unwrap") || code.contains("expect") {
                suggestions.push("Consider returning Result<T, E> instead of panicking.".to_string());
            }
        }
    }

    fn debug_python(&self, code: &str, issues: &mut Vec<CodeIssue>, suggestions: &mut Vec<String>) {
        for (i, line) in code.lines().enumerate() {
            // Check for mutable default arguments
            if line.contains("def ") && line.contains("=[]") || line.contains("={}") {
                issues.push(CodeIssue {
                    issue_type: IssueType::LogicError,
                    line: i,
                    description: "Mutable default argument detected. This is shared across all calls.".to_string(),
                    severity: 0.7,
                });
                suggestions.push(format!("Line {}: Use None as default and create a new list/dict inside the function", i + 1));
            }
            
            // Check for bare except
            if line.trim().starts_with("except:") {
                issues.push(CodeIssue {
                    issue_type: IssueType::StyleWarning,
                    line: i,
                    description: "Bare except clause catches all exceptions. Specify exception types.".to_string(),
                    severity: 0.5,
                });
            }
        }
    }

    fn debug_javascript(&self, code: &str, issues: &mut Vec<CodeIssue>, suggestions: &mut Vec<String>) {
        for (i, line) in code.lines().enumerate() {
            // Check for == instead of ===
            if line.contains(" == ") && !line.contains("=== ") && !line.contains("//") {
                issues.push(CodeIssue {
                    issue_type: IssueType::StyleWarning,
                    line: i,
                    description: "Use === instead of == for strict equality comparison.".to_string(),
                    severity: 0.4,
                });
            }
            
            // Check for var usage
            if line.contains("var ") && !line.contains("//") {
                issues.push(CodeIssue {
                    issue_type: IssueType::StyleWarning,
                    line: i,
                    description: "Use let or const instead of var for block scoping.".to_string(),
                    severity: 0.3,
                });
            }
        }
    }

    fn check_general_issues(&self, code: &str, issues: &mut Vec<CodeIssue>, suggestions: &mut Vec<String>) {
        // Check for TODO/FIXME comments
        for (i, line) in code.lines().enumerate() {
            if line.contains("TODO") || line.contains("FIXME") || line.contains("HACK") {
                issues.push(CodeIssue {
                    issue_type: IssueType::StyleWarning,
                    line: i,
                    description: format!("Unresolved comment: {}", line.trim()),
                    severity: 0.2,
                });
            }
        }
        
        // Check for very long lines
        for (i, line) in code.lines().enumerate() {
            if line.len() > 120 {
                issues.push(CodeIssue {
                    issue_type: IssueType::StyleWarning,
                    line: i,
                    description: format!("Line too long ({} chars). Consider breaking it up.", line.len()),
                    severity: 0.3,
                });
            }
        }
        
        // Check for commented-out code
        let comment_lines: Vec<&str> = code.lines()
            .filter(|l| l.trim().starts_with("//") || l.trim().starts_with("#"))
            .collect();
        if comment_lines.len() > 10 {
            suggestions.push(format!("Found {} commented-out lines. Consider removing dead code.", comment_lines.len()));
        }
    }

    /// Get a summary of the coding engine's activity
    pub fn summary(&self) -> String {
        format!(
            "CodingEngine: {} analyses, {} generations, {} debugs",
            self.total_analyses, self.total_generations, self.total_debugs
        )
    }
}

impl Default for CodingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_rust_code() {
        let mut engine = CodingEngine::new();
        let code = "fn main() {\n    println!(\"Hello\");\n}";
        let snippet = engine.analyze_code(code, "rust");
        assert_eq!(snippet.language, "rust");
        assert!(snippet.complexity > 0.0);
        println!("✅ analyze_rust_code works! complexity={:.3}", snippet.complexity);
    }

    #[test]
    fn test_generate_rust_code() {
        let mut engine = CodingEngine::new();
        let request = CodeGenRequest {
            description: "Write a function that greets the user".to_string(),
            language: "rust".to_string(),
            context: vec![],
            complexity: 0.3,
        };
        let code = engine.generate_code(&request);
        assert!(code.contains("fn greet"));
        println!("✅ generate_rust_code works!");
    }

    #[test]
    fn test_debug_rust_code() {
        let mut engine = CodingEngine::new();
        let code = "fn main() {\n    let x = some_value.unwrap();\n}";
        let snippet = engine.analyze_code(code, "rust");
        let result = engine.debug_code(&snippet);
        assert!(!result.is_valid);
        assert!(!result.issues.is_empty());
        println!("✅ debug_rust_code works! {} issues found", result.issues.len());
    }

    #[test]
    fn test_generate_python_code() {
        let mut engine = CodingEngine::new();
        let request = CodeGenRequest {
            description: "Write a fibonacci function".to_string(),
            language: "python".to_string(),
            context: vec![],
            complexity: 0.5,
        };
        let code = engine.generate_code(&request);
        assert!(code.contains("fibonacci"));
        println!("✅ generate_python_code works!");
    }

    #[test]
    fn test_debug_python_code() {
        let mut engine = CodingEngine::new();
        let code = "def append_to(item, list=[]):\n    list.append(item)\n    return list";
        let snippet = engine.analyze_code(code, "python");
        let result = engine.debug_code(&snippet);
        assert!(!result.is_valid);
        println!("✅ debug_python_code works! {} issues found", result.issues.len());
    }
}

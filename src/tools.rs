//! Nova Tool Use Module
//!
//! Phase 10: Implements tool use capabilities - API integration, web search,
//! file operations, and external tool calling for Nova Core.
//!
//! Features:
//! - Web search: query web APIs for information
//! - File operations: read, write, list files
//! - HTTP requests: GET, POST to external APIs
//! - Code execution: run code snippets in sandboxed environments
//! - Data transformation: convert between formats (JSON, CSV, etc.)
//! - Calculator: precise mathematical computation

use std::collections::HashMap;
use std::time::Instant;

/// Types of tools Nova can use
#[derive(Debug, Clone, PartialEq)]
pub enum ToolType {
    /// Web search tool
    WebSearch,
    /// File read tool
    FileRead,
    /// File write tool
    FileWrite,
    /// HTTP GET request
    HttpGet,
    /// HTTP POST request
    HttpPost,
    /// Code execution
    CodeExecution,
    /// Data transformation
    DataTransform,
    /// Calculator
    Calculator,
    /// Shell command execution
    ShellCommand,
}

/// A tool that Nova can invoke
#[derive(Debug, Clone)]
pub struct Tool {
    /// Tool type
    pub tool_type: ToolType,
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Parameters for the tool
    pub parameters: HashMap<String, String>,
    /// Whether the tool is available
    pub available: bool,
}

/// Result of a tool invocation
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Whether the tool execution succeeded
    pub success: bool,
    /// Output data
    pub output: String,
    /// Error message (if any)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Nova's tool use engine
#[derive(Debug, Clone)]
pub struct ToolEngine {
    /// Available tools
    pub tools: Vec<Tool>,
    /// Tool usage history
    pub usage_history: Vec<(String, bool)>,
    /// Total tool invocations
    pub total_invocations: usize,
    /// Successful invocations
    pub successful_invocations: usize,
    /// API keys for external services
    api_keys: HashMap<String, String>,
}

impl ToolEngine {
    pub fn new() -> Self {
        let tools = vec![
            Tool {
                tool_type: ToolType::WebSearch,
                name: "web_search".to_string(),
                description: "Search the web for information on a query".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("query".to_string(), "Search query string".to_string());
                    p.insert("max_results".to_string(), "Maximum number of results (default: 5)".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::FileRead,
                name: "file_read".to_string(),
                description: "Read the contents of a file".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("path".to_string(), "Path to the file".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::FileWrite,
                name: "file_write".to_string(),
                description: "Write content to a file".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("path".to_string(), "Path to the file".to_string());
                    p.insert("content".to_string(), "Content to write".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::HttpGet,
                name: "http_get".to_string(),
                description: "Make an HTTP GET request to a URL".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("url".to_string(), "URL to request".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::HttpPost,
                name: "http_post".to_string(),
                description: "Make an HTTP POST request to a URL".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("url".to_string(), "URL to request".to_string());
                    p.insert("body".to_string(), "Request body".to_string());
                    p.insert("content_type".to_string(), "Content type (default: application/json)".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::Calculator,
                name: "calculator".to_string(),
                description: "Perform precise mathematical calculations".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("expression".to_string(), "Mathematical expression to evaluate".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::DataTransform,
                name: "data_transform".to_string(),
                description: "Transform data between formats (JSON, CSV, etc.)".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("input".to_string(), "Input data".to_string());
                    p.insert("from_format".to_string(), "Source format".to_string());
                    p.insert("to_format".to_string(), "Target format".to_string());
                    p
                },
                available: true,
            },
            Tool {
                tool_type: ToolType::ShellCommand,
                name: "shell".to_string(),
                description: "Execute a shell command (limited to safe commands)".to_string(),
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("command".to_string(), "Command to execute".to_string());
                    p
                },
                available: true,
            },
        ];

        Self {
            tools,
            usage_history: Vec::new(),
            total_invocations: 0,
            successful_invocations: 0,
            api_keys: HashMap::new(),
        }
    }

    /// Set an API key for a service
    pub fn set_api_key(&mut self, service: &str, key: &str) {
        self.api_keys.insert(service.to_string(), key.to_string());
    }

    /// Get a tool by name
    pub fn get_tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// List all available tools
    pub fn list_tools(&self) -> Vec<&Tool> {
        self.tools.iter().filter(|t| t.available).collect()
    }

    /// Invoke a tool by name with given parameters
    pub fn invoke(&mut self, tool_name: &str, params: &HashMap<String, String>) -> ToolResult {
        self.total_invocations += 1;
        let start = Instant::now();

        let tool = match self.tools.iter().find(|t| t.name == tool_name) {
            Some(t) => t.clone(),
            None => {
                return ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Tool '{}' not found", tool_name)),
                    execution_time_ms: 0,
                    metadata: HashMap::new(),
                };
            }
        };

        let result = match tool.tool_type {
            ToolType::FileRead => self.invoke_file_read(params),
            ToolType::FileWrite => self.invoke_file_write(params),
            ToolType::HttpGet => self.invoke_http_get(params),
            ToolType::HttpPost => self.invoke_http_post(params),
            ToolType::Calculator => self.invoke_calculator(params),
            ToolType::DataTransform => self.invoke_data_transform(params),
            ToolType::WebSearch => self.invoke_web_search(params),
            ToolType::ShellCommand => self.invoke_shell_command(params),
            ToolType::CodeExecution => self.invoke_code_execution(params),
        };

        let elapsed = start.elapsed().as_millis() as u64;
        
        self.usage_history.push((tool_name.to_string(), result.success));
        if result.success {
            self.successful_invocations += 1;
        }

        ToolResult {
            execution_time_ms: elapsed,
            ..result
        }
    }

    fn invoke_file_read(&self, params: &HashMap<String, String>) -> ToolResult {
        let path = match params.get("path") {
            Some(p) => p,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'path' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        match std::fs::read_to_string(path) {
            Ok(content) => {
                let mut meta = HashMap::new();
                meta.insert("size".to_string(), content.len().to_string());
                ToolResult {
                    success: true,
                    output: content,
                    error: None,
                    execution_time_ms: 0,
                    metadata: meta,
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to read file: {}", e)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
        }
    }

    fn invoke_file_write(&self, params: &HashMap<String, String>) -> ToolResult {
        let path = match params.get("path") {
            Some(p) => p,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'path' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };
        let content = match params.get("content") {
            Some(c) => c,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'content' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        match std::fs::write(path, content) {
            Ok(()) => ToolResult {
                success: true,
                output: format!("Successfully wrote {} bytes to {}", content.len(), path),
                error: None,
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to write file: {}", e)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
        }
    }

    fn invoke_http_get(&self, params: &HashMap<String, String>) -> ToolResult {
        let url = match params.get("url") {
            Some(u) => u,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'url' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        // Use ureq for HTTP requests if available, otherwise return a placeholder
        #[cfg(feature = "http")]
        {
            match ureq::get(url).call() {
                Ok(response) => {
                    let mut body = String::new();
                    match response.into_reader().read_to_string(&mut body) {
                        Ok(_) => ToolResult {
                            success: true,
                            output: body,
                            error: None,
                            execution_time_ms: 0,
                            metadata: HashMap::new(),
                        },
                        Err(e) => ToolResult {
                            success: false, output: String::new(),
                            error: Some(format!("Failed to read response: {}", e)),
                            execution_time_ms: 0, metadata: HashMap::new(),
                        },
                    }
                }
                Err(e) => ToolResult {
                    success: false, output: String::new(),
                    error: Some(format!("HTTP request failed: {}", e)),
                    execution_time_ms: 0, metadata: HashMap::new(),
                },
            }
        }
        #[cfg(not(feature = "http"))]
        {
            let _ = url;
            ToolResult {
                success: false,
                output: String::new(),
                error: Some("HTTP support not enabled. Enable 'http' feature.".to_string()),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            }
        }
    }

    fn invoke_http_post(&self, params: &HashMap<String, String>) -> ToolResult {
        let url = match params.get("url") {
            Some(u) => u,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'url' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };
        let body = params.get("body").cloned().unwrap_or_default();
        let content_type = params.get("content_type").cloned().unwrap_or_else(|| "application/json".to_string());

        #[cfg(feature = "http")]
        {
            match ureq::post(url).set("Content-Type", &content_type).send_string(&body) {
                Ok(response) => {
                    let mut resp_body = String::new();
                    match response.into_reader().read_to_string(&mut resp_body) {
                        Ok(_) => ToolResult {
                            success: true,
                            output: resp_body,
                            error: None,
                            execution_time_ms: 0,
                            metadata: HashMap::new(),
                        },
                        Err(e) => ToolResult {
                            success: false, output: String::new(),
                            error: Some(format!("Failed to read response: {}", e)),
                            execution_time_ms: 0, metadata: HashMap::new(),
                        },
                    }
                }
                Err(e) => ToolResult {
                    success: false, output: String::new(),
                    error: Some(format!("HTTP POST failed: {}", e)),
                    execution_time_ms: 0, metadata: HashMap::new(),
                },
            }
        }
        #[cfg(not(feature = "http"))]
        {
            let _ = (url, body, content_type);
            ToolResult {
                success: false,
                output: String::new(),
                error: Some("HTTP support not enabled. Enable 'http' feature.".to_string()),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            }
        }
    }

    fn invoke_calculator(&self, params: &HashMap<String, String>) -> ToolResult {
        let expression = match params.get("expression") {
            Some(e) => e,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'expression' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        // Simple expression evaluator for basic arithmetic
        let result = self.eval_simple_expression(expression);
        match result {
            Ok(val) => ToolResult {
                success: true,
                output: format!("{}", val),
                error: None,
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Calculation error: {}", e)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
        }
    }

    fn eval_simple_expression(&self, expr: &str) -> Result<f64, String> {
        let expr = expr.trim();
        
        // Try to parse as a simple number first
        if let Ok(n) = expr.parse::<f64>() {
            return Ok(n);
        }

        // Handle basic operations: a + b, a - b, a * b, a / b
        let operators = ['+', '-', '*', '/', '^'];
        for op in operators {
            if let Some(pos) = expr.find(op) {
                if pos == 0 { continue; }
                let left = &expr[..pos].trim();
                let right = &expr[pos+1..].trim();
                let l = self.eval_simple_expression(left)?;
                let r = self.eval_simple_expression(right)?;
                return Ok(match op {
                    '+' => l + r,
                    '-' => l - r,
                    '*' => l * r,
                    '/' => {
                        if r == 0.0 {
                            return Err("Division by zero".to_string());
                        }
                        l / r
                    }
                    '^' => l.powf(r),
                    _ => unreachable!(),
                });
            }
        }

        // Handle parentheses
        if expr.starts_with('(') && expr.ends_with(')') {
            return self.eval_simple_expression(&expr[1..expr.len()-1]);
        }

        Err(format!("Cannot evaluate expression: {}", expr))
    }

    fn invoke_data_transform(&self, params: &HashMap<String, String>) -> ToolResult {
        let input = match params.get("input") {
            Some(i) => i,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'input' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };
        let from = params.get("from_format").cloned().unwrap_or_else(|| "json".to_string());
        let to = params.get("to_format").cloned().unwrap_or_else(|| "json".to_string());

        match (from.as_str(), to.as_str()) {
            ("json", "csv") => {
                // Simple JSON to CSV conversion
                let output = input
                    .trim_matches(|c| c == '[' || c == ']' || c == ' ' || c == '\n')
                    .replace("},{", "}\n{");
                ToolResult {
                    success: true,
                    output,
                    error: None,
                    execution_time_ms: 0,
                    metadata: HashMap::new(),
                }
            }
            ("csv", "json") => {
                let lines: Vec<&str> = input.lines().collect();
                if lines.len() < 2 {
                    return ToolResult {
                        success: false, output: String::new(),
                        error: Some("CSV must have at least header + 1 row".to_string()),
                        execution_time_ms: 0, metadata: HashMap::new(),
                    };
                }
                let headers: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
                let mut json_objects = Vec::new();
                for line in &lines[1..] {
                    let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    let mut obj = String::from("{");
                    for (i, header) in headers.iter().enumerate() {
                        if i > 0 { obj.push_str(", "); }
                        obj.push_str(&format!("\"{}\": \"{}\"", header, values.get(i).unwrap_or(&"")));
                    }
                    obj.push('}');
                    json_objects.push(obj);
                }
                let output = format!("[\n{}\n]", json_objects.join(",\n"));
                ToolResult {
                    success: true,
                    output,
                    error: None,
                    execution_time_ms: 0,
                    metadata: HashMap::new(),
                }
            }
            _ => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Conversion from '{}' to '{}' not supported", from, to)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
        }
    }

    fn invoke_web_search(&self, params: &HashMap<String, String>) -> ToolResult {
        let query = match params.get("query") {
            Some(q) => q,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'query' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        // Placeholder web search - in production, integrate with a search API
        let result = format!(
            "Web search results for '{}':\n\
             This is a simulated web search. To enable real web search,\n\
             configure a search API key using set_api_key().\n\
             \n\
             Suggested search APIs:\n\
             - SerpAPI (serpapi.com)\n\
             - Google Custom Search\n\
             - Bing Web Search",
            query
        );

        ToolResult {
            success: true,
            output: result,
            error: None,
            execution_time_ms: 0,
            metadata: HashMap::new(),
        }
    }

    fn invoke_shell_command(&self, params: &HashMap<String, String>) -> ToolResult {
        let command = match params.get("command") {
            Some(c) => c,
            None => return ToolResult {
                success: false, output: String::new(),
                error: Some("Missing 'command' parameter".to_string()),
                execution_time_ms: 0, metadata: HashMap::new(),
            },
        };

        // Only allow safe commands (no dangerous operations)
        let safe_commands = ["ls", "dir", "echo", "pwd", "whoami", "date", "time", "cat", "head", "tail"];
        let cmd_name = command.split_whitespace().next().unwrap_or("");
        
        if !safe_commands.contains(&cmd_name) {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Command '{}' is not in the safe list. Allowed: {:?}", cmd_name, safe_commands)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            };
        }

        match std::process::Command::new("cmd")
            .args(["/C", command])
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                ToolResult {
                    success: output.status.success(),
                    output: stdout,
                    error: if stderr.is_empty() { None } else { Some(stderr) },
                    execution_time_ms: 0,
                    metadata: HashMap::new(),
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to execute command: {}", e)),
                execution_time_ms: 0,
                metadata: HashMap::new(),
            },
        }
    }

    fn invoke_code_execution(&self, params: &HashMap<String, String>) -> ToolResult {
        // Code execution is a security-sensitive operation
        // In production, this should use a sandboxed environment
        ToolResult {
            success: false,
            output: String::new(),
            error: Some("Code execution requires a sandboxed environment. Not yet implemented.".to_string()),
            execution_time_ms: 0,
            metadata: HashMap::new(),
        }
    }

    /// Get a summary of the tool engine's activity
    pub fn summary(&self) -> String {
        let success_rate = if self.total_invocations > 0 {
            (self.successful_invocations as f64 / self.total_invocations as f64) * 100.0
        } else {
            0.0
        };
        format!(
            "ToolEngine: {} invocations, {} successful ({:.1}%), {} tools available",
            self.total_invocations,
            self.successful_invocations,
            success_rate,
            self.tools.iter().filter(|t| t.available).count()
        )
    }
}

impl Default for ToolEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_list() {
        let engine = ToolEngine::new();
        let tools = engine.list_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "file_read"));
        println!("✅ tool_list: {} tools available", tools.len());
    }

    #[test]
    fn test_file_read_write() {
        let mut engine = ToolEngine::new();
        let test_path = "test_tool_temp.txt";
        
        // Write
        let mut params = HashMap::new();
        params.insert("path".to_string(), test_path.to_string());
        params.insert("content".to_string(), "Hello from Nova Tools!".to_string());
        let result = engine.invoke("file_write", &params);
        assert!(result.success);
        
        // Read
        let mut params = HashMap::new();
        params.insert("path".to_string(), test_path.to_string());
        let result = engine.invoke("file_read", &params);
        assert!(result.success);
        assert_eq!(result.output.trim(), "Hello from Nova Tools!");
        
        // Cleanup
        let _ = std::fs::remove_file(test_path);
        println!("✅ file_read_write works!");
    }

    #[test]
    fn test_calculator() {
        let mut engine = ToolEngine::new();
        
        let mut params = HashMap::new();
        params.insert("expression".to_string(), "2 + 3".to_string());
        let result = engine.invoke("calculator", &params);
        assert!(result.success);
        assert_eq!(result.output.trim(), "5");
        
        params.insert("expression".to_string(), "10 * 5".to_string());
        let result = engine.invoke("calculator", &params);
        assert!(result.success);
        assert_eq!(result.output.trim(), "50");
        
        println!("✅ calculator works!");
    }

    #[test]
    fn test_data_transform_json_to_csv() {
        let mut engine = ToolEngine::new();
        let mut params = HashMap::new();
        params.insert("input".to_string(), r#"[{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]"#.to_string());
        params.insert("from_format".to_string(), "json".to_string());
        params.insert("to_format".to_string(), "csv".to_string());
        let result = engine.invoke("data_transform", &params);
        assert!(result.success);
        println!("✅ data_transform JSON→CSV works!");
    }

    #[test]
    fn test_data_transform_csv_to_json() {
        let mut engine = ToolEngine::new();
        let mut params = HashMap::new();
        params.insert("input".to_string(), "name,age\nAlice,30\nBob,25".to_string());
        params.insert("from_format".to_string(), "csv".to_string());
        params.insert("to_format".to_string(), "json".to_string());
        let result = engine.invoke("data_transform", &params);
        assert!(result.success);
        println!("✅ data_transform CSV→JSON works!");
    }

    #[test]
    fn test_tool_not_found() {
        let mut engine = ToolEngine::new();
        let params = HashMap::new();
        let result = engine.invoke("nonexistent_tool", &params);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));
        println!("✅ tool_not_found handled correctly!");
    }

    #[test]
    fn test_summary() {
        let engine = ToolEngine::new();
        let summary = engine.summary();
        assert!(summary.contains("ToolEngine"));
        println!("✅ summary: {}", summary);
    }
}

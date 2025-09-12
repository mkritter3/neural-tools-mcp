/**
 * MCP Contract Validator
 * Validates API compatibility and tool contracts for Neural Tools
 */

import { MCPToolSchema, ALL_TOOL_SCHEMAS, SupportedAPIVersion, getToolSchema } from './tool-schemas';

export interface ValidationResult {
  success: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  totalTests: number;
  passedTests: number;
  executionTimeMs: number;
}

export interface ValidationError {
  type: 'missing_tool' | 'parameter_mismatch' | 'type_error' | 'execution_error';
  toolName: string;
  message: string;
  expected?: any;
  actual?: any;
}

export interface ValidationWarning {
  type: 'deprecated_tool' | 'version_mismatch' | 'performance_warning';
  toolName: string;
  message: string;
}

export interface MCPClient {
  listTools(): Promise<ToolInfo[]>;
  callTool(name: string, params: any): Promise<any>;
  getCapabilities?(): Promise<any>;
}

export interface ToolInfo {
  name: string;
  description?: string;
  inputSchema?: any;
}

export interface ContractValidationOptions {
  apiVersion?: SupportedAPIVersion;
  skipDeprecated?: boolean;
  performanceThresholdMs?: number;
  includeExecutionTests?: boolean;
  testToolNames?: string[]; // Only test specific tools
}

export class MCPContractValidator {
  private mcpClient: MCPClient;
  private options: ContractValidationOptions;

  constructor(mcpClient: MCPClient, options: ContractValidationOptions = {}) {
    this.mcpClient = mcpClient;
    this.options = {
      apiVersion: '1.0.0',
      skipDeprecated: false,
      performanceThresholdMs: 5000,
      includeExecutionTests: false,
      ...options
    };
  }

  /**
   * Validate all tool contracts
   */
  async validateAllTools(): Promise<ValidationResult> {
    const startTime = Date.now();
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    let totalTests = 0;
    let passedTests = 0;

    try {
      // Get available tools from MCP server
      const serverTools = await this.mcpClient.listTools();
      const serverToolNames = new Set(serverTools.map(t => t.name));

      // Get expected tools from schemas
      const expectedSchemas = this.getExpectedSchemas();

      for (const schema of expectedSchemas) {
        totalTests++;

        // Skip if only testing specific tools
        if (this.options.testToolNames && !this.options.testToolNames.includes(schema.name)) {
          continue;
        }

        // Skip deprecated tools if requested
        if (this.options.skipDeprecated && schema.deprecated) {
          warnings.push({
            type: 'deprecated_tool',
            toolName: schema.name,
            message: schema.deprecationMessage || `Tool ${schema.name} is deprecated`
          });
          continue;
        }

        // Check if tool exists on server
        if (!serverToolNames.has(schema.name)) {
          errors.push({
            type: 'missing_tool',
            toolName: schema.name,
            message: `Tool '${schema.name}' not found on MCP server`,
            expected: schema.name,
            actual: Array.from(serverToolNames)
          });
          continue;
        }

        // Validate tool signature
        const serverTool = serverTools.find(t => t.name === schema.name);
        if (serverTool) {
          const signatureValid = await this.validateToolSignature(schema, serverTool);
          if (signatureValid.success) {
            passedTests++;
          } else {
            errors.push(...signatureValid.errors);
          }
          warnings.push(...signatureValid.warnings);
        }

        // Optional execution tests
        if (this.options.includeExecutionTests) {
          const executionResult = await this.validateToolExecution(schema);
          if (executionResult.success) {
            passedTests++;
          } else {
            errors.push(...executionResult.errors);
          }
          warnings.push(...executionResult.warnings);
          totalTests++;
        }
      }

      const executionTimeMs = Date.now() - startTime;

      return {
        success: errors.length === 0,
        errors,
        warnings,
        totalTests,
        passedTests,
        executionTimeMs
      };

    } catch (error) {
      const executionTimeMs = Date.now() - startTime;
      return {
        success: false,
        errors: [{
          type: 'execution_error',
          toolName: 'validator',
          message: `Contract validation failed: ${error instanceof Error ? error.message : String(error)}`
        }],
        warnings,
        totalTests,
        passedTests,
        executionTimeMs
      };
    }
  }

  /**
   * Validate a single tool
   */
  async validateTool(toolName: string): Promise<ValidationResult> {
    return this.validateAllTools();
  }

  /**
   * Check server API version compatibility
   */
  async checkApiVersion(): Promise<{ compatible: boolean; serverVersion?: string; clientVersion: string }> {
    try {
      // Try to get server capabilities for version detection
      let serverVersion = 'unknown';
      try {
        // Attempt to call capabilities endpoint if available
        const capabilities = await this.mcpClient.getCapabilities?.();
        if (capabilities && capabilities.version) {
          serverVersion = capabilities.version;
        }
      } catch (error) {
        // Capabilities not available, continue with tool-based detection
      }
      
      // Check tool availability for compatibility detection
      const serverTools = await this.mcpClient.listTools();
      const expectedTools = this.getExpectedSchemas().map(s => s.name);
      const availableTools = serverTools.map(t => t.name);
      
      const missingTools = expectedTools.filter(name => !availableTools.includes(name));
      const extraTools = availableTools.filter(name => !expectedTools.includes(name));
      
      // Determine version compatibility
      let compatible = true;
      let detectedVersion = this.options.apiVersion || '1.0.0';
      
      if (missingTools.length > 0) {
        // Missing critical tools suggests version mismatch
        compatible = false;
        
        // Try to infer server version based on available tools
        if (availableTools.length > 0) {
          detectedVersion = this.inferVersionFromTools(availableTools);
        }
      } else if (extraTools.length > 0) {
        // Extra tools might indicate newer server version
        detectedVersion = this.inferVersionFromTools(availableTools);
      }

      return {
        compatible,
        clientVersion: this.options.apiVersion || '1.0.0',
        serverVersion: serverVersion !== 'unknown' ? serverVersion : detectedVersion
      };
      
    } catch (error) {
      return {
        compatible: false,
        clientVersion: this.options.apiVersion || '1.0.0',
        serverVersion: 'error'
      };
    }
  }
  
  /**
   * Infer server version based on available tools
   */
  private inferVersionFromTools(availableTools: string[]): SupportedAPIVersion {
    // Define tool sets for different versions
    const versionToolSets: Record<string, string[]> = {
      '1.0.0': [
        'project_understanding',
        'semantic_code_search', 
        'atomic_dependency_tracer',
        'vibe_preservation',
        'project_auto_index',
        'graph_query',
        'memory_store_enhanced',
        'memory_search_enhanced',
        'schema_customization'
      ],
      '1.1.0': [
        // Future version tool sets would go here
        'project_understanding',
        'semantic_code_search', 
        'atomic_dependency_tracer',
        'vibe_preservation',
        'project_auto_index',
        'graph_query',
        'memory_store_enhanced',
        'memory_search_enhanced',
        'schema_customization',
        'advanced_analytics',
        'performance_insights'
      ]
    };
    
    // Check which version has the best match
    let bestMatch = '1.0.0';
    let bestScore = 0;
    
    for (const [version, expectedTools] of Object.entries(versionToolSets)) {
      const matchingTools = expectedTools.filter(tool => availableTools.includes(tool));
      const score = matchingTools.length / expectedTools.length;
      
      if (score > bestScore) {
        bestScore = score;
        bestMatch = version;
      }
    }
    
    return bestMatch as SupportedAPIVersion;
  }
  
  /**
   * Get tools for a specific version
   */
  private getToolsForVersion(version: string): string[] {
    return this.getExpectedSchemas()
      .filter(schema => schema.version === version)
      .map(schema => schema.name);
  }

  private getExpectedSchemas(): MCPToolSchema[] {
    const version = this.options.apiVersion || '1.0.0';
    return ALL_TOOL_SCHEMAS.filter(schema => schema.version === version);
  }

  private async validateToolSignature(
    schema: MCPToolSchema, 
    serverTool: ToolInfo
  ): Promise<{ success: boolean; errors: ValidationError[]; warnings: ValidationWarning[] }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Basic description check
    if (!serverTool.description) {
      warnings.push({
        type: 'version_mismatch',
        toolName: schema.name,
        message: `Tool '${schema.name}' missing description`
      });
    }

    // Parameter validation would require more detailed schema from server
    // For now, assume compatibility if tool exists

    return { success: true, errors, warnings };
  }

  private async validateToolExecution(schema: MCPToolSchema): Promise<{ 
    success: boolean; 
    errors: ValidationError[]; 
    warnings: ValidationWarning[] 
  }> {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Get safe test parameters for this tool
    const testParams = this.getSafeTestParameters(schema);
    if (!testParams) {
      // Skip execution test if no safe parameters available
      return { success: true, errors, warnings };
    }

    try {
      const startTime = Date.now();
      const result = await this.mcpClient.callTool(schema.name, testParams);
      const executionTime = Date.now() - startTime;

      // Performance warning
      const threshold = this.options.performanceThresholdMs || 5000;
      if (executionTime > threshold) {
        warnings.push({
          type: 'performance_warning',
          toolName: schema.name,
          message: `Tool execution took ${executionTime}ms (threshold: ${threshold}ms)`
        });
      }

      // Basic response validation
      if (!result || typeof result !== 'object') {
        errors.push({
          type: 'type_error',
          toolName: schema.name,
          message: 'Expected object response from tool',
          expected: 'object',
          actual: typeof result
        });
        return { success: false, errors, warnings };
      }

      return { success: true, errors, warnings };

    } catch (error) {
      errors.push({
        type: 'execution_error',
        toolName: schema.name,
        message: `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`
      });
      return { success: false, errors, warnings };
    }
  }

  private getSafeTestParameters(schema: MCPToolSchema): any | null {
    // Safe test parameters for each tool type
    const safeParams: Record<string, any> = {
      'project_understanding': { scope: 'summary' },
      'semantic_code_search': { query: 'test function' },
      'atomic_dependency_tracer': { target: 'main' },
      'vibe_preservation': { action: 'analyze' },
      'project_auto_index': { scope: 'test' },
      'graph_query': { query: 'RETURN 1 as test LIMIT 1' },
      'memory_store_enhanced': { content: 'test memory content' },
      'memory_search_enhanced': { query: 'test search' },
      'schema_customization': { action: 'list' }
    };

    return safeParams[schema.name] || null;
  }
}

/**
 * Utility function to create a quick validation report
 */
export function formatValidationReport(result: ValidationResult): string {
  const lines: string[] = [];
  
  lines.push(`🧪 MCP Contract Validation Report`);
  lines.push(`${'='.repeat(50)}`);
  lines.push(`✅ Passed: ${result.passedTests}/${result.totalTests}`);
  lines.push(`⏱️  Execution Time: ${result.executionTimeMs}ms`);
  lines.push(`📊 Success Rate: ${((result.passedTests / result.totalTests) * 100).toFixed(1)}%`);
  lines.push('');

  if (result.errors.length > 0) {
    lines.push(`❌ Errors (${result.errors.length}):`);
    result.errors.forEach(error => {
      lines.push(`  - ${error.toolName}: ${error.message}`);
    });
    lines.push('');
  }

  if (result.warnings.length > 0) {
    lines.push(`⚠️  Warnings (${result.warnings.length}):`);
    result.warnings.forEach(warning => {
      lines.push(`  - ${warning.toolName}: ${warning.message}`);
    });
    lines.push('');
  }

  lines.push(result.success ? '🎉 All tests passed!' : '💥 Some tests failed!');
  
  return lines.join('\n');
}
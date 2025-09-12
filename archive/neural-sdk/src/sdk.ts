/**
 * Core Neural SDK - Orchestrates all components
 */

import { ContainerManager } from './container-manager';
import { MCPConfigManager } from './mcp-config';
import { ProjectDetector, ProjectInfo } from './project-detector';
import { FileWatcher } from './file-watcher';
import { createNeuralMCPClient, testMCPConnection } from './mcp-client';
import { MCPContractValidator, ValidationResult, formatValidationReport } from './contract/validator';
import chalk from 'chalk';
import path from 'path';
import os from 'os';

export interface InitializeOptions {
  global?: boolean;
  port?: number;
  autoStart?: boolean;
  customImage?: string;
  enableContractValidation?: boolean;
}

export interface SystemStatus {
  containers: Array<{
    name: string;
    running: boolean;
    status: string;
    image: string;
  }>;
  mcp: {
    configured: boolean;
    configPath: string;
    contractValid?: boolean;
  };
  project: {
    name: string;
    path: string;
    indexedFiles: number;
    lastIndexed: string;
  };
}

export class NeuralSDK {
  private containerManager: ContainerManager;
  private mcpConfig: MCPConfigManager;
  
  constructor() {
    this.containerManager = new ContainerManager();
    this.mcpConfig = new MCPConfigManager();
  }
  
  /**
   * Zero-config initialization - handles everything automatically
   */
  async initialize(projectInfo: ProjectInfo, options: InitializeOptions): Promise<void> {
    const steps = [
      'Checking Docker availability',
      'Pulling Neural Tools container',
      'Creating project-specific services', 
      'Configuring MCP integration',
      'Starting services',
      'Validating setup'
    ];
    
    let currentStep = 0;
    const logStep = (message: string) => {
      console.log(chalk.blue(`[${currentStep + 1}/${steps.length}]`), message);
      currentStep++;
    };
    
    // Step 1: Check Docker
    logStep('Checking Docker availability...');
    await this.containerManager.validateDocker();
    
    // Step 2: Pull/build containers  
    logStep('Pulling Neural Tools container...');
    await this.containerManager.ensureImages();
    
    // Step 3: Create project services
    logStep('Creating project-specific services...');
    await this.containerManager.createProjectServices(projectInfo, {
      port: options.port || 3000
    });
    
    // Step 4: Configure MCP
    logStep('Configuring MCP integration...');
    const configPath = options.global 
      ? path.join(os.homedir(), '.claude', 'mcp_config.json')
      : path.join(projectInfo.path, '.mcp.json');
      
    await this.mcpConfig.configure(projectInfo, {
      configPath,
      port: options.port || 3000
    });
    
    // Step 5: Start services
    if (options.autoStart) {
      logStep('Starting services...');
      await this.containerManager.start(projectInfo.name);
    }
    
    // Step 6: Validate
    logStep('Validating setup...');
    await this.validateSetup(projectInfo, options.enableContractValidation);
    
    console.log(chalk.green('🎉 Neural Tools ready!'));
  }
  
  /**
   * Get comprehensive system status
   */
  async getSystemStatus(): Promise<SystemStatus> {
    const projectDetector = new ProjectDetector();
    const projectInfo = await projectDetector.detectProject(process.cwd());
    
    // Get container status
    const containers = await this.containerManager.getContainerStatus(projectInfo.name);
    
    // Get MCP config status
    const mcpStatus = await this.mcpConfig.getStatus(projectInfo);
    
    // Get project metrics
    const projectStats = await this.getProjectStats(projectInfo);
    
    // Test contract validation if configured
    let contractValid: boolean | undefined;
    if (mcpStatus.configured) {
      try {
        const contractResult = await this.validateMCPContract(projectInfo);
        contractValid = contractResult.success;
      } catch (error) {
        contractValid = false;
      }
    }
    
    return {
      containers,
      mcp: {
        ...mcpStatus,
        contractValid
      },
      project: {
        name: projectInfo.name,
        path: projectInfo.path,
        indexedFiles: projectStats.indexedFiles,
        lastIndexed: projectStats.lastIndexed
      }
    };
  }
  
  /**
   * Stop all services
   */
  async stop(): Promise<void> {
    const projectDetector = new ProjectDetector();
    const projectInfo = await projectDetector.detectProject(process.cwd());
    
    await this.containerManager.stop(projectInfo.name);
  }
  
  /**
   * Show logs with optional following
   */
  async showLogs(options: { follow?: boolean; lines?: number }): Promise<void> {
    const projectDetector = new ProjectDetector();
    const projectInfo = await projectDetector.detectProject(process.cwd());
    
    await this.containerManager.showLogs(projectInfo.name, options);
  }
  
  /**
   * Complete reset - removes all data
   */
  async reset(): Promise<void> {
    const projectDetector = new ProjectDetector();
    const projectInfo = await projectDetector.detectProject(process.cwd());
    
    // Stop services
    await this.containerManager.stop(projectInfo.name);
    
    // Remove containers and volumes
    await this.containerManager.cleanup(projectInfo.name);
    
    // Remove MCP config
    await this.mcpConfig.remove(projectInfo);
    
    console.log(chalk.green('✅ Reset complete'));
  }
  
  /**
   * Validate that everything is working
   */
  private async validateSetup(projectInfo: ProjectInfo, enableContractValidation?: boolean): Promise<void> {
    // Test container connectivity
    const healthy = await this.containerManager.healthCheck(projectInfo.name);
    if (!healthy) {
      throw new Error('Container health check failed');
    }
    
    // Test MCP configuration
    const mcpValid = await this.mcpConfig.validate(projectInfo);
    if (!mcpValid) {
      throw new Error('MCP configuration validation failed');
    }
    
    // Test basic functionality
    await this.testBasicFunctionality(projectInfo);
    
    // Optional contract validation
    if (enableContractValidation) {
      console.log(chalk.blue('Running contract validation...'));
      const contractResult = await this.validateMCPContract(projectInfo);
      
      if (!contractResult.success) {
        console.log(chalk.yellow('⚠️  Contract validation warnings:'));
        console.log(formatValidationReport(contractResult));
      } else {
        console.log(chalk.green('✅ All contracts validated successfully'));
      }
    }
  }
  
  /**
   * Test that basic GraphRAG functionality works
   */
  private async testBasicFunctionality(projectInfo: ProjectInfo): Promise<void> {
    // This would make a simple test call to the MCP server
    // to ensure indexing and search are working
    try {
      await this.containerManager.testMCPConnection(projectInfo.name);
    } catch (error) {
      throw new Error(`MCP functionality test failed: ${error.message}`);
    }
  }
  
  /**
   * Validate MCP contract compatibility
   */
  async validateMCPContract(projectInfo: ProjectInfo, options?: {
    includeExecutionTests?: boolean;
    performanceThresholdMs?: number;
    testToolNames?: string[];
  }): Promise<ValidationResult> {
    try {
      // Create MCP client for contract testing
      const client = await createNeuralMCPClient(projectInfo.path);
      await client.connect();
      
      // First check API version compatibility
      const validator = new MCPContractValidator(client, {
        apiVersion: '1.0.0',
        includeExecutionTests: options?.includeExecutionTests || false,
        performanceThresholdMs: options?.performanceThresholdMs || 5000,
        testToolNames: options?.testToolNames
      });
      
      const versionCheck = await validator.checkApiVersion();
      
      // Run contract validation
      const result = await validator.validateAllTools();
      
      // Add version compatibility warnings
      if (!versionCheck.compatible) {
        result.warnings.unshift({
          type: 'version_mismatch',
          toolName: 'version_compatibility',
          message: `Version mismatch: Client expects ${versionCheck.clientVersion}, Server reports ${versionCheck.serverVersion}`
        });
      }
      
      // Cleanup
      await client.disconnect();
      
      return result;
      
    } catch (error) {
      return {
        success: false,
        errors: [{
          type: 'execution_error',
          toolName: 'contract_validator',
          message: `Contract validation failed: ${error instanceof Error ? error.message : String(error)}`
        }],
        warnings: [],
        totalTests: 0,
        passedTests: 0,
        executionTimeMs: 0
      };
    }
  }
  
  /**
   * Check API version compatibility
   */
  async checkApiCompatibility(projectInfo: ProjectInfo): Promise<{
    compatible: boolean;
    clientVersion: string;
    serverVersion?: string;
    message?: string;
  }> {
    try {
      const client = await createNeuralMCPClient(projectInfo.path);
      await client.connect();
      
      const validator = new MCPContractValidator(client, {
        apiVersion: '1.0.0'
      });
      
      const versionCheck = await validator.checkApiVersion();
      
      await client.disconnect();
      
      return {
        ...versionCheck,
        message: versionCheck.compatible 
          ? `API versions compatible (${versionCheck.clientVersion})`
          : `API version mismatch: client ${versionCheck.clientVersion}, server ${versionCheck.serverVersion}`
      };
      
    } catch (error) {
      return {
        compatible: false,
        clientVersion: '1.0.0',
        serverVersion: 'error',
        message: `Version check failed: ${error instanceof Error ? error.message : String(error)}`
      };
    }
  }
  
  /**
   * Test MCP connection and basic functionality
   */
  async testMCPConnection(projectInfo: ProjectInfo): Promise<{
    connected: boolean;
    error?: string;
    tools?: string[];
    responseTime?: number;
  }> {
    try {
      const client = await createNeuralMCPClient(projectInfo.path);
      return await testMCPConnection(client);
    } catch (error) {
      return {
        connected: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
  
  /**
   * Get project statistics
   */
  private async getProjectStats(projectInfo: ProjectInfo): Promise<{
    indexedFiles: number;
    lastIndexed: string;
  }> {
    try {
      const stats = await this.containerManager.getProjectStats(projectInfo.name);
      return {
        indexedFiles: stats.indexedFiles || 0,
        lastIndexed: stats.lastIndexed || 'Never'
      };
    } catch (error) {
      return {
        indexedFiles: 0,
        lastIndexed: 'Never'
      };
    }
  }
}
/**
 * MCP Client for Neural Tools
 * Provides direct communication with MCP server for tool calls and contract validation
 */

import { spawn, ChildProcess } from 'child_process';
import { MCPClient, ToolInfo } from './contract/validator';

export interface MCPClientOptions {
  serverCommand?: string[];
  serverArgs?: string[];
  timeout?: number;
  cwd?: string;
}

export interface MCPResponse {
  jsonrpc: string;
  id: string | number;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

export interface MCPRequest {
  jsonrpc: string;
  id: string | number;
  method: string;
  params?: any;
}

export class NeuralMCPClient implements MCPClient {
  private serverProcess: ChildProcess | null = null;
  private requestId = 1;
  private pendingRequests = new Map<string | number, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>();
  
  private options: MCPClientOptions;

  constructor(options: MCPClientOptions = {}) {
    this.options = {
      serverCommand: ['python3', '-m', 'src.neural_mcp.neural_server_stdio'],
      timeout: 30000,
      cwd: process.cwd(),
      ...options
    };
  }

  /**
   * Connect to MCP server
   */
  async connect(): Promise<void> {
    if (this.serverProcess) {
      await this.disconnect();
    }

    return new Promise((resolve, reject) => {
      const [command, ...args] = this.options.serverCommand!;
      
      this.serverProcess = spawn(command, args, {
        cwd: this.options.cwd,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env }
      });

      this.serverProcess.on('error', (error) => {
        reject(new Error(`Failed to start MCP server: ${error.message}`));
      });

      this.serverProcess.on('exit', (code) => {
        if (code !== 0 && code !== null) {
          reject(new Error(`MCP server exited with code ${code}`));
        }
      });

      // Handle responses
      let buffer = '';
      this.serverProcess.stdout?.on('data', (data) => {
        buffer += data.toString();
        
        // Process complete JSON messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const response: MCPResponse = JSON.parse(line);
              this.handleResponse(response);
            } catch (error) {
              console.warn('Failed to parse MCP response:', line);
            }
          }
        }
      });

      this.serverProcess.stderr?.on('data', (data) => {
        console.warn('MCP server stderr:', data.toString());
      });

      // Initialize connection
      this.sendRequest('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {}
        },
        clientInfo: {
          name: 'neural-sdk-contract-validator',
          version: '1.0.0'
        }
      }).then(() => {
        resolve();
      }).catch(reject);
    });
  }

  /**
   * Disconnect from MCP server
   */
  async disconnect(): Promise<void> {
    if (this.serverProcess) {
      // Clear pending requests
      for (const [id, request] of this.pendingRequests) {
        clearTimeout(request.timeout);
        request.reject(new Error('Client disconnecting'));
      }
      this.pendingRequests.clear();

      // Terminate server process
      this.serverProcess.kill();
      this.serverProcess = null;
    }
  }

  /**
   * List available tools
   */
  async listTools(): Promise<ToolInfo[]> {
    try {
      const response = await this.sendRequest('tools/list', {});
      return response.tools || [];
    } catch (error) {
      throw new Error(`Failed to list tools: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Call a specific tool
   */
  async callTool(name: string, params: any = {}): Promise<any> {
    try {
      const response = await this.sendRequest('tools/call', {
        name,
        arguments: params
      });
      return response.content || response;
    } catch (error) {
      throw new Error(`Tool call failed for '${name}': ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Get server capabilities
   */
  async getCapabilities(): Promise<any> {
    try {
      return await this.sendRequest('capabilities', {});
    } catch (error) {
      throw new Error(`Failed to get capabilities: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Ping server for health check
   */
  async ping(): Promise<boolean> {
    try {
      await this.sendRequest('ping', {});
      return true;
    } catch (error) {
      return false;
    }
  }

  private async sendRequest(method: string, params: any = {}): Promise<any> {
    if (!this.serverProcess) {
      throw new Error('MCP client not connected');
    }

    const id = this.requestId++;
    const request: MCPRequest = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request timeout after ${this.options.timeout}ms`));
      }, this.options.timeout);

      this.pendingRequests.set(id, { resolve, reject, timeout });

      const requestJson = JSON.stringify(request) + '\n';
      this.serverProcess!.stdin?.write(requestJson);
    });
  }

  private handleResponse(response: MCPResponse): void {
    const pending = this.pendingRequests.get(response.id);
    if (!pending) {
      return; // Unexpected response
    }

    this.pendingRequests.delete(response.id);
    clearTimeout(pending.timeout);

    if (response.error) {
      pending.reject(new Error(`MCP Error ${response.error.code}: ${response.error.message}`));
    } else {
      pending.resolve(response.result);
    }
  }
}

/**
 * Auto-detect Neural Tools server location and create client
 */
export async function createNeuralMCPClient(projectPath?: string): Promise<NeuralMCPClient> {
  const basePath = projectPath || process.cwd();
  
  // Try to find neural-tools directory
  const possiblePaths = [
    `${basePath}/neural-tools`,
    `${basePath}/../neural-tools`,
    `${basePath}/../../neural-tools`
  ];

  let neuralToolsPath: string | null = null;
  for (const path of possiblePaths) {
    try {
      const fs = require('fs');
      if (fs.existsSync(`${path}/src/neural_mcp/neural_server_stdio.py`)) {
        neuralToolsPath = path;
        break;
      }
    } catch (error) {
      // Continue searching
    }
  }

  if (!neuralToolsPath) {
    throw new Error('Neural Tools server not found. Ensure neural-tools directory is accessible.');
  }

  return new NeuralMCPClient({
    serverCommand: ['python3', '-m', 'src.neural_mcp.neural_server_stdio'],
    cwd: neuralToolsPath,
    timeout: 30000
  });
}

/**
 * Quick connection test
 */
export async function testMCPConnection(client: NeuralMCPClient): Promise<{
  connected: boolean;
  error?: string;
  tools?: string[];
  responseTime?: number;
}> {
  const startTime = Date.now();
  
  try {
    await client.connect();
    const tools = await client.listTools();
    const responseTime = Date.now() - startTime;
    
    return {
      connected: true,
      tools: tools.map(t => t.name),
      responseTime
    };
  } catch (error) {
    return {
      connected: false,
      error: error instanceof Error ? error.message : String(error)
    };
  } finally {
    await client.disconnect();
  }
}
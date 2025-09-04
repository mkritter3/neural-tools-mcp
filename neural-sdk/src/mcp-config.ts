/**
 * MCP Configuration Manager - Auto-configures Claude Code MCP integration
 */

import fs from 'fs-extra';
import path from 'path';
import os from 'os';
import { ProjectInfo } from './project-detector';

export interface MCPConfigOptions {
  configPath: string;
  port: number;
  customArgs?: string[];
}

export interface MCPStatus {
  configured: boolean;
  configPath: string;
  serverName?: string;
}

export class MCPConfigManager {
  
  /**
   * Auto-configure MCP for the project
   */
  async configure(projectInfo: ProjectInfo, options: MCPConfigOptions): Promise<void> {
    const serverName = `neural-tools-${projectInfo.name}`;
    
    // Ensure config directory exists
    await fs.ensureDir(path.dirname(options.configPath));
    
    // Load or create MCP config
    let config: any = { mcpServers: {} };
    if (await fs.pathExists(options.configPath)) {
      try {
        config = await fs.readJSON(options.configPath);
      } catch (error) {
        // Invalid JSON, start fresh
        config = { mcpServers: {} };
      }
    }
    
    // Ensure mcpServers exists
    if (!config.mcpServers) {
      config.mcpServers = {};
    }
    
    // Create server configuration
    const serverConfig = {
      command: 'docker',
      args: [
        'exec',
        '-i',
        `neural-${projectInfo.name}`,
        'python3',
        '-u',
        '/app/src/neural_mcp/neural_server_stdio.py',
        ...(options.customArgs || [])
      ],
      env: {
        PROJECT_NAME: projectInfo.name,
        PROJECT_DIR: `/workspace/${projectInfo.name}`,
        PYTHONUNBUFFERED: '1'
      },
      description: `Neural Tools GraphRAG for ${projectInfo.name} - Auto-configured multi-project support`
    };
    
    // Add server to config
    config.mcpServers[serverName] = serverConfig;
    
    // Save config with proper formatting
    await fs.writeJSON(options.configPath, config, { spaces: 2 });
    
    console.log(`✅ MCP configured at ${options.configPath}`);
  }
  
  /**
   * Get MCP configuration status
   */
  async getStatus(projectInfo: ProjectInfo): Promise<MCPStatus> {
    const possibleConfigs = [
      path.join(projectInfo.path, '.mcp.json'),
      path.join(os.homedir(), '.claude', 'mcp_config.json'),
      path.join(os.homedir(), '.mcp.json')
    ];
    
    for (const configPath of possibleConfigs) {
      if (await fs.pathExists(configPath)) {
        try {
          const config = await fs.readJSON(configPath);
          const serverName = `neural-tools-${projectInfo.name}`;
          
          if (config.mcpServers && config.mcpServers[serverName]) {
            return {
              configured: true,
              configPath,
              serverName
            };
          }
        } catch (error) {
          // Invalid config file
        }
      }
    }
    
    return {
      configured: false,
      configPath: possibleConfigs[0] // Default to project config
    };
  }
  
  /**
   * Validate MCP configuration
   */
  async validate(projectInfo: ProjectInfo): Promise<boolean> {
    const status = await this.getStatus(projectInfo);
    
    if (!status.configured) {
      return false;
    }
    
    try {
      const config = await fs.readJSON(status.configPath);
      const serverConfig = config.mcpServers[status.serverName!];
      
      // Validate required fields
      if (!serverConfig.command || !serverConfig.args) {
        return false;
      }
      
      // Validate Docker command structure
      if (serverConfig.command !== 'docker') {
        return false;
      }
      
      if (!Array.isArray(serverConfig.args) || serverConfig.args.length < 4) {
        return false;
      }
      
      // Check if container name is correct
      const expectedContainerName = `neural-${projectInfo.name}`;
      if (!serverConfig.args.includes(expectedContainerName)) {
        return false;
      }
      
      return true;
      
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Remove MCP configuration for project
   */
  async remove(projectInfo: ProjectInfo): Promise<void> {
    const status = await this.getStatus(projectInfo);
    
    if (!status.configured) {
      return; // Nothing to remove
    }
    
    try {
      const config = await fs.readJSON(status.configPath);
      
      if (config.mcpServers && config.mcpServers[status.serverName!]) {
        delete config.mcpServers[status.serverName!];
        
        // Clean up empty mcpServers object
        if (Object.keys(config.mcpServers).length === 0) {
          // If this was the only server, you might want to keep the structure
          // or remove the file entirely based on preference
        }
        
        await fs.writeJSON(status.configPath, config, { spaces: 2 });
        console.log(`✅ Removed MCP config for ${projectInfo.name}`);
      }
      
    } catch (error) {
      throw new Error(`Failed to remove MCP config: ${error.message}`);
    }
  }
  
  /**
   * Update existing MCP configuration
   */
  async update(projectInfo: ProjectInfo, updates: Partial<MCPConfigOptions>): Promise<void> {
    const status = await this.getStatus(projectInfo);
    
    if (!status.configured) {
      throw new Error('MCP not configured for this project');
    }
    
    try {
      const config = await fs.readJSON(status.configPath);
      const serverConfig = config.mcpServers[status.serverName!];
      
      // Apply updates
      if (updates.port) {
        // Port updates might require container recreation
        // This is handled by the ContainerManager
      }
      
      if (updates.customArgs) {
        // Update args while preserving the core Docker exec structure
        const baseArgs = [
          'exec',
          '-i',
          `neural-${projectInfo.name}`,
          'python3',
          '-u',
          '/app/src/neural_mcp/neural_server_stdio.py'
        ];
        serverConfig.args = [...baseArgs, ...updates.customArgs];
      }
      
      await fs.writeJSON(status.configPath, config, { spaces: 2 });
      console.log(`✅ Updated MCP config for ${projectInfo.name}`);
      
    } catch (error) {
      throw new Error(`Failed to update MCP config: ${error.message}`);
    }
  }
  
  /**
   * List all Neural Tools MCP configurations
   */
  async listConfigurations(): Promise<Array<{ projectName: string; configPath: string; status: string }>> {
    const possibleConfigs = [
      path.join(process.cwd(), '.mcp.json'),
      path.join(os.homedir(), '.claude', 'mcp_config.json'),
      path.join(os.homedir(), '.mcp.json')
    ];
    
    const configurations: Array<{ projectName: string; configPath: string; status: string }> = [];
    
    for (const configPath of possibleConfigs) {
      if (await fs.pathExists(configPath)) {
        try {
          const config = await fs.readJSON(configPath);
          
          if (config.mcpServers) {
            for (const [serverName, serverConfig] of Object.entries(config.mcpServers)) {
              if (serverName.startsWith('neural-tools-')) {
                const projectName = serverName.replace('neural-tools-', '');
                configurations.push({
                  projectName,
                  configPath,
                  status: 'configured'
                });
              }
            }
          }
          
        } catch (error) {
          // Skip invalid config files
        }
      }
    }
    
    return configurations;
  }
  
  /**
   * Generate MCP configuration for manual use
   */
  generateConfig(projectInfo: ProjectInfo, options: MCPConfigOptions): object {
    const serverName = `neural-tools-${projectInfo.name}`;
    
    return {
      mcpServers: {
        [serverName]: {
          command: 'docker',
          args: [
            'exec',
            '-i',
            `neural-${projectInfo.name}`,
            'python3',
            '-u',
            '/app/src/neural_mcp/neural_server_stdio.py',
            ...(options.customArgs || [])
          ],
          env: {
            PROJECT_NAME: projectInfo.name,
            PROJECT_DIR: `/workspace/${projectInfo.name}`,
            PYTHONUNBUFFERED: '1'
          },
          description: `Neural Tools GraphRAG for ${projectInfo.name} - Auto-configured multi-project support`
        }
      }
    };
  }
}
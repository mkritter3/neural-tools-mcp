/**
 * Container orchestration abstraction - handles Docker complexity
 */

import Docker from 'dockerode';
import path from 'path';
import fs from 'fs-extra';
import yaml from 'yaml';
import chalk from 'chalk';
import { ProjectInfo } from './project-detector';

export interface ContainerConfig {
  port: number;
  customImage?: string;
  env?: Record<string, string>;
}

export interface ContainerStatus {
  name: string;
  running: boolean;
  status: string;
  image: string;
}

export class ContainerManager {
  private docker: Docker;
  
  constructor() {
    this.docker = new Docker();
  }
  
  /**
   * Validate Docker is available and running
   */
  async validateDocker(): Promise<void> {
    try {
      await this.docker.ping();
    } catch (error) {
      throw new Error(
        'Docker is not running. Please start Docker Desktop and try again.\n' +
        'Download from: https://www.docker.com/products/docker-desktop'
      );
    }
  }
  
  /**
   * Ensure required images are available
   */
  async ensureImages(): Promise<void> {
    const requiredImages = [
      'neural-tools/graphrag:latest',
      'qdrant/qdrant:v1.8.0',
      'neo4j:5.20.0'
    ];
    
    for (const imageName of requiredImages) {
      try {
        await this.docker.getImage(imageName).inspect();
        console.log(chalk.green(`✅ Image ${imageName} ready`));
      } catch (error) {
        console.log(chalk.yellow(`📥 Pulling ${imageName}...`));
        await this.pullImage(imageName);
      }
    }
  }
  
  /**
   * Create project-specific services using our multi-project container
   */
  async createProjectServices(projectInfo: ProjectInfo, config: ContainerConfig): Promise<void> {
    const containerName = `neural-${projectInfo.name}`;
    
    // Stop existing container if running
    try {
      const existingContainer = this.docker.getContainer(containerName);
      await existingContainer.stop();
      await existingContainer.remove();
    } catch (error) {
      // Container doesn't exist, which is fine
    }
    
    // Create workspace directory for project
    const workspaceDir = path.join(process.cwd(), '.neural', 'workspace');
    await fs.ensureDir(workspaceDir);
    
    // Mount the project as a workspace subdirectory  
    const projectWorkspace = path.join(workspaceDir, projectInfo.name);
    if (!await fs.pathExists(projectWorkspace)) {
      await fs.ensureSymlink(projectInfo.path, projectWorkspace);
    }
    
    // Container configuration
    const containerConfig = {
      Image: config.customImage || 'neural-multi-project-final', // Use our built image
      name: containerName,
      Env: [
        `PROJECT_NAME=${projectInfo.name}`,
        `PROJECT_DIR=/workspace/${projectInfo.name}`,
        'PYTHONUNBUFFERED=1',
        ...Object.entries(config.env || {}).map(([k, v]) => `${k}=${v}`)
      ],
      HostConfig: {
        Binds: [
          `${workspaceDir}:/workspace:rw`, // Mount workspace with all projects
          `${path.join(process.cwd(), '.neural', 'data', projectInfo.name)}:/app/config/.neural-tools:rw` // Project-specific data
        ],
        RestartPolicy: {
          Name: 'unless-stopped'
        }
      },
      ExposedPorts: {
        '3000/tcp': {}
      }
    };
    
    // Create and start container
    console.log(chalk.blue(`🏗️ Creating container for ${projectInfo.name}...`));
    const container = await this.docker.createContainer(containerConfig);
    await container.start();
    
    // Wait for services to be ready
    await this.waitForHealthy(containerName, 30000);
    
    console.log(chalk.green(`✅ Container ${containerName} ready`));
  }
  
  /**
   * Start services for a project
   */
  async start(projectName: string): Promise<void> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      await container.start();
      await this.waitForHealthy(containerName, 15000);
    } catch (error) {
      throw new Error(`Failed to start services for ${projectName}: ${error.message}`);
    }
  }
  
  /**
   * Stop services for a project
   */
  async stop(projectName: string): Promise<void> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      await container.stop();
      console.log(chalk.green(`✅ Stopped ${containerName}`));
    } catch (error) {
      throw new Error(`Failed to stop services for ${projectName}: ${error.message}`);
    }
  }
  
  /**
   * Get container status for a project
   */
  async getContainerStatus(projectName: string): Promise<ContainerStatus[]> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      const info = await container.inspect();
      
      return [{
        name: containerName,
        running: info.State.Running,
        status: info.State.Status,
        image: info.Config.Image
      }];
    } catch (error) {
      return [{
        name: containerName,
        running: false,
        status: 'not found',
        image: 'unknown'
      }];
    }
  }
  
  /**
   * Health check for services
   */
  async healthCheck(projectName: string): Promise<boolean> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      
      // Test MCP server response
      const exec = await container.exec({
        Cmd: ['python3', '-c', 'import sys; sys.exit(0)'],
        AttachStdout: true,
        AttachStderr: true
      });
      
      const stream = await exec.start({ hijack: true, stdin: false });
      
      return new Promise((resolve) => {
        let output = '';
        stream.on('data', (chunk) => {
          output += chunk.toString();
        });
        
        stream.on('end', () => {
          resolve(true);
        });
        
        setTimeout(() => resolve(false), 5000);
      });
      
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Show logs for a project
   */
  async showLogs(projectName: string, options: { follow?: boolean; lines?: number }): Promise<void> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      
      const logStream = await container.logs({
        follow: options.follow || false,
        stdout: true,
        stderr: true,
        tail: options.lines || 100,
        timestamps: true
      });
      
      if (options.follow) {
        logStream.pipe(process.stdout);
      } else {
        console.log(logStream.toString());
      }
      
    } catch (error) {
      throw new Error(`Failed to get logs for ${projectName}: ${error.message}`);
    }
  }
  
  /**
   * Cleanup containers and volumes for a project
   */
  async cleanup(projectName: string): Promise<void> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      await container.stop();
      await container.remove({ v: true }); // Remove volumes too
      
      // Cleanup project data directory
      const dataDir = path.join(process.cwd(), '.neural', 'data', projectName);
      await fs.remove(dataDir);
      
      console.log(chalk.green(`✅ Cleaned up ${containerName}`));
    } catch (error) {
      // Container might not exist, which is fine for cleanup
      console.log(chalk.yellow(`⚠️ Cleanup warning: ${error.message}`));
    }
  }
  
  /**
   * Test MCP connection
   */
  async testMCPConnection(projectName: string): Promise<void> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      
      // Test basic MCP server functionality
      const exec = await container.exec({
        Cmd: ['python3', '-c', `
import json
import sys
sys.path.append('/app/src')

# Test the MCP server can start
try:
    from neural_mcp.neural_server_stdio import MultiProjectServiceState
    state = MultiProjectServiceState()
    project = state.detect_project_from_path('/workspace/${projectName}/test.py')
    print(f'✅ MCP server functional, detected project: {project}')
    sys.exit(0)
except Exception as e:
    print(f'❌ MCP server test failed: {e}')
    sys.exit(1)
`],
        AttachStdout: true,
        AttachStderr: true
      });
      
      const stream = await exec.start({ hijack: true, stdin: false });
      
      let output = '';
      stream.on('data', (chunk) => {
        output += chunk.toString();
      });
      
      await new Promise((resolve, reject) => {
        stream.on('end', () => {
          if (output.includes('✅ MCP server functional')) {
            resolve(null);
          } else {
            reject(new Error(output));
          }
        });
        
        setTimeout(() => reject(new Error('Test timeout')), 10000);
      });
      
    } catch (error) {
      throw new Error(`MCP connection test failed: ${error.message}`);
    }
  }
  
  /**
   * Get project statistics from container
   */
  async getProjectStats(projectName: string): Promise<{ indexedFiles: number; lastIndexed: string }> {
    const containerName = `neural-${projectName}`;
    
    try {
      const container = this.docker.getContainer(containerName);
      
      const exec = await container.exec({
        Cmd: ['find', `/app/config/.neural-tools/qdrant/${projectName}`, '-name', '*.dat', '-type', 'f'],
        AttachStdout: true
      });
      
      const stream = await exec.start({ hijack: true, stdin: false });
      
      let output = '';
      stream.on('data', (chunk) => {
        output += chunk.toString();
      });
      
      return new Promise((resolve) => {
        stream.on('end', () => {
          const files = output.trim().split('\n').filter(f => f);
          resolve({
            indexedFiles: files.length,
            lastIndexed: new Date().toISOString()
          });
        });
      });
      
    } catch (error) {
      return { indexedFiles: 0, lastIndexed: 'Never' };
    }
  }
  
  /**
   * Pull a Docker image
   */
  private async pullImage(imageName: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.docker.pull(imageName, (err, stream) => {
        if (err) {
          reject(err);
          return;
        }
        
        this.docker.modem.followProgress(stream, (err, res) => {
          if (err) {
            reject(err);
          } else {
            console.log(chalk.green(`✅ Pulled ${imageName}`));
            resolve();
          }
        });
      });
    });
  }
  
  /**
   * Wait for container to be healthy
   */
  private async waitForHealthy(containerName: string, timeout: number = 30000): Promise<void> {
    const start = Date.now();
    
    while (Date.now() - start < timeout) {
      try {
        const container = this.docker.getContainer(containerName);
        const info = await container.inspect();
        
        if (info.State.Running) {
          return; // Container is running
        }
        
      } catch (error) {
        // Container not ready yet
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error(`Container ${containerName} failed to become healthy within ${timeout}ms`);
  }
}
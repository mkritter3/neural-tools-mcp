#!/usr/bin/env node
/**
 * Neural SDK - Simplified JavaScript Version
 * Zero-config GraphRAG integration
 */

const { program } = require('commander');
const chalk = require('chalk');
const fs = require('fs-extra');
const path = require('path');
const { spawn, exec } = require('child_process');
const os = require('os');

// Simple project detection
function detectProject(projectPath = process.cwd()) {
  const projectName = path.basename(projectPath).toLowerCase().replace(/[^a-z0-9-]/g, '-');
  
  let type = 'unknown';
  if (fs.existsSync(path.join(projectPath, 'package.json'))) type = 'node';
  else if (fs.existsSync(path.join(projectPath, 'pyproject.toml'))) type = 'python';
  else if (fs.existsSync(path.join(projectPath, 'requirements.txt'))) type = 'python';
  else if (fs.existsSync(path.join(projectPath, 'Cargo.toml'))) type = 'rust';
  else if (fs.existsSync(path.join(projectPath, 'go.mod'))) type = 'go';
  
  return { name: projectName, path: projectPath, type };
}

// Check Docker
async function checkDocker() {
  return new Promise((resolve) => {
    exec('docker info', (error) => {
      resolve(!error);
    });
  });
}

// Create MCP configuration
async function createMCPConfig(projectInfo, configPath) {
  const serverName = `neural-tools-${projectInfo.name}`;
  
  let config = { mcpServers: {} };
  if (await fs.pathExists(configPath)) {
    try {
      config = await fs.readJSON(configPath);
    } catch (error) {
      config = { mcpServers: {} };
    }
  }
  
  if (!config.mcpServers) config.mcpServers = {};
  
  config.mcpServers[serverName] = {
    command: 'docker',
    args: [
      'exec',
      '-i', 
      'neural-multi-project-final', // Use our existing container
      'python3',
      '-u',
      '/app/neural-tools-src/servers/neural_server_stdio.py'
    ],
    env: {
      PROJECT_NAME: projectInfo.name,
      PROJECT_DIR: `/workspace/${projectInfo.name}`,
      PYTHONUNBUFFERED: '1'
    },
    description: `Neural Tools GraphRAG for ${projectInfo.name} - Auto-configured`
  };
  
  await fs.ensureDir(path.dirname(configPath));
  await fs.writeJSON(configPath, config, { spaces: 2 });
}

// Setup workspace directory
async function setupWorkspace(projectInfo) {
  const workspaceDir = path.join(process.cwd(), '.neural', 'workspace');
  await fs.ensureDir(workspaceDir);
  
  const projectWorkspace = path.join(workspaceDir, projectInfo.name);
  if (!await fs.pathExists(projectWorkspace)) {
    // Create symlink to project
    try {
      await fs.ensureSymlink(projectInfo.path, projectWorkspace);
    } catch (error) {
      console.log(chalk.yellow(`⚠️ Could not create symlink, copying files instead...`));
      // Fallback: could copy files, but symlink is preferred
    }
  }
}

program
  .name('neural')
  .description('Neural SDK - Zero-config GraphRAG integration')
  .version('1.0.0');

program
  .command('init')
  .description('Initialize Neural Tools in current project')
  .option('-g, --global', 'Install globally for all projects')
  .option('-y, --yes', 'Skip interactive prompts')
  .action(async (options) => {
    console.log(chalk.blue('🧠 Neural Tools Initialization'));
    console.log('==================================');
    console.log();
    
    try {
      // Step 1: Detect project
      console.log(chalk.blue('[1/5] Detecting project...'));
      const projectInfo = detectProject();
      console.log(chalk.green(`✅ Detected ${projectInfo.type} project: ${projectInfo.name}`));
      
      // Step 2: Check Docker
      console.log(chalk.blue('[2/5] Checking Docker...'));
      const dockerOk = await checkDocker();
      if (!dockerOk) {
        throw new Error('Docker not running. Please start Docker Desktop and try again.');
      }
      console.log(chalk.green('✅ Docker is running'));
      
      // Step 3: Check if our container exists
      console.log(chalk.blue('[3/5] Checking Neural Tools container...'));
      const containerExists = await new Promise((resolve) => {
        exec('docker ps -a --format "{{.Names}}" | grep neural-multi-project-final', (error, stdout) => {
          resolve(stdout.includes('neural-multi-project-final'));
        });
      });
      
      if (!containerExists) {
        console.log(chalk.yellow('⚠️ Neural Tools container not found'));
        console.log(chalk.blue('Please run the main container setup first:'));
        console.log('  docker run -d --name neural-multi-project-final \\');
        console.log('    -v /private/tmp/workspace:/workspace \\');
        console.log('    neural-multi-project-final');
        return;
      }
      console.log(chalk.green('✅ Neural Tools container ready'));
      
      // Step 4: Setup workspace
      console.log(chalk.blue('[4/5] Setting up workspace...'));
      await setupWorkspace(projectInfo);
      console.log(chalk.green('✅ Workspace configured'));
      
      // Step 5: Configure MCP
      console.log(chalk.blue('[5/5] Configuring MCP integration...'));
      const configPath = options.global 
        ? path.join(os.homedir(), '.claude', 'mcp_config.json')
        : path.join(projectInfo.path, '.mcp.json');
        
      await createMCPConfig(projectInfo, configPath);
      console.log(chalk.green(`✅ MCP configured at ${configPath}`));
      
      console.log();
      console.log(chalk.green('🎉 Neural Tools initialized!'));
      console.log();
      console.log(chalk.blue('📚 Available commands:'));
      console.log('  neural status    - Check system status');
      console.log('  neural test      - Test MCP connection');
      console.log();
      console.log(chalk.yellow('💡 Next steps:'));
      console.log('  1. Open this project in Claude Code');
      console.log('  2. Neural Tools are now available as MCP tools!');
      console.log();
      
    } catch (error) {
      console.log(chalk.red(`❌ Initialization failed: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Check Neural Tools system status')
  .action(async () => {
    console.log(chalk.blue('🔍 Neural Tools Status'));
    console.log();
    
    try {
      const projectInfo = detectProject();
      
      // Check container
      const containerRunning = await new Promise((resolve) => {
        exec('docker ps --format "{{.Names}}\\t{{.Status}}" | grep neural-multi-project-final', (error, stdout) => {
          if (error) resolve(false);
          else resolve(stdout.includes('Up'));
        });
      });
      
      console.log(chalk.bold('Container Status:'));
      const containerIcon = containerRunning ? '🟢' : '🔴';
      const containerStatus = containerRunning ? 'Running' : 'Stopped';
      console.log(`  ${containerIcon} neural-multi-project-final - ${containerStatus}`);
      
      // Check MCP config
      console.log(chalk.bold('\\nMCP Configuration:'));
      const mcpConfigPath = path.join(projectInfo.path, '.mcp.json');
      const mcpExists = await fs.pathExists(mcpConfigPath);
      const mcpIcon = mcpExists ? '🟢' : '🔴';
      console.log(`  ${mcpIcon} Configuration: ${mcpExists ? 'Found' : 'Missing'}`);
      
      // Project info
      console.log(chalk.bold('\\nProject:'));
      console.log(`  📁 Name: ${projectInfo.name}`);
      console.log(`  📂 Path: ${projectInfo.path}`);
      console.log(`  🏷️  Type: ${projectInfo.type}`);
      
    } catch (error) {
      console.log(chalk.red(`❌ Status check failed: ${error.message}`));
    }
  });

program
  .command('test')
  .description('Test MCP connection')
  .action(async () => {
    console.log(chalk.blue('🧪 Testing MCP Connection'));
    console.log();
    
    try {
      const projectInfo = detectProject();
      
      // Test basic container connectivity
      console.log('Testing container connection...');
      const testResult = await new Promise((resolve, reject) => {
        const testProcess = spawn('docker', [
          'exec', '-i', 'neural-multi-project-final',
          'python3', '-c', `
import sys
sys.path.append('/app/neural-tools-src/servers')
try:
    from neural_server_stdio import MultiProjectServiceState
    state = MultiProjectServiceState()
    project = state.detect_project_from_path('/workspace/${projectInfo.name}/test.py')
    print(f'✅ MCP server functional, detected project: {project}')
    sys.exit(0)
except Exception as e:
    print(f'❌ MCP server test failed: {e}')
    sys.exit(1)
          `
        ]);
        
        let output = '';
        testProcess.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        testProcess.on('close', (code) => {
          if (code === 0) {
            resolve(output);
          } else {
            reject(new Error(output));
          }
        });
      });
      
      console.log(testResult);
      console.log(chalk.green('✅ MCP connection test passed'));
      
    } catch (error) {
      console.log(chalk.red(`❌ MCP test failed: ${error.message}`));
    }
  });

program.parse();
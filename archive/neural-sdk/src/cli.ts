#!/usr/bin/env node
/**
 * Neural SDK - Zero-config GraphRAG integration
 * Modern CLI following 2025 patterns: auto-discovery, container abstraction, zero-config defaults
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import { NeuralSDK } from './sdk';
import { ProjectDetector } from './project-detector';
import { ContainerManager } from './container-manager';
import { MCPConfigManager } from './mcp-config';
import { FileWatcher } from './file-watcher';

const program = new Command();
const sdk = new NeuralSDK();

program
  .name('neural')
  .description('Zero-config Neural Tools GraphRAG SDK')
  .version('1.0.0');

program
  .command('init')
  .description('Initialize Neural Tools in current project (zero-config)')
  .option('-g, --global', 'Install globally for all projects')
  .option('-y, --yes', 'Skip interactive prompts')
  .option('--port <port>', 'Custom MCP port', '3000')
  .option('--contract-validation', 'Enable contract validation during setup')
  .action(async (options) => {
    const spinner = ora('Initializing Neural Tools...').start();
    
    try {
      // Auto-detect project context
      const detector = new ProjectDetector();
      const projectInfo = await detector.detectProject(process.cwd());
      
      spinner.text = `Detected ${projectInfo.type} project: ${projectInfo.name}`;
      
      // Interactive confirmation unless --yes
      if (!options.yes) {
        const { confirmed } = await inquirer.prompt([{
          type: 'confirm',
          name: 'confirmed',
          message: `Initialize Neural Tools for "${projectInfo.name}"?`,
          default: true
        }]);
        
        if (!confirmed) {
          spinner.fail('Installation cancelled');
          return;
        }
      }
      
      // Auto-setup everything
      await sdk.initialize(projectInfo, {
        global: options.global,
        port: parseInt(options.port),
        autoStart: true,
        enableContractValidation: options.contractValidation
      });
      
      spinner.succeed(`Neural Tools initialized for ${projectInfo.name}`);
      
      console.log(chalk.green('\n✅ Ready to go!'));
      console.log(chalk.blue('\n📚 Available commands:'));
      console.log('  neural status    - Check system status');
      console.log('  neural watch     - Start file watching');
      console.log('  neural stop      - Stop services');
      console.log('  neural logs      - View logs');
      console.log('  neural validate  - Validate contracts');
      console.log('\n💡 Your project files are now automatically indexed!');
      
    } catch (error) {
      spinner.fail(`Initialization failed: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Check Neural Tools system status')
  .action(async () => {
    const spinner = ora('Checking system status...').start();
    
    try {
      const status = await sdk.getSystemStatus();
      spinner.stop();
      
      console.log(chalk.blue('🔍 Neural Tools Status\n'));
      
      // Container status
      console.log(chalk.bold('Containers:'));
      status.containers.forEach(container => {
        const icon = container.running ? '🟢' : '🔴';
        console.log(`  ${icon} ${container.name} - ${container.status}`);
      });
      
      // MCP status
      console.log(chalk.bold('\nMCP Integration:'));
      const mcpIcon = status.mcp.configured ? '🟢' : '🔴';
      console.log(`  ${mcpIcon} Configuration: ${status.mcp.configured ? 'Active' : 'Missing'}`);
      
      if (status.mcp.configured && status.mcp.contractValid !== undefined) {
        const contractIcon = status.mcp.contractValid ? '🟢' : '🔴';
        console.log(`  ${contractIcon} Contract Validation: ${status.mcp.contractValid ? 'Valid' : 'Failed'}`);
      }
      
      // Project status
      console.log(chalk.bold('\nProject:'));
      console.log(`  📁 Name: ${status.project.name}`);
      console.log(`  📊 Indexed files: ${status.project.indexedFiles}`);
      console.log(`  🕒 Last updated: ${status.project.lastIndexed}`);
      
    } catch (error) {
      spinner.fail(`Status check failed: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('watch')
  .description('Start intelligent file watching and auto-indexing')
  .option('--patterns <patterns>', 'File patterns to watch (comma-separated)', '**/*.{js,ts,py,md,txt}')
  .option('--ignore <patterns>', 'Patterns to ignore', 'node_modules,dist,.git')
  .action(async (options) => {
    const spinner = ora('Starting file watcher...').start();
    
    try {
      const detector = new ProjectDetector();
      const projectInfo = await detector.detectProject(process.cwd());
      
      const watcher = new FileWatcher(projectInfo);
      
      await watcher.start({
        patterns: options.patterns.split(','),
        ignore: options.ignore.split(',')
      });
      
      spinner.succeed('File watcher started');
      console.log(chalk.green('👀 Watching for changes...'));
      console.log(chalk.gray('Press Ctrl+C to stop'));
      
      // Keep process alive
      process.on('SIGINT', async () => {
        console.log(chalk.yellow('\n🛑 Stopping file watcher...'));
        await watcher.stop();
        console.log(chalk.green('✅ File watcher stopped'));
        process.exit(0);
      });
      
    } catch (error) {
      spinner.fail(`File watcher failed: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('stop')
  .description('Stop Neural Tools services')
  .action(async () => {
    const spinner = ora('Stopping Neural Tools...').start();
    
    try {
      await sdk.stop();
      spinner.succeed('Neural Tools stopped');
    } catch (error) {
      spinner.fail(`Stop failed: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('logs')
  .description('View Neural Tools logs')
  .option('-f, --follow', 'Follow log output')
  .option('--lines <lines>', 'Number of lines to show', '100')
  .action(async (options) => {
    try {
      await sdk.showLogs({
        follow: options.follow,
        lines: parseInt(options.lines)
      });
    } catch (error) {
      console.error(chalk.red(`Logs failed: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate MCP contracts and compatibility')
  .option('--execution-tests', 'Include execution tests (slower but more thorough)')
  .option('--performance-threshold <ms>', 'Performance threshold in milliseconds', '5000')
  .option('--tools <tools>', 'Comma-separated list of specific tools to test')
  .action(async (options) => {
    const spinner = ora('Running contract validation...').start();
    
    try {
      const detector = new ProjectDetector();
      const projectInfo = await detector.detectProject(process.cwd());
      
      const result = await sdk.validateMCPContract(projectInfo);
      spinner.stop();
      
      console.log(chalk.blue('🧪 MCP Contract Validation Results\n'));
      
      if (result.success) {
        console.log(chalk.green('✅ All contracts validated successfully!'));
      } else {
        console.log(chalk.red('❌ Contract validation failed'));
      }
      
      // Import formatValidationReport
      const { formatValidationReport } = await import('./contract/validator');
      console.log('\n' + formatValidationReport(result));
      
      // Exit with error code if validation failed
      if (!result.success) {
        process.exit(1);
      }
      
    } catch (error) {
      spinner.fail(`Contract validation failed: ${error.message}`);
      process.exit(1);
    }
  });

program
  .command('reset')
  .description('Reset Neural Tools (removes all data)')
  .action(async () => {
    const { confirmed } = await inquirer.prompt([{
      type: 'confirm',
      name: 'confirmed',
      message: chalk.red('⚠️  This will delete all indexed data. Continue?'),
      default: false
    }]);
    
    if (confirmed) {
      const spinner = ora('Resetting Neural Tools...').start();
      try {
        await sdk.reset();
        spinner.succeed('Neural Tools reset complete');
      } catch (error) {
        spinner.fail(`Reset failed: ${error.message}`);
        process.exit(1);
      }
    }
  });

// Global error handling
process.on('uncaughtException', (error) => {
  console.error(chalk.red('💥 Unexpected error:'), error.message);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error(chalk.red('💥 Unhandled promise rejection:'), reason);
  process.exit(1);
});

program.parse();
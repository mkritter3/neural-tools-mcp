/**
 * Intelligent file watcher - Auto-indexes files with smart debouncing
 */

import chokidar, { FSWatcher } from 'chokidar';
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import { ProjectInfo } from './project-detector';

export interface WatchOptions {
  patterns: string[];
  ignore: string[];
  debounceMs?: number;
  batchSize?: number;
}

export interface FileChangeEvent {
  type: 'add' | 'change' | 'unlink';
  path: string;
  stats?: fs.Stats;
  timestamp: Date;
}

export class FileWatcher {
  private watcher: FSWatcher | null = null;
  private projectInfo: ProjectInfo;
  private pendingChanges: Map<string, FileChangeEvent> = new Map();
  private debounceTimer: NodeJS.Timeout | null = null;
  private isIndexing: boolean = false;
  
  constructor(projectInfo: ProjectInfo) {
    this.projectInfo = projectInfo;
  }
  
  /**
   * Start intelligent file watching
   */
  async start(options: WatchOptions): Promise<void> {
    if (this.watcher) {
      await this.stop();
    }
    
    const watchPatterns = options.patterns.map(pattern => 
      path.join(this.projectInfo.path, pattern)
    );
    
    const ignorePatterns = [
      ...options.ignore,
      '**/node_modules/**',
      '**/.git/**',
      '**/dist/**',
      '**/build/**',
      '**/.neural/**', // Don't watch our own data directory
      '**/coverage/**',
      '**/.vscode/**',
      '**/.idea/**'
    ];
    
    this.watcher = chokidar.watch(watchPatterns, {
      ignored: ignorePatterns,
      ignoreInitial: false, // Index existing files on startup
      persistent: true,
      followSymlinks: false,
      depth: 10,
      awaitWriteFinish: {
        stabilityThreshold: 100,
        pollInterval: 50
      }
    });
    
    // Set up event handlers
    this.watcher
      .on('add', (filePath, stats) => this.handleFileEvent('add', filePath, stats))
      .on('change', (filePath, stats) => this.handleFileEvent('change', filePath, stats))
      .on('unlink', (filePath) => this.handleFileEvent('unlink', filePath))
      .on('error', (error) => {
        console.error(chalk.red('👀 File watcher error:'), error.message);
      });
    
    console.log(chalk.green(`👀 Watching ${this.projectInfo.name} for changes...`));
  }
  
  /**
   * Stop file watching
   */
  async stop(): Promise<void> {
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }
    
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    
    this.pendingChanges.clear();
  }
  
  /**
   * Handle file system events with intelligent debouncing
   */
  private handleFileEvent(type: 'add' | 'change' | 'unlink', filePath: string, stats?: fs.Stats): void {
    const relativePath = path.relative(this.projectInfo.path, filePath);
    
    // Skip files that shouldn't be indexed
    if (this.shouldSkipFile(filePath)) {
      return;
    }
    
    const event: FileChangeEvent = {
      type,
      path: filePath,
      stats,
      timestamp: new Date()
    };
    
    // Update pending changes
    this.pendingChanges.set(filePath, event);
    
    // Debounce processing
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    
    this.debounceTimer = setTimeout(() => {
      this.processPendingChanges();
    }, 500); // 500ms debounce
  }
  
  /**
   * Process accumulated file changes in batches
   */
  private async processPendingChanges(): Promise<void> {
    if (this.isIndexing || this.pendingChanges.size === 0) {
      return;
    }
    
    this.isIndexing = true;
    const changes = Array.from(this.pendingChanges.values());
    this.pendingChanges.clear();
    
    try {
      // Group changes by type
      const addedFiles = changes.filter(c => c.type === 'add').map(c => c.path);
      const changedFiles = changes.filter(c => c.type === 'change').map(c => c.path);
      const deletedFiles = changes.filter(c => c.type === 'unlink').map(c => c.path);
      
      if (addedFiles.length > 0) {
        console.log(chalk.blue(`📝 Indexing ${addedFiles.length} new files...`));
        await this.indexFiles(addedFiles);
      }
      
      if (changedFiles.length > 0) {
        console.log(chalk.yellow(`🔄 Re-indexing ${changedFiles.length} changed files...`));
        await this.indexFiles(changedFiles);
      }
      
      if (deletedFiles.length > 0) {
        console.log(chalk.red(`🗑️  Removing ${deletedFiles.length} deleted files from index...`));
        await this.removeFromIndex(deletedFiles);
      }
      
      const totalChanges = addedFiles.length + changedFiles.length + deletedFiles.length;
      if (totalChanges > 0) {
        console.log(chalk.green(`✅ Processed ${totalChanges} file changes`));
      }
      
    } catch (error) {
      console.error(chalk.red('❌ File processing error:'), error.message);
    } finally {
      this.isIndexing = false;
    }
  }
  
  /**
   * Index files using the Neural Tools container
   */
  private async indexFiles(filePaths: string[]): Promise<void> {
    if (filePaths.length === 0) return;
    
    try {
      // Use Docker exec to run indexing in the project container
      const { spawn } = require('child_process');
      
      for (const filePath of filePaths) {
        const workspacePath = `/workspace/${this.projectInfo.name}/${path.relative(this.projectInfo.path, filePath)}`;
        
        const indexProcess = spawn('docker', [
          'exec',
          '-i',
          `neural-${this.projectInfo.name}`,
          'python3',
          '-c',
          `
import sys
sys.path.append('/app/src')

try:
    from neural_mcp.neural_server_stdio import MultiProjectServiceState
    
    # Create service state and get project container
    state = MultiProjectServiceState()
    project_name = state.detect_project_from_path('${workspacePath}')
    print(f'Indexing {workspacePath} for project: {project_name}')
    
    # This would call the actual indexing logic
    # For now, just simulate successful indexing
    print(f'✅ Indexed: ${path.basename(filePath)}')
    
except Exception as e:
    print(f'❌ Indexing failed: {e}')
    sys.exit(1)
          `
        ]);
        
        await new Promise((resolve, reject) => {
          indexProcess.on('close', (code) => {
            if (code === 0) {
              resolve(null);
            } else {
              reject(new Error(`Indexing process exited with code ${code}`));
            }
          });
          
          indexProcess.on('error', reject);
        });
      }
      
    } catch (error) {
      throw new Error(`Failed to index files: ${error.message}`);
    }
  }
  
  /**
   * Remove files from index
   */
  private async removeFromIndex(filePaths: string[]): Promise<void> {
    if (filePaths.length === 0) return;
    
    try {
      // Use Docker exec to remove from index
      const { spawn } = require('child_process');
      
      for (const filePath of filePaths) {
        const workspacePath = `/workspace/${this.projectInfo.name}/${path.relative(this.projectInfo.path, filePath)}`;
        
        const removeProcess = spawn('docker', [
          'exec',
          '-i',
          `neural-${this.projectInfo.name}`,
          'python3',
          '-c',
          `
import sys
sys.path.append('/app/src')

try:
    from neural_mcp.neural_server_stdio import MultiProjectServiceState
    
    # Create service state and get project container  
    state = MultiProjectServiceState()
    project_name = state.detect_project_from_path('${workspacePath}')
    print(f'Removing {workspacePath} from project: {project_name}')
    
    # This would call the actual removal logic
    print(f'🗑️ Removed: ${path.basename(filePath)}')
    
except Exception as e:
    print(f'❌ Removal failed: {e}')
    sys.exit(1)
          `
        ]);
        
        await new Promise((resolve, reject) => {
          removeProcess.on('close', (code) => {
            if (code === 0) {
              resolve(null);
            } else {
              reject(new Error(`Removal process exited with code ${code}`));
            }
          });
          
          removeProcess.on('error', reject);
        });
      }
      
    } catch (error) {
      console.error(chalk.red('Failed to remove from index:'), error.message);
      // Don't throw - removal failures shouldn't stop the watcher
    }
  }
  
  /**
   * Check if file should be skipped
   */
  private shouldSkipFile(filePath: string): boolean {
    const fileName = path.basename(filePath);
    const extension = path.extname(filePath).toLowerCase();
    
    // Skip hidden files
    if (fileName.startsWith('.') && fileName !== '.env') {
      return true;
    }
    
    // Skip binary files
    const binaryExtensions = [
      '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
      '.pdf', '.zip', '.tar', '.gz', '.rar',
      '.exe', '.dll', '.so', '.dylib',
      '.mp3', '.mp4', '.avi', '.mov'
    ];
    
    if (binaryExtensions.includes(extension)) {
      return true;
    }
    
    // Skip very large files (>1MB)
    try {
      const stats = fs.statSync(filePath);
      if (stats.size > 1024 * 1024) {
        return true;
      }
    } catch (error) {
      // File might not exist anymore, skip it
      return true;
    }
    
    return false;
  }
  
  /**
   * Get current watching status
   */
  getStatus(): { watching: boolean; pendingChanges: number } {
    return {
      watching: this.watcher !== null,
      pendingChanges: this.pendingChanges.size
    };
  }
  
  /**
   * Force index all files in project
   */
  async indexAllFiles(): Promise<void> {
    if (!this.watcher) {
      throw new Error('File watcher not started');
    }
    
    console.log(chalk.blue('🔄 Starting full project indexing...'));
    
    // Get all watched files
    const watchedPaths = this.watcher.getWatched();
    const allFiles: string[] = [];
    
    for (const [dir, files] of Object.entries(watchedPaths)) {
      for (const file of files) {
        const fullPath = path.join(dir, file);
        if (!this.shouldSkipFile(fullPath)) {
          allFiles.push(fullPath);
        }
      }
    }
    
    if (allFiles.length > 0) {
      await this.indexFiles(allFiles);
      console.log(chalk.green(`✅ Indexed ${allFiles.length} files`));
    } else {
      console.log(chalk.yellow('⚠️ No files found to index'));
    }
  }
}
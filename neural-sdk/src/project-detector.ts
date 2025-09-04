/**
 * Intelligent project detection - auto-discovers project type and configuration
 */

import fs from 'fs-extra';
import path from 'path';
import glob from 'glob';

export interface ProjectInfo {
  name: string;
  path: string;
  type: ProjectType;
  language: string[];
  frameworks: string[];
  packageManager: PackageManager;
  structure: ProjectStructure;
}

export type ProjectType = 
  | 'node'
  | 'python' 
  | 'rust'
  | 'go'
  | 'java'
  | 'csharp'
  | 'cpp'
  | 'mixed'
  | 'unknown';

export type PackageManager = 
  | 'npm'
  | 'yarn' 
  | 'pnpm'
  | 'pip'
  | 'poetry'
  | 'cargo'
  | 'go-mod'
  | 'maven'
  | 'gradle'
  | 'nuget'
  | 'cmake'
  | 'unknown';

export interface ProjectStructure {
  hasTests: boolean;
  hasDocs: boolean;
  sourceDir: string[];
  testDir: string[];
  configFiles: string[];
}

export class ProjectDetector {
  
  /**
   * Auto-detect project information from directory
   */
  async detectProject(projectPath: string): Promise<ProjectInfo> {
    const normalizedPath = path.resolve(projectPath);
    const projectName = this.sanitizeProjectName(path.basename(normalizedPath));
    
    // Detect project type and language
    const type = await this.detectProjectType(normalizedPath);
    const languages = await this.detectLanguages(normalizedPath);
    const frameworks = await this.detectFrameworks(normalizedPath, type);
    const packageManager = await this.detectPackageManager(normalizedPath);
    const structure = await this.analyzeStructure(normalizedPath);
    
    return {
      name: projectName,
      path: normalizedPath,
      type,
      language: languages,
      frameworks,
      packageManager,
      structure
    };
  }
  
  /**
   * Detect primary project type
   */
  private async detectProjectType(projectPath: string): Promise<ProjectType> {
    const indicators = [
      { file: 'package.json', type: 'node' as ProjectType },
      { file: 'pyproject.toml', type: 'python' as ProjectType },
      { file: 'requirements.txt', type: 'python' as ProjectType },
      { file: 'Cargo.toml', type: 'rust' as ProjectType },
      { file: 'go.mod', type: 'go' as ProjectType },
      { file: 'pom.xml', type: 'java' as ProjectType },
      { file: 'build.gradle', type: 'java' as ProjectType },
      { file: '*.csproj', type: 'csharp' as ProjectType },
      { file: '*.sln', type: 'csharp' as ProjectType },
      { file: 'CMakeLists.txt', type: 'cpp' as ProjectType },
      { file: 'Makefile', type: 'cpp' as ProjectType }
    ];
    
    const detectedTypes = new Set<ProjectType>();
    
    for (const indicator of indicators) {
      if (indicator.file.includes('*')) {
        const matches = glob.sync(indicator.file, { cwd: projectPath });
        if (matches.length > 0) {
          detectedTypes.add(indicator.type);
        }
      } else {
        const filePath = path.join(projectPath, indicator.file);
        if (await fs.pathExists(filePath)) {
          detectedTypes.add(indicator.type);
        }
      }
    }
    
    if (detectedTypes.size === 0) return 'unknown';
    if (detectedTypes.size === 1) return Array.from(detectedTypes)[0];
    return 'mixed';
  }
  
  /**
   * Detect languages used in project
   */
  private async detectLanguages(projectPath: string): Promise<string[]> {
    const languageExtensions = {
      'TypeScript': ['ts', 'tsx'],
      'JavaScript': ['js', 'jsx', 'mjs'],
      'Python': ['py', 'pyx', 'pyi'],
      'Rust': ['rs'],
      'Go': ['go'],
      'Java': ['java'],
      'C#': ['cs'],
      'C++': ['cpp', 'cc', 'cxx', 'hpp', 'h'],
      'C': ['c', 'h'],
      'Shell': ['sh', 'bash'],
      'YAML': ['yml', 'yaml'],
      'JSON': ['json'],
      'Markdown': ['md', 'mdx']
    };
    
    const detected = new Set<string>();
    
    for (const [language, extensions] of Object.entries(languageExtensions)) {
      for (const ext of extensions) {
        const pattern = `**/*.${ext}`;
        const matches = glob.sync(pattern, { 
          cwd: projectPath,
          ignore: ['node_modules/**', 'dist/**', 'build/**', '.git/**']
        });
        if (matches.length > 0) {
          detected.add(language);
          break;
        }
      }
    }
    
    return Array.from(detected);
  }
  
  /**
   * Detect frameworks and libraries
   */
  private async detectFrameworks(projectPath: string, projectType: ProjectType): Promise<string[]> {
    const frameworks = new Set<string>();
    
    try {
      if (projectType === 'node') {
        const packageJsonPath = path.join(projectPath, 'package.json');
        if (await fs.pathExists(packageJsonPath)) {
          const packageJson = await fs.readJSON(packageJsonPath);
          const deps = { ...packageJson.dependencies, ...packageJson.devDependencies };
          
          // Common frameworks
          if (deps.react) frameworks.add('React');
          if (deps.vue) frameworks.add('Vue');
          if (deps.angular) frameworks.add('Angular');
          if (deps.svelte) frameworks.add('Svelte');
          if (deps.next) frameworks.add('Next.js');
          if (deps.nuxt) frameworks.add('Nuxt.js');
          if (deps.express) frameworks.add('Express');
          if (deps.fastify) frameworks.add('Fastify');
          if (deps.nest) frameworks.add('NestJS');
        }
      }
      
      if (projectType === 'python') {
        // Check for Python frameworks
        const files = await fs.readdir(projectPath);
        if (files.includes('manage.py')) frameworks.add('Django');
        
        const requirementsPath = path.join(projectPath, 'requirements.txt');
        if (await fs.pathExists(requirementsPath)) {
          const requirements = await fs.readFile(requirementsPath, 'utf8');
          if (requirements.includes('flask')) frameworks.add('Flask');
          if (requirements.includes('fastapi')) frameworks.add('FastAPI');
          if (requirements.includes('streamlit')) frameworks.add('Streamlit');
        }
      }
      
    } catch (error) {
      // Ignore errors in framework detection
    }
    
    return Array.from(frameworks);
  }
  
  /**
   * Detect package manager
   */
  private async detectPackageManager(projectPath: string): Promise<PackageManager> {
    const managers: Array<{ file: string; manager: PackageManager }> = [
      { file: 'yarn.lock', manager: 'yarn' },
      { file: 'pnpm-lock.yaml', manager: 'pnpm' },
      { file: 'package-lock.json', manager: 'npm' },
      { file: 'poetry.lock', manager: 'poetry' },
      { file: 'Pipfile.lock', manager: 'pip' },
      { file: 'Cargo.lock', manager: 'cargo' },
      { file: 'go.sum', manager: 'go-mod' },
      { file: 'pom.xml', manager: 'maven' },
      { file: 'build.gradle', manager: 'gradle' }
    ];
    
    for (const { file, manager } of managers) {
      if (await fs.pathExists(path.join(projectPath, file))) {
        return manager;
      }
    }
    
    // Fallback detection
    if (await fs.pathExists(path.join(projectPath, 'package.json'))) return 'npm';
    if (await fs.pathExists(path.join(projectPath, 'requirements.txt'))) return 'pip';
    
    return 'unknown';
  }
  
  /**
   * Analyze project structure
   */
  private async analyzeStructure(projectPath: string): Promise<ProjectStructure> {
    const structure: ProjectStructure = {
      hasTests: false,
      hasDocs: false,
      sourceDir: [],
      testDir: [],
      configFiles: []
    };
    
    try {
      const items = await fs.readdir(projectPath);
      
      // Common source directories
      const sourceDirs = ['src', 'lib', 'app', 'source', 'code'];
      const testDirs = ['test', 'tests', '__tests__', 'spec', 'specs'];
      const docDirs = ['docs', 'doc', 'documentation'];
      const configFiles = ['.env', 'config.json', '.gitignore', 'tsconfig.json', 'webpack.config.js'];
      
      for (const item of items) {
        const itemPath = path.join(projectPath, item);
        const stat = await fs.stat(itemPath);
        
        if (stat.isDirectory()) {
          if (sourceDirs.includes(item)) structure.sourceDir.push(item);
          if (testDirs.includes(item)) {
            structure.testDir.push(item);
            structure.hasTests = true;
          }
          if (docDirs.includes(item)) structure.hasDocs = true;
        } else {
          if (configFiles.includes(item)) structure.configFiles.push(item);
        }
      }
      
      // Check for test files if no test directory
      if (!structure.hasTests) {
        const testFiles = glob.sync('**/*.{test,spec}.{js,ts,py}', {
          cwd: projectPath,
          ignore: ['node_modules/**']
        });
        structure.hasTests = testFiles.length > 0;
      }
      
    } catch (error) {
      // Return default structure on error
    }
    
    return structure;
  }
  
  /**
   * Sanitize project name for container/config usage
   */
  private sanitizeProjectName(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9-]/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .substring(0, 63); // Docker name limit
  }
}
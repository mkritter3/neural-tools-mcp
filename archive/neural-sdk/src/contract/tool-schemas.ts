/**
 * MCP Tool Schema Definitions
 * Type-safe contracts for Neural Tools MCP server
 */

export interface MCPToolParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object';
  required: boolean;
  description?: string;
  default?: any;
}

export interface MCPToolSchema {
  name: string;
  description: string;
  parameters: MCPToolParameter[];
  returnType: string;
  version: string;
  deprecated?: boolean;
  deprecationMessage?: string;
}

/**
 * Core Tools - v1.0.0
 */
export const CORE_TOOL_SCHEMAS: MCPToolSchema[] = [
  {
    name: 'project_understanding',
    description: 'Generate condensed project understanding without reading all files',
    version: '1.0.0',
    parameters: [
      {
        name: 'scope',
        type: 'string',
        required: false,
        description: 'Analysis scope: full, summary, architecture, dependencies, core_logic',
        default: 'full'
      },
      {
        name: 'max_tokens',
        type: 'string',
        required: false,
        description: 'Maximum tokens for response',
        default: '2000'
      }
    ],
    returnType: 'ProjectUnderstanding'
  },
  {
    name: 'semantic_code_search',
    description: 'Search code by semantic meaning using embeddings',
    version: '1.0.0',
    parameters: [
      {
        name: 'query',
        type: 'string',
        required: true,
        description: 'Natural language search query'
      },
      {
        name: 'search_type',
        type: 'string',
        required: false,
        description: 'Search algorithm: semantic, hybrid, vector',
        default: 'semantic'
      },
      {
        name: 'limit',
        type: 'string',
        required: false,
        description: 'Maximum number of results',
        default: '10'
      },
      {
        name: 'filters',
        type: 'string',
        required: false,
        description: 'JSON filter conditions'
      }
    ],
    returnType: 'SearchResults'
  },
  {
    name: 'atomic_dependency_tracer',
    description: 'Trace code dependencies and call relationships',
    version: '1.0.0',
    parameters: [
      {
        name: 'target',
        type: 'string',
        required: true,
        description: 'Function, class, or module name to trace'
      },
      {
        name: 'trace_type',
        type: 'string',
        required: false,
        description: 'Trace direction: calls, imports, full',
        default: 'calls'
      },
      {
        name: 'depth',
        type: 'string',
        required: false,
        description: 'Maximum trace depth',
        default: '3'
      }
    ],
    returnType: 'DependencyGraph'
  },
  {
    name: 'vibe_preservation',
    description: 'Preserve code style and patterns across modifications',
    version: '1.0.0',
    parameters: [
      {
        name: 'action',
        type: 'string',
        required: true,
        description: 'Action to perform: analyze, generate, check'
      },
      {
        name: 'code_sample',
        type: 'string',
        required: false,
        description: 'Code to analyze or reference'
      },
      {
        name: 'context',
        type: 'string',
        required: false,
        description: 'Additional context for analysis'
      }
    ],
    returnType: 'StyleAnalysis'
  },
  {
    name: 'project_auto_index',
    description: 'Automatically index project files for search',
    version: '1.0.0',
    parameters: [
      {
        name: 'scope',
        type: 'string',
        required: false,
        description: 'Indexing scope: modified, all, incremental',
        default: 'modified'
      },
      {
        name: 'since_minutes',
        type: 'string',
        required: false,
        description: 'Only index files modified within N minutes'
      },
      {
        name: 'force_reindex',
        type: 'string',
        required: false,
        description: 'Force reindexing of all files: true, false',
        default: 'false'
      }
    ],
    returnType: 'IndexingResult'
  },
  {
    name: 'graph_query',
    description: 'Execute Cypher queries on Neo4j knowledge graph',
    version: '1.0.0',
    parameters: [
      {
        name: 'query',
        type: 'string',
        required: true,
        description: 'Cypher query to execute'
      }
    ],
    returnType: 'GraphQueryResult'
  }
];

/**
 * Memory Tools - v1.0.0
 */
export const MEMORY_TOOL_SCHEMAS: MCPToolSchema[] = [
  {
    name: 'memory_store_enhanced',
    description: 'Store content with enhanced semantic indexing',
    version: '1.0.0',
    parameters: [
      {
        name: 'content',
        type: 'string',
        required: true,
        description: 'Content to store in memory'
      },
      {
        name: 'category',
        type: 'string',
        required: false,
        description: 'Memory category for organization',
        default: 'general'
      },
      {
        name: 'metadata',
        type: 'string',
        required: false,
        description: 'JSON metadata for enhanced searchability'
      }
    ],
    returnType: 'MemoryStoreResult'
  },
  {
    name: 'memory_search_enhanced',
    description: 'Search stored memories using semantic similarity',
    version: '1.0.0',
    parameters: [
      {
        name: 'query',
        type: 'string',
        required: true,
        description: 'Search query for memory retrieval'
      },
      {
        name: 'category',
        type: 'string',
        required: false,
        description: 'Limit search to specific category'
      },
      {
        name: 'limit',
        type: 'string',
        required: false,
        description: 'Maximum number of results',
        default: '10'
      }
    ],
    returnType: 'MemorySearchResult'
  },
  {
    name: 'schema_customization',
    description: 'Customize collection schemas and data models',
    version: '1.0.0',
    parameters: [
      {
        name: 'action',
        type: 'string',
        required: true,
        description: 'Schema action: create, update, delete, list'
      },
      {
        name: 'collection_name',
        type: 'string',
        required: false,
        description: 'Target collection name'
      },
      {
        name: 'schema_data',
        type: 'string',
        required: false,
        description: 'JSON schema definition'
      }
    ],
    returnType: 'SchemaOperationResult'
  }
];

/**
 * All available tool schemas
 */
export const ALL_TOOL_SCHEMAS = [
  ...CORE_TOOL_SCHEMAS,
  ...MEMORY_TOOL_SCHEMAS
];

/**
 * Version registry for backward compatibility
 */
export const SUPPORTED_API_VERSIONS = ['1.0.0', '1.1.0'] as const;
export type SupportedAPIVersion = typeof SUPPORTED_API_VERSIONS[number];

/**
 * Get tool schema by name and version
 */
export function getToolSchema(toolName: string, version: SupportedAPIVersion = '1.0.0'): MCPToolSchema | undefined {
  return ALL_TOOL_SCHEMAS.find(schema => 
    schema.name === toolName && schema.version === version
  );
}

/**
 * Get all schemas for a specific version
 */
export function getSchemasForVersion(version: SupportedAPIVersion): MCPToolSchema[] {
  return ALL_TOOL_SCHEMAS.filter(schema => schema.version === version);
}

/**
 * Check if a tool exists in any supported version
 */
export function isToolSupported(toolName: string): boolean {
  return ALL_TOOL_SCHEMAS.some(schema => schema.name === toolName);
}

/**
 * Get tool parameter validation info
 */
export function getRequiredParameters(toolName: string, version: SupportedAPIVersion = '1.0.0'): string[] {
  const schema = getToolSchema(toolName, version);
  if (!schema) return [];
  
  return schema.parameters
    .filter(param => param.required)
    .map(param => param.name);
}

/**
 * Response type definitions for type safety
 */
export interface ProjectUnderstanding {
  project: string;
  scope: string;
  indexed_categories: string[];
  code_patterns?: any[];
  architecture_overview?: string;
  dependencies?: string[];
}

export interface SearchResults {
  query: string;
  results: Array<{
    content: string;
    score: number;
    source: string;
    metadata?: any;
  }>;
  search_type: string;
  total_time_ms: number;
}

export interface DependencyGraph {
  target: string;
  dependencies: Array<{
    name: string;
    type: string;
    relationship: string;
    location?: string;
  }>;
  trace_type: string;
  depth: number;
}

export interface MemoryStoreResult {
  id: string;
  status: 'success' | 'error';
  message?: string;
  category: string;
}

export interface MemorySearchResult {
  query: string;
  results: Array<{
    id: string;
    content: string;
    score: number;
    category: string;
    metadata?: any;
  }>;
  total_results: number;
}
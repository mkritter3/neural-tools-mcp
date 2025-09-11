# ADR-0020: Per-Project Custom GraphRAG Schemas

## Status
Proposed

## Context

Currently, all projects connected via MCP use the same GraphRAG schema structure:
- Fixed node types (CodeElement, Function, Class, Module, etc.)
- Fixed relationship types (CALLS, IMPORTS, CONTAINS, etc.)
- Fixed vector collection schemas (code, docs, general)

However, different projects have different needs:
- A React project needs Component, Hook, Context nodes
- A Python ML project needs Model, Dataset, Pipeline nodes
- A microservices project needs Service, API, Message nodes
- An infrastructure project needs Resource, Config, Deployment nodes

Each project should be able to define its own:
1. **Node types** with custom properties
2. **Relationship types** with semantics
3. **Vector collection schemas** 
4. **Indexing rules** for what to extract
5. **Query patterns** optimized for their domain

## Decision

Implement a **Project Schema Configuration System** that allows each project to define custom GraphRAG schemas through configuration files and auto-detection.

### Architecture Overview

```
Project Root/
├── .mcp.json                    # MCP configuration
├── .graphrag/                   # GraphRAG configuration directory
│   ├── schema.yaml              # Main schema definition
│   ├── node_types.yaml          # Custom node type definitions
│   ├── relationships.yaml       # Custom relationship definitions
│   ├── collections.yaml         # Vector collection configurations
│   ├── extractors/              # Custom extraction rules
│   │   ├── react.yaml          # React-specific extractors
│   │   ├── python.yaml         # Python-specific extractors
│   │   └── custom.py           # Custom Python extractors
│   └── queries/                 # Predefined query patterns
│       ├── impact_analysis.cypher
│       └── dependency_graph.cypher
```

## Implementation Design

### Phase 1: Schema Definition System

```yaml
# .graphrag/schema.yaml
version: "1.0"
project_type: "react-typescript"
description: "E-commerce frontend application"

# Import base schemas
extends:
  - "@graphrag/react"      # Built-in React schema
  - "@graphrag/typescript"  # Built-in TypeScript schema

# Custom node types
node_types:
  Component:
    properties:
      name: string
      type: enum[functional, class, hook]
      props: json
      state: json
      file_path: string
      line_number: int
    indexes:
      - name
      - type
      - file_path
    
  Hook:
    properties:
      name: string
      returns: string
      dependencies: list[string]
      custom: boolean
    
  Context:
    properties:
      name: string
      provider: string
      default_value: json
      
  Store:
    properties:
      name: string
      type: enum[redux, zustand, mobx]
      slices: json

# Custom relationships
relationships:
  USES_HOOK:
    from: [Component, Hook]
    to: Hook
    properties:
      line_number: int
      
  CONSUMES_CONTEXT:
    from: Component
    to: Context
    
  DISPATCHES_ACTION:
    from: Component
    to: Store
    properties:
      action_type: string
      
  RENDERS:
    from: Component
    to: Component
    properties:
      conditional: boolean
      in_loop: boolean

# Vector collections with custom fields
collections:
  components:
    vector_size: 768
    distance: cosine
    fields:
      - component_name
      - component_type
      - props_schema
      - hooks_used
      - render_logic
      
  hooks:
    vector_size: 768
    fields:
      - hook_name
      - dependencies
      - return_type
      - usage_examples
```

### Phase 2: Auto-Detection System

```python
# SchemaDetector class
class ProjectSchemaDetector:
    """Auto-detect project type and suggest schema"""
    
    def detect_project_type(self, project_path: str) -> ProjectType:
        """Detect project type from files and dependencies"""
        indicators = {
            'react': ['package.json:react', '*.jsx', '*.tsx'],
            'django': ['manage.py', 'settings.py', 'models.py'],
            'fastapi': ['main.py:FastAPI', 'requirements.txt:fastapi'],
            'vue': ['package.json:vue', '*.vue'],
            'angular': ['angular.json', '*.component.ts'],
            'flask': ['app.py:Flask', 'requirements.txt:flask'],
            'springboot': ['pom.xml:spring-boot', 'Application.java'],
            'rails': ['Gemfile:rails', 'config/routes.rb'],
            'nextjs': ['next.config.js', 'pages/', 'app/']
        }
        
        detected_types = []
        for project_type, patterns in indicators.items():
            if self._matches_patterns(project_path, patterns):
                detected_types.append(project_type)
        
        return self._resolve_project_type(detected_types)
    
    def suggest_schema(self, project_type: ProjectType) -> SchemaConfig:
        """Suggest optimal schema for project type"""
        schema_templates = {
            'react': ReactSchemaTemplate(),
            'django': DjangoSchemaTemplate(),
            'fastapi': FastAPISchemaTemplate(),
            'microservices': MicroservicesSchemaTemplate()
        }
        
        template = schema_templates.get(project_type, GenericSchemaTemplate())
        return template.generate()
    
    def extract_custom_patterns(self, project_path: str) -> List[Pattern]:
        """Extract custom patterns from codebase"""
        patterns = []
        
        # Analyze imports
        patterns.extend(self._analyze_import_patterns(project_path))
        
        # Analyze class hierarchies
        patterns.extend(self._analyze_class_patterns(project_path))
        
        # Analyze function signatures
        patterns.extend(self._analyze_function_patterns(project_path))
        
        # Analyze data flows
        patterns.extend(self._analyze_data_flow_patterns(project_path))
        
        return patterns
```

### Phase 3: Schema Management Interface

```python
class SchemaManager:
    """Manage per-project GraphRAG schemas"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.schema_path = f".graphrag/schema.yaml"
        self.current_schema = None
        
    async def initialize_schema(self):
        """Initialize or load project schema"""
        if os.path.exists(self.schema_path):
            self.current_schema = await self.load_schema()
        else:
            # Auto-detect and create schema
            project_type = ProjectSchemaDetector().detect_project_type(".")
            self.current_schema = await self.create_schema(project_type)
    
    async def create_schema(self, project_type: str = None) -> Schema:
        """Create new schema for project"""
        if project_type:
            # Use template
            template = get_schema_template(project_type)
            schema = template.generate()
        else:
            # Interactive creation
            schema = await self.interactive_schema_builder()
        
        # Apply to Neo4j
        await self.apply_neo4j_constraints(schema)
        
        # Configure Qdrant collections
        await self.setup_qdrant_collections(schema)
        
        return schema
    
    async def apply_neo4j_constraints(self, schema: Schema):
        """Apply schema constraints to Neo4j"""
        cypher_commands = []
        
        # Create node constraints
        for node_type in schema.node_types:
            cypher_commands.append(
                f"CREATE CONSTRAINT {node_type.name}_unique "
                f"IF NOT EXISTS FOR (n:{node_type.name}) "
                f"REQUIRE n.id IS UNIQUE"
            )
            
            # Create indexes
            for index_field in node_type.indexes:
                cypher_commands.append(
                    f"CREATE INDEX {node_type.name}_{index_field}_idx "
                    f"IF NOT EXISTS FOR (n:{node_type.name}) "
                    f"ON (n.{index_field})"
                )
        
        # Execute in Neo4j
        for cmd in cypher_commands:
            await self.neo4j.execute(cmd)
    
    async def migrate_schema(self, new_schema: Schema):
        """Migrate from current schema to new schema"""
        migration = SchemaMigration(self.current_schema, new_schema)
        
        # Generate migration plan
        plan = migration.generate_plan()
        
        # Execute migration
        await migration.execute(plan)
        
        self.current_schema = new_schema
```

### Phase 4: Custom Extractors

```python
# .graphrag/extractors/react_hooks.py
class ReactHooksExtractor(BaseExtractor):
    """Extract React hooks and their relationships"""
    
    def extract(self, ast_node, file_path):
        hooks = []
        
        if self.is_hook_definition(ast_node):
            hook = {
                'type': 'Hook',
                'name': ast_node.name,
                'file': file_path,
                'dependencies': self.extract_dependencies(ast_node),
                'returns': self.extract_return_type(ast_node)
            }
            hooks.append(hook)
            
        if self.is_hook_usage(ast_node):
            usage = {
                'type': 'USES_HOOK',
                'from': self.get_component_name(ast_node),
                'to': ast_node.call.name,
                'line': ast_node.line
            }
            hooks.append(usage)
            
        return hooks
```

### Phase 5: Query Pattern Library

```cypher
-- .graphrag/queries/component_impact.cypher
-- Find all components affected by changing a hook
MATCH (h:Hook {name: $hookName})
MATCH (c:Component)-[:USES_HOOK]->(h)
MATCH (c)-[:RENDERS*0..3]->(affected:Component)
RETURN DISTINCT affected.name as component, 
       affected.file_path as file,
       COUNT(DISTINCT c) as impact_paths
ORDER BY impact_paths DESC

-- .graphrag/queries/circular_dependencies.cypher
-- Detect circular dependencies in components
MATCH (c1:Component)-[:IMPORTS]->(c2:Component)
MATCH path = (c2)-[:IMPORTS*1..]->(c1)
RETURN c1.name as component1, 
       c2.name as component2,
       [n in nodes(path) | n.name] as cycle
```

## Configuration in .mcp.json

```json
{
  "mcpServers": {
    "neural-tools": {
      "env": {
        "PROJECT_NAME": "my-react-app",
        "GRAPHRAG_SCHEMA_PATH": ".graphrag/schema.yaml",
        "ENABLE_SCHEMA_AUTO_DETECTION": "true",
        "SCHEMA_TYPE": "react-typescript",
        "CUSTOM_EXTRACTORS": ".graphrag/extractors",
        "ENABLE_SCHEMA_VALIDATION": "true"
      }
    }
  }
}
```

## Tool Integration

### New MCP Tools

1. **schema_init**: Initialize project schema
   ```json
   {
     "project_type": "react",
     "auto_detect": true,
     "template": "@graphrag/react"
   }
   ```

2. **schema_update**: Update existing schema
   ```json
   {
     "add_node_type": {
       "name": "APIEndpoint",
       "properties": {...}
     }
   }
   ```

3. **schema_validate**: Validate data against schema
   ```json
   {
     "validate_nodes": true,
     "validate_relationships": true,
     "fix_issues": false
   }
   ```

4. **schema_query**: Run schema-aware queries
   ```json
   {
     "pattern": "component_impact",
     "params": {"component": "UserProfile"}
   }
   ```

## Migration Strategy

### Phase 1: Detection & Templates (Week 1)
- Implement project type detection
- Create schema templates for common frameworks
- Add schema loading from YAML/JSON

### Phase 2: Custom Definitions (Week 2)
- Support custom node types
- Support custom relationships
- Implement schema validation

### Phase 3: Extractors (Week 3)
- Build extractor framework
- Create language-specific extractors
- Support custom Python extractors

### Phase 4: Query Patterns (Week 4)
- Build query pattern library
- Create domain-specific queries
- Add query optimization

## Benefits

### For Developers
- **Domain-specific schemas** match their mental model
- **Better search results** with project-specific context
- **Custom relationships** capture their architecture
- **Reusable patterns** across similar projects

### For Teams
- **Shared understanding** through explicit schemas
- **Onboarding tool** for new developers
- **Architecture documentation** that's always current
- **Evolution tracking** as schema changes

### For Analysis
- **Precise queries** with typed relationships
- **Better impact analysis** with custom relationships
- **Domain insights** from specialized extractors
- **Performance optimization** with targeted indexes

## Consequences

### Positive
- Projects get schemas that match their domain
- Better code intelligence and search
- Extensible and customizable system
- Schema evolution is tracked
- Can share schemas across similar projects

### Negative
- Initial setup complexity for new projects
- Schema maintenance overhead
- Need to handle schema migrations
- More complex than one-size-fits-all

### Neutral
- Requires learning schema definition language
- Projects can still use default schema
- Schema changes need coordination

## Example Schemas

### React + Redux Schema
```yaml
extends: ["@graphrag/react", "@graphrag/redux"]
node_types:
  Component: {...}
  Action: {type, payload}
  Reducer: {slice, initialState}
  Selector: {input, output}
relationships:
  DISPATCHES: {from: Component, to: Action}
  HANDLES: {from: Reducer, to: Action}
  SELECTS: {from: Component, to: Selector}
```

### FastAPI Microservice Schema
```yaml
extends: "@graphrag/fastapi"
node_types:
  Endpoint: {path, method, response_model}
  Schema: {fields, validators}
  Dependency: {scope, provider}
  BackgroundTask: {queue, schedule}
relationships:
  VALIDATES: {from: Endpoint, to: Schema}
  DEPENDS_ON: {from: Endpoint, to: Dependency}
  TRIGGERS: {from: Endpoint, to: BackgroundTask}
```

### Django Schema
```yaml
extends: "@graphrag/django"
node_types:
  Model: {table, fields, meta}
  View: {url_pattern, template}
  Serializer: {model, fields}
  Signal: {sender, receiver}
relationships:
  QUERIES: {from: View, to: Model}
  SERIALIZES: {from: Serializer, to: Model}
  LISTENS: {from: Signal, to: Model}
```

## Testing Strategy

```python
async def test_custom_schema():
    # Create test project
    project = TestProject("react-app")
    
    # Initialize schema
    schema = await SchemaManager(project).initialize_schema()
    
    # Verify node types
    assert "Component" in schema.node_types
    assert "Hook" in schema.node_types
    
    # Test extraction
    extractor = ReactExtractor(schema)
    nodes = await extractor.extract("App.tsx")
    assert nodes[0].type == "Component"
    
    # Test relationships
    graph = await build_graph(project, schema)
    assert graph.has_relationship("App", "USES_HOOK", "useState")
```

## References

- GraphRAG Paper: "From Local to Global: A Graph RAG Approach"
- Neo4j GraphQL Schema Definition
- TypeScript Compiler API for AST parsing
- React DevTools Component Tree Structure

## Decision Outcome

Implement per-project custom GraphRAG schemas with auto-detection, templates, and extensible extractors. Start with React/TypeScript as the pilot implementation.

**Target: 4 weeks for full implementation, 1 week for React template**
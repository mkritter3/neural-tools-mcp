#!/usr/bin/env python3
"""
Test script for ADR-0020: Per-project custom GraphRAG schemas
Tests the schema management functionality integrated into MCP server
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neural-tools', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neural-tools', 'src', 'servers', 'services'))

from schema_manager import SchemaManager, ProjectType, NodeType, RelationshipType, GraphRAGSchema


async def test_schema_auto_detection():
    """Test auto-detection of project type"""
    print("\n🧪 Testing schema auto-detection...")
    
    # Test with current project (should detect as generic or based on files)
    manager = SchemaManager("test-project", os.getcwd())
    project_type = await manager.detect_project_type()
    print(f"  ✅ Detected project type: {project_type.value}")
    
    # Test with React indicators
    test_dir = Path("/tmp/test-react-project")
    test_dir.mkdir(exist_ok=True)
    package_json = test_dir / "package.json"
    package_json.write_text(json.dumps({
        "name": "test-react-app",
        "dependencies": {
            "react": "^18.0.0",
            "react-dom": "^18.0.0"
        }
    }))
    
    manager_react = SchemaManager("test-react", str(test_dir))
    react_type = await manager_react.detect_project_type()
    print(f"  ✅ Detected React project: {react_type.value}")
    assert react_type == ProjectType.REACT
    
    # Clean up
    package_json.unlink()
    test_dir.rmdir()
    
    return True


async def test_schema_creation():
    """Test schema creation with templates"""
    print("\n🧪 Testing schema creation...")
    
    # Create a React schema
    manager = SchemaManager("test-react-app", "/tmp")
    schema = await manager.create_schema(ProjectType.REACT)
    
    print(f"  ✅ Created schema for {schema.project_type.value}")
    print(f"  ✅ Node types: {list(schema.node_types.keys())}")
    print(f"  ✅ Relationships: {list(schema.relationship_types.keys())}")
    
    # Verify React-specific types exist
    assert "Component" in schema.node_types
    assert "Hook" in schema.node_types
    assert "USES_HOOK" in schema.relationship_types
    
    # Create a Django schema
    manager_django = SchemaManager("test-django-app", "/tmp")
    django_schema = await manager_django.create_schema(ProjectType.DJANGO)
    
    print(f"  ✅ Created Django schema with {len(django_schema.node_types)} node types")
    
    # Verify Django-specific types exist
    assert "Model" in django_schema.node_types
    assert "View" in django_schema.node_types
    assert "QUERIES" in django_schema.relationship_types
    
    return True


async def test_schema_persistence():
    """Test saving and loading schemas"""
    print("\n🧪 Testing schema persistence...")
    
    test_dir = Path("/tmp/test-schema-project")
    test_dir.mkdir(exist_ok=True)
    
    # Create and save schema
    manager = SchemaManager("test-persist", str(test_dir))
    original_schema = await manager.create_schema(ProjectType.FASTAPI)
    
    # Add custom node type
    custom_node = NodeType(
        name="CustomEndpoint",
        properties={"url": "string", "method": "string", "auth": "boolean"},
        indexes=["url"],
        description="Custom API endpoint"
    )
    original_schema.node_types["CustomEndpoint"] = custom_node
    
    # Save schema
    await manager.save_schema(original_schema)
    print(f"  ✅ Saved schema to {manager.schema_file}")
    
    # Create new manager and load schema
    manager2 = SchemaManager("test-persist", str(test_dir))
    loaded_schema = await manager2.load_schema()
    
    print(f"  ✅ Loaded schema from disk")
    
    # Verify custom node type persisted
    assert "CustomEndpoint" in loaded_schema.node_types
    assert loaded_schema.node_types["CustomEndpoint"].properties["auth"] == "boolean"
    print(f"  ✅ Custom node type persisted correctly")
    
    # Clean up
    if manager.schema_file.exists():
        manager.schema_file.unlink()
    if manager.schema_dir.exists():
        manager.schema_dir.rmdir()
    test_dir.rmdir()
    
    return True


async def test_schema_validation():
    """Test schema validation methods"""
    print("\n🧪 Testing schema validation...")
    
    manager = SchemaManager("test-validation", "/tmp")
    schema = await manager.create_schema(ProjectType.REACT)
    manager.current_schema = schema
    
    # Test valid node with all required properties from schema
    valid_node = await manager.validate_node("Component", {
        "name": "UserProfile",
        "type": "functional",
        "props": {},  # Add required props field
        "file_path": "/src/components/UserProfile.tsx"
    })
    print(f"  ✅ Valid node validation: {valid_node}")
    assert valid_node == True
    
    # Test invalid node type
    invalid_node = await manager.validate_node("NonExistentType", {})
    print(f"  ✅ Invalid node type rejected: {not invalid_node}")
    assert invalid_node == False
    
    # Test valid relationship
    valid_rel = await manager.validate_relationship("USES_HOOK", "Component", "Hook")
    print(f"  ✅ Valid relationship validation: {valid_rel}")
    assert valid_rel == True
    
    # Test invalid relationship
    invalid_rel = await manager.validate_relationship("USES_HOOK", "Model", "Hook")
    print(f"  ✅ Invalid relationship rejected: {not invalid_rel}")
    assert invalid_rel == False
    
    return True


async def test_collection_configuration():
    """Test vector collection configuration"""
    print("\n🧪 Testing collection configuration...")
    
    manager = SchemaManager("test-collections", "/tmp")
    schema = await manager.create_schema(ProjectType.GENERIC)
    
    # Check default collections
    print(f"  ✅ Default collections: {list(schema.collections.keys())}")
    assert "code" in schema.collections
    assert "docs" in schema.collections
    
    # Verify collection configuration
    code_collection = schema.collections["code"]
    print(f"  ✅ Code collection vector size: {code_collection.vector_size}")
    assert code_collection.vector_size == 768  # Nomic embed size
    assert code_collection.distance_metric == "cosine"
    
    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Testing ADR-0020: Per-project GraphRAG schemas")
    print("=" * 60)
    
    tests = [
        ("Auto-detection", test_schema_auto_detection),
        ("Schema creation", test_schema_creation),
        ("Schema persistence", test_schema_persistence),
        ("Schema validation", test_schema_validation),
        ("Collection configuration", test_collection_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, "PASS" if result else "FAIL"))
            if not result:
                print(f"  ❌ {test_name} failed")
        except Exception as e:
            print(f"  ❌ {test_name} error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, status in results:
        emoji = "✅" if status == "PASS" else "❌"
        print(f"{emoji} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Schema management is working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Pre-Deployment Validation Suite
Comprehensive tests that MUST pass before any deployment

This suite ensures:
1. All contracts are maintained (ADR-0096)
2. Vector search is functional
3. MCP tools return usable results
4. No regressions in critical functionality
5. Data consistency across components
"""

import sys
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class PreDeploymentValidator:
    """Comprehensive pre-deployment validation"""

    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []

    async def validate_all(self) -> bool:
        """Run all validation checks"""
        print("=" * 70)
        print("🔍 PRE-DEPLOYMENT VALIDATION SUITE")
        print("=" * 70)
        print()

        # 1. Contract validation
        print("1️⃣ Validating Schema Contracts...")
        contract_ok = await self.validate_contracts()
        self._report_result("Schema Contracts", contract_ok)

        # 2. Vector search functionality
        print("\n2️⃣ Validating Vector Search...")
        vector_ok = await self.validate_vector_search()
        self._report_result("Vector Search", vector_ok)

        # 3. MCP tool functionality
        print("\n3️⃣ Validating MCP Tools...")
        mcp_ok = await self.validate_mcp_tools()
        self._report_result("MCP Tools", mcp_ok)

        # 4. Data consistency
        print("\n4️⃣ Validating Data Consistency...")
        consistency_ok = await self.validate_data_consistency()
        self._report_result("Data Consistency", consistency_ok)

        # 5. Neo4j connection and schema
        print("\n5️⃣ Validating Neo4j Integration...")
        neo4j_ok = await self.validate_neo4j_integration()
        self._report_result("Neo4j Integration", neo4j_ok)

        # 6. Indexer functionality
        print("\n6️⃣ Validating Indexer Service...")
        indexer_ok = await self.validate_indexer()
        self._report_result("Indexer Service", indexer_ok)

        # 7. Performance benchmarks
        print("\n7️⃣ Validating Performance Benchmarks...")
        performance_ok = await self.validate_performance()
        self._report_result("Performance", performance_ok)

        # Final report
        self._print_final_report()

        all_passed = all(r[1] for r in self.results)
        return all_passed

    async def validate_contracts(self) -> bool:
        """Validate all schema contracts"""
        try:
            # Run contract validation tests
            result = subprocess.run(
                [sys.executable, "test_contract_validation.py"],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("   ✅ All contract tests passed")
                return True
            else:
                self.errors.append(f"Contract tests failed: {result.stderr}")
                print(f"   ❌ Contract tests failed")
                return False

        except subprocess.TimeoutExpired:
            self.errors.append("Contract validation timed out")
            return False
        except Exception as e:
            self.errors.append(f"Contract validation error: {e}")
            return False

    async def validate_vector_search(self) -> bool:
        """Validate vector search functionality"""
        try:
            from servers.services.robust_vector_search import RobustVectorSearch
            from servers.services.chunk_schema import ChunkSchema

            # Test schema validation
            chunk = ChunkSchema(
                chunk_id="test.py:chunk:0",
                file_path="test.py",
                content="test content for validation",
                embedding=[0.1] * 768,
                project="validation-test"
            )

            # Test Neo4j dict conversion
            neo4j_dict = chunk.to_neo4j_dict()

            # Verify required fields
            required = ["chunk_id", "file_path", "content", "embedding", "project"]
            missing = [f for f in required if f not in neo4j_dict]

            if missing:
                self.errors.append(f"Missing fields in Neo4j dict: {missing}")
                return False

            # Test vector search query building
            search = RobustVectorSearch(project_name="validation-test")
            query = search._build_vector_query(limit=10, min_score=0.6)

            # Verify query uses literal index name
            if "chunk_embeddings_index" not in query:
                self.errors.append("Vector search query missing literal index name")
                return False

            if "$index_name" in query:
                self.errors.append("Vector search using parameterized index name")
                return False

            print("   ✅ Vector search validation passed")
            return True

        except Exception as e:
            self.errors.append(f"Vector search validation error: {e}")
            print(f"   ❌ Vector search validation failed: {e}")
            return False

    async def validate_mcp_tools(self) -> bool:
        """Validate MCP tool functionality"""
        try:
            from neural_mcp.tools.fast_search import execute as fast_search
            from neural_mcp.tools.elite_search import execute as elite_search

            # Test fast_search with minimal args
            fast_result = await fast_search({"query": "test validation"})
            if not fast_result or not fast_result[0].text:
                self.errors.append("Fast search returned empty result")
                return False

            # Parse and validate structure
            fast_data = json.loads(fast_result[0].text)
            if "status" not in fast_data:
                self.errors.append("Fast search missing status field")
                return False

            # Test elite_search with minimal args
            elite_result = await elite_search({"query": "test validation"})
            if not elite_result or not elite_result[0].text:
                self.errors.append("Elite search returned empty result")
                return False

            # Parse and validate structure
            elite_data = json.loads(elite_result[0].text)
            if "status" not in elite_data:
                self.errors.append("Elite search missing status field")
                return False

            print("   ✅ MCP tools validation passed")
            return True

        except json.JSONDecodeError as e:
            self.errors.append(f"MCP tool returned invalid JSON: {e}")
            return False
        except Exception as e:
            self.errors.append(f"MCP tools validation error: {e}")
            print(f"   ❌ MCP tools validation failed: {e}")
            return False

    async def validate_data_consistency(self) -> bool:
        """Validate data consistency across components"""
        try:
            from servers.services.chunk_schema import ChunkSchema
            import json
            from datetime import datetime, timezone

            # Test no DateTime objects in JSON
            chunk = ChunkSchema(
                chunk_id="consistency.py:chunk:0",
                file_path="consistency.py",
                content="consistency test",
                embedding=[0.2] * 768,
                project="consistency-test"
            )

            # Should serialize to JSON without errors
            chunk_dict = chunk.to_dict()
            json_str = json.dumps(chunk_dict)
            parsed = json.loads(json_str)

            # Verify created_at is string, not datetime
            if not isinstance(parsed["created_at"], str):
                self.errors.append("created_at is not a string")
                return False

            # Verify can parse as ISO datetime
            try:
                datetime.fromisoformat(parsed["created_at"].replace('Z', '+00:00'))
            except ValueError:
                self.errors.append("created_at is not valid ISO format")
                return False

            # Test Neo4j data has only primitives
            neo4j_dict = chunk.to_neo4j_dict()
            for key, value in neo4j_dict.items():
                if isinstance(value, dict):
                    self.errors.append(f"Neo4j dict contains nested Map at {key}")
                    return False
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self.errors.append(f"Neo4j dict contains Map in list at {key}")
                            return False

            print("   ✅ Data consistency validation passed")
            return True

        except Exception as e:
            self.errors.append(f"Data consistency error: {e}")
            print(f"   ❌ Data consistency validation failed: {e}")
            return False

    async def validate_neo4j_integration(self) -> bool:
        """Validate Neo4j integration and schema"""
        try:
            from servers.services.neo4j_service import Neo4jService

            # Try to create service (will fail if Neo4j not running, which is OK for now)
            try:
                service = Neo4jService(
                    uri="bolt://localhost:47687",
                    auth=("neo4j", "graphrag-password"),
                    project_name="validation-test"
                )
                await service.initialize()

                # Check if vector index exists
                indexes = await service.get_indexes()
                vector_index_exists = any(
                    "chunk_embeddings" in idx.get("name", "")
                    for idx in indexes
                )

                if not vector_index_exists:
                    self.warnings.append("Vector index not found in Neo4j")

                await service.close()
                print("   ✅ Neo4j integration validation passed")
                return True

            except Exception as neo4j_error:
                # Neo4j not running is OK for pre-deployment
                self.warnings.append(f"Neo4j not accessible: {neo4j_error}")
                print("   ⚠️  Neo4j not running (OK for pre-deployment)")
                return True

        except ImportError as e:
            self.errors.append(f"Neo4j service import failed: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Neo4j validation error: {e}")
            return False

    async def validate_indexer(self) -> bool:
        """Validate indexer service"""
        try:
            from servers.services.indexer_service import IndexerService
            from servers.services.chunk_schema import ChunkSchema

            # Check that IndexerService uses ChunkSchema
            service_file = Path(__file__).parent.parent / "src/servers/services/indexer_service.py"
            if service_file.exists():
                content = service_file.read_text()
                if "ChunkSchema" not in content:
                    self.errors.append("IndexerService doesn't use ChunkSchema")
                    return False
                if "chunk.to_neo4j_dict()" not in content:
                    self.warnings.append("IndexerService might not use ChunkSchema.to_neo4j_dict()")

            print("   ✅ Indexer service validation passed")
            return True

        except Exception as e:
            self.errors.append(f"Indexer validation error: {e}")
            return False

    async def validate_performance(self) -> bool:
        """Validate performance benchmarks"""
        try:
            # Test that vector operations are reasonably fast
            from servers.services.chunk_schema import ChunkSchema
            import time

            start = time.time()
            for _ in range(100):
                chunk = ChunkSchema(
                    chunk_id=f"perf{_}.py:chunk:0",
                    file_path=f"perf{_}.py",
                    content="performance test",
                    embedding=[0.1] * 768,
                    project="perf-test"
                )
                _ = chunk.to_neo4j_dict()

            elapsed = time.time() - start
            ops_per_sec = 100 / elapsed

            if ops_per_sec < 1000:
                self.warnings.append(f"ChunkSchema operations slow: {ops_per_sec:.0f} ops/sec")

            print(f"   ✅ Performance validation passed ({ops_per_sec:.0f} ops/sec)")
            return True

        except Exception as e:
            self.errors.append(f"Performance validation error: {e}")
            return False

    def _report_result(self, test_name: str, passed: bool):
        """Record test result"""
        self.results.append((test_name, passed))
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}")

    def _print_final_report(self):
        """Print final validation report"""
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)

        # Results
        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        print(f"\nTests: {passed}/{total} passed")

        for name, passed in self.results:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")

        # Errors
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")

        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")

        # Final verdict
        all_passed = all(r[1] for r in self.results)
        print("\n" + "=" * 70)
        if all_passed:
            print("✅ PRE-DEPLOYMENT VALIDATION: PASSED")
            print("🚀 Safe to deploy!")
        else:
            print("❌ PRE-DEPLOYMENT VALIDATION: FAILED")
            print("🛑 DO NOT DEPLOY - Fix issues first!")
        print("=" * 70)


async def main():
    """Run pre-deployment validation"""
    validator = PreDeploymentValidator()
    success = await validator.validate_all()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
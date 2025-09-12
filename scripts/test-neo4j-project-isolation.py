#!/usr/bin/env python3
"""
Test script to verify ADR-0029 Neo4j multi-project isolation implementation.

This script verifies that:
1. File nodes have project properties
2. Queries are properly isolated by project
3. No data leakage between projects
4. Composite constraints are working
"""

import asyncio
import sys
from neo4j import AsyncGraphDatabase
from typing import Dict, List, Any
import json

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:47687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag-password"

class Neo4jIsolationTester:
    def __init__(self):
        self.driver = None
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "details": []
        }

    async def connect(self):
        """Connect to Neo4j database."""
        print(f"\n🔌 Connecting to Neo4j at {NEO4J_URI}...")
        self.driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        # Verify connection
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            await result.single()
        print("✅ Connected to Neo4j successfully")

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()

    async def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            if result["passed"]:
                self.results["tests_passed"] += 1
                print(f"  ✅ PASS: {result['message']}")
            else:
                self.results["tests_failed"] += 1
                print(f"  ❌ FAIL: {result['message']}")
            self.results["details"].append({
                "test": test_name,
                "passed": result["passed"],
                "message": result["message"],
                "data": result.get("data", {})
            })
        except Exception as e:
            self.results["tests_failed"] += 1
            print(f"  ❌ ERROR: {str(e)}")
            self.results["details"].append({
                "test": test_name,
                "passed": False,
                "message": f"Error: {str(e)}",
                "data": {}
            })

    async def test_project_properties_exist(self) -> Dict:
        """Test 1: Verify File nodes have project properties."""
        async with self.driver.session() as session:
            # Check for File nodes with project property
            result = await session.run("""
                MATCH (f:File)
                WHERE f.project IS NOT NULL
                RETURN f.project AS project, COUNT(f) AS count
                ORDER BY project
            """)
            records = [record async for record in result]
            
            if not records:
                return {
                    "passed": False,
                    "message": "No File nodes with project property found",
                    "data": {}
                }
            
            projects = {r["project"]: r["count"] for r in records}
            return {
                "passed": True,
                "message": f"Found {len(projects)} project(s) with File nodes: {projects}",
                "data": projects
            }

    async def test_project_isolation_query(self) -> Dict:
        """Test 2: Verify queries filter by project correctly."""
        async with self.driver.session() as session:
            # Get all unique projects
            projects_result = await session.run("""
                MATCH (n)
                WHERE n.project IS NOT NULL
                RETURN DISTINCT n.project AS project
            """)
            projects = [r["project"] async for r in projects_result]
            
            if not projects:
                return {
                    "passed": False,
                    "message": "No nodes with project property found",
                    "data": {}
                }
            
            # For each project, verify we only get nodes from that project
            isolation_results = {}
            for project in projects:
                # Query with project filter
                filtered_result = await session.run("""
                    MATCH (n {project: $project})
                    RETURN COUNT(n) AS count
                """, project=project)
                filtered_count = (await filtered_result.single())["count"]
                
                # Query to check if any nodes from this project exist without filter
                # (This would catch nodes that don't have project property)
                check_result = await session.run("""
                    MATCH (n)
                    WHERE (n:File OR n:Class OR n:Method OR n:CodeChunk)
                    AND (n.project = $project OR n.project IS NULL)
                    RETURN COUNT(CASE WHEN n.project IS NULL THEN 1 END) AS null_count,
                           COUNT(CASE WHEN n.project = $project THEN 1 END) AS project_count
                """, project=project)
                check_record = await check_result.single()
                
                isolation_results[project] = {
                    "filtered_count": filtered_count,
                    "null_count": check_record["null_count"],
                    "project_count": check_record["project_count"]
                }
            
            # Check if any nodes have null project
            total_null = sum(r["null_count"] for r in isolation_results.values())
            
            if total_null > 0:
                return {
                    "passed": False,
                    "message": f"Found {total_null} nodes without project property",
                    "data": isolation_results
                }
            
            return {
                "passed": True,
                "message": f"All nodes properly isolated by project",
                "data": isolation_results
            }

    async def test_composite_constraints(self) -> Dict:
        """Test 3: Verify composite constraints are in place."""
        async with self.driver.session() as session:
            # Get all constraints
            result = await session.run("SHOW CONSTRAINTS")
            constraints = []
            async for record in result:
                constraints.append({
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties")
                })
            
            # Check for our expected composite constraints
            expected_constraints = [
                ("File", ["project", "path"]),
                ("Class", ["project", "name", "file_path"]),
                ("Method", ["project", "name", "class_name", "file_path"])
            ]
            
            found_constraints = {}
            for label, expected_props in expected_constraints:
                # Find constraint for this label
                label_constraint = None
                for c in constraints:
                    if c.get("labelsOrTypes") and label in c["labelsOrTypes"]:
                        props = c.get("properties", [])
                        if all(p in props for p in expected_props):
                            label_constraint = c
                            break
                
                found_constraints[label] = {
                    "expected": expected_props,
                    "found": label_constraint is not None,
                    "constraint": label_constraint
                }
            
            all_found = all(v["found"] for v in found_constraints.values())
            
            return {
                "passed": all_found,
                "message": "All composite constraints found" if all_found else "Some constraints missing",
                "data": {
                    "constraints": found_constraints,
                    "total_constraints": len(constraints)
                }
            }

    async def test_cross_project_isolation(self) -> Dict:
        """Test 4: Verify no cross-project data leakage."""
        async with self.driver.session() as session:
            # Get all projects
            projects_result = await session.run("""
                MATCH (n)
                WHERE n.project IS NOT NULL
                RETURN DISTINCT n.project AS project
            """)
            projects = [r["project"] async for r in projects_result]
            
            if len(projects) < 2:
                return {
                    "passed": True,
                    "message": f"Only {len(projects)} project(s) found - need at least 2 for cross-project test",
                    "data": {"projects": projects}
                }
            
            # Check relationships between nodes of different projects
            leak_result = await session.run("""
                MATCH (n1)-[r]-(n2)
                WHERE n1.project IS NOT NULL 
                AND n2.project IS NOT NULL 
                AND n1.project <> n2.project
                RETURN n1.project AS project1, 
                       n2.project AS project2, 
                       type(r) AS relationship,
                       labels(n1) AS labels1,
                       labels(n2) AS labels2,
                       COUNT(*) AS count
            """)
            
            leaks = []
            async for record in leak_result:
                leaks.append({
                    "project1": record["project1"],
                    "project2": record["project2"],
                    "relationship": record["relationship"],
                    "labels1": record["labels1"],
                    "labels2": record["labels2"],
                    "count": record["count"]
                })
            
            if leaks:
                return {
                    "passed": False,
                    "message": f"Found {len(leaks)} cross-project relationships!",
                    "data": {"leaks": leaks}
                }
            
            return {
                "passed": True,
                "message": "No cross-project data leakage detected",
                "data": {"projects_tested": projects}
            }

    async def test_project_specific_queries(self) -> Dict:
        """Test 5: Verify project-specific queries work correctly."""
        async with self.driver.session() as session:
            # Test a typical query pattern with project filter
            test_project = "eventfully-yours"  # We know this exists
            
            # Query Files for specific project
            files_result = await session.run("""
                MATCH (f:File {project: $project})
                RETURN f.path AS path, f.name AS name
                LIMIT 5
            """, project=test_project)
            
            files = []
            async for record in files_result:
                files.append({
                    "path": record["path"],
                    "name": record["name"]
                })
            
            # Query with relationships
            rel_result = await session.run("""
                MATCH (f:File {project: $project})
                OPTIONAL MATCH (f)<-[:DEFINED_IN]-(func:Function)
                WHERE func.project = $project OR func.project IS NULL
                RETURN f.path AS file, COUNT(func) AS functions
                LIMIT 5
            """, project=test_project)
            
            file_functions = []
            async for record in rel_result:
                file_functions.append({
                    "file": record["file"],
                    "functions": record["functions"]
                })
            
            return {
                "passed": len(files) > 0,
                "message": f"Found {len(files)} files in {test_project}",
                "data": {
                    "project": test_project,
                    "sample_files": files,
                    "file_functions": file_functions
                }
            }

    async def run_all_tests(self):
        """Run all isolation tests."""
        print("\n" + "="*60)
        print("🔬 ADR-0029 Neo4j Multi-Project Isolation Test Suite")
        print("="*60)
        
        await self.connect()
        
        # Run all tests
        await self.run_test(
            "Project Properties Exist",
            self.test_project_properties_exist
        )
        
        await self.run_test(
            "Project Isolation Query",
            self.test_project_isolation_query
        )
        
        await self.run_test(
            "Composite Constraints",
            self.test_composite_constraints
        )
        
        await self.run_test(
            "Cross-Project Isolation",
            self.test_cross_project_isolation
        )
        
        await self.run_test(
            "Project-Specific Queries",
            self.test_project_specific_queries
        )
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        print(f"✅ Tests Passed: {self.results['tests_passed']}")
        print(f"❌ Tests Failed: {self.results['tests_failed']}")
        print(f"📈 Success Rate: {self.results['tests_passed']/(self.results['tests_passed']+self.results['tests_failed'])*100:.1f}%")
        
        # Print detailed results
        print("\n📝 Detailed Results:")
        print(json.dumps(self.results["details"], indent=2))
        
        await self.close()
        
        # Return exit code
        return 0 if self.results["tests_failed"] == 0 else 1

async def main():
    tester = Neo4jIsolationTester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())

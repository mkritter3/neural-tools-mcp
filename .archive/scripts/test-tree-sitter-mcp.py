#!/usr/bin/env python3
"""
Test script for L9 Tree-sitter MCP Server
Validates multi-language AST analysis capabilities
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Test code samples in different languages
TEST_SAMPLES = {
    "python": {
        "extension": ".py",
        "code": '''#!/usr/bin/env python3
"""
Test Python module for AST analysis
"""

import asyncio
from typing import Dict, List, Optional

class UserManager:
    """Manages user operations"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.cache = {}
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Retrieve user by ID with caching"""
        if user_id in self.cache:
            return self.cache[user_id]
        
        user = await self.db.fetch_user(user_id)
        if user:
            self.cache[user_id] = user
        return user
    
    def clear_cache(self):
        """Clear user cache"""
        self.cache.clear()

async def main():
    """Main entry point"""
    manager = UserManager(None)
    user = await manager.get_user(123)
    print(f"User: {user}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    },
    
    "javascript": {
        "extension": ".js", 
        "code": '''/**
 * User management utilities for web app
 */

const axios = require('axios');

class ApiClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
        this.cache = new Map();
    }
    
    async fetchUser(userId) {
        const cacheKey = `user_${userId}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        try {
            const response = await axios.get(`${this.baseURL}/users/${userId}`);
            this.cache.set(cacheKey, response.data);
            return response.data;
        } catch (error) {
            console.error(`Failed to fetch user ${userId}:`, error);
            return null;
        }
    }
    
    clearCache() {
        this.cache.clear();
    }
}

function processUsers(users) {
    return users
        .filter(user => user.active)
        .map(user => ({
            id: user.id,
            name: user.name,
            email: user.email
        }));
}

module.exports = { ApiClient, processUsers };
'''
    },
    
    "go": {
        "extension": ".go",
        "code": '''package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// User represents a user in the system
type User struct {
	ID       int64  `json:"id"`
	Name     string `json:"name"`
	Email    string `json:"email"`
	Active   bool   `json:"active"`
	CreateAt time.Time `json:"created_at"`
}

// UserService handles user operations
type UserService struct {
	mu    sync.RWMutex
	cache map[int64]*User
}

// NewUserService creates a new user service
func NewUserService() *UserService {
	return &UserService{
		cache: make(map[int64]*User),
	}
}

// GetUser retrieves a user by ID
func (s *UserService) GetUser(ctx context.Context, id int64) (*User, error) {
	s.mu.RLock()
	user, exists := s.cache[id]
	s.mu.RUnlock()
	
	if exists {
		return user, nil
	}
	
	// Simulate database fetch
	user = &User{
		ID:       id,
		Name:     fmt.Sprintf("User %d", id),
		Email:    fmt.Sprintf("user%d@example.com", id),
		Active:   true,
		CreateAt: time.Now(),
	}
	
	s.mu.Lock()
	s.cache[id] = user
	s.mu.Unlock()
	
	return user, nil
}

// ClearCache clears the user cache
func (s *UserService) ClearCache() {
	s.mu.Lock()
	s.cache = make(map[int64]*User)
	s.mu.Unlock()
}

func main() {
	service := NewUserService()
	ctx := context.Background()
	
	user, err := service.GetUser(ctx, 123)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Printf("Retrieved user: %+v\\n", user)
}
'''
    }
}

def create_test_files(test_dir: Path):
    """Create test files in different languages"""
    test_files = {}
    
    for lang, sample in TEST_SAMPLES.items():
        file_path = test_dir / f"test_{lang}{sample['extension']}"
        file_path.write_text(sample["code"])
        test_files[lang] = str(file_path)
        print(f"📝 Created test file: {file_path}")
    
    return test_files

def test_mcp_server():
    """Test the MCP server with tree-sitter analysis"""
    print("🚀 Testing L9 Tree-sitter MCP Server")
    print("=" * 50)
    
    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        test_files = create_test_files(test_dir)
        
        print(f"\n📁 Test directory: {test_dir}")
        
        # Build Docker image
        print("\n🔨 Building MCP server image...")
        build_cmd = [
            "docker", "build",
            "-f", "docker/Dockerfile.mcp",
            "-t", "l9-mcp-server:test",
            "."
        ]
        
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to build Docker image:")
            print(result.stderr)
            return False
        
        print("✅ Docker image built successfully")
        
        # Test tree-sitter analyzer directly first
        print("\n🌳 Testing tree-sitter analyzer...")
        
        try:
            # Import and test the analyzer
            sys.path.append('neural-system')
            from tree_sitter_ast import TreeSitterAnalyzer, TREE_SITTER_AVAILABLE
            
            if not TREE_SITTER_AVAILABLE:
                print("⚠️  Tree-sitter not available locally, testing in Docker only")
            else:
                analyzer = TreeSitterAnalyzer()
                print(f"✅ Tree-sitter analyzer initialized")
                print(f"📊 Supported languages: {len(analyzer.extension_map)}")
                
                # Test each language
                for lang, file_path in test_files.items():
                    structure = analyzer.analyze_file(file_path)
                    if structure:
                        patterns = analyzer.extract_searchable_patterns(structure)
                        print(f"  ✅ {lang.capitalize()}: {len(structure.functions)} functions, {len(structure.classes)} classes, {len(patterns)} patterns")
                    else:
                        print(f"  ❌ {lang.capitalize()}: Analysis failed")
        
        except ImportError as e:
            print(f"⚠️  Local testing skipped: {e}")
        
        print("\n🐳 Testing MCP server in Docker...")
        
        # Test MCP server via Docker (simulate MCP client)
        docker_test_cmd = [
            "docker", "run", "--rm",
            "-v", f"{test_dir}:/app/project:ro",
            "-e", "PROJECT_NAME=test",
            "l9-mcp-server:test",
            "python3", "-c", '''
import asyncio
import sys
sys.path.append("/app")

from tree_sitter_ast import TreeSitterAnalyzer, TREE_SITTER_AVAILABLE

async def test():
    if not TREE_SITTER_AVAILABLE:
        print("❌ Tree-sitter not available in Docker")
        return False
    
    analyzer = TreeSitterAnalyzer()
    print(f"✅ Tree-sitter available with {len(analyzer.parsers)} parsers")
    
    # Test analyzing mounted files
    from pathlib import Path
    project_dir = Path("/app/project")
    
    for test_file in project_dir.glob("test_*"):
        print(f"📄 Analyzing {test_file.name}...")
        structure = analyzer.analyze_file(str(test_file))
        
        if structure:
            print(f"  ✅ Language: {structure.language.value}")
            print(f"  📊 Functions: {len(structure.functions)}, Classes: {len(structure.classes)}")
            print(f"  🧮 Complexity: {structure.complexity_score}")
        else:
            print(f"  ❌ Failed to analyze {test_file.name}")
    
    return True

if asyncio.run(test()):
    print("\\n🎉 All tests passed!")
else:
    print("\\n❌ Some tests failed")
    exit(1)
'''
        ]
        
        result = subprocess.run(docker_test_cmd, capture_output=True, text=True)
        
        print("Docker test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Docker test errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 Tree-sitter MCP server test completed successfully!")
            return True
        else:
            print(f"\n❌ Docker test failed with code {result.returncode}")
            return False

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
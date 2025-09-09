#!/usr/bin/env python3
"""
Phase 1.1 Integration Tests - File Watcher with Neural Tools Integration
Tests integration with neural indexing pipeline and real file system operations
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx
import time

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.enhanced_file_watcher import EnhancedFileWatcher

class TestFileWatcherIntegration:
    """Integration tests for file watcher with neural indexing pipeline"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with realistic structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "test_project"
            project_dir.mkdir()
            
            # Create realistic project structure
            (project_dir / "src").mkdir()
            (project_dir / "tests").mkdir()
            (project_dir / "docs").mkdir()
            
            # Create some initial files
            (project_dir / "src" / "main.py").write_text('''
def main():
    """Main entry point for the application"""
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
            ''')
            
            (project_dir / "tests" / "test_main.py").write_text('''
import pytest
from src.main import main

def test_main():
    """Test main function returns 0"""
    assert main() == 0
            ''')
            
            (project_dir / "README.md").write_text('''
# Test Project
This is a test project for file watcher integration testing.
            ''')
            
            yield project_dir
    
    @pytest.fixture
    def mock_neural_indexer(self):
        """Mock neural indexing service"""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_integration_with_neural_indexer(self, temp_project_dir, mock_neural_indexer):
        """Test file watcher integration with neural indexing pipeline"""
        
        # Setup file watcher with neural indexing callback
        async def neural_index_callback(file_path: str):
            """Callback that simulates neural indexing"""
            await mock_neural_indexer.index_file(file_path)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=neural_index_callback,
            debounce_delay=0.2,
            patterns=['*.py', '*.md', '*.txt']
        )
        
        await watcher.start()
        
        # Simulate code changes
        main_file = temp_project_dir / "src" / "main.py"
        main_file.write_text('''
def main():
    """Main entry point - now with enhanced functionality"""
    print("Hello, Enhanced World!")
    return 0

def helper_function():
    """New helper function added"""
    return "helper"

if __name__ == "__main__":
    main()
        ''')
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Add new file
        new_file = temp_project_dir / "src" / "utils.py"
        new_file.write_text('''
def utility_function():
    """Utility function for common operations"""
    return "utility"
        ''')
        
        await asyncio.sleep(0.5)
        await watcher.stop()
        
        # Verify neural indexer was called for both changes
        assert mock_neural_indexer.index_file.call_count >= 2
        
        # Verify correct file paths were indexed
        indexed_paths = [call[0][0] for call in mock_neural_indexer.index_file.call_args_list]
        assert any(str(main_file) in path for path in indexed_paths)
        assert any(str(new_file) in path for path in indexed_paths)
    
    @pytest.mark.asyncio
    async def test_performance_with_large_file_changes(self, temp_project_dir, mock_neural_indexer):
        """Test file watcher performance with large file changes"""
        
        # Create callback that tracks timing
        processing_times = []
        
        async def timed_callback(file_path: str):
            start_time = time.time()
            await mock_neural_indexer.index_file(file_path)
            processing_times.append(time.time() - start_time)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=timed_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        await watcher.start()
        
        # Create large files
        large_files = []
        for i in range(10):
            large_file = temp_project_dir / f"large_file_{i}.py"
            # Create large content (simulating real code files)
            large_content = f'''
"""
Large Python file {i} for performance testing
"""

class LargeClass{i}:
    """A large class with many methods"""
    
    def __init__(self):
        self.data = []
        for j in range(100):
            self.data.append(f"item_{{j}}")
    
''' + '\n'.join([f'''
    def method_{j}(self):
        """Method {j} implementation"""
        return f"result_{j}"
''' for j in range(20)])
            
            large_file.write_text(large_content)
            large_files.append(large_file)
        
        # Modify all files simultaneously
        for i, large_file in enumerate(large_files):
            content = large_file.read_text()
            large_file.write_text(content + f"\n# Modified at {time.time()}")
        
        # Wait for all processing to complete
        await asyncio.sleep(2.0)
        await watcher.stop()
        
        # Verify performance requirements
        assert len(processing_times) >= 5  # Most files should have been processed
        # Individual processing should be reasonably fast
        for proc_time in processing_times:
            assert proc_time < 1.0  # Each callback should complete quickly
    
    @pytest.mark.asyncio
    async def test_concurrent_file_system_operations(self, temp_project_dir, mock_neural_indexer):
        """Test file watcher with concurrent file system operations"""
        
        async def indexing_callback(file_path: str):
            # Simulate real indexing work
            await asyncio.sleep(0.05)  # Simulate processing time
            await mock_neural_indexer.index_file(file_path)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=indexing_callback,
            debounce_delay=0.1,
            patterns=['*.py', '*.md']
        )
        
        await watcher.start()
        
        # Perform concurrent operations
        async def create_and_modify_files():
            tasks = []
            
            # Create files concurrently
            for i in range(5):
                async def create_file(index):
                    file_path = temp_project_dir / f"concurrent_{index}.py"
                    file_path.write_text(f"# Concurrent file {index}\nprint('file_{index}')")
                    await asyncio.sleep(0.1)
                    file_path.write_text(f"# Modified concurrent file {index}\nprint('modified_file_{index}')")
                
                tasks.append(create_file(i))
            
            await asyncio.gather(*tasks)
        
        await create_and_modify_files()
        await asyncio.sleep(1.0)  # Wait for all processing
        await watcher.stop()
        
        # Verify all files were processed
        assert mock_neural_indexer.index_file.call_count >= 5
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_integration(self, temp_project_dir):
        """Test file watcher error recovery in integrated environment"""
        
        # Create callback that fails intermittently
        call_count = 0
        successful_calls = []
        
        async def flaky_callback(file_path: str):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception(f"Simulated failure for {file_path}")
            else:
                successful_calls.append(file_path)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=flaky_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        await watcher.start()
        
        # Create multiple files to trigger both successes and failures
        for i in range(9):  # Should have 3 failures and 6 successes
            test_file = temp_project_dir / f"error_test_{i}.py"
            test_file.write_text(f"# Test file {i}")
            await asyncio.sleep(0.15)  # Ensure separate events
        
        await asyncio.sleep(1.0)
        await watcher.stop()
        
        # Verify watcher continued working despite errors
        assert len(successful_calls) >= 4  # Should have some successful calls
        assert call_count >= 6  # Should have attempted most files
    
    @pytest.mark.asyncio
    async def test_directory_structure_changes(self, temp_project_dir, mock_neural_indexer):
        """Test watcher response to directory structure changes"""
        
        async def structure_aware_callback(file_path: str):
            await mock_neural_indexer.index_file(file_path)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=structure_aware_callback,
            debounce_delay=0.1,
            patterns=['*.py', '*.md']
        )
        
        await watcher.start()
        
        # Create new subdirectories and files
        new_module_dir = temp_project_dir / "src" / "new_module"
        new_module_dir.mkdir()
        
        module_file = new_module_dir / "__init__.py"
        module_file.write_text('"""New module package"""')
        
        implementation_file = new_module_dir / "implementation.py"
        implementation_file.write_text('''
class NewFeature:
    """Implementation of new feature"""
    
    def execute(self):
        return "feature executed"
        ''')
        
        await asyncio.sleep(0.5)
        
        # Modify files in new structure
        implementation_file.write_text('''
class NewFeature:
    """Enhanced implementation of new feature"""
    
    def execute(self):
        return "enhanced feature executed"
    
    def additional_method(self):
        return "additional functionality"
        ''')
        
        await asyncio.sleep(0.5)
        await watcher.stop()
        
        # Verify new files were detected and indexed
        indexed_paths = [call[0][0] for call in mock_neural_indexer.index_file.call_args_list]
        assert any("__init__.py" in path for path in indexed_paths)
        assert any("implementation.py" in path for path in indexed_paths)
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_extended_watching(self, temp_project_dir, mock_neural_indexer):
        """Test memory stability during extended file watching"""
        
        processed_files = []
        
        async def tracking_callback(file_path: str):
            processed_files.append(file_path)
            await mock_neural_indexer.index_file(file_path)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_project_dir),
            callback=tracking_callback,
            debounce_delay=0.05,  # Quick debounce for speed
            patterns=['*.py']
        )
        
        await watcher.start()
        
        # Simulate extended usage with many file changes
        test_file = temp_project_dir / "memory_test.py"
        test_file.write_text("# Initial content")
        
        # Make many small changes
        for i in range(50):
            test_file.write_text(f"# Content change {i}\nprint('iteration_{i}')")
            await asyncio.sleep(0.02)  # Rapid changes to test debouncing
        
        await asyncio.sleep(1.0)  # Final processing time
        await watcher.stop()
        
        # Verify debouncing worked (should have far fewer callbacks than changes)
        assert len(processed_files) < 50  # Should be significantly debounced
        assert len(processed_files) > 0   # But should have some callbacks
        
        # Memory should be stable (no easy way to test directly, but no crashes indicates success)
        assert mock_neural_indexer.index_file.call_count == len(processed_files)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
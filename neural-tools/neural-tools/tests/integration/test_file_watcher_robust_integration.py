#!/usr/bin/env python3
"""
Phase 1.1 Robust Integration Tests - File Watcher with Real File System
Tests integration with actual file system operations and threading
"""

import pytest
import asyncio
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import AsyncMock
import os
import psutil

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.enhanced_file_watcher import EnhancedFileWatcher

class TestRobustFileWatcherIntegration:
    """Robust integration tests addressing threading and real-world scenarios"""
    
    @pytest.fixture
    def persistent_temp_dir(self):
        """Create a temporary directory that persists during the test"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_file_watcher_with_real_file_operations(self, persistent_temp_dir):
        """Test file watcher with actual file system operations"""
        callback_results = []
        
        async def tracking_callback(file_path: str):
            callback_results.append({
                'file_path': file_path,
                'timestamp': time.time(),
                'thread_id': threading.get_ident()
            })
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=tracking_callback,
            debounce_delay=0.2,
            patterns=['*.py', '*.txt']
        )
        
        try:
            await watcher.start()
            await asyncio.sleep(0.5)  # Let watcher fully initialize
            
            # Create a file and verify detection
            test_file = persistent_temp_dir / "real_test.py"
            test_file.write_text("print('initial')")
            
            await asyncio.sleep(0.5)  # Wait for file system event
            
            # Modify the file
            test_file.write_text("print('modified')")
            
            await asyncio.sleep(0.5)  # Wait for modification detection
            
            # Create another file with different extension
            text_file = persistent_temp_dir / "readme.txt"
            text_file.write_text("This is a readme file")
            
            await asyncio.sleep(0.5)  # Wait for creation detection
            
        finally:
            await watcher.stop()
        
        print(f"Callback results: {len(callback_results)} events detected")
        for result in callback_results:
            print(f"  - {result['file_path']} at {result['timestamp']}")
        
        # Should have detected at least some file operations
        assert len(callback_results) >= 1, "Should detect at least one file operation"
        
        # Verify correct files were detected
        detected_files = [r['file_path'] for r in callback_results]
        assert any('real_test.py' in path for path in detected_files), "Should detect Python file"
    
    @pytest.mark.asyncio
    async def test_threading_safety_with_concurrent_callbacks(self, persistent_temp_dir):
        """Test that callbacks handle threading correctly"""
        callback_threads = set()
        callback_count = 0
        callback_lock = asyncio.Lock()
        
        async def thread_tracking_callback(file_path: str):
            nonlocal callback_count
            async with callback_lock:
                callback_count += 1
                callback_threads.add(threading.get_ident())
                # Simulate some work
                await asyncio.sleep(0.01)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=thread_tracking_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        try:
            await watcher.start()
            await asyncio.sleep(0.3)  # Let watcher initialize
            
            # Create multiple files concurrently
            files_to_create = []
            for i in range(5):
                file_path = persistent_temp_dir / f"concurrent_{i}.py"
                files_to_create.append(file_path)
            
            # Create files in rapid succession
            for file_path in files_to_create:
                file_path.write_text(f"# File {file_path.name}")
                await asyncio.sleep(0.05)  # Small delay between creations
            
            # Wait for all callbacks to complete
            await asyncio.sleep(1.0)
            
        finally:
            await watcher.stop()
        
        print(f"Threading test results:")
        print(f"  Total callbacks: {callback_count}")
        print(f"  Thread IDs used: {callback_threads}")
        
        # Should have processed some callbacks
        assert callback_count > 0, "Should have processed some callbacks"
        
        # Threading should be handled correctly (no crashes)
        assert len(callback_threads) >= 1, "Should have executed callbacks in at least one thread"
    
    @pytest.mark.asyncio
    async def test_performance_under_realistic_workload(self, persistent_temp_dir):
        """Test performance with realistic file operations"""
        callback_times = []
        
        async def performance_callback(file_path: str):
            start_time = time.time()
            # Simulate realistic indexing work
            await asyncio.sleep(0.002)  # 2ms simulated work
            end_time = time.time()
            callback_times.append(end_time - start_time)
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=performance_callback,
            debounce_delay=0.1,
            patterns=['*.py', '*.js', '*.md']
        )
        
        try:
            await watcher.start()
            await asyncio.sleep(0.3)  # Initialize
            
            # Create a realistic project structure
            (persistent_temp_dir / "src").mkdir()
            (persistent_temp_dir / "tests").mkdir()
            (persistent_temp_dir / "docs").mkdir()
            
            start_time = time.time()
            
            # Create files in different directories
            project_files = [
                "src/main.py",
                "src/utils.py", 
                "src/config.js",
                "tests/test_main.py",
                "tests/test_utils.py",
                "docs/README.md",
                "docs/API.md"
            ]
            
            for file_path in project_files:
                full_path = persistent_temp_dir / file_path
                full_path.write_text(f"# Content for {file_path}")
                await asyncio.sleep(0.05)  # Realistic creation timing
            
            # Modify some files
            for file_path in project_files[:3]:
                full_path = persistent_temp_dir / file_path
                content = full_path.read_text()
                full_path.write_text(content + "\n# Modified")
                await asyncio.sleep(0.05)
            
            total_time = time.time() - start_time
            
            # Wait for all processing
            await asyncio.sleep(1.0)
            
        finally:
            await watcher.stop()
        
        print(f"Performance test results:")
        print(f"  Total project setup time: {total_time:.2f}s")
        print(f"  Callbacks executed: {len(callback_times)}")
        if callback_times:
            avg_callback_time = sum(callback_times) / len(callback_times)
            max_callback_time = max(callback_times)
            print(f"  Average callback time: {avg_callback_time*1000:.1f}ms")
            print(f"  Max callback time: {max_callback_time*1000:.1f}ms")
            
            # Performance requirements
            assert avg_callback_time < 0.1, f"Average callback time {avg_callback_time*1000:.1f}ms too high"
            assert max_callback_time < 0.2, f"Max callback time {max_callback_time*1000:.1f}ms too high"
        
        # Should have processed most files
        assert len(callback_times) >= 5, f"Should process at least 5 files, got {len(callback_times)}"
    
    @pytest.mark.asyncio
    async def test_file_system_stress_test(self, persistent_temp_dir):
        """Stress test with many file operations"""
        processed_files = set()
        error_count = 0
        
        async def stress_callback(file_path: str):
            nonlocal error_count
            try:
                processed_files.add(file_path)
                # Small amount of work
                await asyncio.sleep(0.001)
            except Exception as e:
                error_count += 1
                print(f"Callback error: {e}")
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=stress_callback,
            debounce_delay=0.05,  # Shorter debounce for stress test
            patterns=['*.tmp']
        )
        
        try:
            await watcher.start()
            await asyncio.sleep(0.2)
            
            # Create many files rapidly
            start_time = time.time()
            for i in range(50):
                file_path = persistent_temp_dir / f"stress_{i}.tmp"
                file_path.write_text(f"stress test file {i}")
                
                # Every 10 files, modify some previous ones
                if i % 10 == 0 and i > 0:
                    for j in range(max(0, i-5), i):
                        mod_path = persistent_temp_dir / f"stress_{j}.tmp"
                        if mod_path.exists():
                            mod_path.write_text(f"modified stress test file {j}")
                
                # Small delay to avoid overwhelming the system
                if i % 10 == 0:
                    await asyncio.sleep(0.1)
            
            creation_time = time.time() - start_time
            
            # Wait for processing to complete
            await asyncio.sleep(2.0)
            
        finally:
            await watcher.stop()
        
        print(f"Stress test results:")
        print(f"  File creation time: {creation_time:.2f}s")
        print(f"  Files processed: {len(processed_files)}")
        print(f"  Errors encountered: {error_count}")
        
        # Should handle stress without excessive errors
        assert error_count < 5, f"Too many callback errors: {error_count}"
        assert len(processed_files) >= 20, f"Should process at least 20 files, got {len(processed_files)}"
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_extended_operation(self, persistent_temp_dir):
        """Test memory usage doesn't grow excessively during extended operation"""
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async def memory_test_callback(file_path: str):
            # Simulate work that could potentially leak memory
            data = {"path": file_path, "content": "x" * 1000}
            await asyncio.sleep(0.001)
            return data  # Return data that should be garbage collected
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=memory_test_callback,
            debounce_delay=0.02,
            patterns=['*.mem']
        )
        
        memory_samples = []
        
        try:
            await watcher.start()
            await asyncio.sleep(0.2)
            
            # Run extended operation
            for batch in range(10):
                # Create files
                for i in range(20):
                    file_path = persistent_temp_dir / f"memory_test_{batch}_{i}.mem"
                    file_path.write_text(f"memory test batch {batch} file {i} " + "x" * 500)
                
                await asyncio.sleep(0.3)  # Process batch
                
                # Sample memory usage
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                print(f"Batch {batch}: Memory usage = {current_memory:.1f}MB")
        
        finally:
            await watcher.stop()
            
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_samples) if memory_samples else final_memory
        
        print(f"Memory usage analysis:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        
        # Memory growth should be reasonable for extended operation
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB, should be under 100MB"
        assert max_memory < initial_memory + 150, f"Peak memory {max_memory:.1f}MB too high"
    
    @pytest.mark.asyncio
    async def test_edge_cases_and_error_conditions(self, persistent_temp_dir):
        """Test edge cases and error conditions"""
        callback_calls = []
        callback_errors = []
        
        async def edge_case_callback(file_path: str):
            try:
                callback_calls.append(file_path)
                # Simulate occasional failures
                if len(callback_calls) % 7 == 0:  # Fail every 7th call
                    raise ValueError(f"Simulated error for {file_path}")
            except Exception as e:
                callback_errors.append(str(e))
                raise
        
        watcher = EnhancedFileWatcher(
            watch_path=str(persistent_temp_dir),
            callback=edge_case_callback,
            debounce_delay=0.1,
            patterns=['*.edge']
        )
        
        try:
            await watcher.start()
            await asyncio.sleep(0.2)
            
            # Test edge cases
            edge_cases = [
                "normal_file.edge",
                "file with spaces.edge",
                "file-with-dashes.edge", 
                "file_with_underscores.edge",
                "UPPERCASE.EDGE",
                "file.with.dots.edge",
                "unicode_файл.edge",  # Unicode filename
                "very_long_filename_that_exceeds_normal_length_limits_but_should_still_work.edge"
            ]
            
            for filename in edge_cases:
                try:
                    file_path = persistent_temp_dir / filename
                    file_path.write_text(f"Content for {filename}")
                    await asyncio.sleep(0.15)  # Allow processing
                except Exception as e:
                    print(f"Could not create {filename}: {e}")
            
            # Test rapid file creation/deletion
            rapid_file = persistent_temp_dir / "rapid.edge"
            for i in range(5):
                rapid_file.write_text(f"rapid content {i}")
                await asyncio.sleep(0.05)
                if i % 2 == 1 and rapid_file.exists():
                    rapid_file.unlink()
                await asyncio.sleep(0.05)
            
            # Wait for all processing
            await asyncio.sleep(1.0)
            
        finally:
            await watcher.stop()
        
        print(f"Edge case test results:")
        print(f"  Successful callbacks: {len(callback_calls)}")
        print(f"  Callback errors: {len(callback_errors)}")
        print(f"  Files processed: {set(os.path.basename(p) for p in callback_calls)}")
        
        # Should handle most edge cases successfully
        assert len(callback_calls) >= 5, f"Should process at least 5 files, got {len(callback_calls)}"
        
        # System should remain stable despite callback errors
        assert len(callback_errors) > 0, "Should have some simulated errors"
        assert len(callback_calls) > len(callback_errors), "Should have more successes than failures"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
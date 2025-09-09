#!/usr/bin/env python3
"""
Phase 1.1 Performance Tests - File Watcher Benchmarks
Tests performance requirements: <500ms response time, debouncing effectiveness
"""

import pytest
import asyncio
import tempfile
import time
import statistics
from pathlib import Path
from unittest.mock import AsyncMock
import concurrent.futures
import psutil
import os

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.enhanced_file_watcher import EnhancedFileWatcher

class TestFileWatcherPerformance:
    """Performance benchmarks for enhanced file watcher"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for performance testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def performance_callback(self):
        """Callback that tracks timing and call metrics"""
        call_times = []
        call_count = 0
        
        async def timed_callback(file_path: str):
            nonlocal call_count
            start_time = time.time()
            call_count += 1
            
            # Simulate realistic indexing work
            await asyncio.sleep(0.01)  # 10ms processing simulation
            
            end_time = time.time()
            call_times.append(end_time - start_time)
        
        timed_callback.call_times = call_times
        timed_callback.call_count = lambda: call_count
        return timed_callback
    
    @pytest.mark.asyncio
    async def test_response_time_under_500ms(self, temp_dir, performance_callback):
        """Test that file changes are detected within 500ms (Exit Criteria 1.1.1)"""
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=performance_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        await watcher.start()
        
        response_times = []
        
        # Test multiple file change detection times
        for i in range(10):
            test_file = temp_dir / f"response_test_{i}.py"
            
            start_time = time.time()
            test_file.write_text(f"# Test file {i}\nprint('response_test_{i}')")
            
            # Wait for callback to be triggered
            initial_count = performance_callback.call_count()
            timeout = 0
            while performance_callback.call_count() <= initial_count and timeout < 1.0:
                await asyncio.sleep(0.01)
                timeout += 0.01
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            await asyncio.sleep(0.2)  # Prevent interference between tests
        
        await watcher.stop()
        
        # Verify performance requirements
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        
        print(f"\nResponse Time Performance:")
        print(f"Average: {avg_response_time*1000:.1f}ms")
        print(f"Maximum: {max_response_time*1000:.1f}ms")
        print(f"All times: {[f'{t*1000:.1f}ms' for t in response_times]}")
        
        # EXIT CRITERIA 1.1.1: Response time must be under 500ms
        assert max_response_time < 0.5, f"Max response time {max_response_time*1000:.1f}ms exceeds 500ms requirement"
        assert avg_response_time < 0.3, f"Average response time {avg_response_time*1000:.1f}ms should be well under limit"
    
    @pytest.mark.asyncio
    async def test_debouncing_effectiveness_90_percent(self, temp_dir, performance_callback):
        """Test that debouncing reduces callbacks by 90% (Exit Criteria 1.1.2)"""
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=performance_callback,
            debounce_delay=0.2,  # Longer debounce for testing
            patterns=['*.py']
        )
        
        await watcher.start()
        
        test_file = temp_dir / "debounce_test.py"
        test_file.write_text("# Initial content")
        
        # Make rapid changes (should be heavily debounced)
        rapid_changes = 50
        for i in range(rapid_changes):
            test_file.write_text(f"# Rapid change {i}\nprint('change_{i}')")
            await asyncio.sleep(0.01)  # 10ms between changes (much faster than debounce)
        
        # Wait for debouncing to complete
        await asyncio.sleep(0.5)
        await watcher.stop()
        
        callback_count = performance_callback.call_count()
        reduction_percentage = (1 - callback_count / rapid_changes) * 100
        
        print(f"\nDebouncing Performance:")
        print(f"Changes made: {rapid_changes}")
        print(f"Callbacks triggered: {callback_count}")
        print(f"Reduction: {reduction_percentage:.1f}%")
        
        # EXIT CRITERIA 1.1.2: Debouncing should reduce callbacks by at least 90%
        assert reduction_percentage >= 90, f"Debouncing only reduced callbacks by {reduction_percentage:.1f}%, need ≥90%"
        assert callback_count <= 5, f"Too many callbacks ({callback_count}) after debouncing"
    
    @pytest.mark.asyncio
    async def test_concurrent_file_handling_performance(self, temp_dir, performance_callback):
        """Test performance with concurrent file modifications"""
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=performance_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        await watcher.start()
        
        # Create many files concurrently
        concurrent_files = 20
        start_time = time.time()
        
        async def create_and_modify_file(index):
            file_path = temp_dir / f"concurrent_{index}.py"
            file_path.write_text(f"# Concurrent file {index}")
            await asyncio.sleep(0.05)  # Small delay
            file_path.write_text(f"# Modified concurrent file {index}")
        
        # Execute concurrent operations
        tasks = [create_and_modify_file(i) for i in range(concurrent_files)]
        await asyncio.gather(*tasks)
        
        # Wait for all callbacks to complete
        await asyncio.sleep(1.0)
        processing_time = time.time() - start_time
        
        await watcher.stop()
        
        callback_count = performance_callback.call_count()
        throughput = callback_count / processing_time
        
        print(f"\nConcurrent Performance:")
        print(f"Files created: {concurrent_files}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Callbacks triggered: {callback_count}")
        print(f"Throughput: {throughput:.1f} callbacks/second")
        
        # Performance expectations
        assert processing_time < 3.0, f"Concurrent processing took {processing_time:.2f}s, should be under 3s"
        assert callback_count >= concurrent_files * 0.5, f"Too few callbacks ({callback_count}) for concurrent operations"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, temp_dir):
        """Test memory usage remains stable during extended operation"""
        
        memory_samples = []
        
        async def memory_tracking_callback(file_path: str):
            # Track memory usage during callbacks
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=memory_tracking_callback,
            debounce_delay=0.05,
            patterns=['*.py']
        )
        
        # Baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        await watcher.start()
        
        # Create sustained file activity
        test_file = temp_dir / "memory_test.py"
        for i in range(100):
            test_file.write_text(f"# Memory test iteration {i}\n" + "x" * (i * 10))
            await asyncio.sleep(0.02)
        
        await asyncio.sleep(0.5)  # Final processing
        await watcher.stop()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory
        
        print(f"\nMemory Usage:")
        print(f"Baseline: {baseline_memory:.1f}MB")
        print(f"Final: {final_memory:.1f}MB")
        print(f"Growth: {memory_growth:.1f}MB")
        if memory_samples:
            print(f"Peak during callbacks: {max(memory_samples):.1f}MB")
        
        # Memory should not grow excessively
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB, should be under 50MB"
    
    @pytest.mark.asyncio
    async def test_large_file_handling_performance(self, temp_dir, performance_callback):
        """Test performance with large files"""
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=performance_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        await watcher.start()
        
        # Create large file content (simulating large source files)
        large_content = "# Large Python file\n" + "\n".join([
            f"def function_{i}():\n    '''Function {i} documentation'''\n    return {i}"
            for i in range(1000)
        ])
        
        large_file = temp_dir / "large_file.py"
        
        start_time = time.time()
        large_file.write_text(large_content)
        
        # Wait for processing
        initial_count = performance_callback.call_count()
        timeout = 0
        while performance_callback.call_count() <= initial_count and timeout < 2.0:
            await asyncio.sleep(0.01)
            timeout += 0.01
        
        processing_time = time.time() - start_time
        
        await watcher.stop()
        
        file_size = len(large_content.encode()) / 1024 / 1024  # MB
        
        print(f"\nLarge File Performance:")
        print(f"File size: {file_size:.2f}MB")
        print(f"Detection time: {processing_time*1000:.1f}ms")
        
        # Large files should still be detected quickly
        assert processing_time < 1.0, f"Large file detection took {processing_time*1000:.1f}ms, should be under 1000ms"
    
    @pytest.mark.asyncio
    async def test_pattern_matching_performance(self, temp_dir, performance_callback):
        """Test performance of pattern matching with many files"""
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=performance_callback,
            debounce_delay=0.05,
            patterns=['*.py', '*.txt', '*.md', '*.json', '*.yaml']
        )
        
        await watcher.start()
        
        # Create many files with different extensions
        extensions = ['py', 'txt', 'md', 'json', 'yaml', 'xml', 'html', 'css', 'js']
        files_per_extension = 10
        
        start_time = time.time()
        
        for ext in extensions:
            for i in range(files_per_extension):
                file_path = temp_dir / f"test_{i}.{ext}"
                file_path.write_text(f"Content for {ext} file {i}")
        
        # Wait for all processing
        await asyncio.sleep(1.0)
        creation_time = time.time() - start_time
        
        await watcher.stop()
        
        callback_count = performance_callback.call_count()
        total_files = len(extensions) * files_per_extension
        matching_files = 5 * files_per_extension  # Only 5 extensions should match patterns
        
        print(f"\nPattern Matching Performance:")
        print(f"Total files created: {total_files}")
        print(f"Expected matches: {matching_files}")
        print(f"Actual callbacks: {callback_count}")
        print(f"Creation time: {creation_time:.2f}s")
        
        # Should only process files matching patterns
        assert callback_count <= matching_files + 5, f"Too many callbacks ({callback_count}) for pattern matching"
        assert callback_count >= matching_files * 0.8, f"Too few callbacks ({callback_count}) for expected matches"
        assert creation_time < 3.0, f"Pattern matching took {creation_time:.2f}s, should be under 3s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
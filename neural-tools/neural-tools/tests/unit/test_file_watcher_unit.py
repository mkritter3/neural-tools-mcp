#!/usr/bin/env python3
"""
Phase 1.1 Unit Tests - File Watcher with Selective Reprocessing
Tests debouncing, selective reprocessing, and callback behaviors
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, call
from pathlib import Path
import time

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.enhanced_file_watcher import EnhancedFileWatcher

class TestEnhancedFileWatcher:
    """Test suite for Enhanced File Watcher with selective reprocessing"""
    
    @pytest.fixture
    def mock_callback(self):
        """Mock callback for file change events"""
        return AsyncMock()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def watcher(self, mock_callback, temp_dir):
        """Create file watcher instance"""
        return EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.1,  # Short delay for testing
            patterns=['*.py', '*.txt']
        )
    
    @pytest.mark.asyncio
    async def test_callback_invokes_index_file_not_directory(self, watcher, mock_callback, temp_dir):
        """Test that callback is called with index_file, not index_directory"""
        # Start watcher first
        await watcher.start()
        await asyncio.sleep(0.1)  # Give observer time to start
        
        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text("print('hello')")
        
        # Wait for initial detection
        await asyncio.sleep(0.2)
        
        # Modify file to ensure change detection
        test_file.write_text("print('modified')")
        
        # Wait for debounce and processing
        await asyncio.sleep(0.3)
        
        # Stop watcher
        await watcher.stop()
        
        # Verify callback was called with file path, not directory
        mock_callback.assert_called()
        call_args = mock_callback.call_args[0]
        assert str(test_file) in call_args[0] or call_args[0] == str(test_file)
        assert str(temp_dir) != call_args[0]  # Should NOT be directory
    
    @pytest.mark.asyncio
    async def test_debouncing_prevents_rapid_reindexing(self, watcher, mock_callback, temp_dir):
        """Test that rapid file changes are debounced"""
        test_file = temp_dir / "rapid_test.py"
        test_file.write_text("initial")
        
        await watcher.start()
        
        # Make rapid changes
        for i in range(5):
            test_file.write_text(f"change_{i}")
            await asyncio.sleep(0.02)  # Faster than debounce delay
        
        # Wait for debounce to complete
        await asyncio.sleep(0.3)
        await watcher.stop()
        
        # Should have been called fewer times than changes made
        assert mock_callback.call_count <= 2  # At most initial + final
    
    @pytest.mark.asyncio
    async def test_selective_reprocessing_changed_files_only(self, watcher, mock_callback, temp_dir):
        """Test that only changed files are reprocessed"""
        # Create multiple files
        file1 = temp_dir / "file1.py"
        file2 = temp_dir / "file2.py"
        file1.write_text("content1")
        file2.write_text("content2")
        
        await watcher.start()
        
        # Modify only file1
        file1.write_text("modified_content1")
        
        await asyncio.sleep(0.2)
        await watcher.stop()
        
        # Verify only file1 was processed
        call_paths = [call[0][0] for call in mock_callback.call_args_list]
        assert any(str(file1) in path for path in call_paths)
        assert not any(str(file2) in path for path in call_paths)
    
    @pytest.mark.asyncio
    async def test_pattern_filtering(self, mock_callback, temp_dir):
        """Test that only files matching patterns are watched"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.1,
            patterns=['*.py']  # Only Python files
        )
        
        # Create files with different extensions
        py_file = temp_dir / "test.py"
        txt_file = temp_dir / "test.txt"
        py_file.write_text("python code")
        txt_file.write_text("text content")
        
        await watcher.start()
        
        # Modify both files
        py_file.write_text("modified python")
        txt_file.write_text("modified text")
        
        await asyncio.sleep(0.2)
        await watcher.stop()
        
        # Only .py file should trigger callback
        call_paths = [call[0][0] for call in mock_callback.call_args_list]
        assert any("test.py" in path for path in call_paths)
        assert not any("test.txt" in path for path in call_paths)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_callback(self, temp_dir):
        """Test that watcher continues working even if callback fails"""
        failing_callback = AsyncMock(side_effect=Exception("Callback error"))
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=failing_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        test_file = temp_dir / "test.py"
        test_file.write_text("initial")
        
        await watcher.start()
        
        # This should not crash the watcher
        test_file.write_text("modified")
        await asyncio.sleep(0.2)
        
        # Make another change to verify watcher is still working
        test_file.write_text("modified_again")
        await asyncio.sleep(0.2)
        
        await watcher.stop()
        
        # Callback should have been called despite errors
        assert failing_callback.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_file_modifications(self, watcher, mock_callback, temp_dir):
        """Test handling of concurrent file modifications"""
        files = []
        for i in range(5):
            file_path = temp_dir / f"concurrent_{i}.py"
            file_path.write_text(f"initial_{i}")
            files.append(file_path)
        
        await watcher.start()
        
        # Modify all files concurrently
        async def modify_file(file_path, content):
            file_path.write_text(content)
        
        tasks = [
            modify_file(file_path, f"modified_{i}")
            for i, file_path in enumerate(files)
        ]
        await asyncio.gather(*tasks)
        
        await asyncio.sleep(0.3)  # Wait for all processing
        await watcher.stop()
        
        # All files should have been processed
        call_paths = [call[0][0] for call in mock_callback.call_args_list]
        processed_files = set(call_paths)
        
        for file_path in files:
            assert any(str(file_path) in path for path in processed_files)
    
    def test_debounce_delay_configuration(self, mock_callback, temp_dir):
        """Test that debounce delay can be configured"""
        watcher1 = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.5
        )
        
        watcher2 = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=1.0
        )
        
        assert watcher1.debounce_delay == 0.5
        assert watcher2.debounce_delay == 1.0
    
    @pytest.mark.asyncio
    async def test_watcher_lifecycle(self, watcher, mock_callback, temp_dir):
        """Test start/stop lifecycle of watcher"""
        assert not watcher.is_running
        
        await watcher.start()
        assert watcher.is_running
        
        # Test file change detection works
        test_file = temp_dir / "lifecycle_test.py"
        test_file.write_text("test content")
        
        await asyncio.sleep(0.2)
        
        await watcher.stop()
        assert not watcher.is_running
        
        # Verify callback was invoked while running
        mock_callback.assert_called()
    
    @pytest.mark.asyncio
    async def test_multiple_pattern_matching(self, mock_callback, temp_dir):
        """Test watcher with multiple file patterns"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.1,
            patterns=['*.py', '*.txt', '*.md']
        )
        
        # Create files matching different patterns
        files = [
            temp_dir / "test.py",
            temp_dir / "test.txt", 
            temp_dir / "test.md",
            temp_dir / "test.json"  # Should be ignored
        ]
        
        for file_path in files:
            file_path.write_text("content")
        
        await watcher.start()
        
        # Modify all files
        for i, file_path in enumerate(files):
            file_path.write_text(f"modified_{i}")
        
        await asyncio.sleep(0.2)
        await watcher.stop()
        
        # Verify only matching patterns were processed
        call_paths = [call[0][0] for call in mock_callback.call_args_list]
        
        assert any("test.py" in path for path in call_paths)
        assert any("test.txt" in path for path in call_paths)
        assert any("test.md" in path for path in call_paths)
        assert not any("test.json" in path for path in call_paths)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
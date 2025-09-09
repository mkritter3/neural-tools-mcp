#!/usr/bin/env python3
"""
Phase 1.1 Unit Tests - File Watcher Debouncing Logic Only
Tests core debouncing functionality without filesystem dependencies
"""

import pytest
import asyncio
from unittest.mock import AsyncMock
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.enhanced_file_watcher import EnhancedFileWatcher

class TestFileWatcherDebouncing:
    """Test debouncing logic without filesystem dependencies"""
    
    @pytest.fixture
    def mock_callback(self):
        """Mock callback for file change events"""
        return AsyncMock()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_debouncing_logic_direct(self, mock_callback, temp_dir):
        """Test debouncing logic by directly calling internal methods"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        # Create test file for hashing
        test_file = temp_dir / "test.py"
        test_file.write_text("initial content")
        
        # Directly test the scheduling method (bypasses filesystem watcher)
        watcher._schedule_file_processing(str(test_file))
        watcher._schedule_file_processing(str(test_file))  # Second call should cancel first
        watcher._schedule_file_processing(str(test_file))  # Third call should cancel second
        
        # Wait for debounce to complete
        await asyncio.sleep(0.2)
        
        # Should only be called once despite 3 schedule calls
        mock_callback.assert_called_once_with(str(test_file))
    
    @pytest.mark.asyncio
    async def test_hash_based_selective_processing(self, mock_callback, temp_dir):
        """Test that files with same content hash are not reprocessed"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.05,
            patterns=['*.py']
        )
        
        test_file = temp_dir / "hash_test.py"
        test_file.write_text("same content")
        
        # First processing - file is new, should process
        watcher._schedule_file_processing(str(test_file))
        await asyncio.sleep(0.1)
        
        # Second processing - same content, should not process
        watcher._schedule_file_processing(str(test_file))
        await asyncio.sleep(0.1)
        
        # Callback should only be called once (for new file)
        assert mock_callback.call_count == 1
    
    @pytest.mark.asyncio
    async def test_pattern_matching_logic(self, mock_callback, temp_dir):
        """Test pattern matching without filesystem watcher"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.05,
            patterns=['*.py', '*.txt']
        )
        
        # Test files that match patterns
        python_file = temp_dir / "test.py"
        text_file = temp_dir / "test.txt"
        json_file = temp_dir / "test.json"  # Should not match
        
        python_file.write_text("python code")
        text_file.write_text("text content") 
        json_file.write_text("json content")
        
        # Test pattern matching directly
        assert watcher._matches_pattern(str(python_file)) == True
        assert watcher._matches_pattern(str(text_file)) == True
        assert watcher._matches_pattern(str(json_file)) == False
    
    @pytest.mark.asyncio
    async def test_file_hash_computation(self, temp_dir):
        """Test file hash computation logic"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=AsyncMock(),
            patterns=['*']
        )
        
        test_file = temp_dir / "hash_test.txt"
        
        # Test with content
        test_file.write_text("test content")
        hash1 = watcher._compute_file_hash(str(test_file))
        assert hash1 is not None
        assert len(hash1) == 64  # SHA256 hash length
        
        # Same content should produce same hash
        hash2 = watcher._compute_file_hash(str(test_file))
        assert hash1 == hash2
        
        # Different content should produce different hash
        test_file.write_text("different content")
        hash3 = watcher._compute_file_hash(str(test_file))
        assert hash1 != hash3
        
        # Non-existent file should return None
        hash4 = watcher._compute_file_hash("/non/existent/file.txt")
        assert hash4 is None
    
    @pytest.mark.asyncio
    async def test_concurrent_scheduling(self, mock_callback, temp_dir):
        """Test concurrent file scheduling"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=mock_callback,
            debounce_delay=0.1,
            patterns=['*.py']
        )
        
        # Create test files
        files = []
        for i in range(5):
            test_file = temp_dir / f"concurrent_{i}.py"
            test_file.write_text(f"content {i}")
            files.append(str(test_file))
        
        # Schedule all files concurrently
        for file_path in files:
            watcher._schedule_file_processing(file_path)
        
        # Wait for all processing to complete
        await asyncio.sleep(0.2)
        
        # All files should have been processed
        assert mock_callback.call_count == len(files)
        
        # Verify each file was called
        called_files = [call[0][0] for call in mock_callback.call_args_list]
        for file_path in files:
            assert file_path in called_files
    
    @pytest.mark.asyncio
    async def test_error_handling_in_callback(self, temp_dir):
        """Test error handling when callback fails"""
        failing_callback = AsyncMock(side_effect=Exception("Callback failed"))
        
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=failing_callback,
            debounce_delay=0.05,
            patterns=['*.py']
        )
        
        test_file = temp_dir / "error_test.py"
        test_file.write_text("test content")
        
        # This should not raise an exception
        watcher._schedule_file_processing(str(test_file))
        await asyncio.sleep(0.1)
        
        # Callback should have been called despite the error
        failing_callback.assert_called_once()
        
        # Should be able to continue processing other files
        test_file2 = temp_dir / "error_test2.py"
        test_file2.write_text("test content 2")
        
        watcher._schedule_file_processing(str(test_file2))
        await asyncio.sleep(0.1)
        
        # Should have been called twice now
        assert failing_callback.call_count == 2
    
    def test_configuration_parameters(self, temp_dir):
        """Test that configuration parameters are set correctly"""
        watcher = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=AsyncMock(),
            debounce_delay=0.5,
            patterns=['*.py', '*.js', '*.txt']
        )
        
        assert watcher.watch_path == Path(temp_dir)
        assert watcher.debounce_delay == 0.5
        assert watcher.patterns == ['*.py', '*.js', '*.txt']
        assert not watcher.is_running
        
        # Test default patterns
        watcher_default = EnhancedFileWatcher(
            watch_path=str(temp_dir),
            callback=AsyncMock()
        )
        assert watcher_default.patterns == ['*']
        assert watcher_default.debounce_delay == 0.5  # Default

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
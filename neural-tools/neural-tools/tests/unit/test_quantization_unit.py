#!/usr/bin/env python3
"""
Phase 1.4 Unit Tests - Vector Quantization Support
Tests quantization algorithms, compression ratios, and performance
"""

import pytest
import numpy as np
import time
import psutil
import os
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.quantization import (
    VectorQuantizer, 
    QuantizationConfig, 
    QuantizationType,
    QuantizationStats,
    QuantizationResult
)

class TestVectorQuantizer:
    """Test suite for vector quantization functionality"""
    
    @pytest.fixture
    def binary_config(self):
        """Binary quantization configuration"""
        return QuantizationConfig(
            quantization_type=QuantizationType.BINARY,
            enable_memory_mapping=True,
            compression_threshold=100,
            preserve_original=False,
            batch_size=500
        )
    
    @pytest.fixture
    def scalar_8bit_config(self):
        """8-bit scalar quantization configuration"""
        return QuantizationConfig(
            quantization_type=QuantizationType.SCALAR_8BIT,
            enable_memory_mapping=True,
            compression_threshold=100,
            preserve_original=False,
            batch_size=500
        )
    
    @pytest.fixture
    def sample_vectors(self):
        """Sample vectors for testing"""
        np.random.seed(42)
        return np.random.randn(1000, 384).astype(np.float32)  # Common embedding dimension
    
    @pytest.fixture
    def quantizer(self, binary_config):
        """Default quantizer instance"""
        return VectorQuantizer(binary_config)
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization with default config"""
        quantizer = VectorQuantizer()
        assert quantizer.config is not None
        assert quantizer.config.quantization_type == QuantizationType.BINARY
        assert quantizer.stats is not None
        assert "vectors_processed" in quantizer.stats
        assert "compression_ratio" in quantizer.stats
    
    def test_binary_quantization(self, quantizer, sample_vectors):
        """Test binary quantization functionality"""
        result = quantizer.quantize_vectors(sample_vectors)
        
        assert result.success
        assert result.quantized_vectors is not None
        assert result.metadata is not None
        
        # Binary quantization should use 1 bit per dimension
        expected_size = len(sample_vectors) * sample_vectors.shape[1] // 8  # bits to bytes
        actual_size = len(result.quantized_vectors)
        
        # Allow some overhead for metadata
        assert actual_size <= expected_size * 1.5
        
        # Compression ratio should be significant
        assert result.metadata.get("compression_ratio", 0) > 8  # At least 8x compression
    
    def test_scalar_8bit_quantization(self, scalar_8bit_config, sample_vectors):
        """Test 8-bit scalar quantization"""
        quantizer = VectorQuantizer(scalar_8bit_config)
        result = quantizer.quantize_vectors(sample_vectors)
        
        assert result.success
        assert result.quantized_vectors is not None
        
        # 8-bit quantization should use 1/4 the space of float32
        expected_compression = 4
        actual_compression = result.metadata.get("compression_ratio", 0)
        assert actual_compression >= expected_compression * 0.8  # Allow 20% variance
        assert actual_compression <= expected_compression * 1.2
    
    def test_vector_dequantization(self, quantizer, sample_vectors):
        """Test vector dequantization accuracy"""
        result = quantizer.quantize_vectors(sample_vectors)
        assert result.success
        
        dequantized = quantizer.dequantize_vectors(
            result.quantized_vectors, 
            result.metadata
        )
        
        assert dequantized is not None
        assert dequantized.shape == sample_vectors.shape
        
        # For binary quantization, expect some precision loss
        # But cosine similarity should remain reasonable
        similarities = []
        for i in range(min(100, len(sample_vectors))):  # Test subset
            original = sample_vectors[i]
            reconstructed = dequantized[i]
            
            # Cosine similarity
            cos_sim = np.dot(original, reconstructed) / (
                np.linalg.norm(original) * np.linalg.norm(reconstructed)
            )
            similarities.append(cos_sim)
        
        avg_similarity = np.mean(similarities)
        # Binary quantization should maintain >0.7 cosine similarity on average
        assert avg_similarity > 0.7, f"Average similarity {avg_similarity:.3f} too low"
    
    def test_memory_usage_reduction(self, quantizer, sample_vectors):
        """Test that quantization reduces memory usage"""
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        # Create large vector set
        large_vectors = np.random.randn(5000, 768).astype(np.float32)
        original_size = large_vectors.nbytes
        
        # Quantize
        result = quantizer.quantize_vectors(large_vectors)
        assert result.success
        
        quantized_size = len(result.quantized_vectors) if isinstance(result.quantized_vectors, bytes) else result.quantized_vectors.nbytes
        
        # Should achieve significant compression
        compression_ratio = original_size / quantized_size
        assert compression_ratio > 4, f"Compression ratio {compression_ratio:.2f} too low"
        
        # Memory usage should not increase excessively
        mem_after = process.memory_info().rss
        mem_increase_mb = (mem_after - mem_before) / 1024 / 1024
        assert mem_increase_mb < 100, f"Memory increased by {mem_increase_mb:.1f}MB"
    
    def test_batch_processing(self, quantizer):
        """Test batch processing of vectors"""
        # Create vectors larger than batch size
        large_vectors = np.random.randn(2500, 384).astype(np.float32)
        
        result = quantizer.quantize_vectors(large_vectors)
        assert result.success
        
        # Should process in batches without errors
        assert result.quantized_vectors is not None
        assert "batches_processed" in result.metadata
        
        expected_batches = (len(large_vectors) + quantizer.config.batch_size - 1) // quantizer.config.batch_size
        actual_batches = result.metadata.get("batches_processed", 0)
        assert actual_batches == expected_batches
    
    def test_compression_threshold(self):
        """Test compression threshold behavior"""
        config = QuantizationConfig(compression_threshold=1000)
        quantizer = VectorQuantizer(config)
        
        # Small vector set (below threshold)
        small_vectors = np.random.randn(500, 384).astype(np.float32)
        result = quantizer.quantize_vectors(small_vectors)
        
        # Should either skip quantization or indicate it's below threshold
        assert result.success
        if result.quantized_vectors is not None:
            # If quantized, should have metadata indicating threshold handling
            assert "below_threshold" in result.metadata or result.metadata.get("compression_ratio", 0) > 1
    
    def test_quantization_stats_tracking(self, quantizer, sample_vectors):
        """Test that quantization statistics are properly tracked"""
        initial_processed = quantizer.stats["vectors_processed"]
        
        result = quantizer.quantize_vectors(sample_vectors)
        assert result.success
        
        # Stats should be updated
        assert quantizer.stats["vectors_processed"] > initial_processed
        assert quantizer.stats["compression_ratio"] > 0
        assert quantizer.stats["total_original_size"] > 0
        assert quantizer.stats["total_compressed_size"] > 0
    
    def test_error_handling_invalid_vectors(self, quantizer):
        """Test error handling with invalid input vectors"""
        # Test with None
        result = quantizer.quantize_vectors(None)
        assert not result.success
        assert "invalid" in result.error.lower() or "none" in result.error.lower()
        
        # Test with empty array
        empty_vectors = np.array([])
        result = quantizer.quantize_vectors(empty_vectors)
        assert not result.success or result.quantized_vectors is None
        
        # Test with wrong dimensions
        wrong_dims = np.random.randn(10, 10, 10)  # 3D instead of 2D
        result = quantizer.quantize_vectors(wrong_dims)
        assert not result.success or "dimension" in str(result.error).lower()
    
    def test_quantization_reproducibility(self, quantizer, sample_vectors):
        """Test that quantization is reproducible"""
        result1 = quantizer.quantize_vectors(sample_vectors)
        result2 = quantizer.quantize_vectors(sample_vectors)
        
        assert result1.success and result2.success
        
        # Results should be identical (for deterministic quantization)
        if isinstance(result1.quantized_vectors, bytes) and isinstance(result2.quantized_vectors, bytes):
            assert result1.quantized_vectors == result2.quantized_vectors
        elif hasattr(result1.quantized_vectors, 'shape') and hasattr(result2.quantized_vectors, 'shape'):
            np.testing.assert_array_equal(result1.quantized_vectors, result2.quantized_vectors)
    
    def test_quantization_performance(self, quantizer):
        """Test quantization performance requirements"""
        # Create large vector set for performance testing
        large_vectors = np.random.randn(10000, 768).astype(np.float32)
        
        start_time = time.time()
        result = quantizer.quantize_vectors(large_vectors)
        quantization_time = time.time() - start_time
        
        assert result.success
        
        # Should quantize 10k vectors in reasonable time
        vectors_per_second = len(large_vectors) / quantization_time
        assert vectors_per_second > 1000, f"Performance too slow: {vectors_per_second:.1f} vectors/sec"
        
        print(f"Quantization performance: {vectors_per_second:.1f} vectors/sec")
    
    def test_memory_mapping_option(self, sample_vectors):
        """Test memory mapping configuration option"""
        # Test with memory mapping enabled
        config_with_mmap = QuantizationConfig(enable_memory_mapping=True)
        quantizer_mmap = VectorQuantizer(config_with_mmap)
        result_mmap = quantizer_mmap.quantize_vectors(sample_vectors)
        
        # Test with memory mapping disabled
        config_no_mmap = QuantizationConfig(enable_memory_mapping=False)
        quantizer_no_mmap = VectorQuantizer(config_no_mmap)
        result_no_mmap = quantizer_no_mmap.quantize_vectors(sample_vectors)
        
        # Both should succeed
        assert result_mmap.success
        assert result_no_mmap.success
        
        # Results should be valid regardless of memory mapping setting
        assert result_mmap.quantized_vectors is not None
        assert result_no_mmap.quantized_vectors is not None

class TestQuantizationConfig:
    """Test quantization configuration validation"""
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Valid configuration
        valid_config = QuantizationConfig(
            quantization_type=QuantizationType.BINARY,
            compression_threshold=100,
            batch_size=1000
        )
        assert valid_config.quantization_type == QuantizationType.BINARY
        assert valid_config.compression_threshold == 100
        assert valid_config.batch_size == 1000
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = QuantizationConfig()
        assert config.quantization_type == QuantizationType.BINARY
        assert config.enable_memory_mapping is True
        assert config.compression_threshold > 0
        assert config.batch_size > 0
        assert config.preserve_original is False

class TestQuantizationTypes:
    """Test different quantization types"""
    
    def test_quantization_type_enum(self):
        """Test quantization type enumeration"""
        assert QuantizationType.NONE.value == "none"
        assert QuantizationType.BINARY.value == "binary"
        assert QuantizationType.SCALAR_8BIT.value == "scalar_8bit"
        assert QuantizationType.SCALAR_4BIT.value == "scalar_4bit"
    
    @pytest.mark.parametrize("qtype", [
        QuantizationType.BINARY,
        QuantizationType.SCALAR_8BIT,
    ])
    def test_quantization_type_functionality(self, qtype):
        """Test that different quantization types work"""
        config = QuantizationConfig(quantization_type=qtype)
        quantizer = VectorQuantizer(config)
        
        test_vectors = np.random.randn(100, 128).astype(np.float32)
        result = quantizer.quantize_vectors(test_vectors)
        
        assert result.success
        assert result.quantized_vectors is not None
        assert result.metadata.get("quantization_type") == qtype.value

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
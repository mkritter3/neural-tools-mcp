#!/usr/bin/env python3
"""
Phase 1.4 Integration Tests - Vector Quantization with Real Performance Validation
Tests quantization integration with real workloads and memory constraints
"""

import pytest
import asyncio
import numpy as np
import time
import psutil
import os
from unittest.mock import patch, Mock
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.quantization import (
    VectorQuantizer,
    QuantizationConfig,
    QuantizationType,
    QuantizationStats
)

class TestQuantizationIntegration:
    """Integration tests for quantization with real performance requirements"""
    
    @pytest.fixture
    def memory_optimized_config(self):
        """Configuration optimized for memory usage"""
        return QuantizationConfig(
            quantization_type=QuantizationType.BINARY,
            enable_memory_mapping=True,
            compression_threshold=500,
            preserve_original=False,
            batch_size=2000
        )
    
    @pytest.fixture
    def accuracy_optimized_config(self):
        """Configuration optimized for accuracy"""
        return QuantizationConfig(
            quantization_type=QuantizationType.SCALAR_8BIT,
            enable_memory_mapping=True,
            compression_threshold=500,
            preserve_original=True,  # Keep originals for accuracy comparison
            batch_size=1000
        )
    
    @pytest.fixture
    def realistic_dataset(self):
        """Realistic dataset for testing"""
        np.random.seed(42)
        # Simulate realistic embedding dimensions and document counts
        num_documents = 10000
        embedding_dim = 768  # Common for transformer models
        
        # Generate vectors with some structure (not pure random)
        centers = np.random.randn(20, embedding_dim) * 2
        assignments = np.random.randint(0, 20, num_documents)
        
        vectors = []
        for i in range(num_documents):
            center = centers[assignments[i]]
            noise = np.random.randn(embedding_dim) * 0.5
            vector = center + noise
            vectors.append(vector)
        
        return np.array(vectors, dtype=np.float32)
    
    @pytest.mark.benchmark
    def test_memory_usage_reduction_target(self, memory_optimized_config, realistic_dataset):
        """Verify >70% memory reduction with quantization (Roadmap requirement)"""
        process = psutil.Process()
        
        # Baseline memory before quantization
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        original_size_mb = realistic_dataset.nbytes / 1024 / 1024
        
        print(f"Original dataset size: {original_size_mb:.1f}MB")
        
        quantizer = VectorQuantizer(memory_optimized_config)
        
        # Perform quantization
        start_time = time.time()
        result = quantizer.quantize_vectors(realistic_dataset)
        quantization_time = time.time() - start_time
        
        assert result.success, f"Quantization failed: {result.error}"
        
        # Calculate compression
        if isinstance(result.quantized_vectors, bytes):
            compressed_size_mb = len(result.quantized_vectors) / 1024 / 1024
        else:
            compressed_size_mb = result.quantized_vectors.nbytes / 1024 / 1024
        
        reduction_percentage = (1 - compressed_size_mb / original_size_mb) * 100
        
        print(f"Quantization results:")
        print(f"  Original: {original_size_mb:.1f}MB")
        print(f"  Compressed: {compressed_size_mb:.1f}MB")
        print(f"  Reduction: {reduction_percentage:.1f}%")
        print(f"  Time: {quantization_time:.2f}s")
        print(f"  Speed: {len(realistic_dataset)/quantization_time:.0f} vectors/sec")
        
        # ROADMAP EXIT CRITERIA: >70% memory reduction
        assert reduction_percentage > 70, f"Memory reduction {reduction_percentage:.1f}% below 70% target"
        
        # Verify performance is reasonable
        vectors_per_sec = len(realistic_dataset) / quantization_time
        assert vectors_per_sec > 2000, f"Quantization too slow: {vectors_per_sec:.0f} vectors/sec"
    
    @pytest.mark.asyncio
    async def test_query_latency_impact(self, accuracy_optimized_config, realistic_dataset):
        """Verify <5% latency increase with quantization (Roadmap requirement)"""
        quantizer = VectorQuantizer(accuracy_optimized_config)
        
        # Quantize the dataset
        result = quantizer.quantize_vectors(realistic_dataset)
        assert result.success
        
        # Prepare query vectors
        query_vectors = realistic_dataset[:100]  # Use subset as queries
        
        # Benchmark original vector search (mock)
        def mock_vector_search(vectors, query, k=10):
            """Mock vector search using cosine similarity"""
            similarities = np.dot(vectors, query) / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
            )
            top_k = np.argpartition(similarities, -k)[-k:]
            return top_k[np.argsort(similarities[top_k])[::-1]]
        
        # Benchmark original search
        original_times = []
        for query in query_vectors[:10]:  # Test subset
            start = time.perf_counter()
            _ = mock_vector_search(realistic_dataset, query)
            original_times.append(time.perf_counter() - start)
        
        avg_original_time = np.mean(original_times) * 1000  # ms
        
        # Benchmark quantized search
        dequantized_vectors = quantizer.dequantize_vectors(
            result.quantized_vectors, result.metadata
        )
        
        quantized_times = []
        for query in query_vectors[:10]:  # Test subset
            start = time.perf_counter()
            _ = mock_vector_search(dequantized_vectors, query)
            quantized_times.append(time.perf_counter() - start)
        
        avg_quantized_time = np.mean(quantized_times) * 1000  # ms
        
        latency_increase = ((avg_quantized_time - avg_original_time) / avg_original_time) * 100
        
        print(f"Latency analysis:")
        print(f"  Original avg: {avg_original_time:.2f}ms")
        print(f"  Quantized avg: {avg_quantized_time:.2f}ms") 
        print(f"  Increase: {latency_increase:.1f}%")
        
        # ROADMAP EXIT CRITERIA: <5% latency increase
        assert latency_increase < 5, f"Latency increase {latency_increase:.1f}% exceeds 5% target"
    
    @pytest.mark.accuracy
    async def test_recall_after_quantization(self, accuracy_optimized_config, realistic_dataset):
        """Verify <2% recall drop with quantization (Roadmap requirement)"""
        quantizer = VectorQuantizer(accuracy_optimized_config)
        
        # Quantize vectors
        result = quantizer.quantize_vectors(realistic_dataset)
        assert result.success
        
        # Dequantize for comparison
        dequantized_vectors = quantizer.dequantize_vectors(
            result.quantized_vectors, result.metadata
        )
        
        # Create ground truth with original vectors
        query_vectors = realistic_dataset[:50]  # Test queries
        k = 10
        
        recall_scores = []
        
        for i, query in enumerate(query_vectors):
            # Ground truth: search in original vectors
            original_similarities = np.dot(realistic_dataset, query) / (
                np.linalg.norm(realistic_dataset, axis=1) * np.linalg.norm(query)
            )
            original_top_k = np.argpartition(original_similarities, -k)[-k:]
            original_top_k = set(original_top_k)
            
            # Quantized results: search in dequantized vectors
            quantized_similarities = np.dot(dequantized_vectors, query) / (
                np.linalg.norm(dequantized_vectors, axis=1) * np.linalg.norm(query)
            )
            quantized_top_k = np.argpartition(quantized_similarities, -k)[-k:]
            quantized_top_k = set(quantized_top_k)
            
            # Calculate recall
            intersection = len(original_top_k & quantized_top_k)
            recall = intersection / k
            recall_scores.append(recall)
        
        avg_recall = np.mean(recall_scores)
        recall_drop = (1 - avg_recall) * 100
        
        print(f"Recall analysis:")
        print(f"  Average recall: {avg_recall:.3f}")
        print(f"  Recall drop: {recall_drop:.1f}%")
        print(f"  Min recall: {np.min(recall_scores):.3f}")
        print(f"  Max recall: {np.max(recall_scores):.3f}")
        
        # ROADMAP EXIT CRITERIA: <2% recall drop
        assert recall_drop < 2, f"Recall drop {recall_drop:.1f}% exceeds 2% target"
        assert avg_recall > 0.98, f"Average recall {avg_recall:.3f} below 0.98 target"
    
    @pytest.mark.benchmark
    def test_benchmark_consistency(self, memory_optimized_config, realistic_dataset):
        """Verify benchmarks consistent over 5 runs with <5% variance (Roadmap requirement)"""
        quantizer = VectorQuantizer(memory_optimized_config)
        
        # Run quantization 5 times
        compression_ratios = []
        quantization_times = []
        
        for run in range(5):
            # Reset quantizer stats for clean measurement
            quantizer.stats = {
                "vectors_processed": 0,
                "compression_ratio": 0.0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "quantization_errors": 0
            }
            
            start_time = time.time()
            result = quantizer.quantize_vectors(realistic_dataset)
            end_time = time.time()
            
            assert result.success, f"Run {run+1} failed: {result.error}"
            
            quantization_times.append(end_time - start_time)
            compression_ratios.append(result.metadata.get("compression_ratio", 0))
            
            print(f"Run {run+1}: {compression_ratios[-1]:.1f}x compression, {quantization_times[-1]:.2f}s")
        
        # Calculate variance
        compression_variance = (np.std(compression_ratios) / np.mean(compression_ratios)) * 100
        time_variance = (np.std(quantization_times) / np.mean(quantization_times)) * 100
        
        print(f"Consistency analysis:")
        print(f"  Compression ratio: {np.mean(compression_ratios):.1f}x ± {compression_variance:.1f}%")
        print(f"  Time: {np.mean(quantization_times):.2f}s ± {time_variance:.1f}%")
        
        # ROADMAP EXIT CRITERIA: <5% variance
        assert compression_variance < 5, f"Compression variance {compression_variance:.1f}% exceeds 5%"
        assert time_variance < 5, f"Time variance {time_variance:.1f}% exceeds 5%"
    
    @pytest.mark.asyncio
    async def test_concurrent_quantization_stress(self, memory_optimized_config):
        """Test quantization under concurrent load"""
        quantizer = VectorQuantizer(memory_optimized_config)
        
        # Create multiple datasets for concurrent processing
        datasets = []
        for i in range(5):
            vectors = np.random.randn(2000, 384).astype(np.float32)
            datasets.append(vectors)
        
        async def quantize_dataset(dataset, dataset_id):
            """Quantize a dataset asynchronously"""
            try:
                # Simulate async processing
                await asyncio.sleep(0.01)
                result = quantizer.quantize_vectors(dataset)
                return {"id": dataset_id, "success": result.success, "result": result}
            except Exception as e:
                return {"id": dataset_id, "success": False, "error": str(e)}
        
        # Run concurrent quantization
        start_time = time.time()
        tasks = [quantize_dataset(dataset, i) for i, dataset in enumerate(datasets)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify all succeeded
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        assert success_count == len(datasets), f"Only {success_count}/{len(datasets)} concurrent tasks succeeded"
        
        # Performance should be reasonable
        total_vectors = sum(len(d) for d in datasets)
        throughput = total_vectors / total_time
        print(f"Concurrent throughput: {throughput:.0f} vectors/sec across {len(datasets)} tasks")
        
        assert throughput > 5000, f"Concurrent throughput {throughput:.0f} too low"
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, memory_optimized_config):
        """Test quantization behavior under memory pressure"""
        quantizer = VectorQuantizer(memory_optimized_config)
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        
        # Process increasingly large datasets
        dataset_sizes = [1000, 5000, 10000, 20000]
        
        for size in dataset_sizes:
            vectors = np.random.randn(size, 768).astype(np.float32)
            
            result = quantizer.quantize_vectors(vectors)
            assert result.success, f"Failed on dataset size {size}: {result.error}"
            
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            print(f"Size {size}: Memory {current_memory:.1f}MB")
            
            # Clean up
            del vectors
            
        memory_growth = peak_memory - initial_memory
        print(f"Memory pressure test: {memory_growth:.1f}MB peak growth")
        
        # Memory growth should be reasonable
        assert memory_growth < 500, f"Memory grew by {memory_growth:.1f}MB"
    
    def test_quantization_with_edge_cases(self, memory_optimized_config):
        """Test quantization with edge case vectors"""
        quantizer = VectorQuantizer(memory_optimized_config)
        
        # Test cases
        edge_cases = {
            "zeros": np.zeros((100, 384)),
            "ones": np.ones((100, 384)), 
            "alternating": np.tile([1, -1], (100, 192)),
            "large_values": np.random.randn(100, 384) * 1000,
            "small_values": np.random.randn(100, 384) * 0.001,
            "mixed_range": np.concatenate([
                np.random.randn(50, 384) * 1000,  # Large values
                np.random.randn(50, 384) * 0.001  # Small values
            ])
        }
        
        for case_name, vectors in edge_cases.items():
            vectors = vectors.astype(np.float32)
            result = quantizer.quantize_vectors(vectors)
            
            print(f"Edge case '{case_name}': {'✓' if result.success else '✗'}")
            
            if result.success:
                assert result.quantized_vectors is not None
                assert result.metadata.get("compression_ratio", 0) > 0
            else:
                # Some edge cases might reasonably fail
                assert "edge case" in result.error.lower() or "invalid" in result.error.lower()
    
    def test_quantization_metadata_completeness(self, accuracy_optimized_config, realistic_dataset):
        """Test that quantization metadata is complete and accurate"""
        quantizer = VectorQuantizer(accuracy_optimized_config)
        
        result = quantizer.quantize_vectors(realistic_dataset)
        assert result.success
        
        # Required metadata fields
        required_fields = [
            "quantization_type",
            "compression_ratio", 
            "original_shape",
            "quantized_size",
            "processing_time"
        ]
        
        for field in required_fields:
            assert field in result.metadata, f"Missing required metadata field: {field}"
            
        # Validate metadata values
        assert result.metadata["quantization_type"] == accuracy_optimized_config.quantization_type.value
        assert result.metadata["compression_ratio"] > 0
        assert result.metadata["original_shape"] == list(realistic_dataset.shape)
        assert result.metadata["quantized_size"] > 0
        assert result.metadata["processing_time"] > 0
        
        print(f"Metadata validation passed:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
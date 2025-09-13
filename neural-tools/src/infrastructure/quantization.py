#!/usr/bin/env python3
"""
Vector Quantization for Neural Tools
Implements binary and scalar quantization to reduce memory usage and improve performance
Following roadmap Phase 1.4 specifications for 2025 production standards
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Supported quantization types"""
    NONE = "none"
    BINARY = "binary"  # 1 bit per dimension
    SCALAR_8BIT = "scalar_8bit"  # 8 bits per dimension
    SCALAR_4BIT = "scalar_4bit"  # 4 bits per dimension (experimental)

@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    quantization_type: QuantizationType = QuantizationType.BINARY
    enable_memory_mapping: bool = True
    compression_threshold: int = 1000  # Only quantize collections with >1000 vectors
    preserve_original: bool = False    # Keep original vectors alongside quantized
    batch_size: int = 1000            # Batch size for quantization operations

@dataclass
class QuantizationStats:
    """Statistics for quantization performance"""
    vectors_processed: int = 0
    compression_ratio: float = 0.0
    total_original_size: int = 0
    total_compressed_size: int = 0
    quantization_errors: int = 0
    processing_time: float = 0.0

@dataclass
class QuantizationResult:
    """Result of quantization operation"""
    success: bool
    quantized_vectors: Optional[Union[bytes, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class VectorQuantizer:
    """
    High-performance vector quantization for embeddings
    Reduces memory usage by 8-32x while maintaining search accuracy
    """
    
    def __init__(self, config: QuantizationConfig = None):
        """
        Initialize quantizer with configuration
        
        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self.stats = {
            'vectors_processed': 0,
            'compression_ratio': 0.0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'quantization_errors': 0
        }
        
        logger.info(f"Initialized vector quantizer: {self.config.quantization_type.value}")
    
    def quantize_vectors(self, vectors: np.ndarray) -> QuantizationResult:
        """
        Quantize multiple vectors at once
        
        Args:
            vectors: Array of vectors to quantize (N, D)
            
        Returns:
            QuantizationResult with quantized data and metadata
        """
        if vectors is None:
            return QuantizationResult(success=False, error="Input vectors cannot be None")
            
        if len(vectors.shape) != 2:
            return QuantizationResult(success=False, error="Input must be 2D array (N, D)")
            
        if len(vectors) == 0:
            return QuantizationResult(success=False, error="Input vectors array is empty")
        
        try:
            start_time = time.time()
            original_size = vectors.nbytes
            
            # Check compression threshold
            if len(vectors) < self.config.compression_threshold:
                metadata = {
                    "below_threshold": True,
                    "quantization_type": self.config.quantization_type.value,
                    "original_shape": list(vectors.shape),
                    "compression_ratio": 1.0,
                    "processing_time": time.time() - start_time,
                    "quantized_size": original_size
                }
                return QuantizationResult(
                    success=True,
                    quantized_vectors=vectors.copy(),
                    metadata=metadata
                )
            
            # Process in batches
            quantized_batches = []
            batch_metadata = []
            batches_processed = 0
            
            for i in range(0, len(vectors), self.config.batch_size):
                batch = vectors[i:i + self.config.batch_size]
                
                if self.config.quantization_type == QuantizationType.BINARY:
                    quantized_batch, metadata = self._quantize_binary_batch(batch)
                elif self.config.quantization_type == QuantizationType.SCALAR_8BIT:
                    quantized_batch, metadata = self._quantize_scalar_8bit_batch(batch)
                else:
                    return QuantizationResult(
                        success=False, 
                        error=f"Unsupported quantization type: {self.config.quantization_type}"
                    )
                
                quantized_batches.append(quantized_batch)
                batch_metadata.append(metadata)
                batches_processed += 1
            
            # Combine batches
            if self.config.quantization_type == QuantizationType.BINARY:
                combined_quantized = b''.join(quantized_batches)
            else:
                combined_quantized = np.concatenate(quantized_batches)
            
            # Calculate final metrics
            quantized_size = len(combined_quantized) if isinstance(combined_quantized, bytes) else combined_quantized.nbytes
            compression_ratio = original_size / quantized_size
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['vectors_processed'] += len(vectors)
            self.stats['total_original_size'] += original_size
            self.stats['total_compressed_size'] += quantized_size
            self.stats['compression_ratio'] = self.stats['total_original_size'] / max(1, self.stats['total_compressed_size'])
            
            metadata = {
                "quantization_type": self.config.quantization_type.value,
                "original_shape": list(vectors.shape),
                "compression_ratio": compression_ratio,
                "processing_time": processing_time,
                "quantized_size": quantized_size,
                "batches_processed": batches_processed,
                "batch_metadata": batch_metadata
            }
            
            return QuantizationResult(
                success=True,
                quantized_vectors=combined_quantized,
                metadata=metadata
            )
            
        except Exception as e:
            self.stats['quantization_errors'] += 1
            logger.error(f"Quantization error: {e}")
            return QuantizationResult(success=False, error=str(e))
    
    def dequantize_vectors(self, quantized_data: Union[bytes, np.ndarray], metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Dequantize vectors back to original format
        
        Args:
            quantized_data: Quantized vector data
            metadata: Metadata from quantization
            
        Returns:
            Dequantized vectors array or None on error
        """
        try:
            quantization_type = metadata.get("quantization_type")
            original_shape = metadata.get("original_shape")
            
            if not original_shape:
                logger.error("Missing original_shape in metadata")
                return None
                
            if quantization_type == QuantizationType.BINARY.value:
                return self._dequantize_binary_batch(quantized_data, metadata)
            elif quantization_type == QuantizationType.SCALAR_8BIT.value:
                return self._dequantize_scalar_8bit_batch(quantized_data, metadata)
            else:
                logger.error(f"Unsupported quantization type: {quantization_type}")
                return None
                
        except Exception as e:
            logger.error(f"Dequantization error: {e}")
            return None
    
    def _quantize_binary_batch(self, vectors: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Quantize batch of vectors using binary quantization"""
        # Simple binary quantization: >0 -> 1, <=0 -> 0
        binary_vectors = (vectors > 0).astype(np.uint8)
        
        # Pack bits
        packed_data = np.packbits(binary_vectors, axis=1)
        
        metadata = {
            "batch_shape": list(vectors.shape),
            "packed_shape": list(packed_data.shape)
        }
        
        return packed_data.tobytes(), metadata
    
    def _dequantize_binary_batch(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize binary batch"""
        # Get shape info from metadata hierarchy
        batch_metadata = metadata.get("batch_metadata", [])
        original_shape = metadata.get("original_shape")
        
        if batch_metadata and len(batch_metadata) > 0:
            # Use batch metadata for multi-batch reconstruction
            all_unpacked = []
            byte_offset = 0
            
            for batch_meta in batch_metadata:
                batch_shape = batch_meta.get("batch_shape")
                packed_shape = batch_meta.get("packed_shape")
                
                if not batch_shape or not packed_shape:
                    continue
                    
                # Calculate size for this batch
                batch_size = np.prod(packed_shape)
                batch_data = data[byte_offset:byte_offset + batch_size]
                byte_offset += batch_size
                
                # Reconstruct this batch
                packed_array = np.frombuffer(batch_data, dtype=np.uint8).reshape(packed_shape)
                unpacked = np.unpackbits(packed_array, axis=1)
                unpacked = unpacked[:, :batch_shape[1]]
                all_unpacked.append(unpacked)
            
            if all_unpacked:
                combined = np.concatenate(all_unpacked, axis=0)
                return (combined.astype(np.float32) * 2) - 1
        
        # Fallback for single batch or legacy metadata
        if original_shape:
            # Estimate packed shape from original
            packed_dim = (original_shape[1] + 7) // 8  # Round up for bit packing
            packed_shape = [original_shape[0], packed_dim]
            
            packed_array = np.frombuffer(data, dtype=np.uint8).reshape(packed_shape)
            unpacked = np.unpackbits(packed_array, axis=1)
            unpacked = unpacked[:, :original_shape[1]]
            
            return (unpacked.astype(np.float32) * 2) - 1
        
        raise ValueError("Missing shape information in metadata")
    
    def _quantize_scalar_8bit_batch(self, vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize batch using 8-bit scalar quantization"""
        # Calculate min/max for scaling
        min_vals = np.min(vectors, axis=1, keepdims=True)
        max_vals = np.max(vectors, axis=1, keepdims=True)
        
        # Avoid division by zero
        ranges = np.maximum(max_vals - min_vals, 1e-8)
        
        # Scale to [0, 255]
        scaled = ((vectors - min_vals) / ranges * 255).round()
        quantized = np.clip(scaled, 0, 255).astype(np.uint8)
        
        metadata = {
            "batch_shape": list(vectors.shape),
            "min_vals": min_vals.flatten().tolist(),
            "max_vals": max_vals.flatten().tolist()
        }
        
        return quantized, metadata
    
    def _dequantize_scalar_8bit_batch(self, data: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Dequantize 8-bit scalar batch"""
        batch_metadata = metadata.get("batch_metadata", [])
        
        if batch_metadata and len(batch_metadata) > 0:
            # Multi-batch scenario
            all_dequantized = []
            row_offset = 0
            
            for batch_meta in batch_metadata:
                batch_shape = batch_meta.get("batch_shape")
                min_vals = np.array(batch_meta.get("min_vals", [])).reshape(-1, 1)
                max_vals = np.array(batch_meta.get("max_vals", [])).reshape(-1, 1)
                
                if not batch_shape or len(min_vals) == 0 or len(max_vals) == 0:
                    continue
                
                batch_rows = batch_shape[0]
                batch_data = data[row_offset:row_offset + batch_rows]
                row_offset += batch_rows
                
                ranges = max_vals - min_vals
                scaled = batch_data.astype(np.float32) / 255.0
                dequantized_batch = scaled * ranges + min_vals
                all_dequantized.append(dequantized_batch)
            
            if all_dequantized:
                return np.concatenate(all_dequantized, axis=0)
        
        # Fallback for single batch
        min_vals = np.array(metadata.get("min_vals", [])).reshape(-1, 1)
        max_vals = np.array(metadata.get("max_vals", [])).reshape(-1, 1)
        
        if len(min_vals) == 0 or len(max_vals) == 0:
            raise ValueError("Missing min/max values in metadata")
        
        ranges = max_vals - min_vals
        scaled = data.astype(np.float32) / 255.0
        return scaled * ranges + min_vals

    def quantize_vector(self, vector: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """
        Quantize a single vector
        
        Args:
            vector: Input vector (float32)
            
        Returns:
            Tuple of (quantized_bytes, metadata)
        """
        if self.config.quantization_type == QuantizationType.BINARY:
            return self._quantize_binary(vector)
        elif self.config.quantization_type == QuantizationType.SCALAR_8BIT:
            return self._quantize_scalar_8bit(vector)
        elif self.config.quantization_type == QuantizationType.SCALAR_4BIT:
            return self._quantize_scalar_4bit(vector)
        else:
            # No quantization
            return vector.tobytes(), {"type": "none", "shape": vector.shape}
    
    def dequantize_vector(self, quantized_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct vector from quantized data
        
        Args:
            quantized_data: Quantized vector bytes
            metadata: Quantization metadata
            
        Returns:
            Reconstructed vector
        """
        q_type = metadata.get("type", "none")
        
        if q_type == "binary":
            return self._dequantize_binary(quantized_data, metadata)
        elif q_type == "scalar_8bit":
            return self._dequantize_scalar_8bit(quantized_data, metadata)
        elif q_type == "scalar_4bit":
            return self._dequantize_scalar_4bit(quantized_data, metadata)
        else:
            # No quantization - direct reconstruction
            shape = metadata.get("shape", (-1,))
            return np.frombuffer(quantized_data, dtype=np.float32).reshape(shape)
    
    def _quantize_binary(self, vector: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Binary quantization - 1 bit per dimension"""
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            norm = 1.0
        
        normalized = vector / norm
        
        # Binary quantization: sign bit
        binary_bits = (normalized >= 0).astype(np.uint8)
        
        # Pack bits efficiently
        # Pad to multiple of 8
        padded_length = ((len(binary_bits) + 7) // 8) * 8
        padded_bits = np.zeros(padded_length, dtype=np.uint8)
        padded_bits[:len(binary_bits)] = binary_bits
        
        # Pack 8 bits into each byte
        packed_bytes = np.packbits(padded_bits).tobytes()
        
        metadata = {
            "type": "binary",
            "shape": vector.shape,
            "norm": float(norm),
            "original_length": len(vector)
        }
        
        # Track compression ratio
        original_size = vector.nbytes
        compressed_size = len(packed_bytes)
        self.stats['memory_saved_bytes'] += original_size - compressed_size
        
        return packed_bytes, metadata
    
    def _dequantize_binary(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct from binary quantization"""
        # Unpack bits
        packed_array = np.frombuffer(data, dtype=np.uint8)
        unpacked_bits = np.unpackbits(packed_array)
        
        # Trim to original length
        original_length = metadata["original_length"]
        binary_vector = unpacked_bits[:original_length]
        
        # Convert back to float: 0 -> -1, 1 -> 1
        reconstructed = (binary_vector.astype(np.float32) * 2) - 1
        
        # Restore magnitude
        norm = metadata["norm"]
        reconstructed = reconstructed * norm
        
        return reconstructed.reshape(metadata["shape"])
    
    def _quantize_scalar_8bit(self, vector: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """8-bit scalar quantization"""
        # Find min/max for scaling
        v_min = float(np.min(vector))
        v_max = float(np.max(vector))
        
        if v_max == v_min:
            # Constant vector
            quantized = np.zeros(len(vector), dtype=np.uint8)
        else:
            # Scale to [0, 255]
            scale = 255.0 / (v_max - v_min)
            quantized = np.round((vector - v_min) * scale).astype(np.uint8)
        
        metadata = {
            "type": "scalar_8bit",
            "shape": vector.shape,
            "min": v_min,
            "max": v_max
        }
        
        # Track compression
        original_size = vector.nbytes
        compressed_size = quantized.nbytes
        self.stats['memory_saved_bytes'] += original_size - compressed_size
        
        return quantized.tobytes(), metadata
    
    def _dequantize_scalar_8bit(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct from 8-bit scalar quantization"""
        quantized = np.frombuffer(data, dtype=np.uint8)
        
        # Restore original range
        v_min = metadata["min"]
        v_max = metadata["max"]
        
        if v_max == v_min:
            reconstructed = np.full_like(quantized, v_min, dtype=np.float32)
        else:
            scale = (v_max - v_min) / 255.0
            reconstructed = quantized.astype(np.float32) * scale + v_min
        
        return reconstructed.reshape(metadata["shape"])
    
    def _quantize_scalar_4bit(self, vector: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """4-bit scalar quantization (experimental)"""
        # Find min/max for scaling
        v_min = float(np.min(vector))
        v_max = float(np.max(vector))
        
        if v_max == v_min:
            # Constant vector - pack as zeros
            quantized_4bit = np.zeros(len(vector), dtype=np.uint8)
        else:
            # Scale to [0, 15] (4 bits)
            scale = 15.0 / (v_max - v_min)
            quantized_4bit = np.round((vector - v_min) * scale).astype(np.uint8)
            quantized_4bit = np.clip(quantized_4bit, 0, 15)
        
        # Pack two 4-bit values into each byte
        # Pad to even length
        if len(quantized_4bit) % 2 == 1:
            quantized_4bit = np.append(quantized_4bit, 0)
        
        # Pack pairs
        packed = []
        for i in range(0, len(quantized_4bit), 2):
            byte_val = (quantized_4bit[i] << 4) | quantized_4bit[i + 1]
            packed.append(byte_val)
        
        packed_bytes = bytes(packed)
        
        metadata = {
            "type": "scalar_4bit",
            "shape": vector.shape,
            "min": v_min,
            "max": v_max,
            "original_length": len(vector)
        }
        
        # Track compression
        original_size = vector.nbytes
        compressed_size = len(packed_bytes)
        self.stats['memory_saved_bytes'] += original_size - compressed_size
        
        return packed_bytes, metadata
    
    def _dequantize_scalar_4bit(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct from 4-bit scalar quantization"""
        # Unpack pairs
        unpacked = []
        for byte_val in data:
            high_nibble = (byte_val >> 4) & 0x0F
            low_nibble = byte_val & 0x0F
            unpacked.extend([high_nibble, low_nibble])
        
        # Trim to original length
        original_length = metadata["original_length"]
        quantized_4bit = np.array(unpacked[:original_length], dtype=np.uint8)
        
        # Restore original range
        v_min = metadata["min"]
        v_max = metadata["max"]
        
        if v_max == v_min:
            reconstructed = np.full_like(quantized_4bit, v_min, dtype=np.float32)
        else:
            scale = (v_max - v_min) / 15.0
            reconstructed = quantized_4bit.astype(np.float32) * scale + v_min
        
        return reconstructed.reshape(metadata["shape"])
    
    async def quantize_batch(self, vectors: List[np.ndarray]) -> List[Tuple[bytes, Dict[str, Any]]]:
        """
        Quantize a batch of vectors efficiently
        
        Args:
            vectors: List of input vectors
            
        Returns:
            List of (quantized_data, metadata) tuples
        """
        start_time = time.time()
        
        # Process in batches to avoid memory issues
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            # Process batch
            for vector in batch:
                quantized_data, metadata = self.quantize_vector(vector)
                results.append((quantized_data, metadata))
            
            # Allow other async operations
            await asyncio.sleep(0)
        
        # Update stats
        self.stats['vectors_quantized'] += len(vectors)
        self.stats['quantization_time_ms'] += (time.time() - start_time) * 1000
        
        logger.info(f"Quantized {len(vectors)} vectors in {self.stats['quantization_time_ms']:.1f}ms")
        
        return results
    
    def calculate_compression_ratio(self, original_vectors: List[np.ndarray]) -> Dict[str, float]:
        """Calculate compression statistics"""
        if not original_vectors:
            return {}
        
        # Calculate original size
        original_size = sum(v.nbytes for v in original_vectors)
        
        # Estimate compressed size
        sample_vector = original_vectors[0]
        quantized_data, metadata = self.quantize_vector(sample_vector)
        compressed_sample_size = len(quantized_data)
        
        # Estimate total compressed size
        estimated_compressed_size = compressed_sample_size * len(original_vectors)
        
        compression_ratio = original_size / max(estimated_compressed_size, 1)
        memory_saved_mb = (original_size - estimated_compressed_size) / (1024 * 1024)
        
        return {
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': estimated_compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'memory_saved_mb': memory_saved_mb,
            'compression_percent': ((original_size - estimated_compressed_size) / original_size) * 100
        }

class QuantizedVectorStore:
    """
    Vector store with transparent quantization support
    Integrates with existing Qdrant collections
    """
    
    def __init__(self, quantizer: VectorQuantizer, qdrant_service=None):
        """
        Initialize quantized vector store
        
        Args:
            quantizer: Vector quantizer instance
            qdrant_service: Optional Qdrant service for integration
        """
        self.quantizer = quantizer
        self.qdrant = qdrant_service
        self.collections: Dict[str, Dict[str, Any]] = {}  # collection_name -> metadata
        
    async def create_quantized_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Create collection with quantization support
        
        Args:
            collection_name: Name of collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric to use
            
        Returns:
            Creation result
        """
        try:
            # Store collection metadata
            self.collections[collection_name] = {
                'vector_size': vector_size,
                'distance_metric': distance_metric,
                'quantization_type': self.quantizer.config.quantization_type.value,
                'created_at': time.time(),
                'vector_count': 0,
                'quantization_enabled': True
            }
            
            # Create Qdrant collection if service available
            result = {"status": "success", "collection_name": collection_name}
            
            if self.qdrant:
                # Create with optimized settings for quantized vectors
                qdrant_result = await self.qdrant.create_quantized_collection(
                    collection_name,
                    vector_size,
                    quantization_config={
                        'type': self.quantizer.config.quantization_type.value,
                        'memory_mapping': self.quantizer.config.enable_memory_mapping
                    }
                )
                result['qdrant_result'] = qdrant_result
            
            logger.info(f"Created quantized collection: {collection_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create quantized collection: {e}")
            return {"status": "error", "message": str(e)}
    
    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]] = None,
        ids: List[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Add vectors with automatic quantization
        
        Args:
            collection_name: Target collection
            vectors: List of vectors to add
            payloads: Optional metadata for each vector
            ids: Optional IDs for vectors
            
        Returns:
            Addition result with quantization stats
        """
        if collection_name not in self.collections:
            return {"status": "error", "message": f"Collection {collection_name} not found"}
        
        try:
            start_time = time.time()
            
            # Quantize vectors
            quantized_results = await self.quantizer.quantize_batch(vectors)
            
            # Prepare for storage
            quantized_vectors = []
            quantization_metadata = []
            
            for (quantized_data, metadata), original_vector in zip(quantized_results, vectors):
                quantized_vectors.append(quantized_data)
                quantization_metadata.append(metadata)
                
                # Optionally store original vector
                if self.quantizer.config.preserve_original:
                    metadata['original_vector'] = original_vector.tolist()
            
            # Store in Qdrant if available
            qdrant_result = None
            if self.qdrant:
                # Convert quantized data back to vectors for Qdrant
                # In production, Qdrant would handle quantized storage directly
                reconstructed_vectors = [
                    self.quantizer.dequantize_vector(data, meta)
                    for data, meta in zip(quantized_vectors, quantization_metadata)
                ]
                
                qdrant_result = await self.qdrant.upsert_vectors(
                    collection_name=collection_name,
                    vectors=reconstructed_vectors,
                    payloads=payloads,
                    ids=ids
                )
            
            # Update collection stats
            self.collections[collection_name]['vector_count'] += len(vectors)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate compression stats
            compression_stats = self.quantizer.calculate_compression_ratio(vectors)
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "vectors_added": len(vectors),
                "processing_time_ms": processing_time,
                "quantization_type": self.quantizer.config.quantization_type.value,
                "compression_stats": compression_stats,
                "qdrant_result": qdrant_result
            }
            
        except Exception as e:
            logger.error(f"Failed to add quantized vectors: {e}")
            return {"status": "error", "message": str(e)}
    
    async def search_quantized(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        filter_conditions: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Search with quantized vectors for improved performance
        
        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Number of results
            filter_conditions: Optional filter
            
        Returns:
            Search results with performance metrics
        """
        if collection_name not in self.collections:
            return {"status": "error", "message": f"Collection {collection_name} not found"}
        
        try:
            start_time = time.time()
            
            # Quantize query vector for consistency
            quantized_query, query_metadata = self.quantizer.quantize_vector(query_vector)
            reconstructed_query = self.quantizer.dequantize_vector(quantized_query, query_metadata)
            
            # Search using Qdrant
            results = []
            if self.qdrant:
                search_result = await self.qdrant.search_vectors(
                    collection_name=collection_name,
                    query_vector=reconstructed_query,
                    limit=limit,
                    filter=filter_conditions
                )
                results = search_result
            
            search_time = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "query_quantization": query_metadata['type'],
                "results": results,
                "search_time_ms": search_time,
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Quantized search failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get comprehensive quantization statistics"""
        return {
            'quantizer_stats': self.quantizer.stats,
            'collections': self.collections,
            'total_collections': len(self.collections),
            'total_vectors': sum(col.get('vector_count', 0) for col in self.collections.values()),
            'quantization_type': self.quantizer.config.quantization_type.value,
            'memory_mapping_enabled': self.quantizer.config.enable_memory_mapping
        }

# Factory functions
def create_binary_quantizer() -> VectorQuantizer:
    """Create quantizer for binary quantization (highest compression)"""
    config = QuantizationConfig(
        quantization_type=QuantizationType.BINARY,
        enable_memory_mapping=True,
        compression_threshold=1000
    )
    return VectorQuantizer(config)

def create_8bit_quantizer() -> VectorQuantizer:
    """Create quantizer for 8-bit scalar quantization (balanced)"""
    config = QuantizationConfig(
        quantization_type=QuantizationType.SCALAR_8BIT,
        enable_memory_mapping=True,
        compression_threshold=1000
    )
    return VectorQuantizer(config)

def create_4bit_quantizer() -> VectorQuantizer:
    """Create quantizer for 4-bit scalar quantization (experimental)"""
    config = QuantizationConfig(
        quantization_type=QuantizationType.SCALAR_4BIT,
        enable_memory_mapping=True,
        compression_threshold=500  # Lower threshold due to higher compression
    )
    return VectorQuantizer(config)

# Example usage and testing
async def test_quantization():
    """Test vector quantization system"""
    print("🔬 Testing vector quantization...")
    
    # Generate test vectors
    np.random.seed(42)
    test_vectors = [np.random.randn(768).astype(np.float32) for _ in range(100)]
    
    # Test different quantization types
    quantizers = {
        'binary': create_binary_quantizer(),
        '8bit': create_8bit_quantizer(),
        '4bit': create_4bit_quantizer()
    }
    
    for name, quantizer in quantizers.items():
        print(f"\n   Testing {name} quantization:")
        
        # Test single vector
        original = test_vectors[0]
        quantized_data, metadata = quantizer.quantize_vector(original)
        reconstructed = quantizer.dequantize_vector(quantized_data, metadata)
        
        # Calculate accuracy
        mse = np.mean((original - reconstructed) ** 2)
        compression_ratio = original.nbytes / len(quantized_data)
        
        print(f"     Compression: {compression_ratio:.1f}x")
        print(f"     MSE: {mse:.6f}")
        print(f"     Quantized size: {len(quantized_data)} bytes")
        
        # Test batch quantization
        start_time = time.time()
        batch_results = await quantizer.quantize_batch(test_vectors)
        batch_time = (time.time() - start_time) * 1000
        
        print(f"     Batch time: {batch_time:.1f}ms for {len(test_vectors)} vectors")
        
        # Calculate compression stats
        compression_stats = quantizer.calculate_compression_ratio(test_vectors)
        print(f"     Memory saved: {compression_stats['memory_saved_mb']:.1f}MB")
        print(f"     Compression: {compression_stats['compression_percent']:.1f}%")
    
    print("\n✅ Quantization test completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_quantization())
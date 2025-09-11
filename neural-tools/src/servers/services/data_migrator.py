#!/usr/bin/env python3
"""
Data Migration Support for GraphRAG
Handles data transformation during schema migrations
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import asyncio
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


@dataclass
class DataTransformation:
    """Defines a data transformation operation"""
    source_type: str
    target_type: str
    transform_func: Callable
    batch_size: int = 1000
    description: str = ""


class DataMigrator:
    """Handles data transformation during schema migrations"""
    
    def __init__(self, neo4j_service=None, qdrant_service=None):
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
    
    async def transform_nodes(
        self,
        node_type: str,
        transformation: Callable[[Dict], Dict],
        batch_size: int = 1000
    ) -> int:
        """
        Transform existing nodes during migration
        
        Args:
            node_type: Type of nodes to transform
            transformation: Function to transform node properties
            batch_size: Number of nodes to process at once
            
        Returns:
            Number of nodes transformed
        """
        if not self.neo4j:
            logger.warning("Neo4j service not available for node transformation")
            return 0
        
        total_transformed = 0
        offset = 0
        
        while True:
            # Fetch batch of nodes
            query = f"""
            MATCH (n:{node_type})
            RETURN n, id(n) as node_id
            SKIP {offset}
            LIMIT {batch_size}
            """
            
            try:
                result = await self.neo4j.execute_query(query)
                
                if not result:
                    break
                
                # Transform each node
                updates = []
                for record in result:
                    node = record["n"]
                    node_id = record["node_id"]
                    
                    # Apply transformation
                    try:
                        new_props = transformation(dict(node))
                        updates.append((node_id, new_props))
                    except Exception as e:
                        logger.error(f"Failed to transform node {node_id}: {e}")
                        continue
                
                # Batch update nodes
                if updates:
                    await self.batch_update_nodes(node_type, updates)
                    total_transformed += len(updates)
                
                # Check if we've processed all nodes
                if len(result) < batch_size:
                    break
                
                offset += batch_size
                
            except Exception as e:
                logger.error(f"Error during node transformation: {e}")
                break
        
        logger.info(f"Transformed {total_transformed} nodes of type {node_type}")
        return total_transformed
    
    async def batch_update_nodes(self, node_type: str, updates: List[Tuple[int, Dict]]):
        """Batch update nodes with new properties"""
        if not self.neo4j:
            return
        
        # Build batch update query
        query = f"""
        UNWIND $updates AS update
        MATCH (n:{node_type})
        WHERE id(n) = update.node_id
        SET n += update.properties
        """
        
        params = {
            "updates": [
                {"node_id": node_id, "properties": props}
                for node_id, props in updates
            ]
        }
        
        await self.neo4j.execute_query(query, params)
    
    async def add_property_to_nodes(
        self,
        node_type: str,
        property_name: str,
        default_value: Any = None,
        value_generator: Optional[Callable[[Dict], Any]] = None
    ) -> int:
        """
        Add a new property to all nodes of a type
        
        Args:
            node_type: Type of nodes to update
            property_name: Name of the new property
            default_value: Default value for the property
            value_generator: Optional function to generate value based on node
            
        Returns:
            Number of nodes updated
        """
        def transform(node: Dict) -> Dict:
            if value_generator:
                value = value_generator(node)
            else:
                value = default_value
            
            return {property_name: value}
        
        return await self.transform_nodes(node_type, transform)
    
    async def remove_property_from_nodes(
        self,
        node_type: str,
        property_name: str
    ) -> int:
        """Remove a property from all nodes of a type"""
        if not self.neo4j:
            return 0
        
        query = f"""
        MATCH (n:{node_type})
        WHERE n.{property_name} IS NOT NULL
        REMOVE n.{property_name}
        RETURN count(n) as count
        """
        
        result = await self.neo4j.execute_query(query)
        count = result[0]["count"] if result else 0
        
        logger.info(f"Removed property {property_name} from {count} nodes")
        return count
    
    async def rename_node_property(
        self,
        node_type: str,
        old_name: str,
        new_name: str
    ) -> int:
        """Rename a property on all nodes of a type"""
        if not self.neo4j:
            return 0
        
        query = f"""
        MATCH (n:{node_type})
        WHERE n.{old_name} IS NOT NULL
        SET n.{new_name} = n.{old_name}
        REMOVE n.{old_name}
        RETURN count(n) as count
        """
        
        result = await self.neo4j.execute_query(query)
        count = result[0]["count"] if result else 0
        
        logger.info(f"Renamed property {old_name} to {new_name} on {count} nodes")
        return count
    
    async def migrate_relationships(
        self,
        old_type: str,
        new_type: str,
        transform_properties: Optional[Callable[[Dict], Dict]] = None
    ) -> int:
        """
        Migrate relationships from one type to another
        
        Args:
            old_type: Current relationship type
            new_type: New relationship type
            transform_properties: Optional function to transform relationship properties
            
        Returns:
            Number of relationships migrated
        """
        if not self.neo4j:
            return 0
        
        if transform_properties:
            # Complex migration with property transformation
            query = f"""
            MATCH (a)-[r:{old_type}]->(b)
            WITH a, r, b, properties(r) as props
            CREATE (a)-[new:{new_type}]->(b)
            SET new = $transform(props)
            DELETE r
            RETURN count(new) as count
            """
            # Note: This would need custom function support in Neo4j
            # For now, we'll do it in batches
            return await self._migrate_relationships_with_transform(
                old_type, new_type, transform_properties
            )
        else:
            # Simple rename
            query = f"""
            MATCH (a)-[r:{old_type}]->(b)
            CREATE (a)-[new:{new_type}]->(b)
            SET new = r
            DELETE r
            RETURN count(new) as count
            """
            
            result = await self.neo4j.execute_query(query)
            count = result[0]["count"] if result else 0
            
            logger.info(f"Migrated {count} relationships from {old_type} to {new_type}")
            return count
    
    async def _migrate_relationships_with_transform(
        self,
        old_type: str,
        new_type: str,
        transform_func: Callable[[Dict], Dict],
        batch_size: int = 1000
    ) -> int:
        """Migrate relationships with property transformation in batches"""
        total_migrated = 0
        
        while True:
            # Get batch of relationships
            query = f"""
            MATCH (a)-[r:{old_type}]->(b)
            RETURN id(a) as start_id, id(b) as end_id, 
                   properties(r) as props, id(r) as rel_id
            LIMIT {batch_size}
            """
            
            result = await self.neo4j.execute_query(query)
            
            if not result:
                break
            
            # Process each relationship
            for record in result:
                try:
                    # Transform properties
                    new_props = transform_func(record["props"])
                    
                    # Create new relationship and delete old
                    migration_query = f"""
                    MATCH (a) WHERE id(a) = $start_id
                    MATCH (b) WHERE id(b) = $end_id
                    MATCH (a)-[r:{old_type}]->(b) WHERE id(r) = $rel_id
                    CREATE (a)-[new:{new_type}]->(b)
                    SET new = $props
                    DELETE r
                    """
                    
                    await self.neo4j.execute_query(migration_query, {
                        "start_id": record["start_id"],
                        "end_id": record["end_id"],
                        "rel_id": record["rel_id"],
                        "props": new_props
                    })
                    
                    total_migrated += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate relationship: {e}")
            
            if len(result) < batch_size:
                break
        
        return total_migrated
    
    async def migrate_vectors(
        self,
        old_collection: str,
        new_collection: str,
        field_mapping: Optional[Dict[str, str]] = None,
        field_transformer: Optional[Callable[[str, Any], Any]] = None,
        batch_size: int = 100
    ) -> int:
        """
        Migrate vectors between collections with optional field remapping
        
        Args:
            old_collection: Source collection name
            new_collection: Target collection name
            field_mapping: Map of old field names to new field names
            field_transformer: Function to transform field values
            batch_size: Number of vectors to process at once
            
        Returns:
            Number of vectors migrated
        """
        if not self.qdrant:
            logger.warning("Qdrant service not available for vector migration")
            return 0
        
        total_migrated = 0
        offset = None
        
        try:
            # Ensure new collection exists
            collections = await self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if new_collection not in collection_names:
                logger.error(f"Target collection {new_collection} does not exist")
                return 0
            
            while True:
                # Scroll through old collection
                records, next_offset = await self.qdrant.scroll(
                    collection_name=old_collection,
                    offset=offset,
                    limit=batch_size,
                    with_vectors=True
                )
                
                if not records:
                    break
                
                # Transform records
                new_records = []
                for record in records:
                    # Build new payload
                    new_payload = {}
                    
                    if field_mapping:
                        # Apply field mapping
                        for old_field, new_field in field_mapping.items():
                            if old_field in record.payload:
                                value = record.payload[old_field]
                                
                                # Apply transformer if provided
                                if field_transformer:
                                    value = field_transformer(old_field, value)
                                
                                new_payload[new_field] = value
                    else:
                        # Copy all fields
                        new_payload = record.payload.copy()
                        
                        # Apply transformer to all fields if provided
                        if field_transformer:
                            for field, value in new_payload.items():
                                new_payload[field] = field_transformer(field, value)
                    
                    # Create new point
                    new_records.append(PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=new_payload
                    ))
                
                # Insert into new collection
                if new_records:
                    await self.qdrant.upsert(
                        collection_name=new_collection,
                        points=new_records
                    )
                    total_migrated += len(new_records)
                
                # Update offset for next batch
                offset = next_offset
                
                if offset is None:
                    break
                    
        except Exception as e:
            logger.error(f"Error during vector migration: {e}")
        
        logger.info(f"Migrated {total_migrated} vectors from {old_collection} to {new_collection}")
        return total_migrated
    
    async def add_field_to_vectors(
        self,
        collection_name: str,
        field_name: str,
        default_value: Any = None,
        value_generator: Optional[Callable[[Dict], Any]] = None,
        batch_size: int = 100
    ) -> int:
        """Add a new field to all vectors in a collection"""
        if not self.qdrant:
            return 0
        
        total_updated = 0
        offset = None
        
        while True:
            # Get batch of vectors
            records, next_offset = await self.qdrant.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=batch_size,
                with_payload=True
            )
            
            if not records:
                break
            
            # Update each record
            for record in records:
                # Calculate new field value
                if value_generator:
                    value = value_generator(record.payload)
                else:
                    value = default_value
                
                # Update payload
                new_payload = record.payload.copy()
                new_payload[field_name] = value
                
                # Update in Qdrant
                await self.qdrant.set_payload(
                    collection_name=collection_name,
                    points=[record.id],
                    payload=new_payload
                )
                
                total_updated += 1
            
            offset = next_offset
            if offset is None:
                break
        
        logger.info(f"Added field {field_name} to {total_updated} vectors")
        return total_updated
    
    async def remove_field_from_vectors(
        self,
        collection_name: str,
        field_name: str,
        batch_size: int = 100
    ) -> int:
        """Remove a field from all vectors in a collection"""
        if not self.qdrant:
            return 0
        
        total_updated = 0
        offset = None
        
        while True:
            # Get batch of vectors
            records, next_offset = await self.qdrant.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=batch_size,
                with_payload=True
            )
            
            if not records:
                break
            
            # Update each record
            for record in records:
                if field_name in record.payload:
                    # Remove field from payload
                    new_payload = record.payload.copy()
                    del new_payload[field_name]
                    
                    # Update in Qdrant
                    await self.qdrant.set_payload(
                        collection_name=collection_name,
                        points=[record.id],
                        payload=new_payload
                    )
                    
                    total_updated += 1
            
            offset = next_offset
            if offset is None:
                break
        
        logger.info(f"Removed field {field_name} from {total_updated} vectors")
        return total_updated
    
    async def reindex_collection(
        self,
        collection_name: str,
        embedding_service,
        text_field: str = "content",
        batch_size: int = 50
    ) -> int:
        """
        Reindex a collection by regenerating embeddings
        
        Args:
            collection_name: Collection to reindex
            embedding_service: Service to generate embeddings
            text_field: Field containing text to embed
            batch_size: Number of vectors to process at once
            
        Returns:
            Number of vectors reindexed
        """
        if not self.qdrant or not embedding_service:
            return 0
        
        total_reindexed = 0
        offset = None
        
        while True:
            # Get batch of vectors
            records, next_offset = await self.qdrant.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=batch_size,
                with_payload=True
            )
            
            if not records:
                break
            
            # Extract texts for embedding
            texts = []
            valid_records = []
            
            for record in records:
                if text_field in record.payload:
                    texts.append(record.payload[text_field])
                    valid_records.append(record)
            
            if texts:
                # Generate new embeddings
                try:
                    new_embeddings = await embedding_service.get_embeddings(texts)
                    
                    # Update vectors
                    updates = []
                    for i, (record, embedding) in enumerate(zip(valid_records, new_embeddings)):
                        updates.append(PointStruct(
                            id=record.id,
                            vector=embedding,
                            payload=record.payload
                        ))
                    
                    # Upsert updated vectors
                    await self.qdrant.upsert(
                        collection_name=collection_name,
                        points=updates
                    )
                    
                    total_reindexed += len(updates)
                    
                except Exception as e:
                    logger.error(f"Failed to reindex batch: {e}")
            
            offset = next_offset
            if offset is None:
                break
        
        logger.info(f"Reindexed {total_reindexed} vectors in {collection_name}")
        return total_reindexed
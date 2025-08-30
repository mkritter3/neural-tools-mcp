#!/usr/bin/env python3
"""
Safe database clearing utility for Neural MCP System
Based on expert analysis for maximum safety and schema preservation
"""

import sqlite3
import chromadb
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def clear_standard_sqlite_db(db_path: Path):
    """
    Safely deletes all data from all tables in a standard SQLite database.
    
    This function preserves the schema, handles foreign keys, and vacuums
    the database to reclaim disk space. The entire operation is transactional.
    """
    if not db_path.exists():
        logging.warning(f"Database file not found, skipping: {db_path}")
        return
    
    logging.info(f"Connecting to SQLite database: {db_path}")
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all user-defined table names
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'system_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            logging.info(f"Database '{db_path.name}' contains no user tables. Nothing to clear.")
            return
        
        logging.info(f"Starting atomic clear of {len(tables)} tables in '{db_path.name}'...")
        
        # Use a transaction for atomicity and performance
        cursor.execute("BEGIN TRANSACTION;")
        # Disable foreign keys to ensure deletion order doesn't matter
        cursor.execute("PRAGMA foreign_keys = OFF;")
        
        for table in tables:
            cursor.execute(f'DELETE FROM "{table}";')
            logging.info(f"  - Cleared table '{table}'")
        
        # Reset autoincrement counters if sqlite_sequence exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence';")
        if cursor.fetchone():
            cursor.execute("DELETE FROM sqlite_sequence;")
            logging.info("  - Reset autoincrement counters")
        
        # Re-enable foreign keys before committing
        cursor.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        logging.info(f"Transaction committed for '{db_path.name}'.")
        
        # Reclaim disk space
        logging.info(f"Running VACUUM on '{db_path.name}' to reclaim space...")
        cursor.execute("VACUUM;")
        conn.commit()
        logging.info(f"VACUUM complete for '{db_path.name}'.")
        
    except sqlite3.Error as e:
        logging.error(f"Failed to clear database '{db_path.name}': {e}")
        if conn:
            logging.info("Rolling back transaction.")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info(f"Connection to '{db_path.name}' closed.")

def clear_chroma_db(db_path: Path):
    """
    Clears a ChromaDB persistent database by deleting all collections.
    
    WARNING: Direct SQLite manipulation of ChromaDB's database file is dangerous
    and can corrupt its internal state. Always use the ChromaDB API.
    """
    db_dir = db_path.parent
    if not db_path.exists():
        logging.warning(f"ChromaDB file not found, skipping: {db_path}")
        return
    
    logging.info(f"Initializing ChromaDB client for path: {db_dir}")
    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        
        # Try reset first (if enabled)
        try:
            client.reset()
            logging.info(f"Successfully reset ChromaDB at '{db_dir}'.")
            return
        except Exception as reset_error:
            logging.info(f"Reset not available: {reset_error}. Falling back to collection deletion.")
        
        # If reset fails, delete all collections manually
        collections = client.list_collections()
        if not collections:
            logging.info("No collections found in ChromaDB.")
            return
        
        for collection in collections:
            client.delete_collection(name=collection.name)
            logging.info(f"  - Deleted collection '{collection.name}'")
        
        logging.info(f"Successfully cleared {len(collections)} collections from ChromaDB.")
        
    except Exception as e:
        # Catching a broad exception as ChromaDB internals can vary
        logging.error(f"An error occurred while clearing ChromaDB at '{db_dir}': {e}")

def clear_all_neural_databases():
    """
    Orchestrates the clearing of all Neural MCP system databases.
    """
    logging.info("=" * 60)
    logging.info("Starting Neural MCP Database Clearing Process")
    logging.info("=" * 60)
    
    # Define the base path
    base_path = Path(__file__).parent.parent  # Go up to .claude directory
    
    # Define database paths
    memory_db = base_path / "memory" / "neural-dynamic-memory.db"
    project_db = base_path / "memory" / "project-knowledge.db"
    chroma_db = base_path / "chroma" / "chroma.sqlite3"
    
    # Clear standard SQLite databases
    logging.info("\n--- Clearing Neural Dynamic Memory Database ---")
    clear_standard_sqlite_db(memory_db)
    
    logging.info("\n--- Clearing Project Knowledge Database ---")
    clear_standard_sqlite_db(project_db)
    
    # Clear ChromaDB using its specific API
    logging.info("\n--- Clearing ChromaDB Vector Database ---")
    clear_chroma_db(chroma_db)
    
    logging.info("\n" + "=" * 60)
    logging.info("Database Clearing Process Complete!")
    logging.info("All data has been cleared while preserving schemas.")
    logging.info("The system is ready to store new data.")
    logging.info("=" * 60)

def verify_databases_cleared():
    """
    Verifies that all databases have been cleared by checking row counts.
    """
    base_path = Path(__file__).parent.parent
    
    databases = [
        base_path / "memory" / "neural-dynamic-memory.db",
        base_path / "memory" / "project-knowledge.db",
        base_path / "chroma" / "chroma.sqlite3"
    ]
    
    print("\n" + "=" * 60)
    print("Database Verification Report")
    print("=" * 60)
    
    for db_path in databases:
        if not db_path.exists():
            print(f"❌ {db_path.name}: Not found")
            continue
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables and their row counts
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'system_%';"
            )
            tables = cursor.fetchall()
            
            total_rows = 0
            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM \"{table_name}\";")
                count = cursor.fetchone()[0]
                total_rows += count
            
            conn.close()
            
            if total_rows == 0:
                print(f"✅ {db_path.name}: CLEARED (0 rows across {len(tables)} tables)")
            else:
                print(f"⚠️  {db_path.name}: Contains {total_rows} rows")
                
        except Exception as e:
            print(f"❌ {db_path.name}: Error checking - {e}")
    
    print("=" * 60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_databases_cleared()
    else:
        print("\n⚠️  WARNING: This will clear ALL data from the Neural MCP databases!")
        print("The database schemas will be preserved, but all content will be deleted.")
        response = input("\nAre you sure you want to continue? (yes/no): ")
        
        if response.lower() == 'yes':
            clear_all_neural_databases()
            print("\n✅ Databases cleared successfully!")
            print("\nRunning verification...")
            verify_databases_cleared()
        else:
            print("Operation cancelled.")
# ==============================================================================
# DYNAMIC MONGODB MIGRATION SCRIPT
# ==============================================================================
# Migration script that uses external schema and sample data configurations

import pymongo
import json
import os
from decimal import Decimal
from datetime import datetime
from bson import ObjectId
from pathlib import Path
from schema_loader import schema_loader

def connect_mongodb(domain=None):
    """Establish MongoDB connection using schema configuration"""
    try:
        # Load configuration for the domain
        domain = domain or schema_loader.current_domain
        schema = schema_loader.load_schema(domain)
        config = schema_loader.load_config(domain)
        
        # Use configuration for connection
        mongodb_uri = config.get("mongodb_config", {}).get("uri", "mongodb://127.0.0.1:27017")
        database_name = schema.get("database_name", f"{domain}-db")
        
        client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[database_name]
        print(f"✅ Connected to MongoDB: {database_name} (domain: {domain})")
        return client, db, database_name
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return None, None, None

def create_collections_and_indexes(db, domain=None):
    """Create collections and indexes based on schema configuration"""
    print(f"\n📊 Creating collections and indexes for domain: {domain or schema_loader.current_domain}...")
    
    try:
        schema = schema_loader.load_schema(domain)
        collections = schema.get("collections", {})
        created_collections = {}
        
        for collection_name, collection_info in collections.items():
            print(f"   Creating {collection_name} collection...")
            collection = db[collection_name]
            
            # Create indexes based on schema definition
            fields = collection_info.get("fields", {})
            
            # Create unique indexes
            for field_name, field_info in fields.items():
                if field_info.get("unique", False):
                    collection.create_index(field_name, unique=True)
                    print(f"      ✓ Unique index on {field_name}")
                elif field_info.get("index", False):
                    collection.create_index(field_name)
                    print(f"      ✓ Index on {field_name}")
            
            # Create compound indexes if specified
            indexes = collection_info.get("indexes", [])
            for index_def in indexes:
                if isinstance(index_def, list):
                    collection.create_index(index_def)
                    print(f"      ✓ Compound index on {index_def}")
            
            created_collections[collection_name] = collection
            
        print(f"✅ Created {len(created_collections)} collections")
        return created_collections
        
    except Exception as e:
        print(f"❌ Error creating collections: {e}")
        return {}

def load_sample_data(domain=None):
    """Load sample data from external JSON file"""
    try:
        domain = domain or schema_loader.current_domain
        sample_data_file = schema_loader.schema_dir / domain / "sample_data.json"
        
        if not sample_data_file.exists():
            print(f"⚠️ No sample data file found: {sample_data_file}")
            return {}
            
        with open(sample_data_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            
        print(f"✅ Loaded sample data for domain: {domain}")
        return sample_data
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return {}

def insert_data_with_relationships(collections, sample_data, domain=None):
    """Insert data while maintaining relationships between collections"""
    print(f"\n📥 Inserting sample data...")
    
    try:
        schema = schema_loader.load_schema(domain)
        collections_schema = schema.get("collections", {})
        
        # Track inserted IDs for relationship mapping
        inserted_mappings = {}
        
        # Process collections in dependency order
        for collection_name, collection in collections.items():
            if collection_name not in sample_data:
                print(f"   ⏭️ Skipping {collection_name} - no sample data")
                continue
                
            print(f"   📊 Processing {collection_name}...")
            
            # Clear existing data
            collection.delete_many({})
            
            data_items = sample_data[collection_name]
            processed_items = []
            
            for item in data_items:
                processed_item = item.copy()
                
                # Add timestamps
                processed_item['created_at'] = datetime.now()
                processed_item['updated_at'] = datetime.now()
                
                # Handle relationships (convert string references to ObjectIds)
                collection_schema = collections_schema.get(collection_name, {})
                fields = collection_schema.get("fields", {})
                
                for field_name, field_info in fields.items():
                    if field_name in processed_item and field_info.get("type") == "ObjectId":
                        ref_collection = field_info.get("ref")
                        if ref_collection and field_name in processed_item:
                            ref_key = processed_item[field_name]
                            if ref_key in inserted_mappings.get(ref_collection, {}):
                                processed_item[field_name] = inserted_mappings[ref_collection][ref_key]
                
                processed_items.append(processed_item)
            
            # Insert the processed items
            if processed_items:
                result = collection.insert_many(processed_items)
                
                # Track inserted IDs for relationships
                inserted_mappings[collection_name] = {}
                for i, inserted_id in enumerate(result.inserted_ids):
                    # Use a unique key from the original item (name, slug, sku, username, etc.)
                    original_item = data_items[i]
                    key = (original_item.get('name') or 
                          original_item.get('slug') or 
                          original_item.get('sku') or
                          original_item.get('username') or
                          str(i))
                    inserted_mappings[collection_name][key] = inserted_id
                
                print(f"      ✅ Inserted {len(processed_items)} {collection_name}")
            else:
                print(f"      ⚠️ No valid data to insert for {collection_name}")
        
        print(f"✅ Sample data insertion completed")
        return inserted_mappings
        
    except Exception as e:
        print(f"❌ Error inserting data: {e}")
        return {}

def migrate_domain(domain):
    """Migrate a specific domain"""
    print(f"\n🚀 Starting migration for domain: {domain}")
    
    # Set the active domain
    if not schema_loader.set_domain(domain):
        print(f"❌ Domain '{domain}' not found")
        return False
    
    # Connect to MongoDB
    client, db, db_name = connect_mongodb(domain)
    if not client or not db:
        return False
    
    try:
        # Create collections and indexes
        collections = create_collections_and_indexes(db, domain)
        if not collections:
            return False
        
        # Load and insert sample data
        sample_data = load_sample_data(domain)
        if sample_data:
            mappings = insert_data_with_relationships(collections, sample_data, domain)
            print(f"✅ Migration completed for domain: {domain}")
            print(f"   Database: {db_name}")
            print(f"   Collections: {list(collections.keys())}")
            return True
        else:
            print(f"⚠️ No sample data found for domain: {domain}")
            return True
            
    except Exception as e:
        print(f"❌ Migration failed for domain {domain}: {e}")
        return False
        
    finally:
        client.close()

def migrate_all_domains():
    """Migrate all available domains"""
    domains = schema_loader.get_available_domains()
    
    if not domains:
        print("❌ No domains found in schema directory")
        return
    
    print(f"🌍 Found {len(domains)} domains: {', '.join(domains)}")
    
    results = {}
    for domain in domains:
        print(f"\n{'='*60}")
        results[domain] = migrate_domain(domain)
    
    print(f"\n{'='*60}")
    print("📊 MIGRATION SUMMARY")
    print(f"{'='*60}")
    
    for domain, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {domain}: {status}")

def main():
    """Main migration function"""
    print("🗄️ DYNAMIC MONGODB MIGRATION TOOL")
    print("=" * 50)
    
    # Show available domains
    domains = schema_loader.get_available_domains()
    print(f"Available domains: {', '.join(domains)}")
    
    # Ask user what to migrate
    print("\nOptions:")
    print("1. Migrate all domains")
    print("2. Migrate specific domain")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        migrate_all_domains()
    elif choice == "2":
        print(f"Available domains: {', '.join(domains)}")
        domain = input("Enter domain name: ").strip()
        if domain in domains:
            migrate_domain(domain)
        else:
            print(f"❌ Domain '{domain}' not found")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
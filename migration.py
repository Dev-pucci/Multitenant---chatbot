# ==============================================================================
# MONGODB MIGRATION SCRIPT - E-COMMERCE PRODUCT DATA
# ==============================================================================
# Complete script to migrate product data from Python dictionaries to MongoDB

import pymongo
import json
from decimal import Decimal
from datetime import datetime
from bson import ObjectId

# MongoDB connection
MONGODB_URI = "mongodb://127.0.0.1:27017"  # Update with your MongoDB URI
DATABASE_NAME = "ecommerce-marketplace"

def connect_mongodb():
    """Establish MongoDB connection"""
    try:
        client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        print(f"Connected to MongoDB: {DATABASE_NAME}")
        return client, db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None, None

def create_collections_and_indexes(db):
    """Create collections and indexes for optimal performance"""
    print("\nCreating collections and indexes...")
    
    # Categories collection
    categories_collection = db['categories']
    categories_collection.create_index("slug", unique=True)
    categories_collection.create_index("name")
    categories_collection.create_index("is_active")
    print("   Categories collection ready")
    
    # Users (vendors) collection
    users_collection = db['users']
    users_collection.create_index("email", unique=True)
    users_collection.create_index("username", unique=True)
    users_collection.create_index("user_type")
    users_collection.create_index("is_active")
    print("   Users collection ready")
    
    # Products collection
    products_collection = db['products']
    products_collection.create_index("slug", unique=True)
    products_collection.create_index("name")
    products_collection.create_index([("category", 1), ("is_active", 1)])
    products_collection.create_index([("vendor", 1), ("is_active", 1)])
    products_collection.create_index([("featured", 1), ("is_active", 1)])
    products_collection.create_index([("in_stock", 1), ("is_active", 1)])
    products_collection.create_index("price")
    products_collection.create_index("stock_quantity")
    products_collection.create_index("created_at")
    print("   Products collection ready")
    
    return categories_collection, users_collection, products_collection

def insert_categories(categories_collection):
    """Insert product categories"""
    print("\nInserting categories...")
    
    categories_data = [
        {
            'name': 'Smartphones',
            'slug': 'smartphones',
            'description': 'Mobile phones and smartphone accessories',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Drinks',
            'slug': 'drinks',
            'description': 'Alcoholic and non-alcoholic beverages',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Kitchen Ware',
            'slug': 'kitchen-ware',
            'description': 'Kitchen utensils, cookware, and appliances',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    ]
    
    # Clear existing categories
    categories_collection.delete_many({})
    
    # Insert categories
    result = categories_collection.insert_many(categories_data)
    category_ids = result.inserted_ids
    
    # Create mapping for easy reference
    category_mapping = {}
    for i, category in enumerate(categories_data):
        category_mapping[category['name']] = category_ids[i]
    
    print(f"   Inserted {len(categories_data)} categories")
    return category_mapping

def insert_vendors(users_collection):
    """Insert vendor/user accounts"""
    print("\nInserting vendors...")
    
    vendors_data = [
        {
            'username': 'electronics_vendor',
            'email': 'electronics@marketplace.com',
            'first_name': 'Electronics',
            'last_name': 'Store',
            'user_type': 'vendor',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'username': 'drinks_vendor',
            'email': 'drinks@marketplace.com',
            'first_name': 'Premium',
            'last_name': 'Drinks',
            'user_type': 'vendor',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'username': 'kitchen_vendor',
            'email': 'kitchen@marketplace.com',
            'first_name': 'Kitchen',
            'last_name': 'Essentials',
            'user_type': 'vendor',
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    ]
    
    # Clear existing vendors
    users_collection.delete_many({'user_type': 'vendor'})
    
    # Insert vendors
    result = users_collection.insert_many(vendors_data)
    vendor_ids = result.inserted_ids
    
    # Create mapping for easy reference
    vendor_mapping = {}
    vendor_names = ['vendor_electronics', 'vendor_drinks', 'vendor_kitchen']
    for i, vendor_name in enumerate(vendor_names):
        vendor_mapping[vendor_name] = vendor_ids[i]
    
    print(f"   Inserted {len(vendors_data)} vendors")
    return vendor_mapping

def decimal_to_float(obj):
    """Convert Decimal objects to float for MongoDB storage"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: decimal_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj

def insert_products(products_collection, category_mapping, vendor_mapping):
    """Insert all product data"""
    print("\nInserting products...")
    
    # Your original products data (converted from your document)
    products_data = [
        # SMARTPHONES - Electronics Vendor
        {
            'name': 'Vivo Y400 Pro',
            'slug': 'vivo-y400-pro',
            'description': '''The Vivo Y400 Pro features a stunning 6.77" AMOLED display with 120Hz refresh rate and 1300 nits peak brightness. 
Powered by Android 15 with Funtouch 15, it offers 8GB RAM and 128GB/256GB storage options. 
The device boasts a 50MP main camera with f/1.8 aperture and a 2MP depth sensor. 
With a 5500mAh battery and 90W fast charging, this phone delivers exceptional performance and battery life.
Key specs: 162g weight, 7.5mm thickness, IP65 dust/water resistance, Nano-SIM support.''',
            'category': category_mapping['Smartphones'],
            'vendor': vendor_mapping['vendor_electronics'],
            'price': 51999.00,
            'stock_quantity': 15,
            'in_stock': True,
            'image_url': 'https://fdn2.gsmarena.com/vv/pics/vivo/vivo-y400-pro-1.jpg',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Motorola Moto G86',
            'slug': 'motorola-moto-g86',
            'description': '''The Motorola Moto G86 comes with a 6.67" P-OLED display featuring 1B colors and 120Hz refresh rate for smooth visuals.
Running Android 15 with a Mediatek Dimensity 7300 chipset, it offers 8GB/12GB RAM options and 256GB/512GB storage.
Photography enthusiasts will love the 50MP main camera with f/1.9 aperture, 25mm wide lens, dual pixel PDAF, and OIS.
The 5200mAh battery with 30W fast charging ensures all-day usage. Additional features include IP68/IP69 water resistance.
Expected release: July 2025. Dimensions: 161.2 x 74.7 x 7.8 mm, Weight: 185g.''',
            'category': category_mapping['Smartphones'],
            'vendor': vendor_mapping['vendor_electronics'],
            'price': 38870.00,
            'stock_quantity': 8,
            'in_stock': True,
            'image_url': 'https://fdn2.gsmarena.com/vv/pics/motorola/motorola-moto-g86-1.jpg',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Motorola Edge 60',
            'slug': 'motorola-edge-60',
            'description': '''The Motorola Edge 60 features a premium 6.67" P-OLED display with 1B colors, 120Hz refresh rate, and HDR10+ support.
Released in April 2025, it runs Android 15 with up to 3 major Android upgrades guaranteed.
Powered by Mediatek Dimensity 7300/7400 chipset with options for 8GB/12GB RAM and 256GB/512GB storage.
Camera setup includes a 50MP main sensor with f/1.8, 24mm wide lens, multi-direction PDAF, and OIS.
The 5200mAh battery with 68W fast charging provides excellent battery life.
Build: Glass front (Gorilla Glass 7), plastic frame, plastic back. Weight: 179g.''',
            'category': category_mapping['Smartphones'],
            'vendor': vendor_mapping['vendor_electronics'],
            'price': 53170.00,
            'stock_quantity': 12,
            'in_stock': True,
            'image_url': 'https://fdn2.gsmarena.com/vv/pics/motorola/motorola-edge-60-1.jpg',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Motorola Moto G45',
            'slug': 'motorola-moto-g45',
            'description': '''The Motorola Moto G45 offers excellent value with its 6.5" IPS LCD display and 120Hz refresh rate.
Released in August 2024, it features Android 14 and Qualcomm SM6375 Snapdragon 6s Gen 3 chipset.
Available in 4GB/8GB RAM variants with 128GB storage options and microSDXC expansion slot.
The camera system includes a 50MP main sensor with f/1.8 aperture and a 2MP macro lens.
Powered by a 5000mAh battery with 18W fast charging for reliable all-day performance.
Build: Glass front (Gorilla Glass 3), plastic frame, silicone polymer back. Weight: 183g.
Features water-repellent design and stereo speakers for enhanced multimedia experience.''',
            'category': category_mapping['Smartphones'],
            'vendor': vendor_mapping['vendor_electronics'],
            'price': 25999.00,
            'stock_quantity': 25,
            'in_stock': True,
            'image_url': 'https://fdn2.gsmarena.com/vv/pics/motorola/motorola-moto-g45-5g-1.jpg',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        
        # DRINKS - Drinks Vendor
        {
            'name': 'Beefeater London Dry Gin - 750ml',
            'slug': 'beefeater-london-dry-gin-750ml',
            'description': '''Beefeater, the only international premium gin still produced in the heart of London, and the world's most awarded gin, is the embodiment of The Spirit of London. 
To this day, Beefeater gin is produced to James Borrough's original 1863 recipe, including the unique process of steeping nine perfectly balanced botanicals for 24 hours. 
Part of the Pernod Ricard group since 2005, Beefeater is a modern, energetic and urban gin with 200 years of distilling heritage and a refreshingly modern take on today's tastes.
Premium London Dry Gin with distinctive juniper flavor and smooth finish. Perfect for cocktails or enjoying neat.''',
            'category': category_mapping['Drinks'],
            'vendor': vendor_mapping['vendor_drinks'],
            'price': 3200.00,
            'stock_quantity': 20,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/17/873585/1.jpg?8695',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'The Glenlivet Founders Reserve 12 Years - 700ml',
            'slug': 'glenlivet-founders-reserve-12-years-700ml',
            'description': '''The Glenlivet Founders Reserve 12 Years is a premium single malt Scotch whisky that represents the perfect introduction to The Glenlivet range.
Aged for 12 years in traditional oak casks, this whisky offers a smooth and approachable flavor profile with notes of citrus, honey, and vanilla.
The Glenlivet distillery, founded in 1824, is renowned for producing exceptional single malt whiskies using traditional methods and the finest ingredients.
This expression showcases the classic Speyside character with its elegant balance and refined taste. Perfect for both whisky enthusiasts and newcomers alike.''',
            'category': category_mapping['Drinks'],
            'vendor': vendor_mapping['vendor_drinks'],
            'price': 8500.00,
            'stock_quantity': 12,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/45/449969/1.jpg?2169',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Jameson Irish Whiskey - 4.5 Litres',
            'slug': 'jameson-irish-whiskey-4-5-litres',
            'description': '''Jameson is the best-selling Irish whiskey in the world. Produced in our distillery in Midleton, County Cork, from malted and unmalted Irish barley.
Jameson's blended whiskeys are triple-distilled, resulting in exceptional smoothness that has made it a favorite worldwide.
The brand has been part of the Pernod Group since 1988 and continues to expand its acclaimed range to offer new taste experiences.
This large 4.5-litre bottle is perfect for parties, events, or stocking up your home bar. Features the signature smooth taste with hints of vanilla and honey.
Ideal for mixing cocktails or enjoying neat with friends and family.''',
            'category': category_mapping['Drinks'],
            'vendor': vendor_mapping['vendor_drinks'],
            'price': 28000.00,
            'stock_quantity': 6,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/57/283585/1.jpg?4526',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Seagrams Imperial Blue Whiskey - 750ml',
            'slug': 'seagrams-imperial-blue-whiskey-750ml',
            'description': '''Pernod Ricard has a unique portfolio of premium brands encompassing every major category of wine and spirits.
Seagrams Imperial Blue represents quality and craftsmanship in whiskey making, offering a smooth and refined drinking experience.
This premium whiskey features a rich amber color with complex flavors of oak, vanilla, and subtle spices.
Perfect for both casual drinking and special occasions, Imperial Blue delivers consistent quality and taste.
The 750ml bottle is ideal for home consumption and makes an excellent gift for whiskey enthusiasts.''',
            'category': category_mapping['Drinks'],
            'vendor': vendor_mapping['vendor_drinks'],
            'price': 2800.00,
            'stock_quantity': 18,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/02/193585/1.jpg?9231',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Jacob\'s Creek Classic Merlot 750ml',
            'slug': 'jacobs-creek-classic-merlot-750ml',
            'description': '''Jacob's Creek Classic Merlot is a premium red wine that showcases the best of Australian winemaking tradition.
This full-bodied wine features rich flavors of dark berries, plums, and subtle oak notes with a smooth, velvety finish.
Perfect for pairing with red meat, pasta dishes, or enjoying on its own during social gatherings.
Jacob's Creek wines are crafted using traditional winemaking techniques combined with modern innovation to deliver consistent quality.
The 750ml bottle is perfect for dinner parties, romantic evenings, or adding to your wine collection.''',
            'category': category_mapping['Drinks'],
            'vendor': vendor_mapping['vendor_drinks'],
            'price': 1800.00,
            'stock_quantity': 24,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/80/283585/1.jpg?9069',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        
        # KITCHEN WARE - Kitchen Vendor
        {
            'name': '2Pcs Set Of Stainless Aluminum Sufuria No Lids + Free Gift',
            'slug': '2pcs-stainless-aluminum-sufuria-set',
            'description': '''This 2-Piece Cookware set is made of pure aluminum encapsulated in the base for fast and even heating and cooking performance.
Solid stainless steel riveted handles stay cool on the stove top, ensuring safe and comfortable cooking.
Ideal even for jiko or firewood cooking during outdoor cooking adventures and camping trips.
The durable construction ensures long-lasting performance while the aluminum core provides excellent heat distribution.
Easy to clean and maintain, these sufurias are perfect for everyday cooking needs. Comes with a free gift to enhance your cooking experience.''',
            'category': category_mapping['Kitchen Ware'],
            'vendor': vendor_mapping['vendor_kitchen'],
            'price': 2500.00,
            'stock_quantity': 30,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/96/7343842/1.jpg?0140',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Mateamoda 5 PCS Kitchenware Utensils Cookware Baking Set',
            'slug': 'mateamoda-5pcs-kitchenware-utensils-set',
            'description': '''Made from very safe food grade silicone, 100% BPA free for healthy cooking and food preparation.
Complete set contains: 1 Brush, 1 Whisk, 1 Leakage Shovel, 1 Big Scraper, 1 Small Scraper.
Hanging hole design allows convenient hanging storage when idle, saving valuable kitchen space.
The material is soft, does not deform or crack, and can be used for a long time without wear.
Adopts one-piece molding technology, making it easy to clean without concealing dirt. Heat resistant and dishwasher safe.
For first use, it is recommended to clean with hot water to remove any manufacturing odor.''',
            'category': category_mapping['Kitchen Ware'],
            'vendor': vendor_mapping['vendor_kitchen'],
            'price': 1200.00,
            'stock_quantity': 45,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/01/538275/1.jpg?7415',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Silicon Ice Cube Maker Tray',
            'slug': 'silicon-ice-cube-maker-tray',
            'description': '''Space Saving Ice Cube Maker that allows you to create a large number of ice cubes and store them inside the tray itself as you freeze.
Using our ice cube maker tray, you can save a ton of space in your freezer while keeping ice readily available.
It will quickly chill bottled beverages and is perfect for parties, gatherings, or daily use.
Simple design allows you to create up to 37 pieces of ice at a time with easy release mechanism.
Releasing ice from the tray is just as easy as pushing out the ice cubes from the sides. Made from food-grade silicone material.
Beat the heat with our premium silicone ice cube maker that's both practical and space-efficient.''',
            'category': category_mapping['Kitchen Ware'],
            'vendor': vendor_mapping['vendor_kitchen'],
            'price': 800.00,
            'stock_quantity': 60,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/97/8119201/1.jpg?7783',
            'featured': False,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Heavy Metal Manual Hand Juice Extractor Squeezer',
            'slug': 'heavy-metal-manual-juice-extractor',
            'description': '''Professional-grade manual juice extractor made from high-quality stainless steel for durability and performance.
Triangle guide nozzle with anti-dripping design ensures easy use and stable pouring without mess.
Great stability with foot design that makes the juicer easily and stably sit on the table without slipping.
Reasonable design features smooth lining and fine hole design for easy slag filtering and maximum juice extraction.
Energy-saving manual operation that's environmentally friendly and doesn't require electricity.
User-friendly handle design with comfortable grip makes juicing easier and more efficient.
Perfect for oranges, watermelons, lemons, and other citrus fruits. Hand wash only for long-term use.''',
            'category': category_mapping['Kitchen Ware'],
            'vendor': vendor_mapping['vendor_kitchen'],
            'price': 3500.00,
            'stock_quantity': 15,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/41/8639882/1.jpg?9629',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        },
        {
            'name': 'Electric Meat Grinder - Cordless Home Cooking Machine',
            'slug': 'electric-meat-grinder-cordless',
            'description': '''Simple and convenient cordless food processor that's easy to use anywhere in your kitchen.
Press and hold the top button to start the motor, release to stop - you control the degree of food processing.
Can chop garlic, onions, meat, and other ingredients in just minutes with precision and consistency.
Food-grade construction with stainless steel blades that cut food evenly from 360 degrees for uniform results.
Upper and lower blades ensure thorough processing while maintaining safety standards. Safe and odorless operation.
Easy to disassemble and clean with waterproof design that allows thorough cleaning and reuse.
USB rechargeable and portable - wireless design makes it perfect for travel, camping, or outdoor activities.
Multi-purpose functionality: chop garlic, onions, peppers, parsley, ginger, peanuts, and make baby food.''',
            'category': category_mapping['Kitchen Ware'],
            'vendor': vendor_mapping['vendor_kitchen'],
            'price': 4200.00,
            'stock_quantity': 12,
            'in_stock': True,
            'image_url': 'https://ke.jumia.is/unsafe/fit-in/500x500/filters:fill(white)/product/12/5588703/1.jpg?3118',
            'featured': True,
            'is_active': True,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
    ]
    
    # Clear existing products
    products_collection.delete_many({})
    
    # Convert Decimal to float and insert products
    converted_products = decimal_to_float(products_data)
    result = products_collection.insert_many(converted_products)
    
    print(f"   Inserted {len(products_data)} products")
    return result.inserted_ids

def verify_data(db):
    """Verify the inserted data"""
    print("\nVerifying inserted data...")
    
    # Count documents in each collection
    categories_count = db.categories.count_documents({})
    users_count = db.users.count_documents({})
    products_count = db.products.count_documents({})
    
    print(f"   Categories: {categories_count}")
    print(f"   Users (Vendors): {users_count}")
    print(f"   Products: {products_count}")
    
    # Sample queries to verify relationships
    print("\nSample data verification:")
    
    # Featured products
    featured_products = list(db.products.find(
        {"featured": True, "is_active": True},
        {"name": 1, "price": 1}
    ).limit(3))
    print(f"   Featured products: {len(featured_products)}")
    for product in featured_products:
        print(f"      - {product['name']}: ${product['price']}")
    
    # Products by category
    pipeline = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$lookup": {"from": "categories", "localField": "_id", "foreignField": "_id", "as": "category_info"}},
        {"$project": {"category_name": {"$arrayElemAt": ["$category_info.name", 0]}, "count": 1}}
    ]
    category_stats = list(db.products.aggregate(pipeline))
    print(f"   Products by category:")
    for stat in category_stats:
        print(f"      - {stat['category_name']}: {stat['count']} products")

def export_sample_queries():
    """Export sample MongoDB queries for testing"""
    sample_queries = {
        "all_active_products": {
            "collection": "products",
            "pipeline": [
                {"$match": {"is_active": True}},
                {"$lookup": {"from": "categories", "localField": "category", "foreignField": "_id", "as": "category_info"}},
                {"$lookup": {"from": "users", "localField": "vendor", "foreignField": "_id", "as": "vendor_info"}},
                {"$project": {
                    "name": 1, 
                    "price": 1, 
                    "stock_quantity": 1,
                    "category_name": {"$arrayElemAt": ["$category_info.name", 0]},
                    "vendor_name": {"$arrayElemAt": ["$vendor_info.first_name", 0]}
                }},
                {"$limit": 10}
            ]
        },
        "featured_products": {
            "collection": "products",
            "pipeline": [
                {"$match": {"featured": True, "is_active": True}},
                {"$project": {"name": 1, "price": 1, "image_url": 1}}
            ]
        },
        "products_by_price_range": {
            "collection": "products",
            "pipeline": [
                {"$match": {"price": {"$gte": 1000, "$lte": 10000}, "is_active": True}},
                {"$project": {"name": 1, "price": 1}},
                {"$sort": {"price": 1}}
            ]
        },
        "low_stock_products": {
            "collection": "products",
            "pipeline": [
                {"$match": {"stock_quantity": {"$lt": 10}, "is_active": True}},
                {"$project": {"name": 1, "stock_quantity": 1}},
                {"$sort": {"stock_quantity": 1}}
            ]
        }
    }
    
    print("\nSample queries exported to 'sample_queries.json'")
    with open('sample_queries.json', 'w') as f:
        json.dump(sample_queries, f, indent=2, default=str)

def main():
    """Main migration function"""
    print("Starting MongoDB migration for e-commerce product data...\n")
    
    # Connect to MongoDB
    client, db = connect_mongodb()
    if not client:
        return
    
    try:
        # Create collections and indexes
        categories_collection, users_collection, products_collection = create_collections_and_indexes(db)
        
        # Insert data in order (categories first, then vendors, then products)
        category_mapping = insert_categories(categories_collection)
        vendor_mapping = insert_vendors(users_collection)
        product_ids = insert_products(products_collection, category_mapping, vendor_mapping)
        
        # Verify the migration
        verify_data(db)
        
        # Export sample queries
        export_sample_queries()
        
        print("\nMigration completed successfully!")
        print(f"    Database: {DATABASE_NAME}")
        print(f"    Collections: categories, users, products")
        print(f"    Total products: {len(product_ids)}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False
    
    finally:
        # Close the connection
        if client:
            client.close()
            print("\nMongoDB connection closed")
    
    return True

if __name__ == "__main__":
    main()
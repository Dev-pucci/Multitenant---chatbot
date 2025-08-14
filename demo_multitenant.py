#!/usr/bin/env python3
# ==============================================================================
# MULTI-TENANT CHATBOT DEMO
# ==============================================================================
# Demonstrates the multi-tenant chatbot system with different customers

from customer_manager import customer_manager, add_customer
from dynamic_migration import migrate_domain
from chatbotlang import main
import os

def setup_demo_customers():
    """Set up demo customers for the multi-tenant system"""
    print("🚀 SETTING UP MULTI-TENANT DEMO")
    print("="*50)
    
    # Check if customers already exist
    customers = customer_manager.list_customers()
    if customers:
        print(f"✅ Found {len(customers)} existing customers:")
        for customer in customers:
            print(f"  • {customer.customer_id} - {customer.name} ({customer.domain})")
        return
    
    # Add demo customers
    print("➕ Adding demo customers...")
    
    # TechMart - Electronics E-commerce
    print("   Adding TechMart Electronics...")
    success1 = add_customer(
        customer_id="techmart",
        name="TechMart Electronics", 
        domain="ecommerce",
        config={
            "mongodb_uri": "mongodb://127.0.0.1:27017",
            "company_details": {
                "industry": "Electronics Retail",
                "size": "Medium",
                "location": "USA"
            },
            "features": {
                "enable_fuzzy_search": True,
                "enable_price_tracking": True,
                "enable_vendor_analytics": True
            }
        }
    )
    
    # Warehouses Inc - Inventory Management
    print("   Adding Warehouses Inc...")
    success2 = add_customer(
        customer_id="warehouses_inc",
        name="Warehouses Inc",
        domain="inventory", 
        config={
            "mongodb_uri": "mongodb://127.0.0.1:27017",
            "company_details": {
                "industry": "Logistics & Warehousing",
                "size": "Large",
                "location": "Canada"
            },
            "features": {
                "enable_low_stock_alerts": True,
                "enable_reorder_automation": True,
                "enable_warehouse_optimization": True
            }
        }
    )
    
    # Shopify Plus - Large E-commerce
    print("   Adding Shopify Plus Store...")
    success3 = add_customer(
        customer_id="shopify_plus",
        name="Shopify Plus Store",
        domain="ecommerce",
        config={
            "mongodb_uri": "mongodb://127.0.0.1:27017", 
            "company_details": {
                "industry": "Multi-Category E-commerce",
                "size": "Enterprise", 
                "location": "Global"
            },
            "features": {
                "enable_fuzzy_search": True,
                "enable_advanced_analytics": True,
                "enable_multi_language": True,
                "enable_recommendation_engine": True
            }
        }
    )
    
    if success1 and success2 and success3:
        print("✅ Demo customers added successfully!")
    else:
        print("❌ Some customers failed to add")

def migrate_demo_data():
    """Migrate sample data for demo customers"""
    print("\n📊 MIGRATING DEMO DATA")
    print("="*50)
    
    # Migrate both domains
    domains = ["ecommerce", "inventory"]
    
    for domain in domains:
        print(f"\n🔄 Migrating {domain} domain...")
        success = migrate_domain(domain)
        if success:
            print(f"✅ {domain} migration completed")
        else:
            print(f"❌ {domain} migration failed")

def show_demo_instructions():
    """Show instructions for using the demo"""
    print("\n🎯 DEMO INSTRUCTIONS")
    print("="*50)
    print("1. The chatbot will start with no customer selected")
    print("2. Use '/customers' to see available customers:")
    print("   • techmart - Electronics e-commerce")
    print("   • warehouses_inc - Inventory management") 
    print("   • shopify_plus - Large e-commerce")
    print()
    print("3. Use '/switch <customer_id>' to switch customers:")
    print("   Example: /switch techmart")
    print()
    print("4. Try different queries for different domains:")
    print("   E-commerce: 'show me smartphones', 'products under $30000'")
    print("   Inventory: 'show low stock items', 'list all warehouses'")
    print()
    print("5. Use '/current' to see current customer context")
    print()
    print("6. The system automatically:")
    print("   • Switches database connections")
    print("   • Loads appropriate schema")
    print("   • Uses domain-specific prompts")
    print()
    print("🚀 Starting the multi-tenant chatbot...")
    print("="*50)

def main():
    """Main demo function"""
    print("🏢 MULTI-TENANT CHATBOT DEMO")
    print("="*60)
    
    # Setup demo customers
    setup_demo_customers()
    
    # Ask if user wants to migrate data
    print("\n❓ Do you want to migrate sample data to databases?")
    print("   This will create databases with sample products/inventory")
    choice = input("   Migrate data? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        migrate_demo_data()
    
    # Show demo instructions
    show_demo_instructions()
    
    # Start the chatbot (this will import and run the main chatbot)
    try:
        from chatbotlang import main as chatbot_main
        chatbot_main()
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Thanks for trying the multi-tenant system!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
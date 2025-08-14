#!/usr/bin/env python3
# ==============================================================================
# CUSTOMER MANAGEMENT CLI
# ==============================================================================
# Command line interface for managing multi-tenant customers

import sys
import argparse
from typing import Optional
from customer_manager import customer_manager, Customer
from schema_loader import schema_loader

def list_customers():
    """List all customers"""
    customers = customer_manager.list_customers()
    
    if not customers:
        print("📋 No customers found")
        return
    
    print("📋 CUSTOMERS:")
    print("-" * 80)
    print(f"{'ID':<20} {'Name':<25} {'Domain':<15} {'Status':<10} {'Database'}")
    print("-" * 80)
    
    for customer in customers:
        print(f"{customer.customer_id:<20} {customer.name:<25} {customer.domain:<15} {customer.status:<10} {customer.database_name}")

def add_customer():
    """Interactive customer addition"""
    print("\n➕ ADD NEW CUSTOMER")
    print("-" * 40)
    
    # Get customer details
    customer_id = input("Customer ID (lowercase, no spaces): ").strip()
    if not customer_id:
        print("❌ Customer ID is required")
        return
    
    name = input("Company Name: ").strip()
    if not name:
        print("❌ Company name is required")
        return
    
    # Show available domains
    domains = schema_loader.get_available_domains()
    print(f"Available domains: {', '.join(domains)}")
    domain = input("Domain: ").strip()
    if domain not in domains:
        print(f"❌ Invalid domain. Available: {', '.join(domains)}")
        return
    
    # Optional configuration
    print("\nOptional configuration (press Enter to skip):")
    mongodb_uri = input("MongoDB URI [mongodb://127.0.0.1:27017]: ").strip()
    industry = input("Industry: ").strip()
    size = input("Company size (Small/Medium/Large/Enterprise): ").strip()
    location = input("Location: ").strip()
    
    # Build config
    config = {
        "mongodb_uri": mongodb_uri or "mongodb://127.0.0.1:27017",
        "company_details": {
            "industry": industry,
            "size": size,
            "location": location
        },
        "features": {
            "enable_fuzzy_search": True,
            "enable_analytics": True
        }
    }
    
    # Add customer
    success = customer_manager.add_customer(customer_id, name, domain, config)
    if success:
        print(f"✅ Customer '{name}' added successfully!")
        print(f"   Database: {customer_id}_{domain}")
    else:
        print("❌ Failed to add customer")

def set_current_customer():
    """Set the current active customer"""
    print("\n🎯 SET CURRENT CUSTOMER")
    print("-" * 40)
    
    customers = customer_manager.list_customers()
    if not customers:
        print("❌ No customers available")
        return
    
    print("Available customers:")
    for i, customer in enumerate(customers, 1):
        status_emoji = "🟢" if customer.status == "active" else "🔴"
        print(f"  {i}. {customer.customer_id} - {customer.name} {status_emoji}")
    
    try:
        choice = input("\nEnter customer number or ID: ").strip()
        
        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(customers):
                customer = customers[idx]
                customer_id = customer.customer_id
            else:
                print("❌ Invalid selection")
                return
        else:
            customer_id = choice
        
        success = customer_manager.set_current_customer(customer_id)
        if success:
            customer = customer_manager.get_current_customer()
            schema_loader.set_domain(customer.domain)  # Also set the schema domain
            print(f"✅ Set current customer: {customer.name}")
            print(f"   Domain: {customer.domain}")
            print(f"   Database: {customer.database_name}")
        else:
            print("❌ Failed to set customer")
            
    except ValueError:
        print("❌ Invalid input")

def show_current_customer():
    """Show current active customer"""
    customer = customer_manager.get_current_customer()
    
    if not customer:
        print("❌ No customer is currently active")
        return
    
    print(f"\n🎯 CURRENT CUSTOMER")
    print("-" * 40)
    print(f"ID: {customer.customer_id}")
    print(f"Name: {customer.name}")
    print(f"Domain: {customer.domain}")
    print(f"Status: {customer.status}")
    print(f"Database: {customer.database_name}")
    print(f"Created: {customer.created_at}")

def update_customer_status():
    """Update customer status"""
    print("\n📝 UPDATE CUSTOMER STATUS")
    print("-" * 40)
    
    customers = customer_manager.list_customers()
    if not customers:
        print("❌ No customers available")
        return
    
    print("Available customers:")
    for i, customer in enumerate(customers, 1):
        status_emoji = "🟢" if customer.status == "active" else "🔴"
        print(f"  {i}. {customer.customer_id} - {customer.name} [{customer.status}] {status_emoji}")
    
    customer_id = input("Enter customer ID: ").strip()
    customer = customer_manager.get_customer(customer_id)
    
    if not customer:
        print("❌ Customer not found")
        return
    
    print(f"Current status: {customer.status}")
    print("Available statuses: active, inactive, suspended")
    new_status = input("New status: ").strip().lower()
    
    if new_status not in ["active", "inactive", "suspended"]:
        print("❌ Invalid status")
        return
    
    success = customer_manager.update_customer_status(customer_id, new_status)
    if success:
        print(f"✅ Updated {customer.name} status to: {new_status}")
    else:
        print("❌ Failed to update status")

def remove_customer():
    """Remove customer configuration"""
    print("\n🗑️  REMOVE CUSTOMER")
    print("-" * 40)
    print("⚠️  This will remove the customer configuration (database will remain)")
    
    customer_id = input("Customer ID to remove: ").strip()
    customer = customer_manager.get_customer(customer_id)
    
    if not customer:
        print("❌ Customer not found")
        return
    
    print(f"Customer: {customer.name} ({customer.customer_id})")
    confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
    
    if confirm == "yes":
        success = customer_manager.remove_customer(customer_id)
        if success:
            print(f"✅ Removed customer: {customer.name}")
        else:
            print("❌ Failed to remove customer")
    else:
        print("❌ Operation cancelled")

def interactive_menu():
    """Interactive customer management menu"""
    while True:
        print("\n" + "="*50)
        print("👥 CUSTOMER MANAGEMENT")
        print("="*50)
        print("1. List customers")
        print("2. Add new customer")  
        print("3. Set current customer")
        print("4. Show current customer")
        print("5. Update customer status")
        print("6. Remove customer")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            list_customers()
        elif choice == "2":
            add_customer()
        elif choice == "3":
            set_current_customer()
        elif choice == "4":
            show_current_customer()
        elif choice == "5":
            update_customer_status()
        elif choice == "6":
            remove_customer()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Customer Management CLI")
    parser.add_argument("--list", action="store_true", help="List all customers")
    parser.add_argument("--current", action="store_true", help="Show current customer")
    parser.add_argument("--set", type=str, help="Set current customer by ID")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.list:
        list_customers()
    elif args.current:
        show_current_customer()
    elif args.set:
        success = customer_manager.set_current_customer(args.set)
        if success:
            customer = customer_manager.get_current_customer()
            print(f"✅ Set current customer: {customer.name}")
        else:
            print("❌ Failed to set customer")
    elif args.interactive:
        interactive_menu()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
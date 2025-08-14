# ==============================================================================
# CUSTOMER MANAGER - Multi-Tenant Support
# ==============================================================================
# Handles customer context, database switching, and tenant isolation

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Customer:
    """Customer configuration data class"""
    customer_id: str
    name: str
    domain: str  # ecommerce, inventory, healthcare, etc.
    database_name: str
    status: str = "active"  # active, inactive, suspended
    created_at: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None

class CustomerManager:
    """Manages customer configurations and database routing"""
    
    def __init__(self, customers_dir: str = "customers"):
        self.customers_dir = Path(customers_dir)
        self.customers_cache = {}
        self.current_customer = None
        self._ensure_customers_directory()
    
    def _ensure_customers_directory(self):
        """Create customers directory if it doesn't exist"""
        self.customers_dir.mkdir(exist_ok=True)
        logger.info(f"Customer directory: {self.customers_dir}")
    
    def add_customer(self, customer_id: str, name: str, domain: str, 
                    config: Optional[Dict] = None) -> bool:
        """Add a new customer configuration"""
        try:
            # Generate database name
            database_name = f"{customer_id}_{domain}"
            
            customer_data = {
                "customer_id": customer_id,
                "name": name,
                "domain": domain,
                "database_name": database_name,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "config": config or {}
            }
            
            customer_file = self.customers_dir / f"{customer_id}.json"
            with open(customer_file, 'w', encoding='utf-8') as f:
                json.dump(customer_data, f, indent=2)
            
            # Cache the customer
            self.customers_cache[customer_id] = Customer(**customer_data)
            
            logger.info(f"✅ Added customer: {name} ({customer_id}) - Domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add customer {customer_id}: {e}")
            return False
    
    def get_customer(self, customer_id: str) -> Optional[Customer]:
        """Get customer by ID"""
        # Check cache first
        if customer_id in self.customers_cache:
            return self.customers_cache[customer_id]
        
        # Load from file
        customer_file = self.customers_dir / f"{customer_id}.json"
        if not customer_file.exists():
            logger.warning(f"Customer {customer_id} not found")
            return None
        
        try:
            with open(customer_file, 'r', encoding='utf-8') as f:
                customer_data = json.load(f)
            
            # Parse created_at if it exists
            if 'created_at' in customer_data and isinstance(customer_data['created_at'], str):
                customer_data['created_at'] = datetime.fromisoformat(customer_data['created_at'])
            
            customer = Customer(**customer_data)
            self.customers_cache[customer_id] = customer
            return customer
            
        except Exception as e:
            logger.error(f"❌ Failed to load customer {customer_id}: {e}")
            return None
    
    def list_customers(self) -> List[Customer]:
        """List all customers"""
        customers = []
        
        if not self.customers_dir.exists():
            return customers
        
        for customer_file in self.customers_dir.glob("*.json"):
            customer_id = customer_file.stem
            customer = self.get_customer(customer_id)
            if customer:
                customers.append(customer)
        
        return sorted(customers, key=lambda c: c.name)
    
    def set_current_customer(self, customer_id: str) -> bool:
        """Set the active customer context"""
        customer = self.get_customer(customer_id)
        if not customer:
            logger.error(f"❌ Cannot set customer {customer_id} - not found")
            return False
        
        if customer.status != "active":
            logger.error(f"❌ Cannot set customer {customer_id} - status: {customer.status}")
            return False
        
        self.current_customer = customer
        logger.info(f"✅ Set current customer: {customer.name} ({customer_id})")
        return True
    
    def get_current_customer(self) -> Optional[Customer]:
        """Get the currently active customer"""
        return self.current_customer
    
    def get_customer_database_info(self, customer_id: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get database connection info for customer"""
        customer = self.get_customer(customer_id) if customer_id else self.current_customer
        
        if not customer:
            return None
        
        return {
            "customer_id": customer.customer_id,
            "database_name": customer.database_name,
            "domain": customer.domain,
            "mongodb_uri": customer.config.get("mongodb_uri", "mongodb://127.0.0.1:27017")
        }
    
    def update_customer_status(self, customer_id: str, status: str) -> bool:
        """Update customer status (active, inactive, suspended)"""
        customer = self.get_customer(customer_id)
        if not customer:
            return False
        
        try:
            customer.status = status
            
            # Update file
            customer_data = {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "domain": customer.domain,
                "database_name": customer.database_name,
                "status": status,
                "created_at": customer.created_at.isoformat() if customer.created_at else None,
                "config": customer.config or {}
            }
            
            customer_file = self.customers_dir / f"{customer_id}.json"
            with open(customer_file, 'w', encoding='utf-8') as f:
                json.dump(customer_data, f, indent=2)
            
            logger.info(f"✅ Updated customer {customer_id} status to: {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update customer {customer_id}: {e}")
            return False
    
    def get_customers_by_domain(self, domain: str) -> List[Customer]:
        """Get all customers for a specific domain"""
        return [c for c in self.list_customers() if c.domain == domain and c.status == "active"]
    
    def remove_customer(self, customer_id: str) -> bool:
        """Remove customer configuration (does not delete database)"""
        try:
            customer_file = self.customers_dir / f"{customer_id}.json"
            if customer_file.exists():
                customer_file.unlink()
            
            # Remove from cache
            if customer_id in self.customers_cache:
                del self.customers_cache[customer_id]
            
            # Clear current customer if it was this one
            if self.current_customer and self.current_customer.customer_id == customer_id:
                self.current_customer = None
            
            logger.info(f"✅ Removed customer: {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove customer {customer_id}: {e}")
            return False

# Global customer manager instance
customer_manager = CustomerManager()

# Convenience functions
def add_customer(customer_id: str, name: str, domain: str, config: Optional[Dict] = None) -> bool:
    """Add a new customer"""
    return customer_manager.add_customer(customer_id, name, domain, config)

def set_customer_context(customer_id: str) -> bool:
    """Set active customer context"""
    return customer_manager.set_current_customer(customer_id)

def get_current_customer() -> Optional[Customer]:
    """Get current customer"""
    return customer_manager.get_current_customer()

def get_customer_db_info() -> Optional[Dict[str, str]]:
    """Get current customer database info"""
    return customer_manager.get_customer_database_info()

def list_all_customers() -> List[Customer]:
    """List all customers"""
    return customer_manager.list_customers()
# Schema Loader Module - Dynamic schema and configuration loading
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class SchemaLoader:
    """Loads and manages external schema configurations"""
    
    def __init__(self, schema_dir: str = "schema", default_domain: str = "ecommerce"):
        self.schema_dir = Path(schema_dir)
        self.default_domain = default_domain
        self.current_domain = default_domain
        self._schemas_cache = {}
        self._configs_cache = {}
        self._prompts_cache = {}
        
    def get_available_domains(self) -> List[str]:
        """Get list of available schema domains"""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory {self.schema_dir} does not exist")
            return []
        
        domains = []
        for item in self.schema_dir.iterdir():
            if item.is_dir() and (item / "schema.json").exists():
                domains.append(item.name)
        
        return sorted(domains)
    
    def set_domain(self, domain: str) -> bool:
        """Switch to a different schema domain"""
        if domain not in self.get_available_domains():
            logger.error(f"Domain '{domain}' not found. Available: {self.get_available_domains()}")
            return False
        
        self.current_domain = domain
        logger.info(f"Switched to domain: {domain}")
        return True
    
    def load_schema(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Load schema configuration for specified domain"""
        domain = domain or self.current_domain
        
        # Check cache first
        if domain in self._schemas_cache:
            return self._schemas_cache[domain]
        
        schema_file = self.schema_dir / domain / "schema.json"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Cache the loaded schema
            self._schemas_cache[domain] = schema
            logger.info(f"Loaded schema for domain: {domain}")
            return schema
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {schema_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load schema from {schema_file}: {e}")
    
    def load_config(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for specified domain"""
        domain = domain or self.current_domain
        
        # Check cache first
        if domain in self._configs_cache:
            return self._configs_cache[domain]
        
        config_file = self.schema_dir / domain / "config.json"
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return self._get_default_config(domain)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Cache the loaded config
            self._configs_cache[domain] = config
            logger.info(f"Loaded config for domain: {domain}")
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_file}: {e}")
            return self._get_default_config(domain)
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return self._get_default_config(domain)
    
    def load_prompts(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Load prompts configuration for specified domain"""
        domain = domain or self.current_domain
        
        # Check cache first
        if domain in self._prompts_cache:
            return self._prompts_cache[domain]
        
        prompts_file = self.schema_dir / domain / "prompts.json"
        if not prompts_file.exists():
            logger.warning(f"Prompts file not found: {prompts_file}, using defaults")
            return self._get_default_prompts(domain)
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            
            # Cache the loaded prompts
            self._prompts_cache[domain] = prompts
            logger.info(f"Loaded prompts for domain: {domain}")
            return prompts
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in prompts file {prompts_file}: {e}")
            return self._get_default_prompts(domain)
        except Exception as e:
            logger.error(f"Failed to load prompts from {prompts_file}: {e}")
            return self._get_default_prompts(domain)
    
    def get_database_name(self, domain: Optional[str] = None) -> str:
        """Get database name for specified domain"""
        schema = self.load_schema(domain)
        return schema.get("database_name", f"{domain or self.current_domain}-db")
    
    def get_collections(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get collections definition for specified domain"""
        schema = self.load_schema(domain)
        return schema.get("collections", {})
    
    def get_collection_names(self, domain: Optional[str] = None) -> List[str]:
        """Get list of collection names for specified domain"""
        collections = self.get_collections(domain)
        return list(collections.keys())
    
    def build_schema_context(self, domain: Optional[str] = None) -> str:
        """Build schema context string for AI prompts"""
        domain = domain or self.current_domain
        schema = self.load_schema(domain)
        config = self.load_config(domain)
        
        context_parts = []
        context_parts.append(f"DOMAIN: {schema.get('description', domain.title())}")
        context_parts.append(f"DATABASE: {schema.get('database_name', domain)}")
        context_parts.append("")
        
        # Add collections info
        collections = schema.get('collections', {})
        for collection_name, collection_info in collections.items():
            context_parts.append(f"COLLECTION: {collection_name}")
            context_parts.append(f"Description: {collection_info.get('description', '')}")
            context_parts.append("Fields:")
            
            fields = collection_info.get('fields', {})
            for field_name, field_info in fields.items():
                field_type = field_info.get('type', 'unknown')
                field_desc = field_info.get('description', '')
                required = ' (Required)' if field_info.get('required', False) else ''
                example = f" Example: {field_info['example']}" if 'example' in field_info else ''
                context_parts.append(f"  • {field_name} ({field_type}){required}: {field_desc}{example}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_default_config(self, domain: str) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "domain": domain,
            "version": "1.0.0",
            "settings": {
                "enable_fuzzy_matching": True,
                "fuzzy_threshold": 0.3,
                "max_results_per_query": 100,
                "enable_follow_up_suggestions": True,
                "default_sort_order": "created_at"
            }
        }
    
    def _get_default_prompts(self, domain: str) -> Dict[str, Any]:
        """Get default prompts configuration"""
        return {
            "system_prompt": f"You are a MongoDB query expert for a {domain} system.",
            "query_generation_instructions": [
                "Generate precise MongoDB aggregation pipelines",
                "Always include proper field filtering",
                "Use appropriate collection names"
            ],
            "follow_up_templates": [],
            "analysis_guidelines": [
                "Create meaningful follow-ups or set to 'NONE'"
            ]
        }
    
    def clear_cache(self):
        """Clear all cached schemas, configs, and prompts"""
        self._schemas_cache.clear()
        self._configs_cache.clear()
        self._prompts_cache.clear()
        logger.info("Schema cache cleared")
    
    def validate_schema(self, domain: Optional[str] = None) -> bool:
        """Validate schema structure and required fields"""
        try:
            domain = domain or self.current_domain
            schema = self.load_schema(domain)
            
            # Check required top-level fields
            required_fields = ['domain', 'collections']
            for field in required_fields:
                if field not in schema:
                    logger.error(f"Missing required field '{field}' in schema for domain {domain}")
                    return False
            
            # Check collections structure
            collections = schema.get('collections', {})
            if not collections:
                logger.error(f"No collections defined in schema for domain {domain}")
                return False
            
            for coll_name, coll_info in collections.items():
                if 'fields' not in coll_info:
                    logger.error(f"Missing 'fields' in collection '{coll_name}' for domain {domain}")
                    return False
            
            logger.info(f"Schema validation passed for domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed for domain {domain}: {e}")
            return False


# Global schema loader instance
schema_loader = SchemaLoader()

# Convenience functions
def get_current_schema() -> Dict[str, Any]:
    """Get current active schema"""
    return schema_loader.load_schema()

def get_current_config() -> Dict[str, Any]:
    """Get current active configuration"""
    return schema_loader.load_config()

def get_current_prompts() -> Dict[str, Any]:
    """Get current active prompts"""
    return schema_loader.load_prompts()

def set_active_domain(domain: str) -> bool:
    """Set active schema domain"""
    return schema_loader.set_domain(domain)

def get_available_domains() -> List[str]:
    """Get list of available domains"""
    return schema_loader.get_available_domains()

def build_schema_context() -> str:
    """Build schema context for AI prompts"""
    return schema_loader.build_schema_context()
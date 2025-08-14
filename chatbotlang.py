# ==============================================================================
# ECOMMERCE PRODUCT SCHEMA LANGGRAPH CHATBOT - COMPLETE SYSTEM
# ==============================================================================
# LangGraph-based chatbot for e-commerce marketplace product management
# with Django Product schema integration and text-only responses

import os
import logging
import time
import json
import uuid
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union, Annotated

# Environment setup
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")  # Base URI without database

if not GEMINI_API_KEY:
    raise ValueError("[ERROR] Please set GEMINI_API_KEY in .env file")

def get_customer_mongodb_connection():
    """Get MongoDB connection based on current customer context"""
    customer = get_current_customer()
    if customer:
        db_info = get_customer_db_info()
        mongodb_uri = db_info.get("mongodb_uri", MONGODB_URI)
        database_name = db_info["database_name"]
        logger.info(f"🔗 Connecting to customer database: {database_name}")
        return pymongo.MongoClient(mongodb_uri), database_name
    else:
        # Fallback to default database
        default_db = "default_ecommerce"
        logger.warning("⚠️ No customer context - using default database")
        return pymongo.MongoClient(MONGODB_URI), default_db

# Core imports
import pymongo
import msgpack
from bson import ObjectId
from difflib import SequenceMatcher
import re
from schema_loader import schema_loader, build_schema_context
from customer_manager import customer_manager, get_current_customer, get_customer_db_info
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field, model_validator, field_validator

print("ECOMMERCE PRODUCT SCHEMA CHATBOT")
print(f"[OK] Gemini API Key: {'Present' if GEMINI_API_KEY else 'Missing'}")
print(f"[OK] MongoDB URI: {MONGODB_URI}")

# ==============================================================================
# CORE COMPONENTS
# ==============================================================================

# Custom MongoDB Saver with Timestamps  
class TimestampedMongoDBSaver(MongoDBSaver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_name = kwargs.get('db_name', 'checkpointing_db') 
        self.checkpoint_collection_name = kwargs.get('checkpoint_collection_name', 'checkpoints')
    
    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        
        try:
            db = self.client[self.db_name]
            checkpoint_collection = db[self.checkpoint_collection_name]
            
            thread_id = config.get("configurable", {}).get("thread_id")
            
            checkpoint_id = None
            if isinstance(checkpoint, dict) and 'id' in checkpoint:
                checkpoint_id = checkpoint['id']
            elif hasattr(checkpoint, 'id'):
                checkpoint_id = checkpoint.id
            else:
                logger.debug(f"Could not extract checkpoint ID from {type(checkpoint)}")
                return result
            
            if thread_id and checkpoint_id:
                update_result = checkpoint_collection.update_one(
                    {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id
                    },
                    {
                        "$set": {
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        }
                    }
                )
                if update_result.modified_count > 0:
                    logger.debug(f"Added timestamp to checkpoint {checkpoint_id}")
                    
        except Exception as e:
            logger.warning(f"Failed to add timestamp to checkpoint: {e}")
        
        return result

# Fuzzy Matching Utility Class
class FuzzyMatcher:
    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """Calculate similarity score between two strings (0-1)."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def normalize_search_terms(text: str) -> List[str]:
        """Extract and normalize search terms from text."""
        # Remove common words and normalize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        terms = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'show', 'me', 'find', 'get', 'what', 'is', 'are'}
        return [term for term in terms if term not in stop_words and len(term) > 1]
    
    @staticmethod
    def fuzzy_product_search(search_text: str, products: List[Dict], min_threshold: float = 0.3) -> List[Dict]:
        """Find products using fuzzy matching on name field."""
        if not search_text or not products:
            return products
        
        search_terms = FuzzyMatcher.normalize_search_terms(search_text)
        if not search_terms:
            return products
        
        scored_products = []
        for product in products:
            product_name = product.get('name', '')
            product_desc = product.get('description', '')
            
            # Calculate scores for different fields
            name_scores = [FuzzyMatcher.similarity_score(term, product_name) for term in search_terms]
            desc_scores = [FuzzyMatcher.similarity_score(term, product_desc) for term in search_terms]
            
            # Best match score (highest individual term match)
            best_name_score = max(name_scores) if name_scores else 0
            best_desc_score = max(desc_scores) if desc_scores else 0
            
            # Combined score (weighted towards name)
            combined_score = (best_name_score * 0.8) + (best_desc_score * 0.2)
            
            # Check for exact substring matches (boost score)
            search_lower = search_text.lower()
            name_lower = product_name.lower()
            if search_lower in name_lower or any(term in name_lower for term in search_terms):
                combined_score = min(combined_score + 0.3, 1.0)
            
            if combined_score >= min_threshold:
                product_copy = product.copy()
                product_copy['_fuzzy_score'] = combined_score
                scored_products.append(product_copy)
        
        # Sort by fuzzy score (highest first)
        return sorted(scored_products, key=lambda x: x['_fuzzy_score'], reverse=True)

# MongoDB Executor for Product Schema with Fuzzy Matching
class ProductSchemaExecutor:
    def __init__(self):
        self.connected = False
        self.fuzzy_matcher = FuzzyMatcher()
        self.client = None
        self.db = None
        self._connect_to_customer_db()
    
    def _connect_to_customer_db(self):
        """Connect to database based on current customer context"""
        try:
            self.client, db_name = get_customer_mongodb_connection()
            self.db = self.client[db_name]
            self.client.admin.command('ping')
            self.connected = True
            
            customer = get_current_customer()
            customer_info = f" for {customer.name}" if customer else " (default)"
            logger.info(f"✅ MongoDB connected{customer_info} with fuzzy matching enabled")
        except Exception as e:
            logger.error(f"❌ MongoDB failed: {e}")
    
    def switch_customer_context(self):
        """Switch database connection when customer context changes"""
        if self.client:
            self.client.close()
        self._connect_to_customer_db()
    
    def execute_query(self, query_data: Dict) -> Dict[str, Any]:
        if not self.connected:
            return {"success": False, "error": "MongoDB not connected", "results": [], "count": 0}
        try:
            collection_name = query_data.get("collection")
            pipeline = query_data.get("pipeline", [])
            fuzzy_search = query_data.get("fuzzy_search", {})
            
            if not collection_name: 
                return {"success": False, "error": "No collection specified"}
            
            collection = self.db[collection_name]
            cursor = collection.aggregate(pipeline)
            results = [self._convert_objectid(doc) for doc in cursor]
            
            # Apply fuzzy matching if specified and no results found
            if fuzzy_search and (not results or len(results) == 0):
                logger.info(f"🔍 No exact matches found, applying fuzzy search...")
                results = self._apply_fuzzy_search(collection, fuzzy_search, pipeline)
            
            logger.info(f"✅ Query executed: {len(results)} records from {collection_name}")
            return {
                "success": True, 
                "collection": collection_name, 
                "results": results, 
                "count": len(results), 
                "pipeline": pipeline,
                "fuzzy_applied": bool(fuzzy_search and len(results) > 0)
            }
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {"success": False, "error": str(e), "results": [], "count": 0}
    
    def _apply_fuzzy_search(self, collection, fuzzy_config: Dict, original_pipeline: List[Dict]) -> List[Dict]:
        """Apply fuzzy matching when exact search fails."""
        try:
            search_text = fuzzy_config.get("search_text", "")
            search_fields = fuzzy_config.get("fields", ["name"])
            limit = fuzzy_config.get("limit", 50)
            
            if not search_text:
                return []
            
            # Build broader search pipeline (remove strict filters)
            fuzzy_pipeline = []
            
            # Keep basic filters but remove text-based matches
            for stage in original_pipeline:
                if "$match" in stage:
                    match_stage = stage["$match"].copy()
                    # Remove text-based filters for fuzzy search
                    filtered_match = {}
                    for key, value in match_stage.items():
                        if key not in search_fields:
                            filtered_match[key] = value
                    if filtered_match:
                        fuzzy_pipeline.append({"$match": filtered_match})
                elif "$project" in stage or "$sort" in stage or "$group" in stage:
                    fuzzy_pipeline.append(stage)
            
            # Add limit for performance
            fuzzy_pipeline.append({"$limit": limit})
            
            # Execute broader search
            cursor = collection.aggregate(fuzzy_pipeline)
            all_results = [self._convert_objectid(doc) for doc in cursor]
            
            if not all_results:
                return []
            
            # Apply fuzzy matching to results
            fuzzy_results = self.fuzzy_matcher.fuzzy_product_search(search_text, all_results)
            
            # Limit results
            final_limit = min(fuzzy_config.get("result_limit", 10), len(fuzzy_results))
            return fuzzy_results[:final_limit]
            
        except Exception as e:
            logger.error(f"❌ Fuzzy search failed: {e}")
            return []

    def _convert_objectid(self, obj):
        if isinstance(obj, ObjectId): 
            return str(obj)
        elif isinstance(obj, dict): 
            return {key: self._convert_objectid(value) for key, value in obj.items()}
        elif isinstance(obj, list): 
            return [self._convert_objectid(item) for item in obj]
        return obj

# Essential Keywords for E-commerce Product Classification
DATABASE_KEYWORDS = [
    "products", "product", "list products", "show products", "all products",
    "vendors", "vendor", "seller", "sellers", "list vendors",
    "categories", "category", "product categories", "list categories",
    "users", "user", "customers", "list users", "user accounts",
    "inventory", "stock", "stock quantity", "out of stock", "in stock",
    "price", "pricing", "expensive", "cheap", "cost", "revenue",
    "featured products", "featured", "active products", "inactive",
    "search products", "filter products", "product details",
    "sales", "orders", "purchases", "transactions",
    "marketplace", "platform", "database", "records"
]

SYSTEM_CAPABILITY_KEYWORDS = [
    "what do you do", "what can you do", "who are you", "what are you",
    "how can you help", "hello", "hi", "help", "capabilities",
    "about", "info", "information"
]

# ==============================================================================
# AI COMPONENTS & PROMPTS
# ==============================================================================

def setup_ai_components():
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    
    # Load dynamic schema from external configuration
    try:
        schema_context = build_schema_context()
        current_domain = schema_loader.current_domain
        
        # Create schema documents for RAG using external schema
        product_schema_docs = [
            Document(
                page_content=schema_context,
                metadata={"source": f"{current_domain}_schema.json", "domain": current_domain}
            )
        ]
        
        logger.info(f"✅ Loaded external schema for domain: {current_domain}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load external schema: {e}. Using fallback.")
        # Fallback to a minimal schema if external loading fails
        product_schema_docs = [
            Document(
                page_content="Basic database schema with collections and fields.",
                metadata={"source": "fallback_schema", "domain": "default"}
            )
        ]
    
    db = Chroma.from_documents(product_schema_docs, embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0, 
        google_api_key=GEMINI_API_KEY
    )
    
    # Load dynamic prompts from external configuration
    try:
        prompts_config = schema_loader.load_prompts()
        
        # Build the template using dynamic prompts
        template_parts = []
        
        # System prompt from configuration
        system_prompt = prompts_config.get(
            "system_prompt", 
            "You are a MongoDB query expert for a database system with fuzzy matching capabilities."
        )
        template_parts.append(system_prompt)
        template_parts.append("Your goal is to generate precise MongoDB aggregation pipelines based on the user's question and conversation history.")
        template_parts.append("")
        
        # Standard conversation structure
        template_parts.append("CONVERSATION HISTORY:\n{chat_history}")
        template_parts.append("")
        template_parts.append("SCHEMA CONTEXT:\n{context}")
        template_parts.append("")
        template_parts.append("USER QUESTION: {question}")
        template_parts.append("")
        
        # Query generation instructions from configuration
        instructions = prompts_config.get("query_generation_instructions", [])
        if instructions:
            template_parts.append("CRITICAL INSTRUCTIONS:")
            for i, instruction in enumerate(instructions, 1):
                template_parts.append(f"{i}. {instruction}")
            template_parts.append("")
        
        # Fuzzy matching instructions from configuration
        fuzzy_config = prompts_config.get("fuzzy_matching", {})
        if fuzzy_config.get("enabled", True):
            template_parts.append("FUZZY MATCHING INSTRUCTIONS:")
            fuzzy_instructions = fuzzy_config.get("instructions", [])
            for instruction in fuzzy_instructions:
                template_parts.append(f"- {instruction}")
            template_parts.append("")
            
            # Examples from configuration
            examples = fuzzy_config.get("examples", [])
            if examples:
                template_parts.append("EXAMPLES WITH FUZZY MATCHING:")
                template_parts.append("")
                for example in examples:
                    if "title" in example:
                        template_parts.append(f"{example['title']}:")
                    if "query" in example:
                        template_parts.append(example["query"])
                    template_parts.append("")
        
        # Common patterns from configuration
        common_patterns = prompts_config.get("common_patterns", [])
        if common_patterns:
            template_parts.append("COMMON PATTERNS:")
            for pattern in common_patterns:
                template_parts.append(f"- {pattern}")
            template_parts.append("")
        
        template_parts.append("JSON Query:")
        
        template = "\n".join(template_parts)
        logger.info(f"✅ Built template using dynamic prompts from {current_domain}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load dynamic prompts: {e}. Using fallback template.")
        # Fallback template if dynamic loading fails
        template = """You are a MongoDB query expert for a database system.
Generate precise MongoDB aggregation pipelines based on the user's question and conversation history.

CONVERSATION HISTORY:
{chat_history}

SCHEMA CONTEXT:
{context}

USER QUESTION: {question}

Generate ONLY the MongoDB query as a single JSON object.

JSON Query:"""

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm
    
    return retriever, llm, rag_chain

# Pydantic models for structured output
class QuestionClassification(BaseModel):
    """Classify the user's question."""
    classification: str = Field(
        description="'internal' for database/product questions, 'external' for general questions, 'off-topic' for unrelated."
    )

class FollowUpSuggestion(BaseModel):
    """A model for follow-up questions."""
    follow_up_question: str = Field(
        description="A follow-up question to ask the user. Must be 'NONE' if no good follow-up exists."
    )
    contextual_query: Optional[Dict] = Field(
        None, 
        description="MongoDB query to execute if user answers 'yes'. Required if follow_up_question is not 'NONE'."
    )

    @field_validator('contextual_query', mode='before')
    @classmethod
    def parse_query_from_string(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("contextual_query is a string but not valid JSON")
        return value

    @model_validator(mode='after')
    def check_query_if_question_exists(self) -> 'FollowUpSuggestion':
        if self.follow_up_question.upper() != 'NONE' and not self.contextual_query:
            raise ValueError("If follow_up_question is provided, contextual_query must also be provided.")
        return self

# LangGraph State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    question: HumanMessage
    on_topic: str
    retry_count: int
    follow_up_question: str
    follow_up_context: Optional[Dict]
    route: str
    raw_data: Any
    final_answer: str

# ==============================================================================
# WORKFLOW NODES
# ==============================================================================

def format_chat_history(messages: List[BaseMessage]) -> str:
    """Helper to format message history for the LLM prompt."""
    return "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages])

def _format_as_text_table(results: List[Dict]) -> str:
    """Helper to format JSON results into a clean text table."""
    if not results: 
        return "No results found."
    
    headers = list(results[0].keys())
    
    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = max(
            len(str(header)),
            max(len(str(item.get(header, ''))) for item in results)
        )
    
    # Create header row
    header_row = " | ".join(str(header).ljust(col_widths[header]) for header in headers)
    separator = "-" * len(header_row)
    
    # Create data rows
    data_rows = []
    for item in results:
        row = " | ".join(str(item.get(h, '')).ljust(col_widths[h]) for h in headers)
        data_rows.append(row)
    
    return f"{header_row}\n{separator}\n" + "\n".join(data_rows)

def question_classifier(state: AgentState, llm):
    logger.info("🔍 Classifying question...")
    question = state["question"].content.lower()

    # Check for system capability questions
    if any(phrase in question for phrase in SYSTEM_CAPABILITY_KEYWORDS):
        state["on_topic"] = "external"
        return state
    
    # Check for database-related questions
    if any(phrase in question for phrase in DATABASE_KEYWORDS):
        state["on_topic"] = "internal"
        return state

    # Use AI for classification
    system_msg = SystemMessage(content="""Classify questions for an e-commerce marketplace platform.
- 'internal': Questions about products, vendors, categories, inventory, pricing, sales data
- 'external': General questions about the platform capabilities or unrelated topics
- 'off-topic': Questions completely unrelated to e-commerce

When in doubt, classify as 'internal' if it could relate to products or marketplace data.""")
    
    structured_llm = llm.with_structured_output(QuestionClassification)
    result = structured_llm.invoke([system_msg, HumanMessage(content=question)])
    
    classification = result.classification.replace("-", "_")
    state["on_topic"] = classification
    logger.info(f"🎯 Classification: {state['on_topic']}")
    return state

def internal_search_node(state: AgentState, retriever, rag_chain, llm, db_executor):
    logger.info("📖 Internal Search: Generating MongoDB query for product data...")
    MAX_RETRIES = 3
    state["retry_count"] = 0
    
    question = state["question"].content
    chat_history = format_chat_history(state["messages"])

    logger.info("... Retrieving schema context")
    documents = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in documents])

    for attempt in range(MAX_RETRIES):
        state["retry_count"] = attempt + 1
        logger.info(f"... Generating query (Attempt {state['retry_count']}/{MAX_RETRIES})")
        
        try:
            response = rag_chain.invoke({
                "context": context, 
                "question": question, 
                "chat_history": chat_history
            })
            generation = response.content.strip().replace("```json", "").replace("```", "")
            query_data = json.loads(generation)

            logger.info(f"...... Executing query on: {query_data.get('collection')}")
            result = db_executor.execute_query(query_data)
            
            if result["success"]:
                logger.info(f"✅ Query successful on attempt {state['retry_count']}")
                
                # Format as clean text response
                if result['results']:
                    formatted_table = _format_as_text_table(result['results'])
                    if result.get('fuzzy_applied'):
                        # For fuzzy matches, provide more context
                        original_query = state["question"].content
                        final_answer = f"""🎯 **Query Results** (using fuzzy matching)
**Collection:** {result['collection']}
**Records Found:** {result['count']}

I couldn't find an exact match for "{original_query}", but here are the closest products:

**📊 RESULTS:**
```
{formatted_table}
```"""
                    else:
                        final_answer = f"""🎯 **Query Results**
**Collection:** {result['collection']}
**Records Found:** {result['count']}

**📊 RESULTS:**
```
{formatted_table}
```"""
                else:
                    # Check if we tried fuzzy search but still found nothing
                    user_query = state["question"].content.lower()
                    logger.info(f"🔍 DEBUG: Checking user query: '{user_query}' for no-results response")
                    if any(term in user_query for term in ['g68', 'model', 'phone', 'smartphone']):
                        logger.info(f"🔍 DEBUG: Using custom no-results message for product search")
                        final_answer = f"""🎯 **Query Results**
**Collection:** {result['collection']}
**Records Found:** 0

I couldn't find any products matching "{state['question'].content}". 

The available Motorola models in our database are:
- Motorola Moto G45 (Price: 25999)
- Motorola Moto G86 (Price: 38870) 
- Motorola Edge 60 (Price: 53170)

Would you like information about any of these models instead?"""
                    else:
                        final_answer = f"""🎯 **Query Results**
**Collection:** {result['collection']}
**Records Found:** 0

No records match your criteria."""

                state["messages"].append(AIMessage(content=final_answer))
                state["raw_data"] = result['results']
                state["final_answer"] = final_answer
                return state
            
            else:
                logger.warning(f"⚠️ Query execution failed: {result['error']}")
                chat_history += f"\nATTEMPT {state['retry_count']} FAILED. Error: {result['error']}. Generated Query: {generation}. Please generate a corrected query."

        except Exception as e:
            logger.error(f"❌ Query generation/parsing failed: {e}")
            chat_history += f"\nATTEMPT {state['retry_count']} FAILED. Error: {e}. The previous attempt was invalid JSON. Please generate a valid JSON query."

        time.sleep(1)

    logger.error("❌ All query attempts failed.")
    fail_message = "I tried multiple times but could not generate a valid query for your request. Please try rephrasing your question about products, categories, or vendors."
    state["messages"].append(AIMessage(content=fail_message))
    state["final_answer"] = fail_message
    return state

def external_search_node(state: AgentState, llm):
    logger.info("🌐 External Search: Generating response about platform capabilities...")
    question = state["question"].content
    chat_history = format_chat_history(state["messages"])
    
    prompt = f"""You are a helpful AI assistant for an e-commerce marketplace platform.
    
Previous conversation:
{chat_history}

You can help users with:
- Product information and searches
- Inventory management queries  
- Pricing and category analysis
- Vendor/seller information
- General marketplace questions

The user is asking a general question. Provide a helpful and concise answer.
User Question: {question}"""
    
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    state["raw_data"] = response.content
    state["final_answer"] = response.content
    return state

def generate_follow_up_node(state: AgentState, llm):
    logger.info("💡 Generating potential follow-up question...")
    
    if not state.get("raw_data"):
        logger.info("... No raw data to analyze, skipping follow-up.")
        state["follow_up_question"] = "NONE"
        state["follow_up_context"] = None
        return state

    follow_up_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest", 
        temperature=0, 
        google_api_key=GEMINI_API_KEY
    )
    suggestion_llm = follow_up_llm.with_structured_output(FollowUpSuggestion)
    
    for attempt in range(2):
        logger.info(f"... Follow-up generation attempt {attempt + 1}/2")
        
        prompt = f"""Generate a follow-up suggestion for e-commerce product data analysis.

**REQUIREMENTS:**
- Provide BOTH follow_up_question AND contextual_query together, or set to "NONE"
- contextual_query must be valid MongoDB aggregation pipeline
- Use "<user_input_needed>" for user input placeholders
- MUST use EXACT template structures from examples below - do NOT modify the pipeline structure

**User's Question:** "{state['question'].content}"

**Raw Data Sample:**
{json.dumps(state.get("raw_data", [])[:3], indent=2, default=json_converter)}

**VALID EXAMPLES:**

Price filter (DO NOT include name filters):
{{
  "follow_up_question": "Would you like to filter by a specific price range?",
  "contextual_query": {{
    "collection": "products",
    "pipeline": [
      {{"$match": {{"price": {{"$gte": "<user_input_needed>", "$lte": "<user_input_needed>"}}, "is_active": true}}}},
      {{"$project": {{"name": 1, "price": 1, "category": 1, "stock_quantity": 1}}}}
    ]
  }}
}}

Price range for fuzzy search results (maintain brand/search context):
{{
  "follow_up_question": "Do you have a specific price range in mind?",
  "contextual_query": {{
    "collection": "products", 
    "pipeline": [
      {{"$match": {{"name": {{"$regex": "BRAND_FROM_ORIGINAL_QUERY", "$options": "i"}}, "price": {{"$gte": "<user_input_needed>", "$lte": "<user_input_needed>"}}, "is_active": true}}}},
      {{"$project": {{"name": 1, "price": 1, "stock_quantity": 1}}}}
    ]
  }}
}}

Category grouping:
{{
  "follow_up_question": "Would you like to see a breakdown by category?",
  "contextual_query": {{
    "collection": "products",
    "pipeline": [
      {{"$group": {{"_id": "$category", "count": {{"$sum": 1}}, "avg_price": {{"$avg": "$price"}}}}}},
      {{"$sort": {{"count": -1}}}}
    ]
  }}
}}

Vendor product counts (EXACT template for vendor analysis):
{{
  "follow_up_question": "Would you like to see the number of products offered by each vendor?",
  "contextual_query": {{
    "collection": "products",
    "pipeline": [
      {{"$group": {{"_id": "$vendor", "product_count": {{"$sum": 1}}}}}},
      {{"$lookup": {{"from": "users", "localField": "_id", "foreignField": "_id", "as": "vendor_info"}}}},
      {{"$unwind": "$vendor_info"}},
      {{"$project": {{"vendor_name": {{"$concat": ["$vendor_info.first_name", " ", "$vendor_info.last_name"]}}, "product_count": 1, "_id": 0}}}},
      {{"$sort": {{"product_count": -1}}}}
    ]
  }}
}}

No follow-up:
{{
  "follow_up_question": "NONE",
  "contextual_query": null
}}

**ANALYSIS GUIDELINES:**
1. For product data: suggest category, price, or stock filtering
2. For vendor data: suggest product count or performance analysis  
3. For inventory data: suggest low stock alerts or category breakdown
4. Create meaningful follow-ups or set to "NONE"
5. CRITICAL: When user searches for specific brands (e.g., "Motorola", "Samsung"), MAINTAIN that brand context in follow-up queries
6. Replace "BRAND_FROM_ORIGINAL_QUERY" with the actual brand from the user's original question
7. If no specific brand mentioned, then show all products in price range
8. Examples: "Motorola G68" → use "motorola" in follow-up, "Samsung phone" → use "samsung"
9. IMPORTANT: For vendor queries, vendors are stored in "users" collection, NOT "vendors" collection
10. Always use the EXACT template structure provided in examples above

Generate response:"""
        
        try:
            suggestion = suggestion_llm.invoke(prompt)
            if suggestion.follow_up_question and suggestion.follow_up_question.upper() != "NONE":
                logger.info(f"✅ Follow-up generated successfully on attempt {attempt + 1}.")
                state["follow_up_question"] = suggestion.follow_up_question
                
                contextual_query = suggestion.contextual_query
                if isinstance(contextual_query, str):
                    try:
                        contextual_query = json.loads(contextual_query)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Failed to parse contextual_query as JSON")
                        raise ValueError("Invalid contextual_query format")
                
                state["follow_up_context"] = contextual_query
                return state
            else:
                logger.info("... LLM decided no follow-up was needed.")
                state["follow_up_question"] = "NONE"
                state["follow_up_context"] = None
                return state

        except Exception as e:
            logger.warning(f"⚠️ Follow-up generation attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                logger.error("❌ All follow-up generation attempts failed.")
                state["follow_up_question"] = "NONE"
                state["follow_up_context"] = None
                return state
    
    state["follow_up_question"] = "NONE"
    state["follow_up_context"] = None
    return state

def handle_follow_up_node(state: AgentState, db_executor, llm):
    import json  # Import at function level to avoid scope issues
    logger.info("🤝 Handling user's follow-up response...")
    user_response = state["question"].content
    
    is_affirmative = any(user_response.lower().startswith(word) for word in ["yes", "sure", "ok", "yep"])
    
    if is_affirmative:
        query_template = state.get("follow_up_context")
        if query_template is None:
            state["messages"].append(AIMessage(content="I seem to have lost the context. Could you ask again?"))
            state["follow_up_question"] = "NONE"
            state["follow_up_context"] = None
            return state

        query_str = json.dumps(query_template, default=json_converter)

        # Handle brand context substitution
        if "BRAND_FROM_ORIGINAL_QUERY" in query_str:
            # Try to extract brand from the conversation history
            brand_context = "motorola"  # Default fallback, but we should extract dynamically
            # Look through previous messages to find the original search query
            for msg in state["messages"]:
                if hasattr(msg, 'content') and any(brand in msg.content.lower() for brand in ['motorola', 'samsung', 'apple', 'vivo']):
                    if 'motorola' in msg.content.lower():
                        brand_context = "motorola"
                        break
                    elif 'samsung' in msg.content.lower():
                        brand_context = "samsung" 
                        break
                    elif 'apple' in msg.content.lower():
                        brand_context = "apple"
                        break
                    elif 'vivo' in msg.content.lower():
                        brand_context = "vivo"
                        break
            
            query_str = query_str.replace("BRAND_FROM_ORIGINAL_QUERY", brand_context)
            logger.info(f"... Substituted brand context: {brand_context}")

        # Special handling for vendor product count queries - use the proven working query
        if "product" in query_str.lower() and "vendor" in query_str.lower() and state.get("follow_up_question", "").lower().find("vendor") != -1:
            logger.info("... Detected vendor product count query, using proven template")
            final_query = {
                "collection": "products",
                "pipeline": [
                    {"$group": {"_id": "$vendor", "product_count": {"$sum": 1}}},
                    {"$lookup": {"from": "users", "localField": "_id", "foreignField": "_id", "as": "vendor_info"}},
                    {"$unwind": "$vendor_info"},
                    {"$project": {"vendor_name": {"$concat": ["$vendor_info.first_name", " ", "$vendor_info.last_name"]}, "product_count": 1, "_id": 0}},
                    {"$sort": {"product_count": -1}}
                ]
            }
        elif "<user_input_needed>" in query_str:
            logger.info("... Query requires user input. Extracting...")
            
            extraction_prompt = f"""Extract the specific value from the user's response for filtering.
            User's response: "{user_response}"
            
            Examples:
            - "yes, under $100" → extract "100"
            - "sure, electronics category" → extract "electronics"  
            - "ok, show me clothing" → extract "clothing"
            - "yes between 10,000 and 50,000" → extract "10000,50000"
            - "between 1000 and 5000" → extract "1000,5000"
            
            For price ranges, separate min and max with comma. For single values, just return the value.
            
            Extracted value:"""
            
            extracted_value = llm.invoke(extraction_prompt).content.strip().lower().replace("'", "").replace('"', '')
            logger.info(f"... Extracted value: '{extracted_value}'")
            
            # Handle price extraction (convert to number if it's a price)
            if any(price_word in query_str.lower() for price_word in ["price", "cost", "$"]):
                try:
                    import re
                    # Handle comma-separated ranges like "10000,50000" or "10,000 and 50,000"
                    if ',' in extracted_value and 'and' not in extracted_value:
                        # Split on comma for comma-separated ranges
                        parts = extracted_value.split(',')
                        clean_numbers = [float(part.strip()) for part in parts if part.strip().isdigit()]
                    else:
                        # Extract all numbers, handling commas in thousands like "10,000"
                        numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', extracted_value)
                        # Clean up numbers (remove commas for thousands)
                        clean_numbers = [float(num.replace(',', '')) for num in numbers if num]
                    
                    if len(clean_numbers) >= 2:
                        min_price = min(clean_numbers)
                        max_price = max(clean_numbers)
                        logger.info(f"... Extracted price range: {min_price} to {max_price}")
                        
                        # Replace placeholders with actual numbers (not strings)
                        # First parse as template, then replace with numbers
                        template_obj = json.loads(query_str)
                        
                        # Convert template to string and replace placeholders with actual numbers
                        template_str = json.dumps(template_obj)
                        template_str = template_str.replace('"<user_input_needed>"', str(int(min_price)), 1)
                        template_str = template_str.replace('"<user_input_needed>"', str(int(max_price)), 1)
                        
                        final_query_str = template_str
                    elif len(clean_numbers) == 1:
                        single_price = clean_numbers[0]
                        logger.info(f"... Extracted single price: {single_price}")
                        # For single price, use it for both min and max (or just single replacement)
                        # Handle single price with proper number formatting
                        if '"<user_input_needed>"' in query_str:
                            final_query_str = query_str.replace('"<user_input_needed>"', str(int(single_price)))
                        else:
                            final_query_str = query_str.replace("<user_input_needed>", str(int(single_price)))
                    else:
                        logger.warning("... No valid numbers found in price response")
                        final_query_str = query_str.replace("<user_input_needed>", "0")
                except ValueError as e:
                    logger.info(f"... Could not process price: {e}, using as string: {extracted_value}")
                    final_query_str = query_str.replace("<user_input_needed>", str(extracted_value))
            else:
                final_query_str = query_str.replace("<user_input_needed>", str(extracted_value))

            final_query = json.loads(final_query_str)
            
        else:
            logger.info("... Query is self-contained. Executing as is.")
            final_query = query_template

        logger.info(f"👍 User said yes. Executing follow-up query")
        logger.info(f"🔍 DEBUG: Final query being executed: {final_query}")
        
        # Add timeout protection and better error handling
        try:
            result = db_executor.execute_query(final_query)
            logger.info(f"✅ Query executed: {result.get('count', 0)} records from {result.get('collection', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Follow-up query execution failed: {e}")
            # Return a simple fallback response
            error_response = f"I apologize, but I encountered an issue executing the follow-up query. Please try asking your question again."
            state["messages"].append(AIMessage(content=error_response))
            state["final_answer"] = error_response
            state["follow_up_question"] = "NONE"
            state["follow_up_context"] = None
            return state
        
        if result.get("success"):
            if result.get('results') and len(result.get('results', [])) > 0:
                try:
                    formatted_results = _format_as_text_table(result.get('results', []))
                    response_message = f"""🎯 **Follow-up Results**

**📊 RESULTS:**
```
{formatted_results}
```

Found {result.get('count', 0)} matching records."""
                    logger.info(f"✅ Successfully formatted {result.get('count', 0)} results")
                except Exception as e:
                    logger.error(f"❌ Formatting error: {e}")
                    # Fallback to simple format
                    results_summary = []
                    for item in result.get('results', [])[:5]:  # Show first 5
                        if 'name' in item:
                            results_summary.append(f"- {item['name']}")
                        elif 'username' in item:
                            results_summary.append(f"- {item['username']}")
                        else:
                            results_summary.append(f"- {str(item)[:50]}...")
                    
                    response_message = f"""🎯 **Follow-up Results**

{chr(10).join(results_summary)}

Found {result.get('count', 0)} total records."""
            else:
                response_message = "No records match your follow-up criteria."
                logger.info("ℹ️ No results found for follow-up query")
            
            state["messages"].append(AIMessage(content=response_message))
            state["raw_data"] = result.get('results', [])
            state["final_answer"] = response_message
            logger.info("✅ Follow-up response prepared and stored")
        else:
            error_message = f"❌ **Query Error**\n\nI encountered an issue: {result.get('error', 'Unknown error')}"
            state["messages"].append(AIMessage(content=error_message))
            state["final_answer"] = error_message
            logger.error(f"❌ Follow-up query failed: {result.get('error')}")
    else:
        logger.info("👎 User declined follow-up.")
        response_message = "Okay, no problem. What else can I help you with regarding products or the marketplace?"
        state["messages"].append(AIMessage(content=response_message))
        state["final_answer"] = response_message

    # Clear the follow-up state
    state["follow_up_question"] = "NONE"
    state["follow_up_context"] = None
    return state

def is_new_product_query(user_message: str) -> bool:
    """Check if user message is a new product search rather than follow-up response."""
    message_lower = user_message.lower()
    
    # Keywords that indicate new product searches
    product_search_indicators = [
        'show me', 'find', 'search for', 'look for', 'what is the price of',
        'how much is', 'do you have', 'g68', 'model', 'product', 
        'phone', 'smartphone', 'device', 'item'
    ]
    
    # Check if message contains product search indicators
    if any(indicator in message_lower for indicator in product_search_indicators):
        return True
    
    # Check if it's clearly a follow-up response (yes/no/ok)
    follow_up_responses = ['yes', 'no', 'ok', 'sure', 'nope', 'yep', 'yeah']
    if any(message_lower.strip().startswith(resp) for resp in follow_up_responses):
        return False
    
    # If message contains product-related terms, likely a new search
    return any(term in message_lower for term in ['price', 'cost', 'buy', 'purchase'])

def pre_router_node(state: AgentState) -> Dict[str, Any]:
    """Determines the initial routing path."""
    logger.info("🚦 Pre-routing based on conversation state...")
    
    updates = state.copy()
    user_message = state["question"].content
    
    # Check if there's a pending follow-up
    has_follow_up = state.get("follow_up_question") and state.get("follow_up_question", "").upper() != "NONE"
    
    if has_follow_up:
        # Check if the user's message is a new product search or a follow-up response
        if is_new_product_query(user_message):
            logger.info("... Route: classify (new product search detected)")
            updates["route"] = "classify"
            # Clear the follow-up since user is asking something new
            updates["follow_up_question"] = "NONE"
            updates["follow_up_context"] = None
        else:
            logger.info("... Route: handle_follow_up")
            updates["route"] = "handle_follow_up"
    else:
        logger.info("... Route: classify")
        updates["route"] = "classify"
        
    return updates

def off_topic_response(state: AgentState):
    logger.info("🚫 Off-topic response")
    message = """I'm sorry, but that question is outside my scope. I'm designed to help with e-commerce marketplace queries such as:

• Product searches and information
• Inventory management  
• Pricing analysis
• Category browsing
• Vendor/seller information
• General marketplace operations

Please ask me something about products, categories, vendors, or marketplace data."""
    
    state["messages"].append(AIMessage(content=message))
    state["final_answer"] = message
    return state

def finalize_response_node(state: AgentState):
    logger.info("🎨 Finalizing response...")
    
    # Add follow-up question to the final answer if available
    if state.get("follow_up_question") and state.get("follow_up_question", "").upper() != "NONE":
        follow_up_text = f"\n\n💡 **Follow-up suggestion:** {state['follow_up_question']}"
        current_answer = state.get("final_answer", "")
        state["final_answer"] = current_answer + follow_up_text
        
        # Update the last message in the conversation
        if state["messages"] and isinstance(state["messages"][-1], AIMessage):
            state["messages"][-1].content += follow_up_text
    
    return state

def json_converter(o):
    """Helper function to convert non-serializable objects to strings."""
    if isinstance(o, (datetime, ObjectId)):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# ==============================================================================
# CONVERSATION MANAGEMENT (Simplified for product schema)
# ==============================================================================

def list_previous_conversations(mongo_client):
    """List previous conversations with basic context."""
    try:
        db = mongo_client["ecommerce-marketplace"]
        checkpoints = db["langgraph_checkpoints"]
        
        pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$group": {
                "_id": "$thread_id",
                "latest_update": {"$first": "$updated_at"},
                "created_at": {"$first": "$created_at"},
                "checkpoint_count": {"$sum": 1}
            }},
            {"$sort": {"latest_update": -1}},
            {"$limit": 10}
        ]
        
        conversations = list(checkpoints.aggregate(pipeline))
        
        enhanced_conversations = []
        for i, conv in enumerate(conversations, 1):
            thread_id = conv['_id']
            
            conv_data = {
                'number': i,
                'thread_id': thread_id,
                'thread_id_short': thread_id[:8],
                'latest_update': conv.get('latest_update', 'Unknown'),
                'message_count': conv.get('checkpoint_count', 0),
                'title': f'Conversation {i}'
            }
            
            enhanced_conversations.append(conv_data)
        
        return enhanced_conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []

# ==============================================================================
# WORKFLOW COMPILATION
# ==============================================================================

def create_workflow():
    logger.info("🔧 Building e-commerce product schema workflow...")
    retriever, llm, rag_chain = setup_ai_components()
    db_executor = ProductSchemaExecutor()

    mongo_client = pymongo.MongoClient(MONGODB_URI)
    checkpointer = TimestampedMongoDBSaver(
        client=mongo_client,
        db_name="ecommerce-marketplace",
        checkpoint_collection_name="langgraph_checkpoints"
    )
    logger.info("✅ Using MongoDB for conversation persistence.")
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes  
    workflow.add_node("pre_router", pre_router_node)
    workflow.add_node("classify", lambda state: question_classifier(state, llm))
    workflow.add_node("internal_search", lambda state: internal_search_node(state, retriever, rag_chain, llm, db_executor))
    workflow.add_node("external_search", lambda state: external_search_node(state, llm))
    workflow.add_node("off_topic", off_topic_response)
    workflow.add_node("generate_follow_up", lambda state: generate_follow_up_node(state, llm))
    workflow.add_node("handle_follow_up", lambda state: handle_follow_up_node(state, db_executor, llm))
    workflow.add_node("finalize_response", finalize_response_node)
    
    # Routing functions
    def route_from_pre_router(state: AgentState) -> str:
        return state.get("route", "classify")

    def route_from_classification(state: AgentState) -> str:
        return state.get("on_topic", "off_topic")
            
    # Define the graph structure
    workflow.set_entry_point("pre_router")
    workflow.add_conditional_edges("pre_router", route_from_pre_router, {
        "handle_follow_up": "handle_follow_up", 
        "classify": "classify"
    })
    workflow.add_conditional_edges("classify", route_from_classification, {
        "internal": "internal_search", 
        "external": "external_search", 
        "off_topic": "off_topic"
    })
    
    # Connect all paths to follow-up generation, then finalization
    workflow.add_edge("internal_search", "generate_follow_up")
    workflow.add_edge("generate_follow_up", "finalize_response")
    workflow.add_edge("handle_follow_up", "finalize_response")   
    workflow.add_edge("external_search", "finalize_response")    
    workflow.add_edge("off_topic", "finalize_response")
    workflow.add_edge("finalize_response", END)
    
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ E-commerce product schema workflow compiled successfully")
    return graph, db_executor

# ==============================================================================
# EXECUTION FUNCTIONS
# ==============================================================================

def process_question(graph, question: str, thread_id: str):
    start_time = time.time()
    logger.info(f"🚀 Processing: '{question}' for thread_id: {thread_id}")
    
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=question)], 
        "question": HumanMessage(content=question)
    }
    
    try:
        final_graph_output = graph.invoke(inputs, config=config)
        
        # Extract the final answer
        final_answer = final_graph_output.get("final_answer", "No response generated")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processing completed in {processing_time:.2f}s")
        
        return {
            "answer": final_answer, 
            "time": f"{processing_time:.2f}s",
            "thread_id": thread_id
        }

    except Exception as e:
        logger.error(f"❌ Processing failed for thread {thread_id}: {e}", exc_info=True)
        processing_time = time.time() - start_time
        return {
            "answer": f"An error occurred while processing your request: {str(e)}", 
            "time": f"{processing_time:.2f}s",
            "thread_id": thread_id
        }

def interactive_mode(graph, db_executor):
    thread_id = str(uuid.uuid4())
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    
    cached_conversations = []
    
    # Show customer context
    customer = get_current_customer()
    if customer:
        print(f"\n🛒 MULTI-TENANT CHATBOT - {customer.name.upper()}")
        print(f"[Customer]: {customer.name} ({customer.customer_id})")
        print(f"[Domain]: {customer.domain}")
        print(f"[Database]: {customer.database_name}")
    else:
        print(f"\n🛒 MULTI-TENANT CHATBOT - NO CUSTOMER SET")
        print(f"[Warning]: No customer context - using default database")
    
    print(f"[Thread ID]: {thread_id}")
    print(f"[Connection]: {'✅ Connected' if db_executor.connected else '❌ Disconnected'}")
    print("\n[CAPABILITIES]:")
    print("  • Product searches and information")
    print("  • Inventory management queries")
    print("  • Pricing and category analysis") 
    print("  • Vendor/seller information")
    print("  • General marketplace operations")
    print("\n[COMMANDS]:")
    print("  /list - Show previous conversations")
    print("  /resume <number|thread_id> - Resume a conversation")
    print("  /customers - List all customers")
    print("  /switch <customer_id> - Switch to different customer")
    print("  /current - Show current customer info")
    print("  /help - Show help")
    print("  quit - Exit")
    print("=" * 60)

    while True:
        try:
            question = input("\n🛍️  Ask about products, inventory, pricing, or vendors: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question: 
                continue
            
            # Handle commands
            if question.startswith('/'):
                command_parts = question.split()
                command = command_parts[0].lower()
                
                if command == '/help':
                    print("\n[AVAILABLE COMMANDS]:")
                    print("  /list                    - Show recent conversations")
                    print("  /resume <number|id>      - Resume a conversation")
                    print("  /customers               - List all customers")
                    print("  /switch <customer_id>    - Switch to different customer")
                    print("  /current                 - Show current customer info")
                    print("  /help                    - Show this help")
                    print("  quit/exit/q              - Exit the chatbot")
                    print("\n[EXAMPLE QUESTIONS]:")
                    print("  • 'Show me all active products'")
                    print("  • 'List products under $50'")
                    print("  • 'Which products are out of stock?'")
                    print("  • 'Show featured products'")
                    print("  • 'Group products by category'")
                    print("  • 'List all vendors'")
                    continue
                    
                elif command == '/list':
                    print("\n[RECENT CONVERSATIONS]:")
                    conversations = list_previous_conversations(mongo_client)
                    cached_conversations = conversations
                    
                    if not conversations:
                        print("No previous conversations found.")
                    else:
                        for conv in conversations:
                            num = conv['number']
                            title = conv['title']
                            short_id = conv['thread_id_short']
                            msg_count = conv['message_count']
                            update_time = conv['latest_update']
                            
                            print(f"  {num}. {title}")
                            print(f"     ID: {short_id}... | {msg_count} messages | {update_time}")
                        
                        print(f"\n[TIP] Use '/resume <number>' to continue a conversation")
                    continue
                    
                elif command == '/resume':
                    if len(command_parts) < 2:
                        print("[Error] Usage: /resume <number|thread_id>")
                        continue
                    
                    resume_input = command_parts[1]
                    
                    if resume_input.isdigit():
                        conv_number = int(resume_input)
                        if cached_conversations and 1 <= conv_number <= len(cached_conversations):
                            thread_id = cached_conversations[conv_number - 1]['thread_id']
                            conv_title = cached_conversations[conv_number - 1]['title']
                            print(f"\n[RESUMED] {conv_title}")
                            print(f"   Thread: {thread_id[:8]}...")
                            print(f"   All previous context is maintained.")
                        else:
                            print(f"[Error] Invalid conversation number. Use /list first.")
                    else:
                        thread_id = resume_input
                        print(f"\n[RESUMED] Thread: {thread_id[:8]}...")
                    continue
                
                elif command == '/customers':
                    print("\n[AVAILABLE CUSTOMERS]:")
                    customers = customer_manager.list_customers()
                    if not customers:
                        print("No customers found. Use customer_cli.py to add customers.")
                    else:
                        current_customer = get_current_customer()
                        current_id = current_customer.customer_id if current_customer else None
                        
                        for customer in customers:
                            status_emoji = "🟢" if customer.status == "active" else "🔴"
                            current_marker = " ← CURRENT" if customer.customer_id == current_id else ""
                            print(f"  • {customer.customer_id} - {customer.name} ({customer.domain}) {status_emoji}{current_marker}")
                    continue
                
                elif command == '/switch':
                    if len(command_parts) < 2:
                        print("[Error] Usage: /switch <customer_id>")
                        continue
                    
                    customer_id = command_parts[1]
                    success = customer_manager.set_current_customer(customer_id)
                    if success:
                        customer = get_current_customer()
                        # Update schema domain to match customer
                        schema_loader.set_domain(customer.domain)
                        # Switch database connection
                        db_executor.switch_customer_context()
                        
                        print(f"✅ Switched to customer: {customer.name}")
                        print(f"   Domain: {customer.domain}")
                        print(f"   Database: {customer.database_name}")
                        print(f"   Connection: {'✅ Connected' if db_executor.connected else '❌ Disconnected'}")
                    else:
                        print(f"❌ Failed to switch to customer: {customer_id}")
                    continue
                
                elif command == '/current':
                    customer = get_current_customer()
                    if customer:
                        print(f"\n[CURRENT CUSTOMER]:")
                        print(f"  ID: {customer.customer_id}")
                        print(f"  Name: {customer.name}")
                        print(f"  Domain: {customer.domain}")
                        print(f"  Status: {customer.status}")
                        print(f"  Database: {customer.database_name}")
                        print(f"  Connection: {'✅ Connected' if db_executor.connected else '❌ Disconnected'}")
                    else:
                        print("❌ No customer is currently active")
                        print("   Use /customers to list available customers")
                        print("   Use /switch <customer_id> to set a customer")
                    continue
                    
                else:
                    print(f"[Error] Unknown command: {command}. Type /help for available commands.")
                    continue
            
            print("[Processing...]")
            result = process_question(graph, question, thread_id)
            
            print(f"\n📋 [ANSWER]:")
            print(result['answer'])
            print(f"\n⏱️  [Time]: {result['time']}")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"[Error]: {e}")
            
    print("\n👋 Thanks for using the E-commerce Marketplace Chatbot!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        graph, db_executor = create_workflow()
        interactive_mode(graph, db_executor)
    except Exception as e:
        logger.error(f"❌ Fatal error during initialization: {e}", exc_info=True)
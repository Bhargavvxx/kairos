# backend/app/main.py

import logging
import json
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
# Import our router and services
from .routers import documents
from .routers import hackrx
from .services.vector_service import VectorService
from .services.graph_service import GraphService, FileStorageAdapter as GraphStorageAdapter
from .models.schemas import GraphData
from networkx.readwrite import json_graph  # Add this for graph loading
from .dependencies import (
    get_vector_service,
    get_graph_service,
    get_llm_client,
)
load_dotenv()
# --- Application Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Dynamic Configuration from Environment ---
class DynamicConfig:
    """Dynamic configuration that reloads from environment variables on each access."""
    
    def __init__(self):
        self._last_env_mtime = 0
        self._env_file_path = ".env"
        self._reload_env()
    
    def _reload_env(self):
        """Reload environment variables if .env file has changed."""
        try:
            if os.path.exists(self._env_file_path):
                current_mtime = os.path.getmtime(self._env_file_path)
                if current_mtime > self._last_env_mtime:
                    load_dotenv(override=True)  # Override existing env vars
                    self._last_env_mtime = current_mtime
                    logger.info(f"🔄 Reloaded .env file (modified at {current_mtime})")
        except Exception as e:
            logger.warning(f"Could not reload .env file: {e}")
    
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        self._reload_env()
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def ANTHROPIC_API_KEY(self) -> Optional[str]:
        self._reload_env()
        return os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def GOOGLE_API_KEY(self) -> Optional[str]:
        self._reload_env()
        return os.getenv("GOOGLE_API_KEY")
    
    @property
    def VECTOR_PERSIST_DIR(self) -> str:
        self._reload_env()
        return os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
    
    @property
    def ENABLE_GRAPH_EXTRACTION(self) -> bool:
      self._reload_env()
      value = os.getenv("ENABLE_GRAPH_EXTRACTION", "true").lower()
      return value in ("true", "1", "yes", "on")
    
    @property
    def GRAPH_PERSIST_DIR(self) -> str:
        self._reload_env()
        return os.getenv("GRAPH_PERSIST_DIR", "./graph_db")
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        self._reload_env()
        return os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
    
    @property
    def LLM_MODEL(self) -> str:
        self._reload_env()
        return os.getenv("LLM_MODEL", "gpt-4")
    
    @property
    def LLM_PROVIDER(self) -> str:
        self._reload_env()
        return os.getenv("LLM_PROVIDER", "openai")
    
    @property
    def ALLOWED_ORIGINS(self) -> list:
        self._reload_env()
        return os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    @property
    def USE_MOCK_LLM(self) -> bool:
        self._reload_env()
        value = os.getenv("USE_MOCK_LLM", "true").lower()
        return value in ("true", "1", "yes", "on")

config = DynamicConfig()

# Add debug logging
logger.info(f"Config values - USE_MOCK_LLM: {config.USE_MOCK_LLM}")
logger.info(f"Config values - LLM_PROVIDER: {config.LLM_PROVIDER}")
logger.info(f"Config values - GOOGLE_API_KEY: {'***' if config.GOOGLE_API_KEY else 'None'}")
logger.info(f"Config values - OPENAI_API_KEY: {'***' if config.OPENAI_API_KEY else 'None'}")
logger.info(f"Config values - LLM_MODEL: {config.LLM_MODEL}")
logger.info(f"Config values - ENABLE_GRAPH_EXTRACTION: {config.ENABLE_GRAPH_EXTRACTION}")

# --- LLM Client Setup ---

class MockLLMClient:
    """Mock LLM client for testing without API costs."""
    
    async def generate(self, prompt: str) -> str:
        """Generate a mock response for testing."""
        logger.info("MockLLMClient: Generating mock response...")
        
        # Check if this is a question-answering prompt
        if "QUESTION:" in prompt and "ANSWER:" in prompt:
            # This is a Q&A prompt, return a simple answer
            return "Based on the provided context, the answer to your question is available in the document."
        
        # Check if this is a graph extraction prompt
        if "extract" in prompt.lower() and ("knowledge graph" in prompt.lower() or "json" in prompt.lower()):
            # Return appropriate mock response for graph extraction
            if "insurance" in prompt.lower() or "policy" in prompt.lower() or "travel" in prompt.lower():
                # Enhanced mock response for insurance/travel documents
                mock_response = {
                    "nodes": [
                        {
                            "id": "chunk0_person_policyholder",
                            "type": "Person",
                            "label": "Policyholder",
                            "properties": {"role": "Insured"},
                            "confidence": 0.9
                        },
                        {
                            "id": "chunk0_policy_travel_insurance",
                            "type": "Policy",
                            "label": "Travel Insurance Policy",
                            "properties": {
                                "type": "travel",
                                "coverage": "comprehensive",
                                "sum_insured": "100000",
                                "grace_period": "30 days"
                            },
                            "confidence": 0.95
                        },
                        {
                            "id": "chunk0_company_insurance_provider",
                            "type": "Company",
                            "label": "Insurance Provider",
                            "properties": {"type": "insurer"},
                            "confidence": 0.9
                        },
                        {
                            "id": "chunk0_amount_deductible",
                            "type": "Amount",
                            "label": "Deductible Amount",
                            "properties": {"value": "5000", "currency": "INR"},
                            "confidence": 0.85
                        }
                    ],
                    "edges": [
                        {
                            "source": "chunk0_person_policyholder",
                            "target": "chunk0_policy_travel_insurance",
                            "label": "owns",
                            "type": "owns",
                            "confidence": 0.9
                        },
                        {
                            "source": "chunk0_company_insurance_provider",
                            "target": "chunk0_policy_travel_insurance",
                            "label": "issued",
                            "type": "governs",
                            "confidence": 0.95
                        },
                        {
                            "source": "chunk0_policy_travel_insurance",
                            "target": "chunk0_amount_deductible",
                            "label": "has_deductible",
                            "type": "has_property",
                            "confidence": 0.85
                        }
                    ]
                }
            else:
                # Default mock response
                mock_response = {
                    "nodes": [
                        {
                            "id": "chunk0_entity_document",
                            "type": "Document",
                            "label": "Document",
                            "properties": {},
                            "confidence": 0.8
                        }
                    ],
                    "edges": []
                }
            
            return json.dumps(mock_response, indent=2)
        
        # Default response for other prompts
        return "This is a mock response from the LLM client."

class GeminiLLMClient:
    """Gemini LLM client for document analysis."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self.api_key = api_key
        self.model_name = model
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": 0.1,  # Low for consistent extraction
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4096,  # Increased for complex documents
                }
            )
            logger.info(f"Gemini client initialized with model: {model}")
        except ImportError:
            logger.error("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
            raise NotImplementedError(
                "Google GenerativeAI client not available. Please install google-generativeai package"
            )
    
    async def generate(self, prompt: str) -> str:
        """Generate response from Gemini."""
        logger.info(f"GeminiLLMClient: Calling {self.model_name}...")
        
        try:
            # Enhanced prompt for better JSON extraction
            enhanced_prompt = f"""You are a financial document analysis expert. 
Extract entities and relationships as valid JSON only. 
Do not include any markdown formatting, explanations, or additional text.
Return ONLY the JSON response.

{prompt}"""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(enhanced_prompt)
            )
            
            # Clean response to extract JSON
            response_text = response.text.strip()
            
            # Remove markdown formatting if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            return response_text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

class RealLLMClient:
    """Production LLM client using OpenAI or other providers."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        logger.info(f"RealLLMClient initialized with model: {model}")
    
    async def generate(self, prompt: str) -> str:
        """Generate response from actual LLM."""
        logger.info(f"RealLLMClient: Calling {self.model}...")
        
        # Check if we have OpenAI installed
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Enhanced system prompt for better extraction
            system_prompt = """You are a financial document analysis expert. 
Extract entities and relationships as valid JSON only. 
Do not include any markdown formatting, explanations, or additional text.
Return ONLY the JSON response in the exact format requested.
Ensure all JSON is properly formatted and valid."""
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4096  # Increased for complex documents
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean response to extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            return response_text
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            raise NotImplementedError(
                "OpenAI client not available. Please install openai package or use USE_MOCK_LLM=true"
            )
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

# --- Dynamic LLM Client Function ---
def get_current_llm_client():
    """Get the appropriate LLM client based on current config."""
    if config.USE_MOCK_LLM:
        logger.info("Using MockLLMClient")
        return MockLLMClient()
    elif config.LLM_PROVIDER.lower() == "gemini" and config.GOOGLE_API_KEY:
        logger.info(f"Using GeminiLLMClient with {config.LLM_MODEL}")
        return GeminiLLMClient(
            api_key=config.GOOGLE_API_KEY,
            model=config.LLM_MODEL if config.LLM_MODEL.startswith("gemini") else "gemini-1.5-pro"
        )
    elif config.LLM_PROVIDER.lower() == "openai" and config.OPENAI_API_KEY:
        logger.info(f"Using RealLLMClient (OpenAI) with {config.LLM_MODEL}")
        return RealLLMClient(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL
        )
    elif config.GOOGLE_API_KEY:
        # Fallback to Gemini if available
        logger.info("Falling back to GeminiLLMClient")
        return GeminiLLMClient(
            api_key=config.GOOGLE_API_KEY,
            model="gemini-1.5-pro"
        )
    elif config.OPENAI_API_KEY:
        # Fallback to OpenAI if available
        logger.info("Falling back to RealLLMClient (OpenAI)")
        return RealLLMClient(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL
        )
    else:
        logger.warning("No API key found, falling back to mock LLM")
        return MockLLMClient()

# --- Service Initialization ---
# Services will be stored in app.state during lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle - startup and shutdown.
    Using the new lifespan context manager (recommended over @app.on_event).
    """
    # Startup
    logger.info("Starting KAIROS services...")
    
    # Initialize Vector Service with optimized settings
    vector_service = VectorService(
        model_name=config.EMBEDDING_MODEL,
        collection_name="kairos_documents",
        persist_directory=config.VECTOR_PERSIST_DIR,
        batch_size=None,  # Auto-detect
        max_cache_size=10000,
        force_recreate=False  # Don't recreate on every startup
    )
    logger.info(f"Vector service initialized with {config.EMBEDDING_MODEL}")
    
    # Initialize Graph Service with optimized parameters for HackRx
    graph_storage = GraphStorageAdapter(path=config.GRAPH_PERSIST_DIR)
    graph_service = GraphService(
        storage_adapter=graph_storage,
        max_tokens=4000,
        max_concurrent_operations=20,  # Reduced for stability
        chunk_size=2500,  # Optimized for insurance documents
        chunk_overlap=250,  # Good overlap for context
        enable_checkpointing=True,
        enable_sampling=True
    )
    
    # Initialize the graph service properly
    await graph_service.initialize()
    logger.info(f"Graph service initialized with storage at {config.GRAPH_PERSIST_DIR}")
    
    # Store services in app state (better than globals)
    app.state.vector_service = vector_service
    app.state.graph_service = graph_service
    app.state.get_llm_client = get_current_llm_client  # Store function, not client
    
    # Verify services are working
    try:
        # Test vector service
        vector_health = await vector_service.check_health()
        if vector_health.get("healthy"):
            logger.info(f"Vector service healthy with {vector_health.get('document_count', 0)} documents")
        else:
            logger.warning(f"Vector service health check failed: {vector_health.get('error')}")
        
        # Test graph service
        if hasattr(graph_service, '_initialized') and graph_service._initialized:
            stats = graph_service.get_graph_statistics()
            logger.info(f"Graph loaded successfully with {stats['total_nodes']} nodes and {stats['total_edges']} edges")
        else:
            logger.warning("Graph service not properly initialized")
            
        # Test LLM client
        llm_client = get_current_llm_client()
        test_response = await llm_client.generate("Test prompt")
        logger.info("LLM client test successful")
        
    except Exception as e:
        logger.warning(f"Service verification failed: {e}")
    
    logger.info("All services initialized successfully")
    
    yield  # Application runs
    
    # Shutdown
    logger.info("Shutting down KAIROS services...")
    
    # Save graph state
    if hasattr(app.state, 'graph_service'):
        try:
            await app.state.graph_service.save_graph()
            logger.info("Graph state saved")
            # Close graph service if it has a close method
            if hasattr(app.state.graph_service, 'close'):
                await app.state.graph_service.close()
        except Exception as e:
            logger.error(f"Error saving graph state: {e}")
    
    # Close vector service connections properly
    if hasattr(app.state, 'vector_service'):
        vector_service = app.state.vector_service
        try:
            # ChromaDB doesn't have an explicit close method, but we can clear caches
            if hasattr(vector_service, '_cached_encode'):
                vector_service._cached_encode.cache_clear()
            # Clear document registry
            if hasattr(vector_service, '_document_registry'):
                vector_service._document_registry.clear()
            # Shutdown thread pool
            if hasattr(vector_service, 'executor'):
                vector_service.executor.shutdown(wait=False)
            logger.info("Vector service cleaned up")
        except Exception as e:
            logger.error(f"Error during vector service cleanup: {e}")
    
    logger.info("Shutdown complete")

# --- FastAPI App Instance ---
app = FastAPI(
    title="KAIROS API",
    description="The backend engine for the KAIROS document intelligence platform - Bajaj Finserv HackRx 6.0",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiting Middleware ---
# Add rate limiting to protect against DoS
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    # Create limiter instance
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    logger.info("Rate limiting middleware enabled")
except ImportError:
    logger.warning("slowapi not installed. Rate limiting disabled. Install with: pip install slowapi")
    limiter = None

# --- Health Monitoring Middleware ---
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add request processing time to headers."""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.include_router(documents.router)
app.include_router(hackrx.router)

# --- Additional API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "KAIROS API",
        "version": app.version,
        "status": "healthy",
        "description": "Knowledge Graph & RAG Intelligence System for Bajaj Finserv HackRx 6.0"
    }

@app.get("/config", tags=["Configuration"])
async def get_current_config():
    """Get current dynamic configuration."""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "llm_provider": config.LLM_PROVIDER,
        "use_mock_llm": config.USE_MOCK_LLM,
        "enable_graph_extraction": config.ENABLE_GRAPH_EXTRACTION,
        "has_openai_key": bool(config.OPENAI_API_KEY),
        "has_google_key": bool(config.GOOGLE_API_KEY),
        "vector_persist_dir": config.VECTOR_PERSIST_DIR,
        "graph_persist_dir": config.GRAPH_PERSIST_DIR,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/config/reload", tags=["Configuration"])
async def reload_config():
    """Force reload configuration from .env file."""
    config._reload_env()
    return {
        "message": "Configuration reloaded successfully",
        "timestamp": datetime.utcnow().isoformat(),
        "current_config": {
            "use_mock_llm": config.USE_MOCK_LLM,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL
        }
    }

@app.get("/health", tags=["Health"])
async def health_check(
    vector_service: VectorService = Depends(get_vector_service),
    graph_service: GraphService = Depends(get_graph_service)
):
    """Detailed health check endpoint with dynamic config."""
    health_status = {
        "status": "healthy",
        "services": {
            "vector_service": "healthy",
            "graph_service": "healthy",
            "llm_client": "healthy"
        },
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "llm_provider": config.LLM_PROVIDER,
            "use_mock_llm": config.USE_MOCK_LLM
        }
    }
    
    # Check actual service health
    try:
        # Check vector service
        vector_health = await vector_service.check_health()
        if vector_health.get("healthy"):
            stats = vector_health.get("stats", {})
            health_status["services"]["vector_stats"] = {
                "total_chunks": stats.get("total_chunks", 0),
                "unique_documents": stats.get("unique_documents_estimated", 0)
            }
        else:
            health_status["services"]["vector_service"] = f"error: {vector_health.get('error')}"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["vector_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check graph service
        stats = graph_service.get_graph_statistics()
        health_status["services"]["graph_stats"] = {
            "total_nodes": stats.get("total_nodes", 0),
            "total_edges": stats.get("total_edges", 0)
        }
    except Exception as e:
        health_status["services"]["graph_service"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/stats", tags=["Monitoring"])
async def get_system_stats(
    vector_svc: VectorService = Depends(get_vector_service),
    graph_svc: GraphService = Depends(get_graph_service)
):
    """Get comprehensive system statistics."""
    vector_stats = await vector_svc.get_collection_stats()
    graph_stats = graph_svc.get_graph_statistics()
    
    return {
        "vector_service": vector_stats,
        "graph_service": graph_stats,
        "system": {
            "total_documents": vector_stats.get("unique_documents_estimated", 0),
            "total_chunks": vector_stats.get("total_chunks", 0),
            "total_entities": graph_stats.get("total_nodes", 0),
            "total_relationships": graph_stats.get("total_edges", 0),
            "cache_hit_rate": _calculate_cache_hit_rate(vector_stats)
        }
    }

def _calculate_cache_hit_rate(vector_stats: Dict[str, Any]) -> float:
    """Calculate cache hit rate from vector stats."""
    cache_stats = vector_stats.get("cache_stats", {})
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    total = hits + misses
    
    # Guard against division by zero
    if total == 0:
        return 0.0
    
    return round(hits / total, 3)

# --- Error Handlers ---

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# --- Development/Debug Endpoints ---

if os.getenv("ENV", "development") == "development":
    @app.post("/debug/reset", tags=["Debug"])
    async def reset_system(
        vector_svc: VectorService = Depends(get_vector_service),
        graph_svc: GraphService = Depends(get_graph_service)
    ):
        """Reset all data (development only)."""
        # Reset vector collection
        vector_result = vector_svc.reset_collection()
        
        # Reset graph
        await graph_svc.reset_graph()
        
        return {
            "message": "System reset complete",
            "vector_reset": vector_result,
            "graph_reset": {"success": True}
        }
    
    @app.get("/debug/hackrx-docs", tags=["Debug"])
    async def get_hackrx_documents(
        vector_svc: VectorService = Depends(get_vector_service)
    ):
        """Get all HackRx documents in the system."""
        docs = vector_svc.get_documents_by_source("hackrx")
        return {
            "total_documents": len(docs),
            "documents": docs
        }
    
    @app.delete("/debug/clear-hackrx", tags=["Debug"])
    async def clear_hackrx_documents(
        vector_svc: VectorService = Depends(get_vector_service)
    ):
        """Clear all HackRx documents."""
        docs = vector_svc.get_documents_by_source("hackrx")
        deleted = 0
        for doc in docs:
            result = vector_svc.delete_document(doc.get('document_id'))
            if result.get('success'):
                deleted += 1
        
        return {
            "message": f"Cleared {deleted} HackRx documents",
            "total_found": len(docs)
        }
    
    @app.get("/debug/test-llm", tags=["Debug"])
    async def test_llm_client():
        """Test the current LLM client."""
        try:
            llm_client = get_current_llm_client()
            test_prompt = """Extract entities from this insurance text:
            
"John Doe has a travel insurance policy with XYZ Insurance covering $50,000 with a deductible of $500."

Return ONLY JSON:"""
            
            response = await llm_client.generate(test_prompt)
            return {
                "success": True,
                "llm_provider": config.LLM_PROVIDER,
                "llm_model": config.LLM_MODEL,
                "use_mock": config.USE_MOCK_LLM,
                "response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "llm_provider": config.LLM_PROVIDER
            }

# --- Main entry point for development ---
if __name__ == "__main__":
    import uvicorn
    
    # Create sample .env file if it doesn't exist
    if not os.path.exists(".env"):
        sample_env = """# KAIROS Environment Configuration

# LLM Configuration
USE_MOCK_LLM=true  # Set to false for production
LLM_PROVIDER=openai  # openai, gemini, or mock
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
LLM_MODEL=gpt-3.5-turbo  # Faster for HackRx

# Feature Flags
ENABLE_GRAPH_EXTRACTION=true  # Keep true for accuracy

# Storage Configuration
VECTOR_PERSIST_DIR=./chroma_db
GRAPH_PERSIST_DIR=./graph_db

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# CORS Configuration (comma-separated origins)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Environment
ENV=development
"""
        with open(".env", "w") as f:
            f.write(sample_env)
        logger.info("Created .env file with default settings.")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
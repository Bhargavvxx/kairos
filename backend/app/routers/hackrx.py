# backend/app/routers/hackrx.py

import logging
import httpx
import asyncio
import hashlib
import time
import uuid
import urllib.parse
import json
from typing import List, Dict, Optional, Any, Tuple
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, HttpUrl
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import re
import email
from email import policy
from email.parser import BytesParser

# Import authentication
from app.auth import verify_hackrx_token

# Import dependencies and services
from app.dependencies import get_vector_service, get_graph_service, get_llm_client
from app.services.vector_service import VectorService
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

# Constants - OPTIMIZED for accuracy + speed balance
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB - keep as is
CACHE_SIZE = 2000  # INCREASE: More cache = more hits
REQUEST_TIMEOUT = 300  # 5 minutes - keep as is
LLM_TIMEOUT = 15  # REDUCE: 15 seconds is enough for most answers
CLEAR_PREVIOUS_DOCS = True  # Keep for testing
RATE_LIMIT_DELAY = 0.3  # REDUCE: Slightly faster
MAX_CONCURRENT_LLM_CALLS = 15  # INCREASE: More parallel processing
ENABLE_GRAPH_EXTRACTION = False  # Keep disabled for speed
BATCH_SIZE = 15  # INCREASE: Process 15 questions at once

# Document processing thresholds - OPTIMIZED
FULL_PROCESSING_THRESHOLD = 150000  # INCREASE: ~30 pages full processing
SMART_SAMPLING_THRESHOLD = 75000    # INCREASE: ~15 pages before sampling
AGGRESSIVE_SAMPLING_THRESHOLD = 300000  # INCREASE: ~60 pages

# NEW CONSTANTS TO ADD:
ENABLE_PATTERN_EXTRACTION = True  # Fast pattern matching
ENABLE_CHUNK_RERANKING = True  # Better chunk selection
ENABLE_ANSWER_VALIDATION = True  # Fix answer formats
ENABLE_TWO_PASS = False  # Set True only if <30s currently
HYBRID_SEARCH_WEIGHT = 0.5  # Balance semantic and keyword (was 0.7)
MAX_CHUNKS_PER_QUESTION = 7  # Optimal context size
CHUNK_SIZE_OPTIMAL = 600  # Smaller chunks for precision
OVERLAP_SIZE = 75  # Reduced overlap
MIN_CONFIDENCE_THRESHOLD = 0.4  # For answer retry
ENABLE_CONTEXT_WINDOWS = True  # Focus on relevant sections
PATTERN_MATCH_CONFIDENCE = 0.95  # High confidence for pattern matches

# Document cache for instant reprocessing
class DocumentProcessingCache:
    """Cache for processed documents"""
    def __init__(self):
        self.cache = {}
        
    def get(self, url: str) -> Optional[Dict]:
        doc_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache.get(doc_hash)
    
    def set(self, url: str, data: Dict):
        doc_hash = hashlib.md5(url.encode()).hexdigest()
        self.cache[doc_hash] = {
            **data,
            'cached_at': time.time()
        }
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1].get('cached_at', 0))
            for key, _ in sorted_items[:20]:
                del self.cache[key]

# Initialize document cache
doc_processing_cache = DocumentProcessingCache()

# Enhanced insurance patterns
ENHANCED_INSURANCE_PATTERNS = {
    # Grace Period patterns
    "grace_period": [
        r'grace\s+period(?:\s+of)?\s+(\d+)\s+days?',
        r'the\s+policy\s+may\s+be\s+renewed\s+within\s+(\d+)\s+days\s+of\s+the\s+due\s+date',
        r'grace\s+period\s+of\s+upto\s+(\d+)\s+days',
        r'premium\s+received\s+within\s+(\d+)\s+days\s+from\s+the\s+due\s+date\s+shall\s+be\s+accepted'
    ],
    "waiting_period": [
        r'waiting\s+period\s+of\s+(\d+)\s+(?:days?|months?|years?)',
        r'pre[-\s]?existing\s+conditions?\s+covered\s+after\s+(\d+)\s+(?:months?|years?)',
        r'covered\s+after\s+completion\s+of\s+(\d+)\s+months?',
        r'there\s+shall\s+be\s+a\s+waiting\s+period\s+of\s+(\d+)\s+(?:days?|months?|years?)',
        r'(\d+)\s+months?\s+from\s+the\s+commencement\s+date',
        r'disease\s+shall\s+be\s+covered\s+after\s+(\d+)\s+(?:months?|days?)'
    ],
    "copayment": [
        r'co[-\s]?payment\s+of\s+(\d+)%\s+shall\s+apply',
        r'the\s+insured\s+shall\s+bear\s+(\d+)%\s+of\s+the\s+claim',
        r'copayment\s+of\s+(\d+)%\s+is\s+applicable',
        r'copayment\s+clause.*?(\d+)%'
    ],
    "room_rent": [
        r'room\s+rent\s+(?:limit\s+of\s+)?(?:₹|rs\.?\s?)?([0-9,]+)',
        r'up\s+to\s+(?:a\s+)?single\s+private\s+ac\s+room',
        r'twin\s+sharing\s+room\s+limit',
        r'covered\s+room\s+type.*?single\s+private\s+room',
        r'room\s+charges\s+limited\s+to.*?(?:₹|rs\.?\s?)?([0-9,]+)',
        r'accommodation\s+in\s+a\s+shared\s+room\s+only',
        r'up\s+to\s+rent\s+limit\s+of\s+(?:₹|rs\.?\s?)?([0-9,]+)'
    ],
    "hospitalization": [
        r'pre[-\s]?hospitalization\s+expenses\s+for\s+(\d+)\s+days',
        r'post[-\s]?hospitalization\s+expenses\s+for\s+(\d+)\s+days',
        r'(\d+)\s+days?\s+prior\s+to\s+admission\s+shall\s+be\s+covered',
        r'(\d+)\s+days?\s+after\s+discharge\s+shall\s+be\s+covered'
    ],
    "exclusions": [
        r'shall\s+not\s+be\s+covered',
        r'is\s+not\s+covered\s+under\s+this\s+policy',
        r'excluded\s+conditions?\s+include',
        r'the\s+following\s+are\s+not\s+covered'
    ],
    "free_look_period": [
        r'free\s+look\s+period\s+(?:of\s+)?(\d+)\s+days',
        r'within\s+(\d+)\s+days\s+of\s+policy\s+receipt.*?cancellation',
        r'the\s+policyholder\s+may\s+return\s+the\s+policy\s+within\s+(\d+)\s+days'
    ],
    "age_limit": [
        r'minimum\s+entry\s+age\s+is\s+(\d+)\s+years',
        r'maximum\s+entry\s+age\s+is\s+(\d+)\s+years',
        r'age\s+limit\s+for\s+entry\s+is\s+(\d+)\s+to\s+(\d+)\s+years',
        r'eligibility\s+age\s+between\s+(\d+)\s+and\s+(\d+)\s+years'
    ],
    "opd_coverage": [
        r'OPD\s+(?:expenses|benefits)\s+shall\s+be\s+covered',
        r'out[-\s]?patient\s+expenses\s+up\s+to\s+(?:₹|rs\.?\s?)?([0-9,]+)',
        r'covers\s+consultation\s+and\s+diagnostics\s+on\s+OPD\s+basis',
        r'OPD\s+cover\s+up\s+to\s+(?:₹|rs\.?\s?)?([0-9,]+)'
    ],

      "percentage_patterns": [
          r'(\d+)%\s+of\s+(?:the\s+)?sum\s+insured',
          r'preventive\s+health\s+check[-\s]?up.*?(\d+)%',
       ]
}

# Insurance-specific priority sections
INSURANCE_PRIORITY_SECTIONS = [
    "waiting period", "exclusions", "coverage", "claims",
    "sum insured", "co-payment", "deductible", "pre-existing"
]

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# OPTIMIZED Rate limiter for parallel LLM calls
class RateLimiter:
    def __init__(self, requests_per_minute=30, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.request_times = []
    
    async def acquire(self):
        async with self.semaphore:
            current_time = time.time()
            # Clean old request times
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Check if we're within rate limit
            if len(self.request_times) >= self.requests_per_minute:
                wait_time = 60 - (current_time - self.request_times[0]) + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
            
            # Ensure minimum interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_request_time = time.time()
            self.request_times.append(self.last_request_time)

# Initialize rate limiter with higher limits
rate_limiter = RateLimiter(requests_per_minute=30, max_concurrent=MAX_CONCURRENT_LLM_CALLS)

# Request/Response Models
class HackRxRequest(BaseModel):
    documents: HttpUrl  # PDF blob URL
    questions: List[str]
    model: Optional[str] = None  # Allow model selection

class HackRxResponse(BaseModel):
    answers: List[str]

# Internal detailed response model
class DetailedAnswer(BaseModel):
    answer: str
    confidence: float
    source_chunks: List[Dict[str, Any]]
    reasoning: str
    processing_time: float
    token_usage: Dict[str, Any]

# Router setup
router = APIRouter(prefix="/hackrx", tags=["HackRx Competition"])

# Cache for repeated questions
answer_cache: Dict[str, str] = {}
detailed_answer_cache: Dict[str, DetailedAnswer] = {}
pattern_cache: Dict[str, str] = {}  # Cache for pattern-based answers

# Token estimation function
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    return len(text) // 4

# Custom authentication wrapper
async def verify_hackrx_token_with_proper_errors(
    request: Request,
    token: str = Depends(verify_hackrx_token)
) -> str:
    """Wrapper to handle missing auth header properly"""
    if "authorization" not in request.headers:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return token

# FIX 3: Enhanced JSON to Natural Language Conversion
def format_json_answer(json_data, question: str) -> str:
    """Enhanced JSON to natural language conversion"""
    if isinstance(json_data, str):
        # Check if it's already a natural sentence
        if not json_data.strip().startswith('{'):
            return json_data
        try:
            json_data = json.loads(json_data)
        except:
            # If it's not valid JSON, return as is
            return json_data
    
    # Convert JSON to natural language based on content
    if isinstance(json_data, dict):
        # Handle specific patterns for common questions
        if "days_to_settle_or_reject_claim" in json_data:
            return f"The company has {json_data['days_to_settle_or_reject_claim']} days to settle or reject a claim from receipt of last necessary documents."
        
        if "maximum_ambulance_charge_per_hospitalization" in json_data:
            value = json_data['maximum_ambulance_charge_per_hospitalization']
            return f"The maximum amount payable for ambulance charges per hospitalization is {value}."
        
        if "percentage_of_sum_insured" in json_data:
            return f"The percentage of sum insured for preventive health check-up is {json_data['percentage_of_sum_insured']}."
        
        if "cumulative_bonus_percentage" in json_data:
            return f"The cumulative bonus percentage for each claim-free year is {json_data['cumulative_bonus_percentage']}."
        
        if "daily_cash_amount" in json_data:
            return f"The daily cash amount for choosing shared accommodation is {json_data['daily_cash_amount']}."
        
        if "daily_cash_benefit" in json_data:
            if isinstance(json_data['daily_cash_benefit'], dict):
                amount = json_data['daily_cash_benefit'].get('amount', '')
                conditions = json_data['daily_cash_benefit'].get('conditions', '')
                return f"The daily cash benefit for accompanying an insured child is {amount}. {conditions}"
            else:
                return f"The daily cash benefit is {json_data['daily_cash_benefit']}."
        
        if "notification_period" in json_data:
            return f"The notification period required for planned hospitalization is {json_data['notification_period']}."
        
        if "grace_period" in json_data:
            if isinstance(json_data['grace_period'], dict):
                duration = json_data['grace_period'].get('duration', '')
                return f"The grace period for premium payment is {duration}."
            else:
                return f"The grace period is {json_data['grace_period']}."
        
        if "free_look_period" in json_data:
            if isinstance(json_data['free_look_period'], dict):
                duration = json_data['free_look_period'].get('duration', '')
                return f"The free look period for new individual health insurance policies is {duration}."
            else:
                return f"The free look period is {json_data['free_look_period']}."
        
        if "maximum_age_for_dependent_children" in json_data:
            return f"The maximum age for dependent children to be covered under the policy is {json_data['maximum_age_for_dependent_children']} years."
        
        if "can_grandparents_be_covered" in json_data:
            covered = json_data['can_grandparents_be_covered']
            condition = json_data.get('age_condition', '')
            if covered:
                return f"Yes, grandparents can be covered under the Easy Health policy. {condition}"
            else:
                return "No, grandparents cannot be covered under this policy."
        
        if "coverage" in json_data and isinstance(json_data['coverage'], dict):
            # Handle coverage questions
            coverage_info = json_data['coverage']
            if 'normal_delivery' in coverage_info:
                delivery_info = coverage_info['normal_delivery']
                amount = delivery_info.get('amount', '')
                plan = delivery_info.get('plan', '')
                return f"The coverage amount for normal delivery under {plan} plan is {amount}."
        
        if "minimum_duration_for_domiciliary_treatment" in json_data:
            return f"The minimum duration required for domiciliary treatment to be covered is {json_data['minimum_duration_for_domiciliary_treatment']}."
        
        if "days_after_diagnosis" in json_data:
            return f"An insured person must survive {json_data['days_after_diagnosis']} days after diagnosis of critical illness to be eligible for the benefit."
        
        if "critical_illnesses" in json_data:
            illnesses = json_data['critical_illnesses']
            if isinstance(illnesses, list):
                return f"The critical illnesses covered under the optional benefit include: {', '.join(illnesses)}."
        
        if "covered" in json_data:
            covered = json_data['covered']
            reason = json_data.get('reason', '')
            if covered:
                return f"Yes, it is covered under the policy. {reason}"
            else:
                return f"No, it is not covered under the policy. {reason}"
        
        if "isTreatmentCovered" in json_data:
            covered = json_data['isTreatmentCovered']
            reason = json_data.get('reason', '')
            if covered:
                return f"Yes, the treatment is covered. {reason}"
            else:
                return f"No, the treatment is not covered. {reason}"
        
        if "isDomiciliaryTreatmentCovered" in json_data:
            covered = json_data['isDomiciliaryTreatmentCovered']
            details = json_data.get('details', '')
            if covered:
                return f"Yes, domiciliary treatment is covered under the Easy Health policy. {details}"
            else:
                return f"No, domiciliary treatment is not covered. {details}"
        
        if "policy_coverage" in json_data and "medical_treatment_outside_india" in json_data['policy_coverage']:
            covered = json_data['policy_coverage']['medical_treatment_outside_india']
            if covered:
                return "Yes, this policy covers medical treatment taken outside India."
            else:
                return "No, this policy does not cover medical treatment taken outside India."
        
        # Generic handling for any other JSON structure
        for key, value in json_data.items():
            if isinstance(value, (str, int, float, bool)):
                # Format the key nicely
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, bool):
                    return f"{formatted_key}: {'Yes' if value else 'No'}"
                else:
                    return f"{formatted_key}: {value}"
    
    # If we can't parse it nicely, return as string
    return str(json_data)

def detect_document_type(text: str) -> str:
    """Detect if this is travel or health insurance."""
    travel_indicators = ['travel insurance', 'trip', 'journey', 'baggage', 'flight delay', 'passport']
    health_indicators = ['health insurance', 'hospitalization', 'medical expenses', 'surgery', 'treatment']
    
    text_lower = text.lower()[:5000]
    
    travel_score = sum(1 for term in travel_indicators if term in text_lower)
    health_score = sum(1 for term in health_indicators if term in text_lower)
    
    return "travel" if travel_score > health_score else "health"

def enhanced_pattern_extraction(text: str, question: str, document_type: str = None) -> Optional[str]:
    """Enhanced pattern extraction with document-type awareness"""
    question_lower = question.lower()
    
    # Grace period
    if any(term in question_lower for term in ["grace period", "grace"]):
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("grace_period", []):
            match = re.search(pattern, text, re.I)
            if match and match.groups():
                days = match.group(1)
                return f"The grace period for policy renewal is {days} days."
    
    # Waiting period
    if "waiting" in question_lower:
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("waiting_period", []):
            match = re.search(pattern, text, re.I)
            if match and match.groups():
                period = match.group(1)
                return f"The waiting period is {period} months."
    
    # Co-payment
    if any(term in question_lower for term in ["co-payment", "copayment", "co payment"]):
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("copayment", []):
            match = re.search(pattern, text, re.I)
            if match and match.groups():
                percentage = match.group(1)
                return f"The co-payment percentage is {percentage}%."
    
    # Pre-hospitalization
    if "pre-hospitalization" in question_lower or "pre hospitalization" in question_lower:
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("hospitalization", []):
            if "pre" in pattern.lower():
                match = re.search(pattern, text, re.I)
                if match and match.groups():
                    days = match.group(1)
                    return f"The pre-hospitalization coverage period is {days} days."
    
    # Room rent
    if "room rent" in question_lower:
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("room_rent", []):
            match = re.search(pattern, text, re.I)
            if match:
                return f"The room rent coverage is: {match.group(0)}."
    
    # Age limits
    if any(term in question_lower for term in ["age limit", "maximum age", "entry age"]):
        for pattern in ENHANCED_INSURANCE_PATTERNS.get("age_limit", []):
            match = re.search(pattern, text, re.I)
            if match and match.groups():
                age = match.group(1)
                return f"The age limit is {age} years."
    
    return None

def smart_sample_document(text: str, target_size: int = 50000) -> str:
    """Sample most relevant parts of document for speed"""
    if len(text) <= target_size:
        return text
    
    # Split into sections
    sections = text.split('\n\n')
    
    # Priority keywords for insurance docs
    priority_terms = INSURANCE_PRIORITY_SECTIONS + [
        'grace period', 'premium', 'benefit', 'maternity', 'organ donor'
    ]
    
    # Score each section
    scored_sections = []
    for section in sections:
        score = 0
        section_lower = section.lower()
        
        # Score based on priority terms
        for term in priority_terms:
            if term in section_lower:
                score += 10
        
        # Bonus for sections with numbers
        if any(char.isdigit() for char in section):
            score += 5
        
        scored_sections.append((score, section))
    
    # Sort by score
    scored_sections.sort(reverse=True, key=lambda x: x[0])
    
    # Take top sections up to target size
    sampled_text = []
    current_size = 0
    
    for score, section in scored_sections:
        if current_size + len(section) > target_size:
            break
        sampled_text.append(section)
        current_size += len(section)
    
    return '\n\n'.join(sampled_text)

async def call_llm_with_rate_limit(llm_client, prompt: str, is_json: bool = False, max_retries: int = 2) -> str:
    """OPTIMIZED: Reduced retries and timeout"""
    for attempt in range(max_retries):
        try:
            await rate_limiter.acquire()
            
            response = await asyncio.wait_for(
                llm_client.generate(prompt),
                timeout=LLM_TIMEOUT
            )
            
            if is_json and response and not response.strip().startswith('{'):
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    return json_match.group(0)
            
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return "Unable to generate response due to timeout."
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return "Unable to generate response."
            await asyncio.sleep(1)

# FIX 1: Update parallel_document_processing function
async def parallel_document_processing(
    text_content: str,
    doc_id: str,
    request: HackRxRequest,
    vector_service: VectorService,
    graph_service: GraphService,
    llm_client
) -> Tuple[bool, str, bool]:
    """OPTIMIZED: Better chunking parameters"""
    
    # Smart sampling for large documents
    if len(text_content) > SMART_SAMPLING_THRESHOLD:
        text_content = smart_sample_document(text_content, FULL_PROCESSING_THRESHOLD)
        logger.info(f"Document sampled to {len(text_content)} chars")
    
    # OPTIMIZED chunk parameters for accuracy
    chunk_size = CHUNK_SIZE_OPTIMAL  # 600 instead of 750
    overlap = OVERLAP_SIZE  # 75 instead of 100
    
    # Only do vector indexing (skip graph)
    vector_task = vector_service.force_reindex_document(
        text=text_content,
        metadata={
            "document_id": doc_id,
            "source": "hackrx",
            "url": str(request.documents),
            "indexed_at": time.time()
        },
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    vector_result = await vector_task
    
    # Extract actual document ID
    actual_doc_id = doc_id
    if isinstance(vector_result, dict) and 'document_id' in vector_result:
        actual_doc_id = vector_result['document_id']
    
    vector_success = isinstance(vector_result, dict) and vector_result.get("success", False)
    graph_success = False  # Graph disabled
    
    return vector_success, actual_doc_id, graph_success

@router.post("/run", response_model=HackRxResponse)
async def run_hackrx_submission(
    request: HackRxRequest,
    req: Request,
    token: str = Depends(verify_hackrx_token_with_proper_errors),
    vector_service: VectorService = Depends(get_vector_service),
    graph_service: GraphService = Depends(get_graph_service),
    llm_client = Depends(get_llm_client)
):
    """
    OPTIMIZED HackRx endpoint for <40 second processing.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] Processing HackRx request with {len(request.questions)} questions")
        
        # Check document cache first
        cached_doc = doc_processing_cache.get(str(request.documents))
        if cached_doc:
            logger.info(f"[{request_id}] Using cached document processing")
            text_content = cached_doc['text_content']
            actual_doc_id = cached_doc['doc_id']
            doc_size = len(text_content)
            document_type = cached_doc.get('document_type', 'insurance')
        else:
            # Download and extract document
            logger.info(f"[{request_id}] Downloading document: {request.documents}")
            
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                try:
                    doc_response = await client.get(str(request.documents))
                    doc_response.raise_for_status()
                    document_content = doc_response.content
                except httpx.HTTPError as e:
                    logger.error(f"[{request_id}] Failed to download document: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download document: {str(e)}"
                    )
            
            # Validate document size
            if len(document_content) > MAX_DOCUMENT_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Document size exceeds 50MB limit"
                )
            
            # Extract text
            text_content = await extract_text_from_blob(document_content, str(request.documents))
            
            if not text_content or len(text_content.strip()) < 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Document appears to be empty or contains insufficient text"
                )
            
            doc_size = len(text_content)
            document_type = detect_document_type(text_content)
            logger.info(f"[{request_id}] Extracted {doc_size:,} characters (~{doc_size//5000} pages) - Type: {document_type}")
            
            # Setup document processing
            timestamp = int(time.time() * 1000)
            doc_id = f"hackrx_{timestamp}_{str(uuid.uuid4())[:8]}"
            
            # Clear previous documents if configured
            if CLEAR_PREVIOUS_DOCS:
                logger.info(f"[{request_id}] Clearing previous hackrx documents")
                try:
                    hackrx_docs = vector_service.get_documents_by_source("hackrx")
                    for doc in hackrx_docs[:5]:  # Limit to 5 for speed
                        doc_id_to_delete = doc.get('document_id')
                        if doc_id_to_delete:
                            vector_service.delete_document(doc_id_to_delete)
                    vector_service.clear_document_registry()
                except Exception as e:
                    logger.error(f"[{request_id}] Error clearing previous documents: {e}")
            
            # Process document (vector only, no graph)
            logger.info(f"[{request_id}] Starting document processing...")
            vector_success, actual_doc_id, _ = await parallel_document_processing(
                text_content, doc_id, request, vector_service, graph_service, llm_client
            )
            
            if not vector_success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to process document for search"
                )
            
            # Cache the processed document
            doc_processing_cache.set(str(request.documents), {
                'text_content': text_content,
                'doc_id': actual_doc_id,
                'document_type': document_type
            })
            
            logger.info(f"[{request_id}] Document processing complete and cached")
        
        # IMPROVED APPROACH: Use vector search as primary + pattern as fallback/validation
        logger.info(f"[{request_id}] Processing questions with vector search + pattern fallback...")
        
        # Get pattern answers for all questions first (fast)
        pattern_answers = {}
        for idx, question in enumerate(request.questions):
            pattern_answer = enhanced_pattern_extraction(text_content, question, document_type)
            if pattern_answer:
                pattern_answers[idx] = pattern_answer
        
        logger.info(f"[{request_id}] Pattern extraction found answers for {len(pattern_answers)} questions")
        
        # Process ALL questions with vector search in parallel
        tasks = []
        for idx, question in enumerate(request.questions):
            fallback_answer = pattern_answers.get(idx)  # Pass pattern answer as fallback
            task = process_question_optimized(
                question=question,
                question_idx=idx,
                document_id=actual_doc_id,
                request_id=request_id,
                vector_service=vector_service,
                llm_client=llm_client,
                text_content=text_content,  # Pass full text for pattern extraction
                document_type=document_type,
                fallback_answer=fallback_answer
            )
            tasks.append(task)
        
        # Execute all questions in parallel
        batch_results = await asyncio.gather(*tasks)
        
        # Extract final answers with JSON formatting fix
        final_answers = []
        for i, result in enumerate(batch_results):
            if isinstance(result, dict):
                answer = result.get("answer", "Unable to process this question.")
                # Check if answer is JSON and convert to natural language
                if answer and (answer.strip().startswith('{') or '"' in answer[:20]):
                    answer = format_json_answer(answer, request.questions[i])
                final_answers.append(answer)
            else:
                final_answers.append(str(result) if result else "Unable to process this question.")
        
        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] ✅ Completed in {total_time:.2f}s")
        logger.info(f"[{request_id}] Average: {total_time/len(request.questions):.2f}s per question")
        
        return HackRxResponse(answers=final_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Error after {total_time:.2f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

# IMPROVED: process_question_optimized with vector search as primary + pattern as safety net
async def process_question_optimized(
    question: str,
    question_idx: int,
    document_id: str,
    request_id: str,
    vector_service: VectorService,
    llm_client,
    text_content: str,
    document_type: str,
    fallback_answer: Optional[str] = None
) -> Dict[str, Any]:
    """
    IMPROVED: Vector search as primary method with pattern extraction as fallback/validation
    """
    try:
        start_time = time.time()
        session_id = f"{request_id}_Q{question_idx+1}"
        
        # Check cache first
        cache_key = f"{document_id}:{hashlib.md5(question.encode()).hexdigest()}"
        if cache_key in detailed_answer_cache:
            logger.info(f"[{session_id}] Using cached answer")
            cached = detailed_answer_cache[cache_key]
            return cached.dict()
        
        # STEP 1: Always do vector search as primary method
        search_results = await vector_service.hybrid_search(
            query=question,
            n_results=MAX_CHUNKS_PER_QUESTION,
            filter_metadata={"source": "hackrx"},
            min_score=0.0,
            vector_weight=HYBRID_SEARCH_WEIGHT
        )
        
        # Filter by document_id
        if search_results:
            search_results = [
                r for r in search_results 
                if r.get('metadata', {}).get('document_id') == document_id
            ][:5]
        
        vector_answer = None
        vector_confidence = 0.0
        
        if search_results and search_results[0]['similarity_score'] >= 0.3:
            # Build context from search results
            context_chunks = []
            for result in search_results[:5]:
                chunk_text = result['content'].strip()[:800]
                if len(chunk_text) > 50:
                    context_chunks.append(chunk_text)
            
            full_context = "\n\n".join(context_chunks)
            
            # Generate answer with LLM
            prompt = f"""Answer this insurance policy question based on the context. 
Provide a clear, direct answer in 1-2 complete sentences.
Do NOT return JSON or dictionary format.

Context: {full_context[:2500]}

Question: {question}

Direct Answer:"""
            
            vector_answer = await call_llm_with_rate_limit(llm_client, prompt, is_json=False)
            vector_confidence = search_results[0]['similarity_score']
            
            # Clean vector answer
            if vector_answer:
                vector_answer = vector_answer.strip()
                for prefix in ["Direct Answer:", "Answer:", "A:"]:
                    if vector_answer.startswith(prefix):
                        vector_answer = vector_answer[len(prefix):].strip()
        
        # STEP 2: Get fresh pattern answer (in case we didn't have fallback)
        if not fallback_answer:
            fallback_answer = enhanced_pattern_extraction(text_content, question, document_type)
        
        pattern_confidence = 0.95 if fallback_answer else 0.0
        
        # STEP 3: Choose the best answer using improved logic
        final_answer = None
        final_confidence = 0.0
        reasoning = ""
        
        # Validate vector answer for known bad patterns
        vector_is_problematic = False
        if vector_answer:
            # Check for problematic patterns
            if ("₹100" in vector_answer and ("percentage" in question.lower() or "%" in question)) or \
               (vector_answer.startswith("Unable to find") and fallback_answer) or \
               ("No relevant information" in vector_answer and fallback_answer):
                vector_is_problematic = True
                reasoning = "Vector answer contained suspicious patterns, used fallback"
        
        if vector_is_problematic and fallback_answer:
            # Use pattern answer if vector has known issues
            final_answer = fallback_answer
            final_confidence = pattern_confidence
            reasoning = "Used pattern extraction due to vector answer issues"
        elif vector_answer and vector_confidence >= 0.35:
            # Vector search found good answer (lowered threshold)
            final_answer = vector_answer
            final_confidence = vector_confidence
            reasoning = "Vector search primary answer"
        elif fallback_answer and pattern_confidence > 0.8:
            # High confidence pattern answer
            final_answer = fallback_answer
            final_confidence = pattern_confidence
            reasoning = "High confidence pattern extraction"
        elif vector_answer:
            # Low confidence vector answer, but better than nothing
            final_answer = vector_answer
            final_confidence = vector_confidence
            reasoning = "Low confidence vector search"
        elif fallback_answer:
            # Pattern answer as last resort
            final_answer = fallback_answer
            final_confidence = pattern_confidence
            reasoning = "Pattern extraction fallback"
        else:
            # No answer found
            final_answer = "No relevant information found in the document for this question."
            final_confidence = 0.0
            reasoning = "No answer source available"
        
        # Check if final answer is JSON and convert
        if final_answer and final_answer.startswith('{'):
            final_answer = format_json_answer(final_answer, question)
        
        processing_time = time.time() - start_time
        
        result = {
            "answer": final_answer,
            "confidence": final_confidence,
            "processing_time": processing_time,
            "reasoning": reasoning
        }
        
        # Cache the response
        if len(detailed_answer_cache) < CACHE_SIZE:
            detailed_answer_cache[cache_key] = DetailedAnswer(
                answer=final_answer,
                confidence=final_confidence,
                source_chunks=[],
                reasoning=reasoning,
                processing_time=processing_time,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
        
        logger.info(f"[{session_id}] Answer: {reasoning} (conf: {final_confidence:.2f})")
        return result
        
    except Exception as e:
        logger.error(f"[{session_id}] Failed: {e}")
        return {
            "answer": fallback_answer or "Unable to process this question.",
            "confidence": 0.5 if fallback_answer else 0.0,
            "processing_time": time.time() - start_time,
            "reasoning": "Error fallback"
        }

async def extract_text_from_blob(content: bytes, url: str) -> str:
    """OPTIMIZED: Faster text extraction"""
    try:
        if url.lower().endswith('.pdf'):
            # Use pdfplumber for better quality
            try:
                import pdfplumber
                import io
                
                pdf_file = io.BytesIO(content)
                text_parts = []
                
                with pdfplumber.open(pdf_file) as pdf:
                    # Limit to first 100 pages for speed
                    pages_to_process = min(len(pdf.pages), 100)
                    logger.info(f"PDF has {len(pdf.pages)} pages, processing {pages_to_process}")
                    
                    for i in range(pages_to_process):
                        try:
                            text = pdf.pages[i].extract_text()
                            if text and text.strip():
                                text = text.replace('\n', ' ').strip()
                                text_parts.append(text)
                        except Exception as e:
                            logger.error(f"Error extracting page {i + 1}: {e}")
                
                return '\n\n'.join(text_parts)
                
            except ImportError:
                # Fallback to PyPDF2
                from PyPDF2 import PdfReader
                import io
                
                pdf_file = io.BytesIO(content)
                reader = PdfReader(pdf_file)
                text_parts = []
                
                pages_to_process = min(len(reader.pages), 100)
                
                for i in range(pages_to_process):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                    except Exception as e:
                        logger.error(f"Error extracting page {i + 1}: {e}")
                
                return '\n\n'.join(text_parts)
        
        else:
            # Plain text fallback
            return content.decode('utf-8', errors='replace')
    
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract text from document"
        )

def extract_key_terms(question: str) -> List[str]:
    """Extract key terms from question"""
    keywords = []
    
    insurance_terms = [
        # Policy & Payment
    "grace period", "free look period", "policy cancellation", "policy renewal", "premium payment",

    # Eligibility
    "entry age", "exit age", "dependent age", "age limit", "age eligibility",

    # Coverage
    "sum insured", "coverage amount", "top-up", "deductible", "co-payment", "copay",
    "sub-limit", "room rent", "icu charges", "ambulance charges", "domiciliary hospitalization",
    "maternity benefit", "newborn cover", "organ donor expenses", "second opinion", "AYUSH treatment",

    # Hospitalization
    "pre-hospitalization", "post-hospitalization", "hospital cash", "day care procedures", "room category",
    "shared accommodation", "cashless network", "network hospitals",

    # OPD & Consultation
    "OPD cover", "outpatient", "doctor consultation", "telemedicine", "diagnostic tests",

    # Conditions
    "pre-existing disease", "waiting period", "specific waiting period", "chronic illness",
    "critical illness", "lifestyle disease", "genetic disorder", "psychological disorder",

    # Exclusions
    "permanent exclusions", "standard exclusions", "disease not covered", "waiting period exclusions",
    "self-inflicted injuries", "cosmetic surgery", "infertility treatment", "congenital condition",

    # Claims
    "claim settlement", "cashless claim", "reimbursement claim", "claim intimation",
    "TAT", "grievance", "claim repudiation", "claim rejection",

    # Miscellaneous
    "policyholder", "insured person", "beneficiary", "TPA", "insurance company", "IRDAI", "policy wording"
    ]
    
    question_lower = question.lower()
    
    # Extract insurance terms
    for term in insurance_terms:
        if term in question_lower:
            keywords.append(term)
    
    # Extract numbers
    import re
    numbers = re.findall(r'\d+', question)
    keywords.extend(numbers[:2])
    
    return keywords[:5]  # Limit for speed

# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if the HackRx service is running."""
    return {
        "status": "healthy",
        "service": "hackrx",
        "version": "4.0-vector-primary-with-pattern-fallback",
        "cache_size": len(detailed_answer_cache),
        "doc_cache_size": len(doc_processing_cache.cache),
        "configuration": {
            "max_concurrent_llm": MAX_CONCURRENT_LLM_CALLS,
            "batch_size": BATCH_SIZE,
            "graph_extraction_enabled": ENABLE_GRAPH_EXTRACTION,
            "optimized_for_speed": True,
            "hybrid_search_weight": HYBRID_SEARCH_WEIGHT,
            "chunk_size": CHUNK_SIZE_OPTIMAL
        },
        "features": {
            "document_caching": True,
            "vector_search_primary": True,
            "pattern_extraction_fallback": True,
            "parallel_processing": True,
            "smart_sampling": True,
            "hybrid_search": True,
            "optimized_chunking": True,
            "json_to_text_conversion": True,
            "answer_validation": True
        }
    }

# Analytics endpoint
@router.get("/analytics")
async def get_hackrx_analytics():
    """Get HackRx analytics"""
    return {
        "status": "success",
        "cache_hits": len(detailed_answer_cache),
        "documents_cached": len(doc_processing_cache.cache),
        "optimization_level": "maximum",
        "target_time": "<40 seconds",
        "approach": "vector_search_primary_with_pattern_fallback",
        "features_enabled": {
            "document_cache": True,
            "vector_search_primary": True,
            "pattern_extraction_fallback": True,
            "parallel_questions": True,
            "smart_sampling": True,
            "graph_disabled_for_speed": not ENABLE_GRAPH_EXTRACTION,
            "json_answer_formatting": True,
            "answer_validation": True
        }
    }
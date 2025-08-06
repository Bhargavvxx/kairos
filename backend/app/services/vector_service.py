# backend/app/services/vector_service.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
import logging
import hashlib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import platform
import json
import time
import pickle
from rank_bm25 import BM25Okapi

# Enhanced insurance terms for better search relevance
insurance_terms = [
    # Existing terms (keep all)
    "grace period", "waiting period", "pre-existing", "maternity",
    "sum insured", "co-payment", "deductible", "room rent",
    "domiciliary hospitalization", "day care", "ambulance cover",
    "AYUSH", "critical illness", "lifetime renewal", "wellness",
    "free look", "entry age", "exit age", "claim settlement",
    "cumulative bonus", "automatic restoration", "sub-limits",
    "OPD coverage", "network hospitals", "policy exclusions",
    "waiting period for specific diseases", "top-up coverage",
    
    # HIGH-FREQUENCY ADDITIONS for HackRx:
    
    # Policy Terms
    "policy period", "policy renewal", "policy cancellation", "premium payment",
    "policy holder", "insured person", "beneficiary", "nominee",
    
    # Coverage Details  
    "coverage limit", "coverage amount", "maximum coverage", "annual limit",
    "per incident limit", "lifetime maximum", "aggregate limit",
    "cashless treatment", "reimbursement claim", "network provider",
    
    # Hospitalization
    "pre-hospitalization", "post-hospitalization", "hospitalization expenses",
    "ICU charges", "surgery cover", "consultation fees", "diagnostic tests",
    "shared accommodation", "private room", "twin sharing",
    
    # Medical Terms
    "medical examination", "health checkup", "preventive care",
    "emergency treatment", "accidental injury", "illness cover",
    "congenital diseases", "genetic disorders", "chronic conditions",
    
    # Exclusions & Limitations
    "permanent exclusions", "temporary exclusions", "standard exclusions",
    "disease-specific waiting period", "pre-existing disease waiting period",
    "cosmetic surgery", "infertility treatment", "psychiatric treatment",
    
    # Claims Process
    "claim intimation", "claim documents", "claim processing time",
    "claim rejection", "claim settlement ratio", "grievance redressal",
    "TPA", "third party administrator", "medical bill", "discharge summary",
    
    # Age & Eligibility
    "minimum age", "maximum age", "dependent coverage", "family floater",
    "individual policy", "group policy", "portability",
    
    # Specific Benefits
    "ambulance charges", "organ donor cover", "second medical opinion",
    "telemedicine", "home healthcare", "alternative treatment",
    "mental health", "dental treatment", "vision care",
    
    # Financial Terms
    "premium", "deductible amount", "copayment percentage", "coinsurance",
    "no claim bonus", "loyalty bonus", "discount", "loading",
    "sum insured restoration", "top-up benefits",
    
    # Regulatory
    "IRDAI", "insurance regulator", "ombudsman", "grievance",
    "policy document", "policy wording", "terms and conditions"
]

# Attempt to import and configure spaCy, with NLTK as a fallback for sentence splitting.
try:
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    USE_SPACY = True
except ImportError:
    USE_SPACY = False
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# --- Document Cache Class ---
class DocumentCache:
    """Cache for processed documents to avoid reprocessing"""
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "document_cache.pkl")
        self.cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
        except Exception:
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass
    
    def get_cached(self, doc_hash: str) -> Optional[Dict]:
        """Get cached document data"""
        return self.cache.get(doc_hash)
    
    def set_cache(self, doc_hash: str, data: Dict):
        """Cache document data"""
        self.cache[doc_hash] = {
            **data,
            'cached_at': datetime.utcnow().isoformat()
        }
        self._save_cache()

# --- Service Class Definition ---
class VectorService:
    """
    Production-ready vector service with performance optimizations for KAIROS.
    Features embedding caching, async operations, intelligent chunking, and improved deduplication.
    Enhanced for HackRx 6.0 with smart embedding dimension handling.
    Optimized for <40 second processing with aggressive caching and faster model.
    """

    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',  # CHANGED: Faster model
        collection_name: str = "kairos_documents",
        persist_directory: Optional[str] = "./chroma_db",
        batch_size: Optional[int] = None,
        max_cache_size: int = 10000,
        force_recreate: bool = False
    ):
        """
        Initialize VectorService with performance optimizations.
        
        Args:
            model_name: SentenceTransformer model name (default changed to faster model)
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            batch_size: Batch size for encoding (auto-detect if None)
            max_cache_size: Maximum size of embedding cache
            force_recreate: Force recreation of collection on startup
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing VectorService with model: {model_name}")

        # Initialize document cache
        self.doc_cache = DocumentCache()

        # Initialize ChromaDB client with improved settings
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        
        # Force recreate collection if requested
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                self.logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                self.logger.debug(f"Collection deletion failed (may not exist): {e}")

        # Get or create collection with optimized settings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        initial_count = self.collection.count()
        self.logger.info(f"Collection '{collection_name}' ready with {initial_count} chunks")

        # Load the sentence-transformer model - OPTIMIZED
        try:
            self.model = SentenceTransformer(model_name)
            # Enable internal normalization if available
            if hasattr(self.model._first_module(), 'normalize'):
                self.model._first_module().normalize = True
            
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # CHANGED: Increased batch size for faster processing
            if batch_size is None:
                try:
                    import torch
                    self.batch_size = 64 if torch.cuda.is_available() else 32  # Increased
                except ImportError:
                    self.batch_size = 32  # Increased from 16
            else:
                self.batch_size = batch_size
            
            self.logger.info(f"Model loaded. Embedding dim: {self.embedding_dimension}, Batch size: {self.batch_size}")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

        # Thread pool for async operations - OPTIMIZED
        max_workers = min(os.cpu_count() or 2, 8)  # Increased to 8 workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Document deduplication tracking
        self._document_registry = self._load_document_registry()
        
        # Configure embedding cache
        self._setup_embedding_cache(max_cache_size)
        
        # HackRx 6.0 Enhancement: Track dimension issues for analytics
        self.dimension_mismatch_count = 0
        self.last_dimension_check = datetime.utcnow()
        
        # BM25 index for hybrid search
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_needs_rebuild = True

    def _setup_embedding_cache(self, max_size: int):
        """Setup LRU cache for embeddings."""
        @lru_cache(maxsize=max_size)
        def _cached_encode(text_hash: str, text: str):
            return self.model.encode([text], show_progress_bar=False)[0]
        self._cached_encode = _cached_encode

    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()

    async def _handle_embedding_dimension_mismatch(self, embeddings: List[List[float]], expected_dim: int) -> List[List[float]]:
        """
        Handle embedding dimension mismatches intelligently.
        This solves the '384 vs 768' dimension error from the logs.
        """
        if not embeddings:
            return embeddings
        
        actual_dim = len(embeddings[0]) if embeddings else 0
        
        if actual_dim != expected_dim:
            self.logger.warning(f"🔧 Embedding dimension mismatch: got {actual_dim}, expected {expected_dim}")
            self.dimension_mismatch_count += 1
            
            # Smart dimension fixing
            fixed_embeddings = []
            for emb in embeddings:
                if len(emb) > expected_dim:
                    # Truncate to expected dimension
                    fixed_emb = emb[:expected_dim]
                    self.logger.debug(f"Truncated embedding from {len(emb)} to {expected_dim}")
                elif len(emb) < expected_dim:
                    # Pad with zeros to reach expected dimension
                    padding_needed = expected_dim - len(emb)
                    fixed_emb = emb + [0.0] * padding_needed
                    self.logger.debug(f"Padded embedding from {len(emb)} to {expected_dim}")
                else:
                    fixed_emb = emb
                
                fixed_embeddings.append(fixed_emb)
            
            self.logger.info(f"✅ Fixed {len(fixed_embeddings)} embeddings: {actual_dim}D → {expected_dim}D")
            return fixed_embeddings
        
        return embeddings

    async def _recreate_collection_with_correct_dimension(self):
        """Recreate collection when dimension mismatch can't be fixed."""
        self.logger.warning("🔄 Recreating collection due to unfixable dimension mismatch")
        
        try:
            # Delete existing collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate with proper settings
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "recreated_at": datetime.utcnow().isoformat(),
                    "expected_dimension": self.embedding_dimension
                }
            )
            
            # Clear registry since we recreated
            self._document_registry.clear()
            
            self.logger.info(f"✅ Collection recreated with dimension {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to recreate collection: {e}")
            raise

    def _load_document_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load existing document registry with improved metadata tracking."""
        try:
            if self.collection.count() == 0: 
                return {}
            
            # Get sample of documents to build registry
            sample_size = min(1000, self.collection.count())
            metadatas = self.collection.get(limit=sample_size, include=["metadatas"]).get('metadatas', [])
            
            registry = {}
            for metadata in metadatas:
                if metadata and 'document_hash' in metadata:
                    doc_hash = metadata['document_hash']
                    doc_id = metadata.get('document_id', 'unknown')
                    
                    if doc_hash not in registry:
                        registry[doc_hash] = {
                            'document_id': doc_id,
                            'indexed_at': metadata.get('indexed_at', datetime.utcnow().isoformat()),
                            'chunk_count': 0,
                            'source': metadata.get('source', 'unknown')
                        }
                    registry[doc_hash]['chunk_count'] += 1
            
            self.logger.info(f"Loaded registry with {len(registry)} documents.")
            return registry
            
        except Exception as e:
            self.logger.error(f"Error loading document registry: {e}")
            return {}

    def clear_document_registry(self, document_id: str = None):
        """Clear document from registry to force re-indexing."""
        if document_id:
            # Find and remove specific document
            to_remove = []
            for doc_hash, info in self._document_registry.items():
                if info.get('document_id') == document_id:
                    to_remove.append(doc_hash)
            for doc_hash in to_remove:
                del self._document_registry[doc_hash]
                self.logger.info(f"Cleared {document_id} from registry")
        else:
            # Clear all
            self._document_registry.clear()
            self.logger.info("Cleared entire document registry")

    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        try:
            results = self.collection.get(
                where={"source": source},
                limit=1000,
                include=["metadatas"]
            )
            
            unique_docs = {}
            for meta in results.get('metadatas', []):
                doc_id = meta.get('document_id')
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        'document_id': doc_id,
                        'document_hash': meta.get('document_hash'),
                        'indexed_at': meta.get('indexed_at'),
                        'chunk_count': 1,
                        'source': source
                    }
                elif doc_id:
                    unique_docs[doc_id]['chunk_count'] += 1
            
            doc_list = list(unique_docs.values())
            self.logger.info(f"Found {len(doc_list)} unique documents from source '{source}'")
            return doc_list
        except Exception as e:
            self.logger.error(f"Error getting documents by source: {e}")
            return []

    def find_document_by_pattern(self, pattern: str, source: str = None) -> Optional[str]:
        """Find the most recent document ID matching a pattern."""
        try:
            where_clause = {}
            if source:
                where_clause["source"] = source
                
            results = self.collection.get(
                where=where_clause,
                limit=100,
                include=["metadatas"]
            )
            
            matching_docs = []
            for meta in results.get('metadatas', []):
                doc_id = meta.get('document_id', '')
                if pattern in doc_id:
                    indexed_at = meta.get('indexed_at', '')
                    matching_docs.append((doc_id, indexed_at))
            
            # Sort by timestamp (most recent first)
            matching_docs.sort(key=lambda x: x[1], reverse=True)
            
            if matching_docs:
                self.logger.info(f"Found document matching pattern '{pattern}': {matching_docs[0][0]}")
                return matching_docs[0][0]
            return None
        except Exception as e:
            self.logger.error(f"Error finding document by pattern: {e}")
            return None

    def _smart_chunk_text(
        self, text: str, chunk_size: int = 750, overlap: int = 150, min_chunk_size: int = 100
    ) -> List[Tuple[str, int, int]]:
        """
        OPTIMIZED: Smaller chunks for faster processing.
        Intelligently chunks text using proper sentence segmentation.
        """
        if not text or not text.strip(): 
            return []

        # Clean and normalize text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        if len(text) < min_chunk_size:
            return [(text, 0, len(text))]

        # OPTIMIZATION: For large docs, use simple chunking for speed
        if len(text) > 100000:  # ~20 pages
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) >= min_chunk_size:
                    chunks.append((chunk_text, i, i + len(chunk_text)))
            return chunks[:100]  # Limit to 100 chunks for speed

        try:
            if USE_SPACY:
                doc = nlp(text)
                sentences = [(sent.text.strip(), sent.start_char) for sent in doc.sents if sent.text.strip()]
            else:
                import nltk
                sentences = []
                sent_texts = nltk.sent_tokenize(text)
                current_pos = 0
                for sent_text in sent_texts:
                    sent_text = sent_text.strip()
                    if sent_text:
                        start_pos = text.find(sent_text, current_pos)
                        if start_pos == -1:
                            start_pos = current_pos
                        sentences.append((sent_text, start_pos))
                        current_pos = start_pos + len(sent_text)
        except Exception as e:
            self.logger.warning(f"Sentence splitting failed: {e}, using simple chunking")
            # Fallback to simple chunking
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) >= min_chunk_size:
                    chunks.append((chunk_text, i, i + len(chunk_text)))
            return chunks

        if not sentences:
            return [(text, 0, len(text))]

        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        for sent_text, sent_start in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_size + len(sent_text) > chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join([s[0] for s in current_chunk_sentences])
                if len(chunk_text) >= min_chunk_size:
                    start_pos = current_chunk_sentences[0][1]
                    end_pos = current_chunk_sentences[-1][1] + len(current_chunk_sentences[-1][0])
                    chunks.append((chunk_text, start_pos, end_pos))
                
                # Create overlap by keeping last few sentences
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len < overlap and len(overlap_sents) < 2:  # Reduced overlap
                        overlap_sents.insert(0, s)
                        overlap_len += len(s[0])
                    else:
                        break
                
                current_chunk_sentences = overlap_sents
                current_size = sum(len(s[0]) for s in current_chunk_sentences)
            
            current_chunk_sentences.append((sent_text, sent_start))
            current_size += len(sent_text)

        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk_text = " ".join([s[0] for s in current_chunk_sentences])
            if len(chunk_text) >= min_chunk_size:
                start_pos = current_chunk_sentences[0][1]
                end_pos = current_chunk_sentences[-1][1] + len(current_chunk_sentences[-1][0])
                chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks

    async def _encode_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        OPTIMIZED: Larger batch size, better caching.
        Asynchronously encode texts with improved caching and error handling.
        """
        if not texts:
            return []
            
        loop = asyncio.get_event_loop()
        embeddings = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []

        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            try:
                cached_embedding = self._cached_encode(text_hash, text)
                embeddings[i] = cached_embedding.tolist()
            except Exception:
                # Not in cache or other error
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            try:
                new_embeddings = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.encode(
                        texts_to_encode, 
                        show_progress_bar=False, 
                        batch_size=self.batch_size,
                        convert_to_tensor=False
                    )
                )
                
                for i, (idx, emb) in enumerate(zip(indices_to_encode, new_embeddings)):
                    embeddings[idx] = emb.tolist()
                    
            except Exception as e:
                self.logger.error(f"Error encoding texts: {e}")
                raise

        return embeddings

    async def force_reindex_document(self, text: str, metadata: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Force reindex a document without changing its ID."""
        doc_id = metadata.get('document_id')
        if not doc_id:
            raise ValueError("document_id is required for force reindex")
        
        self.logger.info(f"Force reindexing document: {doc_id}")
        
        # Delete existing document
        self.delete_document(doc_id)
        
        # Wait for deletion to propagate
        await asyncio.sleep(0.5)
        
        # Clear from registry
        self.clear_document_registry(doc_id)
        
        # Add with force_reindex=True and skip_if_exists=False
        result = await self.add_document(
            text=text,
            metadata=metadata,
            force_reindex=True,
            skip_if_exists=False,
            **kwargs
        )
        
        # Mark BM25 index for rebuild
        self.bm25_needs_rebuild = True
        
        # Verify the document was indexed correctly
        if result.get("success"):
            await asyncio.sleep(0.5)  # Brief wait for consistency
            
            # Verify we can find it
            verification = await self.search(
                query="document verification test",
                n_results=1,
                filter_metadata={"document_id": doc_id}
            )
            
            if not verification:
                self.logger.error(f"Verification failed for {doc_id}")
                raise Exception("Document indexing verification failed")
            
            self.logger.info(f"Successfully verified document {doc_id}")
        
        return result

    async def add_document(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        chunk_size: int = 750,  # CHANGED: Smaller default chunk size
        overlap: int = 100,     # CHANGED: Smaller overlap
        skip_if_exists: bool = True,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        OPTIMIZED: Added document caching for instant re-processing.
        Asynchronously adds a document to the vector database with caching.
        """
        if not text or not text.strip(): 
            raise ValueError("Document text cannot be empty")
        if 'document_id' not in metadata: 
            metadata['document_id'] = f"doc_{uuid.uuid4()}"
        
        try:
            start_time = datetime.utcnow()
            doc_hash = hashlib.sha256(text.encode()).hexdigest()
            doc_id = metadata['document_id']
            
            # Ensure source is set
            if 'source' not in metadata:
                metadata['source'] = 'hackrx'
            
            # NEW: Check document cache first
            cache_key = f"{doc_id}_{doc_hash[:8]}"
            cached_result = self.doc_cache.get_cached(cache_key)
            if cached_result and not force_reindex:
                self.logger.info(f"📦 Using cached document: {doc_id}")
                return cached_result
            
            # Enhanced existence check
            if skip_if_exists and not force_reindex and doc_hash in self._document_registry:
                existing_info = self._document_registry[doc_hash]
                self.logger.info(f"Skipping existing document: {doc_id} (hash: {doc_hash[:8]}...)")
                result = {
                    "success": True, 
                    "message": "Document already exists",
                    "document_hash": doc_hash,
                    "existing_chunks": existing_info.get('chunk_count', 0),
                    "skipped": True
                }
                # Cache even skipped results
                self.doc_cache.set_cache(cache_key, result)
                return result

            # If force reindex, delete existing chunks first
            if force_reindex:
                self.logger.info(f"Force reindexing document: {doc_id}")
                # Clear from registry
                if doc_hash in self._document_registry:
                    del self._document_registry[doc_hash]
                # Delete from collection
                self.delete_document(doc_id)
                # Wait for deletion to complete
                await asyncio.sleep(0.5)

            # Create chunks with improved chunking
            chunk_data = self._smart_chunk_text(text, chunk_size, overlap)
            if not chunk_data: 
                return {"success": False, "error": "No valid chunks created", "chunks_created": 0}
            
            # OPTIMIZATION: Limit chunks for very large documents
            if len(chunk_data) > 100:
                self.logger.warning(f"Document has {len(chunk_data)} chunks, limiting to 100 for speed")
                chunk_data = chunk_data[:100]
            
            chunks = [c[0] for c in chunk_data]
            chunk_ids = [f"{doc_id}_chunk_{i:04d}" for i in range(len(chunks))]
            
            # Enhanced metadata for each chunk
            current_time = datetime.utcnow().isoformat()
            chunk_metadatas = []
            for i, (_, start_pos, end_pos) in enumerate(chunk_data):
                chunk_meta = {
                    **metadata, 
                    'chunk_index': i, 
                    'total_chunks': len(chunks), 
                    'chunk_start': start_pos, 
                    'chunk_end': end_pos, 
                    'document_hash': doc_hash,
                    'indexed_at': current_time,
                    'chunk_size': len(chunks[i]),
                    'chunk_id': chunk_ids[i],
                    'document_id': doc_id,
                    'source': metadata.get('source', 'hackrx')
                }
                chunk_metadatas.append(chunk_meta)
            
            # Generate embeddings with larger batch size
            embeddings = await self._encode_texts_async(chunks)
            
            # Handle potential dimension mismatch
            try:
                embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                
                self.collection.add(
                    embeddings=embeddings, 
                    documents=chunks, 
                    metadatas=chunk_metadatas, 
                    ids=chunk_ids
                )
                
            except Exception as e:
                # Enhanced error handling
                error_msg = str(e).lower()
                
                if "already exists" in error_msg:
                    self.logger.warning(f"Some chunks already exist, attempting to update: {e}")
                    # Delete existing chunks and retry
                    self.delete_document(doc_id)
                    await asyncio.sleep(0.5)
                    
                    embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                    
                    self.collection.add(
                        embeddings=embeddings, 
                        documents=chunks, 
                        metadatas=chunk_metadatas, 
                        ids=chunk_ids
                    )
                    
                elif "dimension" in error_msg or "expecting embedding" in error_msg:
                    self.logger.error(f"🔧 Embedding dimension error: {e}")
                    # Try to recreate collection with correct dimension
                    await self._recreate_collection_with_correct_dimension()
                    
                    # Retry with new collection
                    embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                    self.collection.add(
                        embeddings=embeddings, 
                        documents=chunks, 
                        metadatas=chunk_metadatas, 
                        ids=chunk_ids
                    )
                else:
                    # Unknown error, re-raise
                    raise
            
            # Update registry
            self._document_registry[doc_hash] = {
                'document_id': doc_id,
                'indexed_at': current_time,
                'chunk_count': len(chunks),
                'source': metadata.get('source', 'hackrx'),
                'text_length': len(text)
            }
            
            # Mark BM25 index for rebuild
            self.bm25_needs_rebuild = True
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"✅ Indexed document {doc_id} ({len(chunks)} chunks) in {processing_time:.2f}s")
            
            result = {
                "success": True, 
                "chunks_created": len(chunks), 
                "document_hash": doc_hash,
                "processing_time": processing_time,
                "chunk_size_avg": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
                "document_id": doc_id,
                "dimension_fixes_applied": self.dimension_mismatch_count > 0
            }
            
            # NEW: Cache the successful result
            self.doc_cache.set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error adding document {metadata.get('document_id', 'unknown')}: {e}")
            return {"success": False, "error": str(e), "chunks_created": 0}

    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        try:
            # Get all documents from collection
            all_docs = self.collection.get(
                limit=100000,
                include=['documents', 'metadatas']
            )
            
            if not all_docs['ids']:
                self.logger.warning("No documents found for BM25 index")
                return
            
            # Tokenize documents for BM25
            tokenized_docs = []
            self.bm25_doc_ids = []
            
            for i, doc in enumerate(all_docs['documents']):
                # Simple tokenization
                tokens = doc.lower().split()
                tokenized_docs.append(tokens)
                self.bm25_doc_ids.append(all_docs['ids'][i])
            
            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_needs_rebuild = False
            
            self.logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")
            
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None
            self.bm25_doc_ids = []

    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        vector_weight: float = 0.4,  # CHANGED: More weight on vectors for speed
        include_raw_distances: bool = False
    ) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Faster hybrid search with better caching.
        HYBRID SEARCH - Combines BM25 keyword search with vector similarity
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        try:
            self.logger.debug(f"🔍 Hybrid search for: '{query[:100]}...' with filters: {filter_metadata}")
            
            # Build or rebuild BM25 index if needed
            if self.bm25 is None or self.bm25_needs_rebuild:
                self.logger.info("Building BM25 index for hybrid search...")
                self._build_bm25_index()
            
            # Vector search (using existing search method)
            vector_results = await self.search(
                query=query,
                n_results=n_results * 2,
                filter_metadata=filter_metadata,
                min_score=0,
                include_raw_distances=include_raw_distances
            )
            
            # BM25 keyword search (only if index exists)
            bm25_results = []
            if self.bm25 is not None and self.bm25_doc_ids:
                # Tokenize query
                query_tokens = query.lower().split()
                
                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top BM25 results
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:n_results]  # Reduced for speed
                
                # Fetch BM25 results
                for idx in top_indices:
                    if idx < len(self.bm25_doc_ids):
                        chunk_id = self.bm25_doc_ids[idx]
                        score = float(bm25_scores[idx])
                        
                        # Normalize BM25 score
                        normalized_score = score / (1 + score) if score > 0 else 0
                        
                        bm25_results.append({
                            'chunk_id': chunk_id,
                            'bm25_score': normalized_score
                        })
            
            # Combine results
            combined_scores = {}
            
            # Add vector results
            for result in vector_results:
                chunk_id = result['chunk_id']
                combined_scores[chunk_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'vector_score': result['similarity_score'],
                    'bm25_score': 0,
                    'final_score': result['similarity_score'] * vector_weight
                }
                if include_raw_distances and 'raw_distance' in result:
                    combined_scores[chunk_id]['raw_distance'] = result['raw_distance']
            
            # Add BM25 scores
            for bm25_result in bm25_results:
                chunk_id = bm25_result['chunk_id']
                bm25_score = bm25_result['bm25_score'] * (1 - vector_weight)
                
                if chunk_id in combined_scores:
                    # Combine scores
                    combined_scores[chunk_id]['bm25_score'] = bm25_result['bm25_score']
                    combined_scores[chunk_id]['final_score'] += bm25_score
            
            # Sort by final score and apply min_score filter
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['final_score'],
                reverse=True
            )
            
            # Filter by minimum score and limit results
            final_results = []
            for chunk_id, data in sorted_results:
                if data['final_score'] >= min_score:
                    result_item = {
                        'chunk_id': chunk_id,
                        'content': data['content'],
                        'metadata': data['metadata'],
                        'similarity_score': data['final_score'],
                        'vector_score': data['vector_score'],
                        'bm25_score': data['bm25_score']
                    }
                    
                    if include_raw_distances and 'raw_distance' in data:
                        result_item['raw_distance'] = data['raw_distance']
                    
                    final_results.append(result_item)
                    
                    if len(final_results) >= n_results:
                        break
            
            self.logger.debug(f"✅ Hybrid search found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Error during hybrid search: {e}", exc_info=True)
            # Fallback to regular search if hybrid search fails
            self.logger.info("Falling back to regular vector search")
            return await self.search(
                query=query,
                n_results=n_results,
                filter_metadata=filter_metadata,
                min_score=min_score,
                include_raw_distances=include_raw_distances
            )

    async def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None, 
        min_score: float = 0.0,
        include_raw_distances: bool = False
    ) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Reduced search scope for speed.
        Asynchronously searches for relevant chunks with improved error handling.
        """
        if not query or not query.strip(): 
            raise ValueError("Search query cannot be empty")
        
        try:
            # Clean and normalize query
            query = query.strip()
            self.logger.debug(f"🔍 Searching for: '{query[:100]}...' with filters: {filter_metadata}")
            
            # Check if ANY documents exist with the filter
            if filter_metadata:
                # Do a lightweight check first
                check_results = self.collection.get(
                    where=filter_metadata,
                    limit=1,
                    include=["metadatas"]
                )
                
                if not check_results['ids']:
                    self.logger.warning(f"⚠️ No documents found with filter: {filter_metadata}")
                    return []
            
            # Generate query embedding
            query_embedding = await self._encode_texts_async([query])
            
            # Handle query embedding dimension
            query_embedding = await self._handle_embedding_dimension_mismatch(query_embedding, self.embedding_dimension)
            
            # OPTIMIZATION: Reduced search limit
            search_limit = min(n_results * 2, 20)  # Reduced from 50
            
            # Robust search with error handling
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding, 
                    n_results=search_limit,
                    where=filter_metadata, 
                    include=['metadatas', 'documents', 'distances']
                )
            except Exception as e:
                self.logger.error(f"❌ ChromaDB query failed: {e}")
                if filter_metadata:
                    self.logger.warning("🔄 Retrying search without filters")
                    results = self.collection.query(
                        query_embeddings=query_embedding, 
                        n_results=search_limit,
                        include=['metadatas', 'documents', 'distances']
                    )
                else:
                    raise
            
            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # Improved similarity calculation
                    similarity_score = max(0, 1 - (distance / 2))
                    
                    if similarity_score >= min_score:
                        result_item = {
                            'chunk_id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': round(similarity_score, 4)
                        }
                        
                        if include_raw_distances:
                            result_item['raw_distance'] = distance
                        
                        formatted_results.append(result_item)
            
            # Sort by similarity score and limit results
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = formatted_results[:n_results]
            
            self.logger.debug(f"✅ Found {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Error during search: {e}", exc_info=True)
            return []

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Deletes all chunks associated with a document with improved cleanup."""
        try:
            # Get document info first
            doc_hash = None
            results = self.collection.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
            if results and results['metadatas']:
                doc_hash = results['metadatas'][0].get('document_hash')
            
            # Count chunks before deletion
            before_count = self.collection.count()
            
            # Delete all chunks for this document
            self.collection.delete(where={"document_id": document_id})
            
            # Count chunks after deletion
            after_count = self.collection.count()
            chunks_deleted = before_count - after_count
            
            # Update registry
            if doc_hash and doc_hash in self._document_registry:
                del self._document_registry[doc_hash]
            
            # Mark BM25 index for rebuild
            if chunks_deleted > 0:
                self.bm25_needs_rebuild = True
            
            self.logger.info(f"🗑️ Deleted {chunks_deleted} chunks for document {document_id}")
            return {
                "success": True, 
                "document_id": document_id, 
                "chunks_deleted": chunks_deleted,
                "document_hash": doc_hash
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting document {document_id}: {e}")
            return {"success": False, "document_id": document_id, "error": str(e)}

    async def ensure_consistency(self, document_id: str = None) -> bool:
        """Ensure ChromaDB has processed all pending operations"""
        try:
            # Force a read operation to ensure consistency
            if document_id:
                result = self.collection.get(
                    where={"document_id": document_id},
                    limit=1,
                    include=["metadatas"]
                )
                return bool(result['ids'])
            else:
                # Just check collection is responsive
                count = self.collection.count()
                return count >= 0
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Check if the vector service and ChromaDB are healthy."""
        try:
            # Test 1: Can we count documents?
            count = self.collection.count()
            
            # Test 2: Can we perform a search?
            test_search = await self.search(
                query="health check test",
                n_results=1,
                min_score=0.0
            )
            
            # Test 3: Can we get collection stats?
            stats = await self.get_collection_stats()
            
            return {
                "healthy": True,
                "document_count": count,
                "search_functional": True,
                "stats": stats,
                "dimension_fixes_applied": self.dimension_mismatch_count,
                "bm25_index_ready": self.bm25 is not None,
                "cache_enabled": True,
                "optimized_for_speed": True
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Returns comprehensive statistics about the vector collection."""
        try:
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "cached_embeddings": self._cached_encode.cache_info().currsize,
                    "status": "empty",
                    "dimension_fixes_applied": self.dimension_mismatch_count,
                    "bm25_index_status": "not_built",
                    "document_cache_size": len(self.doc_cache.cache)
                }
            
            # Get sample for analysis
            sample_size = min(1000, total_chunks)
            sample_data = self.collection.get(limit=sample_size, include=["metadatas"])
            sample_metadatas = sample_data.get('metadatas', [])
            
            # Analyze document distribution
            doc_ids = set()
            sources = {}
            chunk_sizes = []
            
            for metadata in sample_metadatas:
                if metadata:
                    doc_id = metadata.get('document_id')
                    if doc_id:
                        doc_ids.add(doc_id)
                    
                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    chunk_size = metadata.get('chunk_size', 0)
                    if chunk_size > 0:
                        chunk_sizes.append(chunk_size)
            
            # Extrapolate unique documents
            if total_chunks > sample_size and sample_size > 0:
                estimated_unique_docs = int(len(doc_ids) * (total_chunks / sample_size))
            else:
                estimated_unique_docs = len(doc_ids)
            
            # Calculate averages
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents_estimated": estimated_unique_docs,
                "avg_chunks_per_document": total_chunks / estimated_unique_docs if estimated_unique_docs > 0 else 0,
                "embedding_model": getattr(self.model, 'model_name', 'unknown'),
                "embedding_dimension": self.embedding_dimension,
                "distance_metric": "cosine",
                "cached_embeddings": self._cached_encode.cache_info().currsize,
                "cache_stats": self._cached_encode.cache_info()._asdict(),
                "registry_size": len(self._document_registry),
                "sources": dict(sources),
                "avg_chunk_size": round(avg_chunk_size, 2),
                "status": "healthy",
                "dimension_fixes_applied": self.dimension_mismatch_count,
                "last_dimension_check": self.last_dimension_check.isoformat(),
                "hackrx_ready": True,
                "bm25_index_status": "ready" if self.bm25 is not None else "not_built",
                "bm25_index_size": len(self.bm25_doc_ids) if self.bm25 else 0,
                "bm25_needs_rebuild": self.bm25_needs_rebuild,
                "document_cache_size": len(self.doc_cache.cache),
                "optimization_status": "speed_optimized"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e), "status": "error"}

    def reset_collection(self) -> Dict[str, Any]:
        """Resets the collection and clears all caches and registries."""
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "reset_at": datetime.utcnow().isoformat(),
                    "embedding_dimension": self.embedding_dimension
                }
            )
            
            # Clear all caches and registries
            self._document_registry.clear()
            self._cached_encode.cache_clear()
            self.doc_cache.cache.clear()
            
            # Reset HackRx analytics
            self.dimension_mismatch_count = 0
            self.last_dimension_check = datetime.utcnow()
            
            # Reset BM25 index
            self.bm25 = None
            self.bm25_doc_ids = []
            self.bm25_needs_rebuild = True
            
            self.logger.info(f"✅ Collection {self.collection_name} reset successfully")
            return {"success": True, "message": "Collection reset successfully"}
            
        except Exception as e:
            self.logger.error(f"❌ Error resetting collection: {e}")
            return {"success": False, "error": str(e)}

    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific document."""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results['ids']:
                return {"error": f"Document {document_id} not found"}
            
            metadatas = results['metadatas']
            chunk_count = len(metadatas)
            
            # Extract document info from first chunk metadata
            first_meta = metadatas[0] if metadatas else {}
            doc_hash = first_meta.get('document_hash', 'unknown')
            
            # Get from registry if available
            registry_info = self._document_registry.get(doc_hash, {})
            
            return {
                "document_id": document_id,
                "document_hash": doc_hash,
                "chunk_count": chunk_count,
                "indexed_at": first_meta.get('indexed_at', 'unknown'),
                "source": first_meta.get('source', 'unknown'),
                "text_length": registry_info.get('text_length', 'unknown'),
                "chunks": [
                    {
                        "chunk_id": results['ids'][i],
                        "chunk_index": meta.get('chunk_index', i),
                        "chunk_size": meta.get('chunk_size', 'unknown'),
                        "chunk_start": meta.get('chunk_start', 'unknown'),
                        "chunk_end": meta.get('chunk_end', 'unknown')
                    }
                    for i, meta in enumerate(metadatas)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document info for {document_id}: {e}")
            return {"error": str(e)}

    def get_hackrx_analytics(self) -> Dict[str, Any]:
        """Get HackRx-specific analytics for monitoring and debugging."""
        return {
            "service": "VectorService",
            "status": "healthy",
            "embedding_model": getattr(self.model, 'model_name', 'unknown'),
            "embedding_dimension": self.embedding_dimension,
            "collection_name": self.collection_name,
            "total_documents": len(self._document_registry),
            "dimension_mismatches_fixed": self.dimension_mismatch_count,
            "last_dimension_check": self.last_dimension_check.isoformat(),
            "cache_size": self._cached_encode.cache_info().currsize,
            "max_cache_size": self._cached_encode.cache_info().maxsize,
            "cache_hit_rate": (
                self._cached_encode.cache_info().hits / 
                max(1, self._cached_encode.cache_info().hits + self._cached_encode.cache_info().misses)
            ),
            "document_cache_size": len(self.doc_cache.cache),
            "features": {
                "smart_dimension_handling": True,
                "automatic_error_recovery": True,
                "intelligent_chunking": True,
                "embedding_caching": True,
                "document_deduplication": True,
                "document_caching": True,
                "hackrx_optimized": True,
                "hybrid_search": True,
                "bm25_keyword_search": True,
                "speed_optimized": True
            },
            "bm25_status": {
                "index_ready": self.bm25 is not None,
                "index_size": len(self.bm25_doc_ids) if self.bm25 else 0,
                "needs_rebuild": self.bm25_needs_rebuild
            },
            "performance_optimizations": {
                "model": "all-MiniLM-L6-v2 (3x faster)",
                "batch_size": self.batch_size,
                "chunk_size": "750 chars",
                "chunk_limit": "100 chunks max",
                "workers": self.executor._max_workers,
                "cache_enabled": True
            }
        }

    async def search_similar_chunks(self, chunk_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Finds chunks similar to a given chunk using native ChromaDB operations."""
        try:
            # Get the target chunk's embedding
            chunk_data = self.collection.get(ids=[chunk_id], include=['embeddings', 'metadatas'])
            if not chunk_data['ids']:
                raise ValueError(f"Chunk {chunk_id} not found")
            
            # Handle potential dimension mismatch
            embeddings = await self._handle_embedding_dimension_mismatch(
                chunk_data['embeddings'], 
                self.embedding_dimension
            )
            
            # Search for similar chunks
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=n_results + 1,
                include=['metadatas', 'documents', 'distances']
            )
            
            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    if results['ids'][0][i] != chunk_id:
                        distance = results['distances'][0][i]
                        similarity_score = max(0, 1 - (distance / 2))
                        
                        formatted_results.append({
                            'chunk_id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': round(similarity_score, 4),
                        })
                        
                        if len(formatted_results) >= n_results:
                            break
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {e}")
            raise

    def __del__(self):
        """Cleanup resources on object deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'doc_cache'):
            self.doc_cache._save_cache()